using System.Diagnostics;
using System.Net.Http.Headers;
using System.Text.Json;
using HawaqmAI.Api.Configuration;
using HawaqmAI.Api.Models;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Options;
using Serilog;

namespace HawaqmAI.Api.Services;

/// <summary>
/// Calls the AQMS Web API, forwarding the user's JWT Bearer token.
/// Flattens the device-per-site response into summary rows for the LLM formatter.
/// Resolves station names to IDs via <see cref="IStationResolverService"/> when needed.
/// </summary>
public sealed class ExternalApiService : IExternalApiService
{
    private static readonly Serilog.ILogger _log = Log.ForContext<ExternalApiService>();
    private readonly HttpClient _http;
    private readonly AqmsApiOptions _options;
    private readonly IStationResolverService _stationResolver;
    private readonly ISqlExecutorService _sqlExecutor;
    private readonly IHttpContextAccessor _httpContextAccessor;
    private readonly ISiteDataCacheService _siteDataCache;

    public ExternalApiService(
        IHttpClientFactory httpClientFactory,
        IOptions<AqmsApiOptions> options,
        IStationResolverService stationResolver,
        ISqlExecutorService sqlExecutor,
        IHttpContextAccessor httpContextAccessor,
        ISiteDataCacheService siteDataCache)
    {
        _options = options.Value;
        _http = httpClientFactory.CreateClient("AqmsApi");
        _stationResolver = stationResolver;
        _sqlExecutor = sqlExecutor;
        _httpContextAccessor = httpContextAccessor;
        _siteDataCache = siteDataCache;
    }

    /// <inheritdoc/>
    public async Task<ApiCallResult> CallAsync(
        ApiCallDefinition apiCall,
        Dictionary<string, string> llmParams,
        ResolvedScope scope,
        string bearerToken,
        UserContext? user = null,
        string? templateId = null,
        CancellationToken ct = default)
    {
        var sw = Stopwatch.StartNew();
        try
        {
            // ── Single-site lookup via GetAllSiteData (name-only, no IDs) ────────
            // Site queries always resolve by station name — numeric IDs are never
            // accepted from users or injected from scope. This ensures users always
            // get the site they named and IDs are never exposed in responses.
            var nameToResolve = llmParams.GetValueOrDefault("stationName")
                             ?? llmParams.GetValueOrDefault("siteName")
                             ?? llmParams.GetValueOrDefault("station");

            if (apiCall.Path.Contains("{siteId}", StringComparison.OrdinalIgnoreCase))
            {
                if (string.IsNullOrWhiteSpace(nameToResolve))
                {
                    return new ApiCallResult
                    {
                        Success = false,
                        Error = "Please specify a site name to get AQI data (e.g. \"Abu Dhabi Residential\" or \"Al Saad Indian School\").",
                        ExecutionTimeMs = sw.ElapsedMilliseconds
                    };
                }

                // Use cached GetAllSiteData — refreshed every 5 min by SiteDataCacheService.
                // Falls back to a live API call only when the cache is empty (first-startup race).
                var allJson = _siteDataCache.GetRawJson();
                if (allJson is null)
                {
                    _log.Warning("ExternalApiService: site data cache is empty — falling back to live GetAllSiteData call");
                    var allUrl = _options.BaseUrl.TrimEnd('/') + "/api/AirQuality/GetAllSiteData";
                    using var allReq = new HttpRequestMessage(HttpMethod.Get, allUrl);
                    ApplyAuth(allReq, bearerToken);
                    using var allResp = await _http.SendAsync(allReq, ct);
                    if (!allResp.IsSuccessStatusCode)
                    {
                        return new ApiCallResult
                        {
                            Success = false,
                            Error = $"AQMS API returned {(int)allResp.StatusCode} when looking up stations.",
                            ExecutionTimeMs = sw.ElapsedMilliseconds
                        };
                    }
                    allJson = await allResp.Content.ReadAsStringAsync(ct);
                    // Seed the cache so subsequent requests and background refresh benefit
                    _siteDataCache.UpdateCache(allJson);
                }
                else
                {
                    _log.Debug("ExternalApiService: using cached site data for station lookup");
                }

                // Parse fully into records before JsonDocument goes out of scope
                var allDevices = ParseDevicesFromJson(allJson);

                // Find best-matching station by name using Jaccard word overlap.
                // NormaliseQuery is applied to BOTH the user query AND the station name so that
                // misspelled DB names (e.g. "Commerical Instituational") are corrected before
                // scoring — preventing a normalised query from matching a different correct-spelled
                // station (e.g. "Al Ain Commercial") over the intended but misspelled one.
                var query = NormaliseQuery(nameToResolve.Trim().ToLowerInvariant());
                var queryWords = Tokenize(query);

                // Region words in the query — used to hard-exclude stations from the wrong region
                var queryHasAbuDhabi  = query.Contains("abu dhabi",  StringComparison.OrdinalIgnoreCase) || query.Contains("abudhabi", StringComparison.OrdinalIgnoreCase);
                var queryHasAlAin     = query.Contains("al ain",     StringComparison.OrdinalIgnoreCase) || query.Contains("alain",    StringComparison.OrdinalIgnoreCase);
                var queryHasAlDhafra  = query.Contains("al dhafra",  StringComparison.OrdinalIgnoreCase) || query.Contains("aldhafra", StringComparison.OrdinalIgnoreCase);

                // Build the list of all stations from the API response
                var allStations = allDevices
                    .GroupBy(d => d.StationId)
                    .Select(g => new { g.Key, Name = g.First().StationName, Devices = g.ToList() })
                    .Where(s => !string.IsNullOrWhiteSpace(s.Name))
                    .ToList();

                // RBAC: restrict candidate stations to only the user's permitted sites
                var permittedStations = (user is not null && !user.HasAllSitesAccess && user.PermittedSiteIds.Count > 0)
                    ? [.. allStations.Where(s => user.PermittedSiteIds.Contains(s.Key))]
                    : allStations;

                var matched = permittedStations
                    .Select(s =>
                    {
                        // Normalise station name too — fixes DB misspellings before scoring
                        var normalisedStation = NormaliseQuery(s.Name.ToLowerInvariant());
                        var stationWords = Tokenize(normalisedStation);

                        // Hard-exclude stations from a different region when user specifies one
                        var stationHasAbuDhabi = normalisedStation.Contains("abu dhabi", StringComparison.OrdinalIgnoreCase);
                        var stationHasAlAin    = normalisedStation.Contains("al ain",    StringComparison.OrdinalIgnoreCase);
                        var stationHasAlDhafra = normalisedStation.Contains("al dhafra", StringComparison.OrdinalIgnoreCase);

                        if (queryHasAbuDhabi && (stationHasAlAin || stationHasAlDhafra))
                            return new { s.Key, s.Name, s.Devices, Score = 0.0 };
                        if (queryHasAlAin && (stationHasAbuDhabi || stationHasAlDhafra))
                            return new { s.Key, s.Name, s.Devices, Score = 0.0 };
                        if (queryHasAlDhafra && (stationHasAbuDhabi || stationHasAlAin))
                            return new { s.Key, s.Name, s.Devices, Score = 0.0 };

                        // Exact match after normalisation
                        if (normalisedStation.Equals(query, StringComparison.OrdinalIgnoreCase))
                            return new { s.Key, s.Name, s.Devices, Score = 1000.0 };

                        // Jaccard: intersection / union
                        var intersection = queryWords.Intersect(stationWords).Count();
                        var union = queryWords.Union(stationWords).Count();
                        var score = union > 0 ? (double)intersection / union * 100 : 0;
                        return new { s.Key, s.Name, s.Devices, Score = score };
                    })
                    .Where(s => s.Score > 0)
                    .OrderByDescending(s => s.Score)
                    .FirstOrDefault();

                _log.Debug("ExternalApiService: station name candidates for '{Query}':", nameToResolve);
                allDevices
                    .GroupBy(d => d.StationId)
                    .Select(g => new { Name = g.First().StationName, Words = Tokenize(g.First().StationName.ToLowerInvariant()) })
                    .ToList()
                    .ForEach(s =>
                    {
                        var inter = queryWords.Intersect(s.Words).Count();
                        var uni = queryWords.Union(s.Words).Count();
                        _log.Debug("  '{Name}' → score={Score:F1}", s.Name, uni > 0 ? (double)inter / uni * 100 : 0);
                    });

                if (matched == null)
                {
                    // Check if the site exists in the DB but is outside the user's permitted sites
                    var siteExistsResult = await _sqlExecutor.ExecuteAsync(
                        "SELECT TOP 1 StationName FROM DMN_Stations WHERE StationName = @name OR LOWER(StationName) LIKE '%' + LOWER(@name) + '%'",
                        new Dictionary<string, object> { ["name"] = nameToResolve.Trim() }, ct);

                    if (siteExistsResult.Success && siteExistsResult.Rows.Count > 0)
                    {
                        var actualSiteName = siteExistsResult.Rows[0]["StationName"]?.ToString() ?? nameToResolve;
                        _log.Warning(
                            "ExternalApiService: user {UserId} denied access to site '{Site}' — exists in DB but not in permitted sites",
                            user?.UserId, actualSiteName);
                        return new ApiCallResult
                        {
                            Success = false,
                            Error = $"You do not have access to '{actualSiteName}'. Please ask about a site under your assigned location.",
                            ExecutionTimeMs = sw.ElapsedMilliseconds
                        };
                    }

                    // Site genuinely doesn't exist — show what's available
                    var availableNames = allDevices
                        .GroupBy(d => d.StationId)
                        .Select(g => g.First().StationName)
                        .Where(n => !string.IsNullOrWhiteSpace(n))
                        .OrderBy(n => n)
                        .ToList();

                    var hint = availableNames.Count > 0
                        ? $" Available sites: {string.Join(", ", availableNames)}."
                        : string.Empty;

                    return new ApiCallResult
                    {
                        Success = false,
                        Error = $"'{nameToResolve}' doesn't match any known site. Please enter a valid site name.{hint}",
                        ExecutionTimeMs = sw.ElapsedMilliseconds
                    };
                }

                // RBAC: verify the matched site is in the user's permitted sites
                if (user is not null && !user.HasAllSitesAccess && user.PermittedSiteIds.Count > 0
                    && !user.PermittedSiteIds.Contains(matched.Key))
                {
                    _log.Warning(
                        "ExternalApiService: user {UserId} denied access to site '{Site}' (ID={Id}, permitted={Permitted})",
                        user.UserId, matched.Name, matched.Key, string.Join(",", user.PermittedSiteIds));
                    return new ApiCallResult
                    {
                        Success = false,
                        Error = $"You do not have access to '{matched.Name}'. Please ask about a site under your assigned location.",
                        ExecutionTimeMs = sw.ElapsedMilliseconds
                    };
                }

                _log.Debug("ExternalApiService: resolved '{Query}' → '{Station}' (ID={Id}, score={Score})",
                    nameToResolve, matched.Name, matched.Key, matched.Score);

                // Use cache for this site's readings — filter matched station rows from allJson.
                // Falls back to live GetSiteData only when the site has no rows in cache
                // (e.g. a site with no parameter readings at all).
                var cachedSiteDevices = matched.Devices
                    .Select(d => d with { StationName = matched.Name })
                    .ToList();

                List<DeviceReading> siteDevices;
                if (cachedSiteDevices.Count > 0 && cachedSiteDevices.Any(d => d.Params.Count > 0))
                {
                    _log.Debug("ExternalApiService: using cache for site '{Station}' readings ({Count} devices)", matched.Name, cachedSiteDevices.Count);
                    siteDevices = cachedSiteDevices;
                    sw.Stop();
                }
                else
                {
                    _log.Debug("ExternalApiService: cache has no readings for '{Station}' — falling back to live GetSiteData", matched.Name);
                    var siteUrl = _options.BaseUrl.TrimEnd('/') + $"/api/AirQuality/GetSiteData?siteId={matched.Key}";
                    using var siteReq = new HttpRequestMessage(HttpMethod.Get, siteUrl);
                    ApplyAuth(siteReq, bearerToken);

                    using var siteResp = await _http.SendAsync(siteReq, ct);
                    sw.Stop();

                    if (!siteResp.IsSuccessStatusCode)
                    {
                        return new ApiCallResult
                        {
                            Success = false,
                            Error = $"AQMS API returned {(int)siteResp.StatusCode} fetching data for '{matched.Name}'.",
                            ExecutionTimeMs = sw.ElapsedMilliseconds
                        };
                    }

                    var siteJson = await siteResp.Content.ReadAsStringAsync(ct);
                    siteDevices = ParseDevicesFromJson(siteJson)
                        .Select(d => d with { StationName = string.IsNullOrWhiteSpace(d.StationName) ? matched.Name : d.StationName })
                        .ToList();
                }

                _log.Debug("ExternalApiService: GetSiteData siteId={Id} → {Count} devices", matched.Key, siteDevices.Count);

                // Flatten all parameters from all devices as individual rows.
                // The LLM will pick out whichever parameter(s) the user asked for.
                // site_readings_all omits "Site Name" from each row — it's in the intro sentence
                // and repeating it in every table row is redundant.
                var includeSiteName = templateId != "site_readings_all";
                var allParamRows = siteDevices
                    .SelectMany(d => d.Params.Select(p =>
                    {
                        var row = new Dictionary<string, object?>();
                        if (includeSiteName) row["Site Name"] = matched.Name;
                        row["Parameter"]    = p.Key;
                        row["Value"]        = Math.Round(p.Value.Value, 2);
                        row["Unit"]         = p.Value.Unit;
                        row["Last Updated"] = d.LastUpdated;
                        return row;
                    }))
                    .ToList();

                // If the site has multiple devices, collapse per-parameter rows into one.
                // AQI is averaged across all devices (site-level AQI = avg of device AQIs).
                // All other parameters fall back to the most-recently-updated device value.
                if (siteDevices.Count > 1)
                {
                    allParamRows = allParamRows
                        .GroupBy(r => r["Parameter"]?.ToString() ?? "")
                        .Select(g =>
                        {
                            var paramName = g.Key;
                            if (paramName == "AQI Index")
                            {
                                // Average AQI across all devices for this site
                                var validRows = g
                                    .Where(r => r["Value"] is not null && double.TryParse(r["Value"]?.ToString(), out _))
                                    .ToList();
                                if (validRows.Count == 0) return g.First();
                                var avgValue = validRows.Average(r => Convert.ToDouble(r["Value"]));
                                var newest = validRows.OrderByDescending(r => r["Last Updated"]?.ToString() ?? "").First();
                                var averaged = new Dictionary<string, object?>(newest)
                                {
                                    ["Value"] = Math.Round(avgValue, 2)
                                };
                                return averaged;
                            }
                            // For all other parameters, pick the most-recently-updated device
                            return g.OrderByDescending(r => r["Last Updated"]?.ToString() ?? "").First();
                        })
                        .ToList();
                }

                _log.Debug("ExternalApiService: site_aqi_single → {Count} parameter rows for '{Site}'", allParamRows.Count, matched.Name);

                return new ApiCallResult
                {
                    Success = true,
                    Rows = allParamRows,
                    ExecutionTimeMs = sw.ElapsedMilliseconds
                };
            }

            // For region_aqi / compliance_stats shapes: default year to current year if not provided
            if ((apiCall.ResponseShape == "region_aqi" || apiCall.ResponseShape == "compliance_stats")
                && !llmParams.ContainsKey("year"))
                llmParams["year"] = DateTime.UtcNow.Year.ToString();

            // For device_latest shape: resolve deviceName → deviceId via DMN_Devices
            if (apiCall.ResponseShape == "device_latest")
            {
                var deviceName = llmParams.GetValueOrDefault("deviceName") ?? "";
                if (string.IsNullOrWhiteSpace(deviceName))
                    return new ApiCallResult { Success = false, Error = "Please specify a device name (e.g. 'BA 0010', 'SEI100M 0014').", ExecutionTimeMs = sw.ElapsedMilliseconds };

                // Normalise and zero-pad the numeric suffix so partial inputs resolve correctly.
                // Real device name format: "BA 0001"–"BA 0010", "SEI100M 0014"–"SEI100M 0148"
                // User may type: "BA 01", "BA01", "ba1", "SEI100M 14", "sei100m14" etc.
                // Strategy:
                //   1. Strip spaces, uppercase                → "BA01", "SEI100M14"
                //   2. Split into alpha prefix + numeric suffix via regex
                //   3. Zero-pad suffix to 4 digits            → "BA0001", "SEI100M0014"
                //   4. Reconstruct canonical form with space  → "BA 0001", "SEI100M 0014"
                //   5. Try: exact input → normalised (no space) → zero-padded canonical → LIKE fallback
                var normalised = deviceName.Replace(" ", "").ToUpperInvariant();
                var paddedCanonical = normalised; // default: same as normalised
                var prefixMatch = System.Text.RegularExpressions.Regex.Match(normalised, @"^([A-Z]+(?:\d+[A-Z]+)*)(\d+)$");
                if (prefixMatch.Success)
                {
                    var prefix = prefixMatch.Groups[1].Value;          // e.g. "BA" or "SEI100M"
                    var digits = prefixMatch.Groups[2].Value;          // e.g. "1", "01", "001", "0001"
                    var padded = digits.PadLeft(4, '0');               // always 4 digits
                    paddedCanonical      = prefix + padded;            // "BA0001", "SEI100M0014"
                    var canonicalSpaced  = prefix + " " + padded;      // "BA 0001", "SEI100M 0014"

                    // Also try the spaced canonical as an exact lookup candidate
                    llmParams["_canonicalSpaced"] = canonicalSpaced;
                }

                var idResult = await _sqlExecutor.ExecuteAsync(
                    """
                    SELECT TOP 1 d.DeviceId AS DeviceId, d.DeviceName, d.StationID
                    FROM DMN_Devices d
                    WHERE d.Status = 1
                      AND (
                           d.DeviceName = @deviceName
                        OR d.DeviceName = @canonicalSpaced
                        OR REPLACE(UPPER(d.DeviceName),' ','') = @normalised
                        OR REPLACE(UPPER(d.DeviceName),' ','') = @paddedCanonical
                      )
                    ORDER BY
                      CASE WHEN d.DeviceName = @deviceName                           THEN 0
                           WHEN d.DeviceName = @canonicalSpaced                      THEN 1
                           WHEN REPLACE(UPPER(d.DeviceName),' ','') = @normalised    THEN 2
                           WHEN REPLACE(UPPER(d.DeviceName),' ','') = @paddedCanonical THEN 3
                           ELSE 4 END,
                      d.DeviceName
                    """,
                    new Dictionary<string, object>
                    {
                        ["deviceName"]      = deviceName,
                        ["canonicalSpaced"] = llmParams.GetValueOrDefault("_canonicalSpaced") ?? deviceName,
                        ["normalised"]      = normalised,
                        ["paddedCanonical"] = paddedCanonical
                    }, ct);

                llmParams.Remove("_canonicalSpaced");

                if (!idResult.Success || idResult.Rows.Count == 0)
                    return new ApiCallResult { Success = false, Error = $"No device found matching '{deviceName}'. Please check the device name (e.g. 'BA 0010', 'SEI100M 0014').", ExecutionTimeMs = sw.ElapsedMilliseconds };

                var resolvedId   = idResult.Rows[0]["DeviceId"]?.ToString() ?? "";
                var resolvedName = idResult.Rows[0]["DeviceName"]?.ToString() ?? deviceName;
                var stationId    = idResult.Rows[0]["StationID"] is int sid ? sid
                                 : int.TryParse(idResult.Rows[0]["StationID"]?.ToString(), out var parsedSid) ? parsedSid : 0;

                // RBAC: check if device's station is in the user's permitted sites
                if (user is not null && !user.HasAllSitesAccess && user.PermittedSiteIds.Count > 0)
                {
                    if (stationId == 0 || !user.PermittedSiteIds.Contains(stationId))
                    {
                        _log.Warning(
                            "ExternalApiService: user {UserId} denied access to device '{Device}' (StationID={StationId}, permitted={Permitted})",
                            user.UserId, resolvedName, stationId, string.Join(",", user.PermittedSiteIds));
                        return new ApiCallResult
                        {
                            Success = false,
                            Error = $"You do not have access to device '{resolvedName}'. Please ask about a device under your assigned site.",
                            ExecutionTimeMs = sw.ElapsedMilliseconds
                        };
                    }
                }

                llmParams["deviceId"] = resolvedId;
                llmParams["resolvedDeviceName"] = resolvedName;
                _log.Information("ExternalApiService: resolved device '{Name}' → ID={Id}", resolvedName, resolvedId);
            }

            // Build URL — substitute {param} placeholders from llmParams + scope
            var path = SubstitutePath(apiCall.Path, llmParams, scope);

            // If any {placeholder} remains unresolved, the call cannot proceed
            if (path.Contains('{') && path.Contains('}'))
            {
                _log.Warning("ExternalApiService: unresolved placeholder in path '{Path}' — params={Params}",
                    path, string.Join(", ", llmParams.Keys));
                return new ApiCallResult
                {
                    Success = false,
                    Error = "I couldn't identify which site you mean. Please specify the site ID or name more clearly.",
                    ExecutionTimeMs = sw.ElapsedMilliseconds
                };
            }

            var url = _options.BaseUrl.TrimEnd('/') + path;

            _log.Debug("ExternalApiService: {Method} {Url}", apiCall.Method, url);

            // For GetAllSiteData calls use the cache when available — avoids redundant upstream hits
            // for every multi-site query (region AQI, category filters, site list, etc.).
            string json;
            var isGetAllSiteData = path.TrimStart('/').Equals("api/AirQuality/GetAllSiteData", StringComparison.OrdinalIgnoreCase);
            var cachedJson = isGetAllSiteData ? _siteDataCache.GetRawJson() : null;

            if (cachedJson is not null)
            {
                _log.Debug("ExternalApiService: serving GetAllSiteData from cache ({Len} chars)", cachedJson.Length);
                json = cachedJson;
                sw.Stop();
            }
            else
            {
                // Retry on 500 (transient deadlocks from the upstream API)
                HttpResponseMessage response;
                for (int attempt = 0; ; attempt++)
                {
                    var req = new HttpRequestMessage(apiCall.Method == "POST" ? HttpMethod.Post : HttpMethod.Get, url);
                    ApplyAuth(req, bearerToken);
                    response = await _http.SendAsync(req, ct);
                    if (response.IsSuccessStatusCode || attempt >= 2 ||
                        (int)response.StatusCode < 500) break;
                    _log.Warning("ExternalApiService: {Url} returned {Status} on attempt {A}, retrying...", url, response.StatusCode, attempt + 1);
                    response.Dispose();
                    await Task.Delay(500, ct);
                }
                sw.Stop();

                if (!response.IsSuccessStatusCode)
                {
                    var body = await response.Content.ReadAsStringAsync(ct);
                    _log.Warning("ExternalApiService: {Url} returned {Status}: {Body}",
                        url, response.StatusCode, body[..Math.Min(300, body.Length)]);
                    return new ApiCallResult
                    {
                        Success = false,
                        Error = $"AQMS API returned {(int)response.StatusCode}: {response.ReasonPhrase}",
                        ExecutionTimeMs = sw.ElapsedMilliseconds
                    };
                }

                json = await response.Content.ReadAsStringAsync(ct);
                // Seed the cache whenever we fetch GetAllSiteData live so user traffic keeps it warm
                if (isGetAllSiteData)
                    _siteDataCache.UpdateCache(json);
            }
            _log.Information("ExternalApiService: raw API response ({Len} chars): {Preview}",
                json.Length, json[..Math.Min(500, json.Length)]);
            var regionFilter = llmParams.GetValueOrDefault("regionName");

            // region_aqi_live: aggregate AQI Index per region from GetAllSiteData — same source
            // as the Executive Dashboard. Groups stations by RegionName, averages AQI Index
            // across all stations in each region, optionally filtered to one region.
            if (apiCall.ResponseShape == "region_aqi_live")
            {
                var rows2 = BuildRegionAqiLiveRows(json, regionFilter);
                _log.Debug("ExternalApiService: region_aqi_live → {Count} rows region={Region}",
                    rows2.Count, regionFilter ?? "all");
                return new ApiCallResult { Success = true, Rows = rows2, ExecutionTimeMs = sw.ElapsedMilliseconds };
            }

            // compliance_stats: parse Compliancestatas response → single summary row with
            // overallAvailability (Data Success Rate %), deviceActiveCount, AQI, criticalSiteCount.
            if (apiCall.ResponseShape == "compliance_stats")
            {
                var rows2 = ParseComplianceStatsRows(json);
                _log.Debug("ExternalApiService: compliance_stats → {Count} rows", rows2.Count);
                return new ApiCallResult { Success = true, Rows = rows2, ExecutionTimeMs = sw.ElapsedMilliseconds };
            }

            // aqi_category_filter: classify all sites by AQI category, optionally filter to one.
            // "critical" / "hotspot" maps to Unhealthy + Very Unhealthy + Hazardous combined.
            // Optional sectorName filters results to a specific sector (e.g. "Public & Govt-School").
            if (apiCall.ResponseShape == "aqi_category_filter")
            {
                var aqiCategory  = llmParams.GetValueOrDefault("aqiCategory");
                var sectorFilter = llmParams.GetValueOrDefault("sectorName");
                var regionFilter2 = llmParams.GetValueOrDefault("regionName");
                double? minAqi = llmParams.TryGetValue("minValue", out var minStr2) && double.TryParse(minStr2, out var mn2) ? mn2 : null;
                double? maxAqi = llmParams.TryGetValue("maxValue", out var maxStr2) && double.TryParse(maxStr2, out var mx2) ? mx2 : null;

                _log.Information("ExternalApiService: aqi_category_filter — aqiCategory='{Cat}' sector='{Sec}' region='{Reg}' min={Min} max={Max}",
                    aqiCategory ?? "(none)", sectorFilter ?? "(none)", regionFilter2 ?? "(none)", minAqi, maxAqi);

                var rows2 = await BuildAqiCategoryRows(json, aqiCategory, sectorFilter, user, ct, regionFilter2, minAqi, maxAqi);
                _log.Information("ExternalApiService: aqi_category_filter → {Count} rows returned", rows2.Count);
                return new ApiCallResult { Success = true, Rows = rows2, ExecutionTimeMs = sw.ElapsedMilliseconds };
            }

            // aqi_category_pie: aggregate sites by AQI category → {Category, Count} rows for pie chart.
            // Optional sectorName (e.g. "Public & Govt-School") and regionName filters.
            if (apiCall.ResponseShape == "aqi_category_pie")
            {
                var pieSector = llmParams.GetValueOrDefault("sectorName");
                var pieRegion = llmParams.GetValueOrDefault("regionName");

                _log.Information("ExternalApiService: aqi_category_pie — sector='{Sec}' region='{Reg}'",
                    pieSector ?? "(none)", pieRegion ?? "(none)");

                // Reuse BuildAqiCategoryRows without category filter to get all sites
                var allSiteRows = await BuildAqiCategoryRows(json, null, pieSector, user, ct, pieRegion);

                // Aggregate by AQI category in display order
                var categoryOrder = new[]
                {
                    "Good", "Moderate", "Unhealthy for Sensitive Groups",
                    "Unhealthy", "Very Unhealthy", "Hazardous"
                };

                var grouped = allSiteRows
                    .GroupBy(r => r["AQICategory"]?.ToString() ?? "Unknown")
                    .ToDictionary(g => g.Key, g => g.Count());

                var pieRows = categoryOrder
                    .Where(cat => grouped.ContainsKey(cat))
                    .Select(cat => new Dictionary<string, object?>
                    {
                        ["Category"] = cat,
                        ["Count"]    = grouped[cat]
                    })
                    .ToList();

                // Append any unexpected categories not in our ordered list
                foreach (var cat in grouped.Keys.Where(k => !categoryOrder.Contains(k)))
                {
                    pieRows.Add(new Dictionary<string, object?> { ["Category"] = cat, ["Count"] = grouped[cat] });
                }

                _log.Information("ExternalApiService: aqi_category_pie → {Count} category slices", pieRows.Count);
                return new ApiCallResult { Success = true, Rows = pieRows, ExecutionTimeMs = sw.ElapsedMilliseconds };
            }

            // mold_reports: parse GetMoldReportsBySiteName response → flat rows for table display.
            // Each row = one mold test at the site with key metadata (TestName, SampleNo, ReportNo,
            // SamplingTime, FormSubmisionStatus, Remarks). Parameters listed in a sub-column.
            if (apiCall.ResponseShape == "mold_reports")
            {
                var moldRows = ParseMoldReportRows(json);
                _log.Information("ExternalApiService: mold_reports → {Count} rows for siteName='{Site}'",
                    moldRows.Count, llmParams.GetValueOrDefault("siteName") ?? "(all)");
                return new ApiCallResult { Success = true, Rows = moldRows, ExecutionTimeMs = sw.ElapsedMilliseconds };
            }

            // schools_pollutant: filter API results to sites whose name contains "school",
            // extract requested parameter, and optionally filter by min/max value threshold.
            if (apiCall.ResponseShape == "schools_pollutant")
            {
                var parameterName = llmParams.GetValueOrDefault("parameterName") ?? "AQI Index";
                var rows2 = await BuildSchoolPollutantRows(json, parameterName, regionFilter, llmParams, ct);
                _log.Debug("ExternalApiService: schools_pollutant → {Count} rows for param={Param} region={Region}",
                    rows2.Count, parameterName, regionFilter ?? "all");
                return new ApiCallResult { Success = true, Rows = rows2, ExecutionTimeMs = sw.ElapsedMilliseconds };
            }

            // sector_pollutant: filter API results to a named sector, extract requested parameter
            if (apiCall.ResponseShape == "sector_pollutant")
            {
                var parameterName = llmParams.GetValueOrDefault("parameterName") ?? "AQI Index";
                var sectorName    = llmParams.GetValueOrDefault("sectorName") ?? "";
                var rows2 = await BuildSectorPollutantRows(json, parameterName, sectorName, regionFilter, ct);
                _log.Debug("ExternalApiService: sector_pollutant → {Count} rows for param={Param} sector={Sector} region={Region}",
                    rows2.Count, parameterName, sectorName, regionFilter ?? "all");
                return new ApiCallResult { Success = true, Rows = rows2, ExecutionTimeMs = sw.ElapsedMilliseconds };
            }

            // site_aqi_trend_yearly: resolve devices under the site, then query ParameterAveragesYear
            // for AQI average per year and DeviceCompliance for DSR per year over the last N years.
            if (apiCall.ResponseShape == "site_aqi_trend_yearly")
            {
                var yearsBack = llmParams.TryGetValue("years", out var yrStr) && int.TryParse(yrStr, out var yr) ? yr : 3;
                var rows2 = await BuildSiteAqiTrendYearlyRows(json, nameToResolve ?? "", yearsBack, user, ct);
                _log.Information("ExternalApiService: site_aqi_trend_yearly → {Count} rows for site='{Site}' years={Y}",
                    rows2.Count, nameToResolve ?? "(none)", yearsBack);
                return new ApiCallResult { Success = true, Rows = rows2, ExecutionTimeMs = sw.ElapsedMilliseconds };
            }

            // indoor_ambient_compare: compare indoor site AQI with the nearest outdoor ambient station.
            // 1. Finds the indoor site in the GetAllSiteData response.
            // 2. Fetches its coordinates from DMN_Stations.
            // 3. Calls the Abu Dhabi SDI ArcGIS REST service (public, no auth) for ambient stations + AQI.
            // 4. Finds the nearest ambient station by Haversine distance.
            // 5. Returns a single comparison row.
            if (apiCall.ResponseShape == "indoor_ambient_compare")
            {
                var rows2 = await BuildIndoorAmbientCompareRows(json, nameToResolve ?? "", bearerToken, ct);
                _log.Information("ExternalApiService: indoor_ambient_compare → {Count} rows for site='{Site}'",
                    rows2.Count, nameToResolve ?? "(none)");
                return new ApiCallResult { Success = true, Rows = rows2, ExecutionTimeMs = sw.ElapsedMilliseconds };
            }

            // site_list_with_status: mirrors exactly what the Site Overview UI shows.
            // Sources all sites from DMN_Stations (same as api/Deviceslookup → listStations),
            // then overlays LiveStatus from DMN_Devices.IsEnable — so sites with no readings
            // (not in GetAllSiteData) still appear, matching the UI's 14-site list.
            // LiveStatus values:
            //   "Active"   — at least one device with IsEnable=true
            //   "Inactive" — has devices but all IsEnable=false
            //   "No Data"  — no devices registered for this site
            // Optional llmParams: regionName, liveStatus (Active / Inactive / No Data)
            if (apiCall.ResponseShape == "site_list_with_status")
            {
                // Support pipe-separated multiple values: "Abu Dhabi|Al Dhafra", "Active|Inactive"
                var regionFilters = (llmParams.GetValueOrDefault("regionName") ?? "")
                    .Split('|', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                var statusFilters = (llmParams.GetValueOrDefault("liveStatus") ?? "")
                    .Split('|', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                var sectorFilters = (llmParams.GetValueOrDefault("sectorName") ?? "")
                    .Split('|', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

                // Query all non-deleted stations with region and sector — same as UI's GetStations()
                // which filters only on !IsDeleted (no Status=1 check).
                var stationsSql = """
                    SELECT s.ID AS StationId, s.StationName, r.RegionName, sec.SectorName
                    FROM DMN_Stations s
                    JOIN Regions r ON r.Id = s.RegionID
                    LEFT JOIN Sectors sec ON sec.Id = s.SectorID
                    WHERE ISNULL(s.IsDeleted,0) = 0
                    ORDER BY r.RegionName, s.StationName
                    """;
                var stationsResult = await _sqlExecutor.ExecuteAsync(stationsSql, new Dictionary<string, object>(), ct);

                // Replicate AdminDAL.DevicesList() live-status logic:
                //   isActive = any parameter reading within 24h OR LastCommunicationTime within 10min.
                //   Uses ALL parameters (not just PM2.5/DriverID=4) — some devices may not report PM2.5
                //   but still be live via other parameters, and AdminDAL falls back to IsEnable=false
                //   only when NO parameter row exists at all for that device.
                //   Site is Active if ANY device under that station is active.
                var deviceStatusSql = """
                    SELECT
                        d.StationID,
                        MAX(CASE
                            WHEN p.ParameterReadingUpdateTime >= DATEADD(MINUTE, -1440, GETDATE()) THEN 1
                            WHEN p.LastCommunicationTime     >= DATEADD(MINUTE, -10,   GETDATE()) THEN 1
                            ELSE 0
                        END) AS HasActiveDevice
                    FROM DMN_Devices d
                    LEFT JOIN (
                        SELECT DeviceID,
                               MAX(ParameterReadingUpdateTime) AS ParameterReadingUpdateTime,
                               MAX(LastCommunicationTime)      AS LastCommunicationTime
                        FROM DMN_Parameters
                        GROUP BY DeviceID
                    ) p ON p.DeviceID = d.DeviceId
                    WHERE ISNULL(d.IsDeleted, 0) = 0
                    GROUP BY d.StationID
                    """;
                var deviceEnableResult = await _sqlExecutor.ExecuteAsync(deviceStatusSql, new Dictionary<string, object>(), ct);

                var deviceEnableByStation = deviceEnableResult.Rows
                    .ToDictionary(
                        r => Convert.ToInt32(r["StationID"]),
                        r => Convert.ToInt32(r["HasActiveDevice"]) == 1);

                // RBAC: restrict to permitted sites for non-admin users
                var allStationRows = stationsResult.Rows;
                if (user is not null && !user.HasAllSitesAccess && user.PermittedSiteIds.Count > 0)
                    allStationRows = allStationRows
                        .Where(r => user.PermittedSiteIds.Contains(Convert.ToInt32(r["StationId"])))
                        .ToList();

                var siteRows = allStationRows
                    .Select(r =>
                    {
                        var stationId  = Convert.ToInt32(r["StationId"]);
                        var name       = r["StationName"]?.ToString() ?? "";
                        var region     = r["RegionName"]?.ToString() ?? "";
                        var sector     = r["SectorName"]?.ToString() ?? "";
                        string liveStatus;
                        if (deviceEnableByStation.TryGetValue(stationId, out var hasActive))
                            liveStatus = hasActive ? "Active" : "Inactive";
                        else
                            liveStatus = "No Data";

                        return new { StationName = name, RegionName = region, SectorName = sector, LiveStatus = liveStatus };
                    })
                    .Where(s => !string.IsNullOrWhiteSpace(s.StationName))
                    .Where(s => regionFilters.Length == 0
                        || regionFilters.Any(r => s.RegionName.Contains(r, StringComparison.OrdinalIgnoreCase)))
                    .Where(s => sectorFilters.Length == 0
                        || sectorFilters.Any(f => s.SectorName.Contains(f, StringComparison.OrdinalIgnoreCase)))
                    .Where(s => statusFilters.Length == 0
                        || statusFilters.Any(f => s.LiveStatus.Equals(f, StringComparison.OrdinalIgnoreCase)))
                    .Select(s => new Dictionary<string, object?>
                    {
                        ["Site Name"]   = s.StationName,
                        ["Region"]      = s.RegionName,
                        ["Sector"]      = s.SectorName,
                        ["Live Status"] = s.LiveStatus
                    })
                    .ToList();

                _log.Information("ExternalApiService: site_list_with_status → {Count} rows (regions={R} statuses={S})",
                    siteRows.Count,
                    regionFilters.Length > 0 ? string.Join("|", regionFilters) : "all",
                    statusFilters.Length > 0 ? string.Join("|", statusFilters) : "all");
                return new ApiCallResult { Success = true, Rows = siteRows, ExecutionTimeMs = sw.ElapsedMilliseconds };
            }

            // For multi-site shapes (devices/sites/offline/status), filter to permitted sites before flattening.
            // region_aqi and device_latest have their own RBAC checks earlier in this method.
            List<Dictionary<string, object?>> rows;
            if (apiCall.ResponseShape is "sites" or "device_list" or "offline_devices" or "device_status_summary" or "devices"
                && user is not null && !user.HasAllSitesAccess && user.PermittedSiteIds.Count > 0)
            {
                var allDevices = ParseDevicesFromJson(json);
                var permittedDevices = allDevices.Where(d => user.PermittedSiteIds.Contains(d.StationId)).ToList();
                rows = FlattenDevices(permittedDevices, apiCall.ResponseShape);
            }
            else if (apiCall.ResponseShape == "sites")
            {
                // Filter out stations with no real devices in our DB before aggregating AQI.
                // The AQMS API can return paramaterDtos with AQI for stations that have no devices
                // (mold-only sites or misconfigured stations). We cross-check with DMN_Devices.
                var stationIdsWithDevicesResult = await _sqlExecutor.ExecuteAsync(
                    "SELECT DISTINCT StationID FROM DMN_Devices WHERE ISNULL(IsDeleted, 0) = 0",
                    new Dictionary<string, object>(), ct);
                var validStationIds = stationIdsWithDevicesResult.Rows
                    .Select(r => r.TryGetValue("StationID", out var v) ? Convert.ToInt32(v) : 0)
                    .ToHashSet();
                var allDevices = ParseDevicesFromJson(json);
                var devicesWithRealStations = allDevices.Where(d => validStationIds.Contains(d.StationId)).ToList();
                rows = AggregateByStation(devicesWithRealStations);
            }
            else
            {
                rows = Flatten(json, apiCall.ResponseShape, regionFilter);
            }

            // For device_latest: filter to requested parameter if user asked for a specific one
            if (apiCall.ResponseShape == "device_latest")
            {
                var paramFilter = llmParams.GetValueOrDefault("parameterName");
                if (!string.IsNullOrWhiteSpace(paramFilter))
                {
                    rows = rows.Where(r =>
                        string.Equals(r["ParameterName"]?.ToString(), paramFilter, StringComparison.OrdinalIgnoreCase))
                        .ToList();
                }

            }

            _log.Information("ExternalApiService: {Url} → {Rows} rows in {Ms}ms (shape={Shape})", url, rows.Count, sw.ElapsedMilliseconds, apiCall.ResponseShape);
            return new ApiCallResult { Success = true, Rows = rows, ExecutionTimeMs = sw.ElapsedMilliseconds };
        }
        catch (Exception ex)
        {
            sw.Stop();
            _log.Error(ex, "ExternalApiService: call failed");
            return new ApiCallResult
            {
                Success = false,
                Error = "Failed to reach the AQMS data service. Please try again.",
                ExecutionTimeMs = sw.ElapsedMilliseconds
            };
        }
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// <summary>
    /// Forwards auth to the AQMS API.
    /// The main API uses an HttpOnly cookie ("Token"), so we forward it as a Cookie header.
    /// Falls back to Bearer if a token string was somehow extracted.
    /// </summary>
    private void ApplyAuth(HttpRequestMessage req, string bearerToken)
    {
        var httpCtx = _httpContextAccessor.HttpContext;
        if (httpCtx != null)
        {
            var cookieToken = httpCtx.Request.Cookies["Token"] ?? httpCtx.Request.Cookies["token"];
            if (!string.IsNullOrWhiteSpace(cookieToken))
            {
                req.Headers.Add("Cookie", $"Token={cookieToken}");
                return;
            }
        }

        // Fallback: use Bearer if provided
        if (!string.IsNullOrWhiteSpace(bearerToken))
            req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", bearerToken);
    }

    private static string SubstitutePath(
        string pathTemplate,
        Dictionary<string, string> llmParams,
        ResolvedScope scope)
    {
        var path = pathTemplate;

        // Substitute LLM-extracted params first
        foreach (var kv in llmParams)
            path = path.Replace($"{{{kv.Key}}}", Uri.EscapeDataString(kv.Value), StringComparison.OrdinalIgnoreCase);

        // Substitute siteId from scope if still unresolved
        if (path.Contains("{siteId}", StringComparison.OrdinalIgnoreCase))
        {
            if (scope.SiteIds.Count >= 1)
                path = path.Replace("{siteId}", scope.SiteIds[0].ToString(), StringComparison.OrdinalIgnoreCase);
        }

        return path;
    }

    /// <summary>
    /// Flattens the AQMS API response into rows.
    ///
    /// ResponseShape "devices":
    ///   Input:  array of { StationId, DeviceName, IsOnline, paramaterDtos: [...] }
    ///   Output: one row per device with AQI + key pollutants as columns.
    ///
    /// ResponseShape "sites":
    ///   Input:  same array but multiple devices per site — aggregate per StationId.
    ///   Output: one row per site with average AQI + device count.
    /// </summary>
    /// <summary>Splits text into meaningful word tokens (3+ chars) for fuzzy matching.</summary>
    private static HashSet<string> Tokenize(string text) =>
        new(text.Split([' ', '-', '_', ',', '.', '(', ')'], StringSplitOptions.RemoveEmptyEntries)
                .Where(w => w.Length >= 3),
            StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Normalises common abbreviations/typos in station name queries.
    /// e.g. "abudhabi" → "abu dhabi", "alain" → "al ain", "aldhafra" → "al dhafra"
    /// </summary>
    private static string NormaliseQuery(string query) => query
        .Replace("abudhabi", "abu dhabi", StringComparison.OrdinalIgnoreCase)
        .Replace("abu-dhabi", "abu dhabi", StringComparison.OrdinalIgnoreCase)
        .Replace("alain", "al ain", StringComparison.OrdinalIgnoreCase)
        .Replace("al-ain", "al ain", StringComparison.OrdinalIgnoreCase)
        .Replace("aldhafra", "al dhafra", StringComparison.OrdinalIgnoreCase)
        .Replace("al-dhafra", "al dhafra", StringComparison.OrdinalIgnoreCase)
        .Replace("commerical", "commercial", StringComparison.OrdinalIgnoreCase)
        .Replace("instituational", "institutional", StringComparison.OrdinalIgnoreCase)
        .Replace("instituation", "institution", StringComparison.OrdinalIgnoreCase);

    /// <summary>Parses JSON string fully into DeviceReading records before the JsonDocument is disposed.</summary>
    private static List<DeviceReading> ParseDevicesFromJson(string json)
    {
        using var doc = JsonDocument.Parse(json);
        if (doc.RootElement.ValueKind != JsonValueKind.Array)
            return [];
        // ToList() forces immediate materialisation — all data copied out of JsonDocument
        return doc.RootElement.EnumerateArray().Select(ParseDevice).ToList();
    }

    private static List<Dictionary<string, object?>> Flatten(string json, string responseShape, string? regionFilter = null)
    {
        if (responseShape == "region_aqi")
            return ParseRegionAqiRows(json, regionFilter);

        if (responseShape == "device_latest")
            return ParseDeviceLatestRows(json);

        var devices = ParseDevicesFromJson(json);
        return responseShape switch
        {
            "sites"                 => AggregateByStation(devices),
            "device_list"           => DeviceListRows(devices),
            "offline_devices"       => OfflineDeviceRows(devices),
            "device_status_summary" => DeviceStatusSummaryRows(devices),
            _                       => DeviceRows(devices)
        };
    }

    private static List<Dictionary<string, object?>> ParseDeviceLatestRows(string json)
    {
        // ASP.NET may double-serialize string return values — unwrap if root is a JSON string
        using (var probe = JsonDocument.Parse(json))
        {
            if (probe.RootElement.ValueKind == JsonValueKind.String)
                json = probe.RootElement.GetString() ?? json;
        }

        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        // API returns a flat array of DeviceLatestDto — one object per parameter
        if (root.ValueKind != JsonValueKind.Array) return [];

        var rows = new List<Dictionary<string, object?>>();
        foreach (var el in root.EnumerateArray())
        {
            var deviceName  = el.TryGetProperty("DeviceName",     out var dn) ? dn.GetString() : null;
            var paramName   = el.TryGetProperty("ParameterName",  out var pn) ? pn.GetString() : null;
            var unitName    = el.TryGetProperty("UnitName",        out var un) ? un.GetString() : null;
            var timestamp   = el.TryGetProperty("Timestamp",       out var ts) ? ts.GetString() : null;

            double? paramValue = null;
            if (el.TryGetProperty("ParameterValue", out var pv))
            {
                if (pv.ValueKind == JsonValueKind.Number)
                    paramValue = pv.GetDouble();
                else if (pv.ValueKind == JsonValueKind.String && double.TryParse(pv.GetString(), out var d))
                    paramValue = d;
            }

            if (string.IsNullOrWhiteSpace(paramName)) continue;

            rows.Add(new Dictionary<string, object?>
            {
                ["DeviceName"]     = deviceName,
                ["ParameterName"]  = paramName,
                ["ParameterValue"] = paramValue.HasValue ? (object?)Math.Round(paramValue.Value, 2) : null,
                ["UnitName"]       = paramName == "AQI Index" ? "" : (unitName ?? ""),
                ["LastMeasured"]   = timestamp
            });
        }
        return rows;
    }

    private static List<Dictionary<string, object?>> FlattenDevices(List<DeviceReading> devices, string responseShape)
    {
        return responseShape switch
        {
            "sites"                 => AggregateByStation(devices),
            "device_list"           => DeviceListRows(devices),
            "offline_devices"       => OfflineDeviceRows(devices),
            "device_status_summary" => DeviceStatusSummaryRows(devices),
            _                       => DeviceRows(devices)
        };
    }

    /// <summary>
    /// Normalises user-supplied region name variants to the canonical DB names.
    /// e.g. "abudhabi" → "Abu Dhabi", "alain" → "Al Ain", "aldhafra" → "Al Dhafra"
    /// Returns null if the input doesn't match any known region.
    /// </summary>
    private static string? NormaliseRegionName(string input)
    {
        var s = input.Trim().ToLowerInvariant()
            .Replace("-", "").Replace(" ", "");

        // After stripping spaces, hyphens and lowercasing, map to canonical DB region names.
        // "Abu Dhabi" → "abudhabi", "Al Ain" → "alain", "Al Dhafra" → "aldhafra"
        return s switch
        {
            "abudhabi" or "abudabi" or "abudhabii" => "Abudhabi",
            "alain"                                => "AL Ain",
            "aldhafra" or "dhafra"                 => "AlDhafra",
            _ => null
        };
    }

    /// <summary>Parses the GetRegionGeographicalDataAQI response into summary rows, optionally filtered to one region.</summary>
    private static List<Dictionary<string, object?>> ParseRegionAqiRows(string json, string? regionFilter = null)
    {
        using var doc = JsonDocument.Parse(json);
        if (doc.RootElement.ValueKind != JsonValueKind.Array)
            return [];

        var canonical = regionFilter is not null ? NormaliseRegionName(regionFilter) : null;

        var allRows = doc.RootElement.EnumerateArray()
            .Select(el =>
            {
                var regionName = el.TryGetProperty("RegionName", out var rn) ? rn.GetString() ?? "" : "";
                var aqi        = el.TryGetProperty("AQI",        out var aq) ? (object?)aq.GetDouble() : null;
                var active     = el.TryGetProperty("ActiveStationsCount",   out var ac) ? (object?)ac.GetInt32() : null;
                var pct        = el.TryGetProperty("ActiveStationsPercent", out var ap) ? (object?)Math.Round(ap.GetDouble(), 2) : null;

                return new Dictionary<string, object?>
                {
                    ["RegionName"]            = regionName,
                    ["AQI"]                   = aqi,
                    ["AQICategory"]           = aqi is double d ? ClassifyAqi(d) : "Unknown",
                    ["ActiveStationsCount"]   = active,
                    ["ActiveStationsPercent"] = pct
                };
            })
            .ToList();

        if (canonical is null)
            return allRows;

        // Try exact canonical match first
        var filtered = allRows
            .Where(row => string.Equals(row["RegionName"]?.ToString(), canonical, StringComparison.OrdinalIgnoreCase))
            .ToList();

        // Fallback: fuzzy match — strip spaces from both sides and compare
        if (filtered.Count == 0)
        {
            var canonicalStripped = canonical.Replace(" ", "").ToLowerInvariant();
            filtered = allRows
                .Where(row => row["RegionName"]?.ToString()?.Replace(" ", "").ToLowerInvariant() == canonicalStripped)
                .ToList();
        }

        // If still nothing (unexpected API region name), return all rows so user gets data
        return filtered.Count > 0 ? filtered : allRows;
    }

    private record DeviceReading(
        int StationId,
        string StationName,
        string DeviceName,
        bool IsOnline,
        string LastUpdated,
        Dictionary<string, (double Value, string Unit)> Params,
        string RegionName = "",
        bool HasDevices = true);

    private static DeviceReading ParseDevice(JsonElement el)
    {
        var stationId   = el.TryGetProperty("StationId",   out var sid)  ? sid.GetInt32()        : 0;
        var stationName =
            (el.TryGetProperty("StationName", out var sn)  ? sn.GetString()  : null) ??
            (el.TryGetProperty("stationName", out var sn2) ? sn2.GetString() : null) ??
            "";
        var regionName =
            (el.TryGetProperty("RegionName",  out var rn)  ? rn.GetString()  : null) ??
            (el.TryGetProperty("regionName",  out var rn2) ? rn2.GetString() : null) ??
            "";
        var deviceName  =
            (el.TryGetProperty("DeviceName",  out var dn)  ? dn.GetString()  : null) ??
            (el.TryGetProperty("deviceName",  out var dn2) ? dn2.GetString() : null) ??
            (el.TryGetProperty("Device_Name", out var dn3) ? dn3.GetString() : null) ??
            (el.TryGetProperty("Name",        out var dn4) ? dn4.GetString() : null) ??
            "";
        var isOnline    =
            (el.TryGetProperty("IsOnline",    out var io)  ? io.GetBoolean()  :
             el.TryGetProperty("isOnline",    out var io2) ? io2.GetBoolean() : false);

        // Try common timestamp field names at the device level
        var lastUpdated =
            (el.TryGetProperty("LastUpdated",  out var lu)  ? lu.GetString()  : null) ??
            (el.TryGetProperty("LastUpdate",   out var lu2) ? lu2.GetString() : null) ??
            (el.TryGetProperty("CreatedTime",  out var ct)  ? ct.GetString()  : null) ??
            (el.TryGetProperty("ReadingTime",  out var rt)  ? rt.GetString()  : null) ??
            (el.TryGetProperty("Timestamp",    out var ts)  ? ts.GetString()  : null) ??
            string.Empty;

        var parameters = new Dictionary<string, (double, string)>(StringComparer.OrdinalIgnoreCase);
        if (el.TryGetProperty("paramaterDtos", out var dtos))
        {
            foreach (var p in dtos.EnumerateArray())
            {
                var name  = p.TryGetProperty("ParameterName",  out var pn) ? pn.GetString() ?? "" : "";
                var value = p.TryGetProperty("ParameterValue", out var pv) ? pv.GetDouble() : 0;
                var unit  = p.TryGetProperty("Unit",           out var pu) ? pu.GetString() ?? "" : "";

                // Pick up timestamp from parameter level if not found at device level
                if (string.IsNullOrEmpty(lastUpdated))
                {
                    lastUpdated =
                        (p.TryGetProperty("ParameterReadingUpdateTime", out var prut) ? prut.GetString()  : null) ??
                        (p.TryGetProperty("LastUpdated",                out var plu)  ? plu.GetString()   : null) ??
                        (p.TryGetProperty("CreatedTime",                out var pct)  ? pct.GetString()   : null) ??
                        (p.TryGetProperty("ReadingTime",                out var prt)  ? prt.GetString()   : null) ??
                        string.Empty;
                }

                if (!string.IsNullOrWhiteSpace(name))
                    parameters[name] = (value, unit);
            }
        }

        // A site with no devices has an empty paramaterDtos array — exclude it from AQI aggregation
        var hasDevices = parameters.Count > 0;

        return new DeviceReading(stationId, stationName, deviceName, isOnline, lastUpdated, parameters, regionName, hasDevices);
    }

    private static List<Dictionary<string, object?>> DeviceRows(List<DeviceReading> devices)
    {
        return devices.Select(d =>
        {
            var row = new Dictionary<string, object?>
            {
                ["Site Name"]   = d.StationName,
                ["DeviceName"]  = d.DeviceName,
                ["IsOnline"]    = d.IsOnline ? "Online" : "Offline"
            };
            if (!string.IsNullOrWhiteSpace(d.LastUpdated))
                row["LastUpdated"] = d.LastUpdated;
            foreach (var (name, (value, unit)) in d.Params)
                row[$"{name} ({unit})"] = Math.Round(value, 2);
            return row;
        }).ToList();
    }

    /// <summary>Slim view for device-list queries — only Site Name, DeviceName, IsOnline. No parameter columns.</summary>
    private static List<Dictionary<string, object?>> DeviceListRows(List<DeviceReading> devices)
    {
        return devices.Select(d => new Dictionary<string, object?>
        {
            ["Site Name"]  = d.StationName,
            ["DeviceName"] = d.DeviceName,
            ["IsOnline"]   = d.IsOnline ? "Online" : "Offline"
        }).ToList();
    }

    /// <summary>Offline-only view — filters to offline devices, includes StationName, DeviceName, LastActive timestamp.</summary>
    private static List<Dictionary<string, object?>> OfflineDeviceRows(List<DeviceReading> devices)
    {
        return devices
            .Where(d => !d.IsOnline)
            .Select(d =>
            {
                var row = new Dictionary<string, object?>
                {
                    ["DeviceName"] = d.DeviceName,
                    ["Site Name"]  = d.StationName,
                    ["Status"]     = "Offline"
                };
                if (!string.IsNullOrWhiteSpace(d.LastUpdated))
                    row["LastActive"] = d.LastUpdated;
                return row;
            })
            .OrderBy(r => r["StationName"]?.ToString())
            .ThenBy(r => r["DeviceName"]?.ToString())
            .ToList();
    }

    /// <summary>Single-row summary of active vs total device counts across all sites.</summary>
    private static List<Dictionary<string, object?>> DeviceStatusSummaryRows(List<DeviceReading> devices)
    {
        var total   = devices.Count;
        var online  = devices.Count(d => d.IsOnline);
        var offline = total - online;

        return [new Dictionary<string, object?>
        {
            ["ActiveDevices"]  = online,
            ["OfflineDevices"] = offline,
            ["TotalDevices"]   = total
        }];
    }

    private static List<Dictionary<string, object?>> AggregateByStation(List<DeviceReading> devices)
    {
        return devices
            .GroupBy(d => d.StationId)
            .Select(g =>
            {
                var first = g.First();

                // Skip sites with Status=false — these have no installed devices (mold-only or unconfigured)
                if (!first.HasDevices) return null;

                var aqiValues = g
                    .Where(d => d.Params.ContainsKey("AQI Index"))
                    .Select(d => d.Params["AQI Index"].Value)
                    .ToList();

                var avgAqi = aqiValues.Count > 0
                    ? Math.Round(aqiValues.Average(), 1)
                    : (double?)null;

                var stationName = first.StationName;

                // Pick the most recent LastUpdated across all devices in this station
                var lastUpdated = g
                    .Select(d => d.LastUpdated)
                    .Where(t => !string.IsNullOrWhiteSpace(t))
                    .OrderByDescending(t => t)
                    .FirstOrDefault();

                // Also exclude sites with no AQI readings at all
                if (!avgAqi.HasValue) return null;

                var row = new Dictionary<string, object?>
                {
                    ["Site Name"]    = stationName,
                    ["AQI"]          = avgAqi,
                    ["AQI Category"] = ClassifyAqi(avgAqi.Value),
                    ["Last Updated"] = lastUpdated ?? "N/A"
                };

                return row;
            })
            .Where(r => r is not null)
            .Select(r => r!)
            .ToList();
    }

    private static string ClassifyAqi(double aqi) => aqi switch
    {
        <= 50  => "Good",
        <= 100 => "Moderate",
        <= 150 => "Unhealthy for Sensitive Groups",
        <= 200 => "Unhealthy",
        <= 300 => "Very Unhealthy",
        _      => "Hazardous"
    };

    /// <summary>
    /// Filters GetAllSiteData response to sites in the "Public &amp; Govt-School" sector
    /// (via DB lookup on Sectors table), optionally by region, extracts the requested parameter,
    /// and applies optional minValue / maxValue filters from llmParams.
    /// </summary>
    private async Task<List<Dictionary<string, object?>>> BuildSchoolPollutantRows(
        string json,
        string parameterName,
        string? regionFilter,
        Dictionary<string, string> llmParams,
        CancellationToken ct)
    {
        // Fetch all school-sector sites from DB, optionally filtered by region
        var sql = regionFilter is not null
            ? """
              SELECT s.ID, s.StationName, r.RegionName
              FROM DMN_Stations s
              JOIN Sectors sec ON s.SectorID = sec.Id
              LEFT JOIN Regions r ON s.RegionID = r.Id
              WHERE sec.SectorName = 'Public & Govt-School'
                AND s.Status = 1
                AND (REPLACE(LOWER(r.RegionName),' ','') = REPLACE(LOWER(@regionName),' ',''))
              ORDER BY s.StationName
              """
            : """
              SELECT s.ID, s.StationName, r.RegionName
              FROM DMN_Stations s
              JOIN Sectors sec ON s.SectorID = sec.Id
              LEFT JOIN Regions r ON s.RegionID = r.Id
              WHERE sec.SectorName = 'Public & Govt-School'
                AND s.Status = 1
              ORDER BY s.StationName
              """;

        var sqlParams = regionFilter is not null
            ? new Dictionary<string, object> { ["regionName"] = regionFilter }
            : new Dictionary<string, object>();

        var dbResult = await _sqlExecutor.ExecuteAsync(sql, sqlParams, ct);
        if (!dbResult.Success || dbResult.Rows.Count == 0)
            return [];

        var schoolSiteIds = dbResult.Rows
            .Where(r => r["ID"] is not null)
            .ToDictionary(
                r => Convert.ToInt32(r["ID"]),
                r => (
                    Name:   r["StationName"]?.ToString() ?? "",
                    Region: r["RegionName"]?.ToString()  ?? ""
                ));

        // Parse min/max value thresholds from llmParams (e.g. minValue="50", maxValue="100")
        double? minValue = llmParams.TryGetValue("minValue", out var minStr) && double.TryParse(minStr, out var mn) ? mn : null;
        double? maxValue = llmParams.TryGetValue("maxValue", out var maxStr) && double.TryParse(maxStr, out var mx) ? mx : null;

        var allDevices = ParseDevicesFromJson(json);
        var results = new List<Dictionary<string, object?>>();

        var paramVariants = new[] { parameterName }
            .Concat(_parameterAliases.TryGetValue(parameterName, out var alias) ? [alias] : Array.Empty<string>())
            .ToArray();

        foreach (var group in allDevices.GroupBy(d => d.StationId))
        {
            if (!schoolSiteIds.TryGetValue(group.Key, out var siteInfo))
                continue;

            double? value = null;
            string unit = "";
            string lastUpdated = "";

            bool isAqiParam = paramVariants.Any(v => v.Equals("AQI Index", StringComparison.OrdinalIgnoreCase));
            if (isAqiParam)
            {
                // Average AQI across all devices at this station (mirrors site_aqi_single logic)
                var aqiValues = group
                    .SelectMany(d => paramVariants
                        .Where(v => d.Params.ContainsKey(v))
                        .Select(v => d.Params[v]))
                    .ToList();
                if (aqiValues.Count > 0)
                {
                    value       = Math.Round(aqiValues.Select(r => r.Value).Average(), 0);
                    unit        = aqiValues.First().Unit;
                    lastUpdated = group.Select(d => d.LastUpdated)
                                       .Where(t => !string.IsNullOrWhiteSpace(t))
                                       .OrderByDescending(t => t)
                                       .FirstOrDefault() ?? "";
                }
            }
            else
            {
                foreach (var device in group)
                {
                    foreach (var variant in paramVariants)
                    {
                        if (device.Params.TryGetValue(variant, out var reading))
                        {
                            value       = Math.Round(reading.Value, 2);
                            unit        = reading.Unit;
                            lastUpdated = device.LastUpdated;
                            break;
                        }
                    }
                    if (value.HasValue) break;
                }
            }

            if (value is null) continue;

            // Apply threshold filters
            if (minValue.HasValue && value.Value <= minValue.Value) continue;
            if (maxValue.HasValue && value.Value > maxValue.Value) continue;

            results.Add(new Dictionary<string, object?>
            {
                ["Site Name"]    = siteInfo.Name,
                ["Region"]       = siteInfo.Region,
                ["Parameter"]    = parameterName,
                ["Value"]        = value,
                ["Unit"]         = parameterName.Equals("AQI Index", StringComparison.OrdinalIgnoreCase) ? "" : unit,
                ["Last Updated"] = lastUpdated
            });
        }

        return results;
    }

    /// <summary>
    /// Filters GetAllSiteData response to sites in a given sector (via DB lookup),
    /// optionally by region, and extracts the requested parameter per site.
    /// Uses the live API values — same source as the Executive Dashboard.
    /// </summary>
    private async Task<List<Dictionary<string, object?>>> BuildSectorPollutantRows(
        string json,
        string parameterName,
        string sectorName,
        string? regionFilter,
        CancellationToken ct)
    {
        // 1. Fetch site IDs for the requested sector from DB, optionally filtered by region
        var sql = regionFilter is not null
            ? """
              SELECT s.ID, s.StationName, r.RegionName
              FROM DMN_Stations s
              JOIN Sectors sec ON s.SectorID = sec.Id
              LEFT JOIN Regions r ON s.RegionID = r.Id
              WHERE sec.SectorName = @sectorName
                AND s.Status = 1
                AND (REPLACE(LOWER(r.RegionName),' ','') = REPLACE(LOWER(@regionName),' ',''))
              ORDER BY s.StationName
              """
            : """
              SELECT s.ID, s.StationName, r.RegionName
              FROM DMN_Stations s
              JOIN Sectors sec ON s.SectorID = sec.Id
              LEFT JOIN Regions r ON s.RegionID = r.Id
              WHERE sec.SectorName = @sectorName
                AND s.Status = 1
              ORDER BY s.StationName
              """;

        var sqlParams = regionFilter is not null
            ? new Dictionary<string, object> { ["sectorName"] = sectorName, ["regionName"] = regionFilter }
            : new Dictionary<string, object> { ["sectorName"] = sectorName };

        var dbResult = await _sqlExecutor.ExecuteAsync(sql, sqlParams, ct);
        if (!dbResult.Success || dbResult.Rows.Count == 0)
            return [];

        // Build lookup: StationId → (StationName, RegionName)
        var schoolSiteIds = dbResult.Rows
            .Where(r => r["ID"] is not null)
            .ToDictionary(
                r => Convert.ToInt32(r["ID"]),
                r => (
                    Name:   r["StationName"]?.ToString() ?? "",
                    Region: r["RegionName"]?.ToString()  ?? ""
                ));

        // 2. Parse the API response and keep only school sites
        var allDevices = ParseDevicesFromJson(json);

        // Group by station, pick the device that has the requested parameter
        // (multiple devices per site — take the best available value)
        var results = new List<Dictionary<string, object?>>();

        foreach (var group in allDevices.GroupBy(d => d.StationId))
        {
            if (!schoolSiteIds.TryGetValue(group.Key, out var siteInfo))
                continue;

            // Find the parameter across all devices at this site
            // Prefer exact name match; also try Unicode subscript variants
            var paramVariants = new[] { parameterName }
                .Concat(_parameterAliases.TryGetValue(parameterName, out var uni) ? [uni] : Array.Empty<string>())
                .ToArray();

            double? value = null;
            string unit   = "";
            string lastUpdated = "";

            bool isAqiParam = paramVariants.Any(v => v.Equals("AQI Index", StringComparison.OrdinalIgnoreCase));
            if (isAqiParam)
            {
                // Average AQI across all devices at this station (mirrors site_aqi_single logic)
                var aqiValues = group
                    .SelectMany(d => paramVariants
                        .Where(v => d.Params.ContainsKey(v))
                        .Select(v => d.Params[v]))
                    .ToList();
                if (aqiValues.Count > 0)
                {
                    value       = Math.Round(aqiValues.Select(r => r.Value).Average(), 0);
                    unit        = aqiValues.First().Unit;
                    lastUpdated = group.Select(d => d.LastUpdated)
                                       .Where(t => !string.IsNullOrWhiteSpace(t))
                                       .OrderByDescending(t => t)
                                       .FirstOrDefault() ?? "";
                }
            }
            else
            {
                foreach (var device in group)
                {
                    foreach (var variant in paramVariants)
                    {
                        if (device.Params.TryGetValue(variant, out var reading))
                        {
                            value       = Math.Round(reading.Value, 2);
                            unit        = reading.Unit;
                            lastUpdated = device.LastUpdated;
                            break;
                        }
                    }
                    if (value.HasValue) break;
                }
            }

            if (value is null) continue; // site has no reading for this parameter

            results.Add(new Dictionary<string, object?>
            {
                ["Site Name"]    = siteInfo.Name,
                ["Region"]       = siteInfo.Region,
                ["Parameter"]    = parameterName,
                ["Value"]        = value,
                ["Unit"]         = parameterName.Equals("AQI Index", StringComparison.OrdinalIgnoreCase) ? "" : unit,
                ["Last Updated"] = lastUpdated
            });
        }

        return results;
    }

    /// <summary>
    /// Aggregates AQI Index per region from GetAllSiteData response — same logic as Executive Dashboard.
    ///
    /// The API returns one DeviceReading per StationId/DriverId combination (multiple rows per station
    /// when a station has multiple device drivers). We mirror the DAL logic exactly:
    ///   1. Per station: average AQI Index across all drivers whose last update is within 24h of
    ///      the station's most recent update (cutoff = latestUpdateTime - 1440 minutes).
    ///   2. Per region: average the station-level AQI values.
    /// </summary>
    private static List<Dictionary<string, object?>> BuildRegionAqiLiveRows(string json, string? regionFilter)
    {
        var allDevices = ParseDevicesFromJson(json);

        // Step 1: per station — average AQI across drivers (mirrors DAL averaging logic)
        var stationAqiValues = allDevices
            .GroupBy(d => new { d.StationId, d.StationName, d.RegionName })
            .Select(g =>
            {
                // Find the most recent update time across all drivers for this station
                var latestUpdate = g
                    .Select(d => d.LastUpdated)
                    .Where(t => !string.IsNullOrWhiteSpace(t))
                    .OrderByDescending(t => t)
                    .FirstOrDefault() ?? "";

                // cutoff = latestUpdate - 1440 minutes (24h), same as DAL
                DateTime.TryParse(latestUpdate, out var latestDt);
                var cutoff = latestDt == default ? DateTime.MinValue : latestDt.AddMinutes(-1440);

                // Collect AQI values from all drivers within the cutoff window
                double sum = 0;
                int count = 0;
                foreach (var device in g)
                {
                    if (!device.Params.TryGetValue("AQI Index", out var aqiReading)) continue;
                    DateTime.TryParse(device.LastUpdated, out var devDt);
                    if (devDt == default || devDt >= cutoff)
                    {
                        sum += aqiReading.Value;
                        count++;
                    }
                }

                double? stationAqi = count > 0 ? sum / count : null;
                return new
                {
                    g.Key.StationId,
                    g.Key.StationName,
                    g.Key.RegionName,
                    AQI = stationAqi,
                    LastUpdated = latestUpdate
                };
            })
            .Where(s => s.AQI.HasValue && !string.IsNullOrWhiteSpace(s.RegionName))
            .ToList();

        // Step 2: per region — average station-level AQI values
        var regionRows = stationAqiValues
            .GroupBy(s => s.RegionName)
            .Select(g => new Dictionary<string, object?>
            {
                ["RegionName"]  = g.Key,
                ["AQI"]         = Math.Round(g.Average(s => s.AQI!.Value), 0),
                ["SiteCount"]   = g.Count(),
                ["LastUpdated"] = g.OrderByDescending(s => s.LastUpdated).First().LastUpdated
            })
            .OrderBy(r => r["RegionName"]?.ToString())
            .ToList();

        // Filter to one region if specified
        if (!string.IsNullOrWhiteSpace(regionFilter))
        {
            var filterNorm = regionFilter.Replace(" ", "").ToLowerInvariant();
            var filtered = regionRows
                .Where(r => r["RegionName"]?.ToString()?.Replace(" ", "").ToLowerInvariant() == filterNorm)
                .ToList();
            return filtered.Count > 0 ? filtered : regionRows;
        }

        return regionRows;
    }

    /// <summary>
    /// Parses the Compliancestatas API response into a single summary row.
    /// Fields: DataSuccessRate (%), ActiveDevices, TotalDevices, AQI, CriticalSiteCount.
    /// </summary>
    private static List<Dictionary<string, object?>> ParseComplianceStatsRows(string json)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            // API may double-serialize — unwrap if root is a string
            if (root.ValueKind == JsonValueKind.String)
            {
                var inner = root.GetString() ?? json;
                return ParseComplianceStatsRows(inner);
            }

            if (root.ValueKind != JsonValueKind.Object)
                return [];

            double? availability = null;
            if (root.TryGetProperty("overallAvailability", out var av))
                availability = av.ValueKind == JsonValueKind.Number ? av.GetDouble() : null;

            int? activeCount = null;
            if (root.TryGetProperty("deviceActiveCount", out var ac))
                activeCount = ac.ValueKind == JsonValueKind.Number ? ac.GetInt32() : null;

            int? inactiveCount = null;
            if (root.TryGetProperty("deviceInactiveCount", out var ic))
                inactiveCount = ic.ValueKind == JsonValueKind.Number ? ic.GetInt32() : null;

            int? totalDevices = (activeCount.HasValue && inactiveCount.HasValue)
                ? activeCount.Value + inactiveCount.Value
                : null;

            int? aqi = null;
            if (root.TryGetProperty("aqi", out var aqiProp) && aqiProp.ValueKind == JsonValueKind.Number)
                aqi = aqiProp.GetInt32();

            int? criticalCount = null;
            if (root.TryGetProperty("criticalSiteCount", out var cs))
                criticalCount = cs.ValueKind == JsonValueKind.Number ? cs.GetInt32() : null;

            return [new Dictionary<string, object?>
            {
                ["DataSuccessRate"]  = availability.HasValue ? (object?)Math.Round(availability.Value, 2) : null,
                ["ActiveDevices"]    = activeCount,
                ["InactiveDevices"]  = inactiveCount,
                ["TotalDevices"]     = totalDevices,
                ["AQI"]             = aqi,
                ["AQICategory"]      = aqi.HasValue ? ClassifyAqi(aqi.Value) : null,
                ["CriticalSiteCount"] = criticalCount
            }];
        }
        catch
        {
            return [];
        }
    }

    /// <summary>
    /// Classifies all sites by their current AQI category, with optional filters:
    /// aqiCategory (Good/Moderate/Unhealthy/…, pipe-separated for multi), sectorName,
    /// regionName, minValue (AQI &gt; N), maxValue (AQI &lt;= N).
    /// "Critical"/"hotspot" maps to AQI &gt; 100.
    /// Returns one row per site: SiteName, RegionName, AQI, AQICategory, LastUpdated.
    /// </summary>
    private async Task<List<Dictionary<string, object?>>> BuildAqiCategoryRows(
        string json,
        string? aqiCategoryFilter,
        string? sectorFilter,
        UserContext? user,
        CancellationToken ct,
        string? regionFilter = null,
        double? minAqi = null,
        double? maxAqi = null)
    {
        // If a sector filter is requested, fetch permitted site IDs from DB.
        // Try exact match first; if that returns nothing, fall back to LIKE match
        // so "Public & Govt-School" / "Public & Gov-School" / partial names all resolve.
        HashSet<int>? sectorSiteIds = null;
        if (!string.IsNullOrWhiteSpace(sectorFilter))
        {
            // Normalise "school" variants → canonical DB value used in BuildSchoolPollutantRows
            var normSector = sectorFilter.Trim();
            if (normSector.Contains("school", StringComparison.OrdinalIgnoreCase)
                || normSector.Contains("govt", StringComparison.OrdinalIgnoreCase)
                || normSector.Contains("gov", StringComparison.OrdinalIgnoreCase))
                normSector = "Public & Govt-School";

            var sql = """
                SELECT s.ID
                FROM DMN_Stations s
                JOIN Sectors sec ON s.SectorID = sec.Id
                WHERE (sec.SectorName = @sectorName
                    OR sec.SectorName LIKE '%' + @sectorLike + '%')
                  AND s.Status = 1
                """;
            // sectorLike: use a distinctive fragment to avoid over-matching
            // For schools: "School", for Commercial: "Commercial", for Residential: "Residential"
            var sectorLike = normSector.Contains("School", StringComparison.OrdinalIgnoreCase) ? "School"
                           : normSector.Contains("Commercial", StringComparison.OrdinalIgnoreCase) ? "Commercial"
                           : normSector.Contains("Residential", StringComparison.OrdinalIgnoreCase) ? "Residential"
                           : normSector;
            var dbResult = await _sqlExecutor.ExecuteAsync(sql,
                new Dictionary<string, object> { ["sectorName"] = normSector, ["sectorLike"] = sectorLike }, ct);
            if (dbResult.Success && dbResult.Rows.Count > 0)
            {
                sectorSiteIds = dbResult.Rows
                    .Where(r => r["ID"] is not null)
                    .Select(r => Convert.ToInt32(r["ID"]))
                    .ToHashSet();
                _log.Debug("ExternalApiService: aqi_category_filter sector '{Sector}' → {Count} site IDs", normSector, sectorSiteIds.Count);
            }
            else
            {
                _log.Warning("ExternalApiService: aqi_category_filter sector lookup returned 0 rows for '{Sector}' — skipping sector filter", sectorFilter);
                // Don't return empty — fall through without sector restriction
            }
        }

        var allDevices = ParseDevicesFromJson(json);

        // RBAC: restrict to permitted sites
        if (user is not null && !user.HasAllSitesAccess && user.PermittedSiteIds.Count > 0)
            allDevices = allDevices.Where(d => user.PermittedSiteIds.Contains(d.StationId)).ToList();

        // Sector filter: restrict to sites in the requested sector
        if (sectorSiteIds is not null)
            allDevices = allDevices.Where(d => sectorSiteIds.Contains(d.StationId)).ToList();

        // Region filter: restrict to sites in the requested region (case/space-insensitive).
        // Applied at device level first; re-applied at site-row level after grouping as a safety net,
        // because some devices may have an empty RegionName and survive the device-level pass.
        string? normRegion = null;
        if (!string.IsNullOrWhiteSpace(regionFilter))
        {
            normRegion = regionFilter.Replace(" ", "").ToLowerInvariant();
            allDevices = allDevices.Where(d =>
                !string.IsNullOrWhiteSpace(d.RegionName) &&
                d.RegionName.Replace(" ", "").Equals(normRegion, StringComparison.OrdinalIgnoreCase))
                .ToList();
        }

        // One row per station — API already pre-averages parameters per station
        var siteRows = allDevices
            .GroupBy(d => d.StationId)
            .Select(g =>
            {
                var first = g.First();
                // Average AQI Index across all devices at this station (mirrors site_aqi_single logic)
                var aqiValues = g.Where(d => d.Params.ContainsKey("AQI Index"))
                                  .Select(d => d.Params["AQI Index"].Value)
                                  .ToList();
                double? aqi = aqiValues.Count > 0 ? Math.Round(aqiValues.Average(), 0) : (double?)null;
                string lastUpdated = g.Select(d => d.LastUpdated)
                                      .Where(t => !string.IsNullOrWhiteSpace(t))
                                      .OrderByDescending(t => t)
                                      .FirstOrDefault() ?? "";
                if (!aqi.HasValue) return null;

                // AQI numeric range filter — applied before category classification
                if (minAqi.HasValue && aqi.Value <= minAqi.Value) return null;
                if (maxAqi.HasValue && aqi.Value > maxAqi.Value)  return null;

                return (object?)new Dictionary<string, object?>
                {
                    ["SiteName"]    = first.StationName,
                    ["RegionName"]  = first.RegionName,
                    ["AQI"]         = aqi,
                    ["AQICategory"] = ClassifyAqi(aqi.Value),
                    ["LastUpdated"] = lastUpdated
                };
            })
            .Where(r => r is not null)
            .Cast<Dictionary<string, object?>>()
            // Second-pass region filter: catches sites whose first device had an empty RegionName
            // but whose row RegionName was populated from a different device in the group.
            .Where(r => normRegion is null ||
                (!string.IsNullOrWhiteSpace(r["RegionName"]?.ToString()) &&
                 r["RegionName"]!.ToString()!.Replace(" ", "").Equals(normRegion, StringComparison.OrdinalIgnoreCase)))
            .OrderBy(r => r["AQI"] is double d ? d : double.MaxValue)
            .ToList();

        // If no category/range filters, return region-filtered rows as-is
        if (string.IsNullOrWhiteSpace(aqiCategoryFilter) && !minAqi.HasValue && !maxAqi.HasValue)
            return siteRows;

        // If only numeric range was requested (no category filter), siteRows already has range
        // applied during construction — just return as-is (no category filtering needed)
        if (string.IsNullOrWhiteSpace(aqiCategoryFilter))
            return siteRows;

        // Map AQI category keywords to canonical DB labels.
        // Covers both pipe-separated LLM output ("Moderate|Very Unhealthy")
        // and natural language the LLM might pass through verbatim.
        _log.Information("ExternalApiService: BuildAqiCategoryRows filtering {Total} sites by aqiCategory='{Filter}'. Present: [{Cats}]",
            siteRows.Count, aqiCategoryFilter,
            string.Join(", ", siteRows.Select(r => r["AQICategory"]?.ToString()).Distinct()));

        static string? ResolveCategory(string word) => word.Trim().ToLowerInvariant() switch
        {
            "good" or "healthy" or "safe" or "clean" or "green"       => "Good",
            "moderate" or "medium" or "average" or "okay" or "ok"     => "Moderate",
            "unhealthy for sensitive groups" or "sensitive" or "sensitive groups"
                or "vulnerable" or "kids" or "children" or "elderly"  => "Unhealthy for Sensitive Groups",
            "unhealthy" or "poor" or "bad" or "harmful" or "polluted" => "Unhealthy",
            "very unhealthy" or "very bad" or "very poor" or "severe" => "Very Unhealthy",
            "hazardous" or "dangerous" or "deadly" or "toxic"         => "Hazardous",
            "critical" or "hotspot" or "hotspots"                     => "critical",  // special: AQI > 100
            _ => null
        };

        // Split on pipe (LLM multi-category format), comma, " and ", " or "
        var rawSegments = aqiCategoryFilter
            .Split(['|', ','], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .SelectMany(s => s.Split([" and ", " or "], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            .Select(s => s.Trim())
            .Where(s => !string.IsNullOrEmpty(s))
            .ToList();

        // Try to resolve each segment. If a segment contains multiple words (e.g. "Very Unhealthy"),
        // try the full segment first, then fall back word-by-word.
        var resolvedCategories = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var seg in rawSegments)
        {
            var resolved = ResolveCategory(seg);
            if (resolved is not null)
            {
                resolvedCategories.Add(resolved);
                continue;
            }
            // Also try the segment directly as a canonical label (LLM may output exact names)
            if (siteRows.Any(r => string.Equals(r["AQICategory"]?.ToString(), seg, StringComparison.OrdinalIgnoreCase)))
                resolvedCategories.Add(seg);
        }

        _log.Information("ExternalApiService: resolved categories = [{Cats}]", string.Join(", ", resolvedCategories));

        if (resolvedCategories.Count == 0)
            return siteRows; // unknown filter — return all rather than empty

        bool wantsCritical = resolvedCategories.Contains("critical");
        var exactCategories = resolvedCategories
            .Where(c => !c.Equals("critical", StringComparison.OrdinalIgnoreCase))
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        return siteRows.Where(r =>
        {
            var cat = r["AQICategory"]?.ToString() ?? "";
            var aqiVal = r["AQI"] is double d ? d : (r["AQI"] is int i ? (double)i : (double?)null);

            if (wantsCritical && aqiVal.HasValue && aqiVal.Value > 100)
                return true;
            if (exactCategories.Count > 0 && exactCategories.Contains(cat))
                return true;
            return false;
        }).ToList();
    }

    // Reference to the parameter alias map from RbacEngine logic (duplicated here for API-path use)
    private static readonly Dictionary<string, string> _parameterAliases = new(StringComparer.OrdinalIgnoreCase)
    {
        ["CO2"]    = "CO\u2082",
        ["O3"]     = "O\u2083",
        ["NO2"]    = "NO\u2082",
        ["CH2O"]   = "CH\u2082O",
        ["SO2"]    = "SO\u2082",
    };

    /// <summary>
    /// Resolves devices under the named site, then queries ParameterAveragesYear for AQI per year
    /// and DeviceCompliance for DSR per year. Returns one row per year ordered ascending.
    /// </summary>
    private async Task<List<Dictionary<string, object?>>> BuildSiteAqiTrendYearlyRows(
        string allSiteJson,
        string stationName,
        int yearsBack,
        UserContext? user,
        CancellationToken ct)
    {
        // ── 1. Match the site from GetAllSiteData ──────────────────────────────
        var allDevices = ParseDevicesFromJson(allSiteJson);
        var query = NormaliseQuery(stationName.Trim().ToLowerInvariant());

        var allStations = allDevices
            .GroupBy(d => d.StationId)
            .Select(g => new { g.First().StationId, g.First().StationName, Devices = g.ToList() })
            .Where(s => !string.IsNullOrWhiteSpace(s.StationName))
            .ToList();

        var permitted = (user is not null && !user.HasAllSitesAccess && user.PermittedSiteIds.Count > 0)
            ? allStations.Where(s => user.PermittedSiteIds.Contains(s.StationId)).ToList()
            : allStations;

        var matched = permitted
            .Select(s =>
            {
                var norm = NormaliseQuery(s.StationName.ToLowerInvariant());
                var score = norm == query ? 1.0 : norm.Contains(query) || query.Contains(norm) ? 0.9 : 0.0;
                return new { s.StationId, s.StationName, s.Devices, Score = score };
            })
            .Where(s => s.Score > 0)
            .OrderByDescending(s => s.Score)
            .FirstOrDefault();

        if (matched is null)
        {
            _log.Warning("site_aqi_trend_yearly: no site matched '{Name}'", stationName);
            return [new Dictionary<string, object?> { ["Error"] = $"No site found matching '{stationName}'." }];
        }

        // ── 2. Get device IDs (from DMN_Parameters for this station) ──────────
        // Use the StationId already resolved — build a comma-list for the SQL IN clause.
        // We query ParameterAveragesYear via DMN_Parameters (which carries DeviceID → StationID chain).
        var stationId = matched.StationId;

        var currentYear = DateTime.UtcNow.Year;
        var startYear = currentYear - yearsBack;  // exclusive lower bound

        // ── 3. Query ParameterAveragesYear + DeviceCompliance via SQL ──────────
        // Group by year, average AQI, compute days-above-100, and DSR from DeviceCompliance.
        var sql = $"""
            SELECT
                YEAR(pay.Interval)                                                          AS [Year],
                ROUND(AVG(CAST(pay.Parametervalue AS FLOAT)), 1)                            AS AvgAQI,
                COUNT(CASE WHEN pay.Parametervalue > 100 THEN 1 END)                        AS HighDays,
                ROUND(
                    100.0 * SUM(CASE WHEN dc.IsActive = 1 THEN 1 ELSE 0 END)
                    / NULLIF(COUNT(DISTINCT CAST(dc.RecordedDate AS DATE)), 0), 1)          AS DSR_Pct
            FROM ParameterAveragesYear pay
            JOIN DMN_Parameters p   ON pay.ParameterID = p.ID
            JOIN DMN_Devices    d   ON p.DeviceID      = d.DeviceId
            JOIN DMN_Stations   s   ON d.StationID     = s.ID
            LEFT JOIN DeviceCompliance dc
                ON  dc.DeviceId      = d.ID
                AND YEAR(dc.RecordedDate) = YEAR(pay.Interval)
            WHERE p.ParameterName = 'AQI Index'
              AND s.ID            = {stationId}
              AND YEAR(pay.Interval) > {startYear}
              AND YEAR(pay.Interval) <= {currentYear}
            GROUP BY YEAR(pay.Interval)
            ORDER BY YEAR(pay.Interval) ASC
            """;

        List<Dictionary<string, object?>> yearRows;
        try
        {
            var result = await _sqlExecutor.ExecuteAsync(sql, new Dictionary<string, object>(), ct);
            yearRows = result.Rows;
        }
        catch (Exception ex)
        {
            _log.Warning(ex, "site_aqi_trend_yearly: SQL failed for stationId={Id}", stationId);
            return [new Dictionary<string, object?> { ["Error"] = "Could not retrieve yearly AQI data for that site." }];
        }

        if (yearRows.Count == 0)
        {
            return [new Dictionary<string, object?>
            {
                ["SiteName"] = matched.StationName,
                ["Error"]    = $"No yearly AQI data found for {matched.StationName} in the last {yearsBack} years."
            }];
        }

        // Attach site name to each row for the formatter
        foreach (var row in yearRows)
            row["SiteName"] = matched.StationName;

        return yearRows;
    }

    // ── Abu Dhabi SDI ArcGIS REST endpoints (public, no auth) ───────────────
    private const string ArcGisStationsUrl =
        "https://arcgis.sdi.abudhabi.ae/agspublish/rest/services/ADPHC/AirQuality/MapServer/0/query?where=1=1&outFields=*&returnGeometry=true&f=json";
    private const string ArcGisAqiDataUrl =
        "https://arcgis.sdi.abudhabi.ae/agspublish/rest/services/ADPHC/AirQuality/MapServer/1/query?where=1=1&outFields=*&orderByFields=ObservedAt%20DESC&resultRecordCount=100&f=json";

    /// <summary>
    /// Fetches ambient (outdoor) stations from Abu Dhabi SDI ArcGIS, finds the one nearest to the
    /// indoor site's coordinates, and returns a single comparison row containing both AQI values.
    /// </summary>
    private async Task<List<Dictionary<string, object?>>> BuildIndoorAmbientCompareRows(
        string indoorJson,
        string stationName,
        string bearerToken,
        CancellationToken ct)
    {
        // ── Step 1: Get indoor site AQI from the already-fetched GetAllSiteData response ──
        var allDevices = ParseDevicesFromJson(indoorJson);
        var query = NormaliseQuery(stationName.Trim().ToLowerInvariant());

        var matched = allDevices
            .GroupBy(d => d.StationId)
            .Select(g => new { g.First().StationId, g.First().StationName, Devices = g.ToList() })
            .Where(s => !string.IsNullOrWhiteSpace(s.StationName))
            .Select(s =>
            {
                var norm = NormaliseQuery(s.StationName.ToLowerInvariant());
                var score = norm == query ? 1.0
                    : norm.Contains(query) || query.Contains(norm) ? 0.9
                    : 0.0;
                return new { s.StationId, s.StationName, s.Devices, Score = score };
            })
            .Where(s => s.Score > 0)
            .OrderByDescending(s => s.Score)
            .FirstOrDefault();

        if (matched is null)
        {
            _log.Warning("indoor_ambient_compare: no indoor site matched '{Name}'", stationName);
            return [new Dictionary<string, object?> { ["Error"] = $"No indoor site found matching '{stationName}'." }];
        }

        // Get indoor AQI (average of AQI Index across all devices at the site)
        var indoorAqiValues = matched.Devices
            .Where(d => d.Params.ContainsKey("AQI Index"))
            .Select(d => d.Params["AQI Index"].Value)
            .ToList();

        double? indoorAqi = indoorAqiValues.Count > 0 ? indoorAqiValues.Average() : null;
        var indoorLastUpdated = matched.Devices
            .Where(d => !string.IsNullOrWhiteSpace(d.LastUpdated))
            .Select(d => d.LastUpdated)
            .FirstOrDefault();

        // ── Step 2: Get indoor site coordinates from DMN_Stations ──
        double? indoorLng = null, indoorLat = null;
        try
        {
            var coordSql = $"""
                SELECT TOP 1 CoordinateX, CoordinateY
                FROM DMN_Stations
                WHERE StationName LIKE '%{stationName.Replace("'", "''")}%'
                  AND Status = 1
                  AND CoordinateX IS NOT NULL AND CoordinateX <> 0
                  AND CoordinateY IS NOT NULL AND CoordinateY <> 0
                """;
            var coordResult = await _sqlExecutor.ExecuteAsync(coordSql, new Dictionary<string, object>(), ct);
            var coordRows = coordResult.Rows;
            if (coordRows.Count > 0)
            {
                var r = coordRows[0];
                if (r.TryGetValue("CoordinateX", out var cx) && double.TryParse(cx?.ToString(), out var lng))
                    indoorLng = lng;
                if (r.TryGetValue("CoordinateY", out var cy) && double.TryParse(cy?.ToString(), out var lat))
                    indoorLat = lat;
            }
        }
        catch (Exception ex)
        {
            _log.Warning(ex, "indoor_ambient_compare: failed to fetch coordinates for '{Site}'", stationName);
        }

        // ── Step 3: Fetch ambient stations from Abu Dhabi SDI ArcGIS ──
        using var arcgisHttp = new HttpClient();
        arcgisHttp.Timeout = TimeSpan.FromSeconds(15);

        string stationsJson, aqiJson;
        try
        {
            var stationsTask = arcgisHttp.GetStringAsync(ArcGisStationsUrl, ct);
            var aqiTask     = arcgisHttp.GetStringAsync(ArcGisAqiDataUrl, ct);
            await Task.WhenAll(stationsTask, aqiTask);
            stationsJson = stationsTask.Result;
            aqiJson      = aqiTask.Result;
        }
        catch (Exception ex)
        {
            _log.Warning(ex, "indoor_ambient_compare: ArcGIS fetch failed");
            return [new Dictionary<string, object?>
            {
                ["IndoorSite"]    = matched.StationName,
                ["IndoorAQI"]     = indoorAqi.HasValue ? Math.Round(indoorAqi.Value, 0) : (object?)"N/A",
                ["AmbientStation"] = "N/A",
                ["AmbientAQI"]    = "N/A",
                ["Distance_km"]   = "N/A",
                ["Timestamp"]     = indoorLastUpdated ?? "N/A",
                ["Error"]         = "Ambient station data temporarily unavailable."
            }];
        }

        // ── Step 4: Parse ArcGIS responses and join on station name ──
        var aqiByAlias = new Dictionary<string, JsonElement>(StringComparer.OrdinalIgnoreCase);
        using var aqiDoc = JsonDocument.Parse(aqiJson);
        foreach (var feat in aqiDoc.RootElement.GetProperty("features").EnumerateArray())
        {
            var attrs = feat.GetProperty("attributes");
            var alias = attrs.TryGetProperty("StationAlias", out var sa) ? sa.GetString() ?? ""
                      : attrs.TryGetProperty("StationName",  out var sn) ? sn.GetString() ?? "" : "";
            if (!string.IsNullOrWhiteSpace(alias) && !aqiByAlias.ContainsKey(alias))
                aqiByAlias[alias] = attrs;
        }

        // ── Step 5: Merge stations with AQI, compute distance, find nearest ──
        double? bestDist = null;
        string? bestAmbientName = null;
        double? bestAmbientAqi = null;
        string? bestAmbientTimestamp = null;

        using var stationsDoc = JsonDocument.Parse(stationsJson);
        foreach (var feat in stationsDoc.RootElement.GetProperty("features").EnumerateArray())
        {
            var attrs  = feat.GetProperty("attributes");
            var geom   = feat.TryGetProperty("geometry", out var g) ? g : default;

            var name = attrs.TryGetProperty("NAME",        out var n1) ? n1.GetString() ?? ""
                     : attrs.TryGetProperty("StationName", out var n2) ? n2.GetString() ?? "" : "";
            if (string.IsNullOrWhiteSpace(name)) continue;

            // Get coordinates
            double ambLng = 0, ambLat = 0;
            if (geom.ValueKind == JsonValueKind.Object)
            {
                if (geom.TryGetProperty("x", out var gx)) ambLng = gx.GetDouble();
                if (geom.TryGetProperty("y", out var gy)) ambLat = gy.GetDouble();
            }
            if (ambLng == 0 && attrs.TryGetProperty("E", out var ea)) ambLng = ea.GetDouble();
            if (ambLat == 0 && attrs.TryGetProperty("N", out var na)) ambLat = na.GetDouble();
            if (ambLng == 0 || ambLat == 0) continue;

            // Get AQI for this station
            double? ambAqi = null;
            string? ambTimestamp = null;
            if (aqiByAlias.TryGetValue(name, out var aqiAttrs))
            {
                if (aqiAttrs.TryGetProperty("AQI", out var aqiEl) && aqiEl.ValueKind != JsonValueKind.Null)
                    ambAqi = aqiEl.TryGetDouble(out var av) ? av : null;
                if (aqiAttrs.TryGetProperty("ObservedAt", out var tsEl) && tsEl.ValueKind != JsonValueKind.Null)
                    ambTimestamp = tsEl.GetString();
            }

            // Haversine distance (km) — only if indoor coordinates are known
            double dist = double.MaxValue;
            if (indoorLng.HasValue && indoorLat.HasValue)
                dist = HaversineKm(indoorLat.Value, indoorLng.Value, ambLat, ambLng);

            if (bestDist == null || dist < bestDist)
            {
                bestDist = dist;
                bestAmbientName = name;
                bestAmbientAqi = ambAqi;
                bestAmbientTimestamp = ambTimestamp;
            }
        }

        if (bestAmbientName is null)
        {
            return [new Dictionary<string, object?>
            {
                ["IndoorSite"] = matched.StationName,
                ["IndoorAQI"]  = indoorAqi.HasValue ? Math.Round(indoorAqi.Value, 0) : (object?)"N/A",
                ["Error"]      = "No ambient stations returned from the ArcGIS service."
            }];
        }

        // ── Step 6: Compute difference and assemble result row ──
        double? diff = (indoorAqi.HasValue && bestAmbientAqi.HasValue)
            ? Math.Round(Math.Abs(indoorAqi.Value - bestAmbientAqi.Value), 1) : null;
        string higherOrLower = (indoorAqi.HasValue && bestAmbientAqi.HasValue)
            ? (indoorAqi.Value > bestAmbientAqi.Value ? "higher" : indoorAqi.Value < bestAmbientAqi.Value ? "lower" : "the same")
            : "unknown";

        return [new Dictionary<string, object?>
        {
            ["IndoorSite"]      = matched.StationName,
            ["IndoorAQI"]       = indoorAqi.HasValue ? (object?)Math.Round(indoorAqi.Value, 0) : "N/A",
            ["AmbientStation"]  = bestAmbientName,
            ["AmbientAQI"]      = bestAmbientAqi.HasValue ? (object?)Math.Round(bestAmbientAqi.Value, 0) : "N/A",
            ["Difference"]      = diff.HasValue ? (object?)diff.Value : "N/A",
            ["HigherOrLower"]   = higherOrLower,
            ["Distance_km"]     = bestDist.HasValue && bestDist < double.MaxValue
                                    ? (object?)Math.Round(bestDist.Value, 1) : "N/A",
            ["Timestamp"]       = indoorLastUpdated ?? bestAmbientTimestamp ?? "N/A",
        }];
    }

    /// <summary>Haversine great-circle distance in km between two lat/lng points.</summary>
    private static double HaversineKm(double lat1, double lon1, double lat2, double lon2)
    {
        const double R = 6371.0;
        var dLat = (lat2 - lat1) * Math.PI / 180.0;
        var dLon = (lon2 - lon1) * Math.PI / 180.0;
        var a = Math.Sin(dLat / 2) * Math.Sin(dLat / 2)
              + Math.Cos(lat1 * Math.PI / 180.0) * Math.Cos(lat2 * Math.PI / 180.0)
              * Math.Sin(dLon / 2) * Math.Sin(dLon / 2);
        return R * 2 * Math.Atan2(Math.Sqrt(a), Math.Sqrt(1 - a));
    }


    /// <inheritdoc/>
    public async Task<ApiCallResult> GetAQIGraphDataAsync(
        string deviceId,
        string stationId,
        string deviceName,
        string criteria,
        DateTime fromDate,
        DateTime toDate,
        string? parameterNameFilter,
        string bearerToken,
        CancellationToken ct = default)
    {
        var sw = Stopwatch.StartNew();
        try
        {
            // GetAQIGraphData API:
            //   criteria=Daily   → returns 24 hourly rows for the given date (1H averages)
            //   criteria=Weekly  → returns 7 daily rows for the week containing the given date (24H averages)
            //   criteria=Monthly → returns monthly summary rows
            //   criteria=Yearly  → returns yearly summary rows
            //
            // For 1H (Daily): send the target day as both Fromdate and Todate.
            //   The controller then filters the returned 24-row set to the hour the user asked for.
            //
            // For 24H (Weekly): send the week's Monday → the target day's date.
            //   The controller then filters the returned 7-row set to the day the user asked for.
            //
            // Date format expected by this API: MM/dd/yyyy (e.g. 09/18/2026)

            string fromStr, toStr;
            if (criteria == "Daily")
            {
                // Return the full day so the controller can filter to the specific hour
                fromStr = fromDate.ToString("MM/dd/yyyy");
                toStr   = fromDate.ToString("MM/dd/yyyy");  // same day — API gives all 24 hourly rows
            }
            else if (criteria == "Weekly")
            {
                // The user asked about a specific day; send a week window ending on toDate
                // so the target day is included among the 7 returned rows
                fromStr = fromDate.ToString("MM/dd/yyyy");
                toStr   = toDate.ToString("MM/dd/yyyy");
            }
            else
            {
                fromStr = fromDate.ToString("MM/dd/yyyy");
                toStr   = toDate.ToString("MM/dd/yyyy");
            }

            var url = _options.BaseUrl.TrimEnd('/')
                + "/api/AirQuality/GetAQIGraphData"
                + $"?deviceId={Uri.EscapeDataString(deviceId)}"
                + $"&criteria={Uri.EscapeDataString(criteria)}"
                + $"&Fromdate={Uri.EscapeDataString(fromStr)}"
                + $"&Todate={Uri.EscapeDataString(toStr)}"
                + "&IsDateFormatJson=true";

            _log.Information("GetAQIGraphData: {Url}", url);

            using var req = new HttpRequestMessage(HttpMethod.Get, url);
            ApplyAuth(req, bearerToken);
            using var resp = await _http.SendAsync(req, ct);
            resp.EnsureSuccessStatusCode();
            var json = await resp.Content.ReadAsStringAsync(ct);

            _log.Information("GetAQIGraphData: raw response ({Len} chars): {Preview}",
                json.Length, json[..Math.Min(500, json.Length)]);

            var rows = ParseAQIGraphDataRows(json, deviceName, parameterNameFilter);

            sw.Stop();
            _log.Information("GetAQIGraphData: {Rows} rows in {Ms}ms (criteria={Criteria})", rows.Count, sw.ElapsedMilliseconds, criteria);
            return new ApiCallResult { Success = true, Rows = rows, ExecutionTimeMs = sw.ElapsedMilliseconds };
        }
        catch (Exception ex)
        {
            sw.Stop();
            _log.Error(ex, "GetAQIGraphData failed for deviceId={Id} criteria={Criteria}", deviceId, criteria);
            return new ApiCallResult { Success = false, Error = ex.Message, ExecutionTimeMs = sw.ElapsedMilliseconds };
        }
    }

    /// <summary>
    /// Parses the GetAQIGraphData JSON response.
    /// The API returns an array of objects where each object represents one time slot.
    /// Each object has a timestamp field (various possible names) and pollutant name keys with numeric values.
    /// Produces one row per pollutant per timestamp:
    ///   DeviceName, ParameterName, ParameterValue, Timestamp
    /// </summary>
    private static List<Dictionary<string, object?>> ParseAQIGraphDataRows(
        string json, string deviceName, string? paramFilter)
    {
        var rows = new List<Dictionary<string, object?>>();
        try
        {
            // Unwrap double-serialized string if needed
            using (var probe = JsonDocument.Parse(json))
            {
                if (probe.RootElement.ValueKind == JsonValueKind.String)
                    json = probe.RootElement.GetString() ?? json;
            }

            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            // Resolve to the data array — accept root array or common object wrappers
            JsonElement dataArray;
            if (root.ValueKind == JsonValueKind.Array)
            {
                dataArray = root;
            }
            else if (root.ValueKind == JsonValueKind.Object)
            {
                if (root.TryGetProperty("Data", out var d1) && d1.ValueKind == JsonValueKind.Array)
                    dataArray = d1;
                else if (root.TryGetProperty("data", out var d2) && d2.ValueKind == JsonValueKind.Array)
                    dataArray = d2;
                else if (root.TryGetProperty("Result", out var d3) && d3.ValueKind == JsonValueKind.Array)
                    dataArray = d3;
                else
                {
                    _log.Warning("ParseAQIGraphDataRows: unexpected root shape, keys={Keys}",
                        string.Join(",", root.EnumerateObject().Select(p => p.Name).Take(10)));
                    return rows;
                }
            }
            else
            {
                _log.Warning("ParseAQIGraphDataRows: unexpected JSON kind={Kind}", root.ValueKind);
                return rows;
            }

            // ── Detect response shape ────────────────────────────────────────────
            //
            // Shape A — root object, pollutant names as top-level keys, each holding an array of data points:
            //   { "AQI Index": [ { "x": "...", "y": 12.5 }, ... ], "PM2.5": [ ... ], ... }
            //   This is what GetAQIGraphData returns.
            //
            // Shape B — array of objects, each element = one pollutant with nested data:
            //   [ { "ParameterName": "PM2.5", "Data": [ { "x": "...", "y": 12.5 }, ... ] }, ... ]
            //
            // Shape C — columnar array, one row per time slot, pollutants as columns:
            //   [ { "Interval": "2026-07-16T08:00:00", "PM2.5": 12.5, "CO2": 800 }, ... ]

            // Shape A — GetAQIGraphData response:
            //   Root object → pollutant name → date key → array of data points
            //   { "AQI Index": { "07/16/2026": [ { "Period1": "2026-07-16 08:00:00", "PollutantValue": 62.0, ... }, ... ] }, "PM2.5": { ... } }
            if (root.ValueKind == JsonValueKind.Object
                && root.EnumerateObject().Any(p => p.Value.ValueKind == JsonValueKind.Object))
            {
                _log.Information("ParseAQIGraphDataRows: detected shape=root-object-pollutant→date→array");

                foreach (var pollutantProp in root.EnumerateObject())
                {
                    var pollutantName = pollutantProp.Name;
                    if (pollutantProp.Value.ValueKind != JsonValueKind.Object) continue;

                    if (!string.IsNullOrWhiteSpace(paramFilter)
                        && !string.Equals(pollutantName, paramFilter, StringComparison.OrdinalIgnoreCase))
                        continue;

                    // Each key inside the pollutant object is a date string, value is the array of readings
                    foreach (var dateProp in pollutantProp.Value.EnumerateObject())
                    {
                        if (dateProp.Value.ValueKind != JsonValueKind.Array) continue;

                        foreach (var pt in dateProp.Value.EnumerateArray())
                        {
                            if (pt.ValueKind != JsonValueKind.Object) continue;

                            // Timestamp: Period1, Period, Interval, DateTime, Timestamp
                            string? timestamp = null;
                            foreach (var tsKey in new[] { "Period1", "Period", "Interval", "DateTime", "Timestamp", "Date", "x" })
                            {
                                if (pt.TryGetProperty(tsKey, out var tsEl))
                                {
                                    timestamp = tsEl.ValueKind == JsonValueKind.String
                                        ? tsEl.GetString()
                                        : tsEl.ToString();
                                    break;
                                }
                            }

                            // Value: PollutantValue, ParameterValue, Value, y
                            double? value = null;
                            foreach (var vKey in new[] { "PollutantValue", "ParameterValue", "Value", "value", "y", "Y" })
                            {
                                if (pt.TryGetProperty(vKey, out var vEl))
                                {
                                    if (vEl.ValueKind == JsonValueKind.Number)
                                        value = vEl.GetDouble();
                                    else if (vEl.ValueKind == JsonValueKind.String && double.TryParse(vEl.GetString(), out var dv))
                                        value = dv;
                                    if (value.HasValue) break;
                                }
                            }
                            if (!value.HasValue) continue;

                            rows.Add(new Dictionary<string, object?>
                            {
                                ["DeviceName"]     = deviceName,
                                ["ParameterName"]  = pollutantName,
                                ["ParameterValue"] = Math.Round(value.Value, 2),
                                ["Timestamp"]      = timestamp
                            });
                        }
                    }
                }

                _log.Information("ParseAQIGraphDataRows: parsed {Count} rows (shape=root-object-pollutant→date→array)", rows.Count);
                return rows;
            }

            if (dataArray.GetArrayLength() == 0)
            {
                _log.Warning("ParseAQIGraphDataRows: API returned empty array");
                return rows;
            }

            var firstEl = dataArray.EnumerateArray().First();
            _log.Information("ParseAQIGraphDataRows: first element kind={Kind} keys={Keys}",
                firstEl.ValueKind,
                firstEl.ValueKind == JsonValueKind.Object
                    ? string.Join(",", firstEl.EnumerateObject().Select(p => p.Name).Take(10))
                    : firstEl.ToString()[..Math.Min(200, firstEl.ToString().Length)]);

            // Shape B: array where each element = one pollutant with nested data array
            bool isPerPollutantShape = firstEl.ValueKind == JsonValueKind.Object
                && firstEl.EnumerateObject().Any(p =>
                    p.Value.ValueKind == JsonValueKind.Array
                    && (p.Name.Equals("Data",       StringComparison.OrdinalIgnoreCase)
                     || p.Name.Equals("data",       StringComparison.OrdinalIgnoreCase)
                     || p.Name.Equals("dataPoints", StringComparison.OrdinalIgnoreCase)
                     || p.Name.Equals("values",     StringComparison.OrdinalIgnoreCase)
                     || p.Name.Equals("series",     StringComparison.OrdinalIgnoreCase)));

            _log.Information("ParseAQIGraphDataRows: detected shape={Shape}", isPerPollutantShape ? "per-pollutant-array" : "columnar");

            if (isPerPollutantShape)
            {
                foreach (var pollutantEl in dataArray.EnumerateArray())
                {
                    if (pollutantEl.ValueKind != JsonValueKind.Object) continue;

                    string? pollutantName = null;
                    foreach (var nameKey in new[] { "ParameterName", "parameterName", "Name", "name", "label", "Label" })
                    {
                        if (pollutantEl.TryGetProperty(nameKey, out var pnEl) && pnEl.ValueKind == JsonValueKind.String)
                        { pollutantName = pnEl.GetString(); break; }
                    }
                    if (string.IsNullOrWhiteSpace(pollutantName)) continue;

                    if (!string.IsNullOrWhiteSpace(paramFilter)
                        && !string.Equals(pollutantName, paramFilter, StringComparison.OrdinalIgnoreCase))
                        continue;

                    JsonElement dataPoints = default;
                    foreach (var dpKey in new[] { "Data", "data", "dataPoints", "values", "series", "Points", "points" })
                    {
                        if (pollutantEl.TryGetProperty(dpKey, out var dpEl) && dpEl.ValueKind == JsonValueKind.Array)
                        { dataPoints = dpEl; break; }
                    }
                    if (dataPoints.ValueKind != JsonValueKind.Array) continue;

                    foreach (var pt in dataPoints.EnumerateArray())
                    {
                        if (pt.ValueKind != JsonValueKind.Object) continue;
                        string? timestamp = null;
                        foreach (var tsKey in new[] { "x", "X", "label", "Label", "Interval", "DateTime", "Timestamp", "Date", "t" })
                        {
                            if (pt.TryGetProperty(tsKey, out var tsEl))
                            { timestamp = tsEl.ValueKind == JsonValueKind.String ? tsEl.GetString() : tsEl.ToString(); break; }
                        }
                        double? value = null;
                        foreach (var vKey in new[] { "y", "Y", "Value", "value", "ParameterValue", "v" })
                        {
                            if (pt.TryGetProperty(vKey, out var vEl))
                            {
                                if (vEl.ValueKind == JsonValueKind.Number) value = vEl.GetDouble();
                                else if (vEl.ValueKind == JsonValueKind.String && double.TryParse(vEl.GetString(), out var dv)) value = dv;
                                if (value.HasValue) break;
                            }
                        }
                        if (!value.HasValue) continue;
                        rows.Add(new Dictionary<string, object?>
                        {
                            ["DeviceName"] = deviceName, ["ParameterName"] = pollutantName,
                            ["ParameterValue"] = Math.Round(value.Value, 2), ["Timestamp"] = timestamp
                        });
                    }
                }
            }
            else
            {
                // Shape C: columnar — one array element per time slot, pollutants as columns
                var tsKeys = new[] { "Interval", "DateTime", "Timestamp", "Date", "Period", "CreatedTime", "x", "X" };
                var skipFields = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
                    { "Interval", "DateTime", "Timestamp", "Date", "Period", "CreatedTime", "x", "X",
                      "DeviceId", "DeviceName", "StationId", "StationName", "Criteria" };

                foreach (var entry in dataArray.EnumerateArray())
                {
                    if (entry.ValueKind != JsonValueKind.Object) continue;
                    string? timestamp = null;
                    foreach (var tsKey in tsKeys)
                    {
                        if (entry.TryGetProperty(tsKey, out var tsEl))
                        { timestamp = tsEl.ValueKind == JsonValueKind.String ? tsEl.GetString() : tsEl.ToString(); break; }
                    }
                    foreach (var prop in entry.EnumerateObject())
                    {
                        if (skipFields.Contains(prop.Name)) continue;
                        if (!string.IsNullOrWhiteSpace(paramFilter)
                            && !string.Equals(prop.Name, paramFilter, StringComparison.OrdinalIgnoreCase)) continue;
                        double? value = null;
                        if (prop.Value.ValueKind == JsonValueKind.Number) value = prop.Value.GetDouble();
                        else if (prop.Value.ValueKind == JsonValueKind.String && double.TryParse(prop.Value.GetString(), out var d)) value = d;
                        else continue;
                        rows.Add(new Dictionary<string, object?>
                        {
                            ["DeviceName"] = deviceName, ["ParameterName"] = prop.Name,
                            ["ParameterValue"] = Math.Round(value.Value, 2), ["Timestamp"] = timestamp
                        });
                    }
                }
            }

            _log.Information("ParseAQIGraphDataRows: parsed {Count} rows (shape={Shape})",
                rows.Count, isPerPollutantShape ? "per-pollutant-array" : "columnar");
        }
        catch (Exception ex)
        {
            _log.Error(ex, "ParseAQIGraphDataRows: parse failed");
        }
        return rows;
    }

    /// <inheritdoc/>
    public async Task<ApiCallResult> GetDeviceLatestDataAsync(
        string deviceId,
        string deviceName,
        string? parameterNameFilter,
        string bearerToken,
        CancellationToken ct = default)
    {
        var sw = Stopwatch.StartNew();
        try
        {
            var url = _options.BaseUrl.TrimEnd('/')
                + $"/api/AirQuality/GetDeviceLatestData?DeviceID={Uri.EscapeDataString(deviceId)}";

            _log.Information("GetDeviceLatestData: {Url}", url);

            using var req = new HttpRequestMessage(HttpMethod.Get, url);
            ApplyAuth(req, bearerToken);
            using var resp = await _http.SendAsync(req, ct);
            resp.EnsureSuccessStatusCode();
            var json = await resp.Content.ReadAsStringAsync(ct);

            var rows = ParseDeviceLatestDataRows(json, deviceName, parameterNameFilter);

            sw.Stop();
            _log.Information("GetDeviceLatestData: {Rows} rows in {Ms}ms (deviceId={Id})", rows.Count, sw.ElapsedMilliseconds, deviceId);
            return new ApiCallResult { Success = true, Rows = rows, ExecutionTimeMs = sw.ElapsedMilliseconds };
        }
        catch (Exception ex)
        {
            sw.Stop();
            _log.Error(ex, "GetDeviceLatestData failed for deviceId={Id}", deviceId);
            return new ApiCallResult { Success = false, Error = ex.Message, ExecutionTimeMs = sw.ElapsedMilliseconds };
        }
    }

    /// <summary>
    /// Parses the GetDeviceLatestData JSON response.
    /// The API returns an array of objects, each with ParameterName, ParameterValue, UnitName, Timestamp.
    /// Also handles the flat DeviceLatestDto shape already used by ParseDeviceLatestRows.
    /// </summary>
    private static List<Dictionary<string, object?>> ParseDeviceLatestDataRows(
        string json, string deviceName, string? paramFilter)
    {
        var rows = new List<Dictionary<string, object?>>();
        try
        {
            // Unwrap double-serialized string if needed
            using (var probe = JsonDocument.Parse(json))
            {
                if (probe.RootElement.ValueKind == JsonValueKind.String)
                    json = probe.RootElement.GetString() ?? json;
            }

            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            // Resolve to an array element — accept root array or { Data: [...] } wrapper
            JsonElement dataArray;
            if (root.ValueKind == JsonValueKind.Array)
            {
                dataArray = root;
            }
            else if (root.ValueKind == JsonValueKind.Object)
            {
                if (root.TryGetProperty("Data", out var d1) && d1.ValueKind == JsonValueKind.Array)
                    dataArray = d1;
                else if (root.TryGetProperty("data", out var d2) && d2.ValueKind == JsonValueKind.Array)
                    dataArray = d2;
                else if (root.TryGetProperty("Result", out var d3) && d3.ValueKind == JsonValueKind.Array)
                    dataArray = d3;
                else
                {
                    _log.Warning("ParseDeviceLatestDataRows: unexpected root shape");
                    return rows;
                }
            }
            else
            {
                _log.Warning("ParseDeviceLatestDataRows: unexpected JSON kind={Kind}", root.ValueKind);
                return rows;
            }

            foreach (var el in dataArray.EnumerateArray())
            {
                if (el.ValueKind != JsonValueKind.Object) continue;

                // DeviceName may be in the element or fall back to the caller-provided name
                var dn = (el.TryGetProperty("DeviceName", out var dnEl) ? dnEl.GetString() : null)
                      ?? deviceName;

                var paramName = (el.TryGetProperty("ParameterName", out var pnEl) ? pnEl.GetString() : null) ?? "";
                if (string.IsNullOrWhiteSpace(paramName)) continue;

                // Apply pollutant filter early
                if (!string.IsNullOrWhiteSpace(paramFilter)
                    && !string.Equals(paramName, paramFilter, StringComparison.OrdinalIgnoreCase))
                    continue;

                var unitName = el.TryGetProperty("UnitName", out var unEl) ? unEl.GetString() : null;

                double? paramValue = null;
                if (el.TryGetProperty("ParameterValue", out var pvEl))
                {
                    if (pvEl.ValueKind == JsonValueKind.Number)
                        paramValue = pvEl.GetDouble();
                    else if (pvEl.ValueKind == JsonValueKind.String && double.TryParse(pvEl.GetString(), out var pv2))
                        paramValue = pv2;
                }

                // Timestamp field — accept several common names
                string? timestamp = null;
                foreach (var tsKey in new[] { "Timestamp", "LastMeasured", "LastUpdated", "DateTime", "CreatedTime" })
                {
                    if (el.TryGetProperty(tsKey, out var tsEl) && tsEl.ValueKind == JsonValueKind.String)
                    {
                        timestamp = tsEl.GetString();
                        break;
                    }
                }

                rows.Add(new Dictionary<string, object?>
                {
                    ["DeviceName"]     = dn,
                    ["ParameterName"]  = paramName,
                    ["ParameterValue"] = paramValue.HasValue ? (object?)Math.Round(paramValue.Value, 2) : null,
                    ["UnitName"]       = paramName == "AQI Index" ? "" : (unitName ?? ""),
                    ["LastMeasured"]   = timestamp
                });
            }
        }
        catch (Exception ex)
        {
            _log.Error(ex, "ParseDeviceLatestDataRows: parse failed");
        }
        return rows;
    }

    /// <summary>
    /// Flattens GetMoldReportsBySiteName JSON response into one row per mold test.
    /// Columns match the UI table: Site Name, Test Name, Sampling Date, Report Number, Result, Status.
    /// Result is derived from SampleNo (pass/fail value stored there in existing data).
    /// </summary>
    private static List<Dictionary<string, object?>> ParseMoldReportRows(string json)
    {
        // Helper: reads a string property trying both PascalCase and camelCase
        static string? GetStr(JsonElement el, string name)
        {
            if (el.TryGetProperty(name, out var v)) return v.GetString();
            var camel = char.ToLowerInvariant(name[0]) + name[1..];
            if (el.TryGetProperty(camel, out var v2)) return v2.GetString();
            return null;
        }

        var rows = new List<Dictionary<string, object?>>();
        try
        {
            using var doc = JsonDocument.Parse(json);
            if (doc.RootElement.ValueKind != JsonValueKind.Array) return rows;

            foreach (var test in doc.RootElement.EnumerateArray())
            {
                // Newtonsoft serializes DTO PascalCase properties as-is (no camelCase conversion)
                var siteName     = GetStr(test, "StationName");
                var testName     = GetStr(test, "TestName");
                var reportNo     = GetStr(test, "ReportNo");
                var sampleNo     = GetStr(test, "SampleNo");
                var samplingTime = GetStr(test, "SamplingTime");
                var formStatus   = GetStr(test, "FormSubmisionStatus");

                rows.Add(new Dictionary<string, object?>
                {
                    ["Site Name"]     = siteName,
                    ["Test Name"]     = testName,
                    ["Sampling Date"] = samplingTime,
                    ["Report Number"] = reportNo,
                    ["Result"]        = sampleNo,   // UI shows sampleNo in the Result column
                    ["Status"]        = formStatus
                });
            }
        }
        catch (Exception ex)
        {
            Log.Warning(ex, "ParseMoldReportRows: failed to parse mold reports JSON");
        }
        return rows;
    }

}
