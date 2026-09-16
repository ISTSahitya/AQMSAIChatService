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

    public ExternalApiService(
        IHttpClientFactory httpClientFactory,
        IOptions<AqmsApiOptions> options,
        IStationResolverService stationResolver,
        ISqlExecutorService sqlExecutor,
        IHttpContextAccessor httpContextAccessor)
    {
        _options = options.Value;
        _http = httpClientFactory.CreateClient("AqmsApi");
        _stationResolver = stationResolver;
        _sqlExecutor = sqlExecutor;
        _httpContextAccessor = httpContextAccessor;
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

                // Always call GetAllSiteData — it includes StationName per device
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

                var allJson = await allResp.Content.ReadAsStringAsync(ct);

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

                // GetAllSiteData does not include DeviceName — call GetSiteData?siteId=X
                // which is the confirmed source that includes DeviceName per device.
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
                var siteDevices = ParseDevicesFromJson(siteJson)
                    .Select(d => d with { StationName = string.IsNullOrWhiteSpace(d.StationName) ? matched.Name : d.StationName })
                    .ToList();

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

                // If the site has multiple devices, take the value from the most recently
                // updated device per parameter — same logic as the Executive Dashboard.
                // Averaging across devices gives wrong values when devices have different
                // roles or update cycles (e.g. primary vs secondary sensor).
                if (siteDevices.Count > 1)
                {
                    allParamRows = allParamRows
                        .GroupBy(r => r["Parameter"]?.ToString() ?? "")
                        .Select(g =>
                        {
                            // Pick the row whose Last Updated timestamp is most recent
                            var best = g
                                .OrderByDescending(r => r["Last Updated"]?.ToString() ?? "")
                                .First();
                            return best;
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
                    return new ApiCallResult { Success = false, Error = "Please specify a device name (e.g. 'BA0010', 'SEI100M0114').", ExecutionTimeMs = sw.ElapsedMilliseconds };

                var idResult = await _sqlExecutor.ExecuteAsync(
                    "SELECT TOP 1 d.DeviceId AS DeviceId, d.DeviceName, d.StationID FROM DMN_Devices d WHERE d.DeviceName = @deviceName OR REPLACE(LOWER(d.DeviceName),' ','') = REPLACE(LOWER(@deviceName),' ','') ORDER BY CASE WHEN d.DeviceName = @deviceName THEN 0 ELSE 1 END, d.DeviceName",
                    new Dictionary<string, object> { ["deviceName"] = deviceName }, ct);

                if (!idResult.Success || idResult.Rows.Count == 0)
                    return new ApiCallResult { Success = false, Error = $"No device found matching '{deviceName}'. Please check the device name.", ExecutionTimeMs = sw.ElapsedMilliseconds };

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

            var json = await response.Content.ReadAsStringAsync(ct);
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
            if (apiCall.ResponseShape == "aqi_category_filter")
            {
                var aqiCategory = llmParams.GetValueOrDefault("aqiCategory");
                var rows2 = BuildAqiCategoryRows(json, aqiCategory, user);
                _log.Debug("ExternalApiService: aqi_category_filter → {Count} rows category={Cat}",
                    rows2.Count, aqiCategory ?? "all");
                return new ApiCallResult { Success = true, Rows = rows2, ExecutionTimeMs = sw.ElapsedMilliseconds };
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
        string RegionName = "");

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

        return new DeviceReading(stationId, stationName, deviceName, isOnline, lastUpdated, parameters, regionName);
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
                var aqiValues = g
                    .Where(d => d.Params.ContainsKey("AQI Index"))
                    .Select(d => d.Params["AQI Index"].Value)
                    .ToList();

                var avgAqi = aqiValues.Count > 0
                    ? Math.Round(aqiValues.Average(), 1)
                    : (double?)null;

                var onlineCount  = g.Count(d => d.IsOnline);
                var totalDevices = g.Count();

                var stationName = g.First().StationName;

                // Pick the most recent LastUpdated across all devices in this station
                var lastUpdated = g
                    .Select(d => d.LastUpdated)
                    .Where(t => !string.IsNullOrWhiteSpace(t))
                    .OrderByDescending(t => t)
                    .FirstOrDefault();

                var row = new Dictionary<string, object?>
                {
                    ["Site Name"]    = stationName,
                    ["AQI"]          = avgAqi,
                    ["AQI Category"] = avgAqi.HasValue ? ClassifyAqi(avgAqi.Value) : "Unknown",
                    ["Last Updated"] = lastUpdated ?? "N/A"
                };

                return row;
            })
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
    /// Filters GetAllSiteData response to sites whose StationName contains "school"
    /// (case-insensitive DB lookup), optionally by region, extracts the requested parameter,
    /// and applies optional minValue / maxValue filters from llmParams.
    /// </summary>
    private async Task<List<Dictionary<string, object?>>> BuildSchoolPollutantRows(
        string json,
        string parameterName,
        string? regionFilter,
        Dictionary<string, string> llmParams,
        CancellationToken ct)
    {
        // Fetch all sites whose name contains "school" from DB, optionally filtered by region
        var sql = regionFilter is not null
            ? """
              SELECT s.ID, s.StationName, r.RegionName
              FROM DMN_Stations s
              LEFT JOIN Regions r ON s.RegionID = r.Id
              WHERE LOWER(s.StationName) LIKE '%school%'
                AND s.Status = 1
                AND (REPLACE(LOWER(r.RegionName),' ','') = REPLACE(LOWER(@regionName),' ',''))
              ORDER BY s.StationName
              """
            : """
              SELECT s.ID, s.StationName, r.RegionName
              FROM DMN_Stations s
              LEFT JOIN Regions r ON s.RegionID = r.Id
              WHERE LOWER(s.StationName) LIKE '%school%'
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
    /// Classifies all sites by their current AQI category, optionally filtering to one category.
    /// "Critical" and "hotspot" map to Unhealthy + Very Unhealthy + Hazardous (AQI > 100).
    /// Returns one row per site: SiteName, RegionName, AQI, AQICategory, LastUpdated.
    /// </summary>
    private static List<Dictionary<string, object?>> BuildAqiCategoryRows(
        string json,
        string? aqiCategoryFilter,
        UserContext? user)
    {
        var allDevices = ParseDevicesFromJson(json);

        // RBAC: restrict to permitted sites
        if (user is not null && !user.HasAllSitesAccess && user.PermittedSiteIds.Count > 0)
            allDevices = allDevices.Where(d => user.PermittedSiteIds.Contains(d.StationId)).ToList();

        // One row per station — API already pre-averages parameters per station
        var siteRows = allDevices
            .GroupBy(d => d.StationId)
            .Select(g =>
            {
                var first = g.First();
                // Find AQI Index parameter across all device readings for this station
                double? aqi = null;
                string lastUpdated = "";
                foreach (var device in g)
                {
                    if (device.Params.TryGetValue("AQI Index", out var aqiReading))
                    {
                        aqi = Math.Round(aqiReading.Value, 0);
                        lastUpdated = device.LastUpdated;
                        break;
                    }
                }
                if (!aqi.HasValue) return null;

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
            .OrderBy(r => r["AQI"] is double d ? d : double.MaxValue)
            .ToList();

        if (string.IsNullOrWhiteSpace(aqiCategoryFilter))
            return siteRows;

        // "critical" and "hotspot" = AQI > 100 (all unhealthy tiers combined)
        var lower = aqiCategoryFilter.Trim().ToLowerInvariant();
        if (lower == "critical" || lower == "hotspot")
            return siteRows.Where(r => r["AQI"] is double d && d > 100).ToList();

        // Exact category match (case-insensitive)
        return siteRows
            .Where(r => string.Equals(r["AQICategory"]?.ToString(), aqiCategoryFilter, StringComparison.OrdinalIgnoreCase))
            .ToList();
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
}
