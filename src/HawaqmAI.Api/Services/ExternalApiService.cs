using System.Diagnostics;
using System.Net.Http.Headers;
using System.Text.Json;
using HawaqmAI.Api.Configuration;
using HawaqmAI.Api.Models;
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

    public ExternalApiService(
        IHttpClientFactory httpClientFactory,
        IOptions<AqmsApiOptions> options,
        IStationResolverService stationResolver,
        ISqlExecutorService sqlExecutor)
    {
        _options = options.Value;
        _http = httpClientFactory.CreateClient("AqmsApi");
        _stationResolver = stationResolver;
        _sqlExecutor = sqlExecutor;
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
                if (!string.IsNullOrWhiteSpace(bearerToken))
                    allReq.Headers.Authorization = new AuthenticationHeaderValue("Bearer", bearerToken);

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
                if (!string.IsNullOrWhiteSpace(bearerToken))
                    siteReq.Headers.Authorization = new AuthenticationHeaderValue("Bearer", bearerToken);

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
                var allParamRows = siteDevices
                    .SelectMany(d => d.Params.Select(p => new Dictionary<string, object?>
                    {
                        ["Site Name"]    = matched.Name,
                        ["Parameter"]    = p.Key,
                        ["Value"]        = Math.Round(p.Value.Value, 2),
                        ["Unit"]         = p.Value.Unit,
                        ["Last Updated"] = d.LastUpdated
                    }))
                    .ToList();

                // If the site has multiple devices, average each parameter across devices
                if (siteDevices.Count > 1)
                {
                    allParamRows = allParamRows
                        .GroupBy(r => r["Parameter"]?.ToString() ?? "")
                        .Select(g =>
                        {
                            var avgVal = g.Select(r => r["Value"] is double v ? v : 0).Average();
                            return new Dictionary<string, object?>
                            {
                                ["Site Name"]    = matched.Name,
                                ["Parameter"]    = g.Key,
                                ["Value"]        = Math.Round(avgVal, 2),
                                ["Unit"]         = g.First()["Unit"],
                                ["Last Updated"] = g.First()["Last Updated"]
                            };
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

            // For region_aqi shape: default year to current year if not provided by the user
            if (apiCall.ResponseShape == "region_aqi" && !llmParams.ContainsKey("year"))
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
                if (!string.IsNullOrWhiteSpace(bearerToken))
                    req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", bearerToken);
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

            // schools_pollutant: filter API results to school-sector sites via DB, then extract requested parameter
            if (apiCall.ResponseShape == "schools_pollutant")
            {
                var parameterName = llmParams.GetValueOrDefault("parameterName") ?? "CO2";
                var rows2 = await BuildSectorPollutantRows(json, parameterName, "Public & Govt-School", regionFilter, ct);
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
        Dictionary<string, (double Value, string Unit)> Params);

    private static DeviceReading ParseDevice(JsonElement el)
    {
        var stationId   = el.TryGetProperty("StationId",   out var sid)  ? sid.GetInt32()        : 0;
        var stationName =
            (el.TryGetProperty("StationName", out var sn)  ? sn.GetString()  : null) ??
            (el.TryGetProperty("stationName", out var sn2) ? sn2.GetString() : null) ??
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

        return new DeviceReading(stationId, stationName, deviceName, isOnline, lastUpdated, parameters);
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
