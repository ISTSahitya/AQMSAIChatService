using HawaqmAI.Api.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Serilog;

namespace HawaqmAI.Api.Controllers;

/// <summary>
/// Diagnostic endpoint to inspect and manage the site data cache.
/// GET  /api/sitecache                    — returns cache status + metadata
/// GET  /api/sitecache/data               — returns the raw cached JSON from GetAllSiteData
/// POST /api/sitecache/refresh            — forces an immediate cache refresh
/// GET  /api/sitecache/nearest-ambient    — debug: find the nearest ambient station for a site
/// </summary>
[ApiController]
[Route("api/sitecache")]
[AllowAnonymous]
public sealed class SiteCacheController : ControllerBase
{
    private static readonly Serilog.ILogger _log = Log.ForContext<SiteCacheController>();
    private readonly ISiteDataCacheService _cache;
    private readonly ISqlExecutorService _sql;

    public SiteCacheController(ISiteDataCacheService cache, ISqlExecutorService sql)
    {
        _cache = cache;
        _sql   = sql;
    }

    /// <summary>
    /// Returns metadata about the current state of the site data cache.
    /// </summary>
    [HttpGet]
    [ProducesResponseType(typeof(SiteCacheStatusResponse), 200)]
    public IActionResult GetStatus()
    {
        var info = _cache.GetCacheInfo();
        var response = new SiteCacheStatusResponse
        {
            IsPopulated       = info.IsPopulated,
            SiteCount         = info.SiteCount,
            DeviceCount       = info.DeviceCount,
            LastRefreshedAt   = info.LastRefreshedAt,
            NextRefreshAt     = info.NextRefreshAt,
            RefreshCount      = info.RefreshCount,
            LastError         = info.LastError
        };
        return Ok(response);
    }

    /// <summary>
    /// Returns the raw cached JSON from GetAllSiteData.
    /// Useful for debugging what the AI service sees.
    /// </summary>
    [HttpGet("data")]
    [ProducesResponseType(200)]
    [ProducesResponseType(503)]
    public IActionResult GetData()
    {
        var json = _cache.GetRawJson();
        if (json is null)
        {
            return StatusCode(503, new { success = false, error = "Site data cache is not yet populated. Try again shortly." });
        }

        // Return as raw JSON content (not double-serialised)
        return Content(json, "application/json");
    }

    /// <summary>
    /// Debug: looks up an indoor site's coordinates from DMN_Stations, then finds
    /// the nearest active AmbientStation by Haversine distance.
    /// GET /api/sitecache/nearest-ambient?siteName=Al+Saad+Indian+School
    /// </summary>
    [HttpGet("nearest-ambient")]
    [ProducesResponseType(typeof(NearestAmbientResponse), 200)]
    [ProducesResponseType(400)]
    public async Task<IActionResult> GetNearestAmbient([FromQuery] string siteName, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(siteName))
            return BadRequest(new { error = "siteName query parameter is required." });

        // Step 1: resolve indoor site coordinates from DMN_Stations
        var coordSql = $"""
            SELECT TOP 1
                s.StationName,
                TRY_CAST(s.CoordinateX AS FLOAT) AS Lat,
                TRY_CAST(s.CoordinateY AS FLOAT) AS Lng
            FROM DMN_Stations s
            WHERE s.StationName LIKE '%{siteName.Replace("'", "''")}%'
              AND ISNULL(s.IsDeleted, 0) = 0
              AND TRY_CAST(s.CoordinateX AS FLOAT) IS NOT NULL
              AND TRY_CAST(s.CoordinateY AS FLOAT) IS NOT NULL
              AND TRY_CAST(s.CoordinateX AS FLOAT) <> 0
              AND TRY_CAST(s.CoordinateY AS FLOAT) <> 0
            ORDER BY s.StationName
            """;

        var coordResult = await _sql.ExecuteAsync(coordSql, new Dictionary<string, object>(), ct);
        if (coordResult.Rows.Count == 0)
            return Ok(new NearestAmbientResponse { Error = $"No indoor site with coordinates found matching '{siteName}'." });

        var cr = coordResult.Rows[0];
        var resolvedSiteName = cr["StationName"]?.ToString() ?? siteName;
        if (!double.TryParse(cr["Lat"]?.ToString(), out var lat) ||
            !double.TryParse(cr["Lng"]?.ToString(), out var lng))
            return Ok(new NearestAmbientResponse
            {
                IndoorSite = resolvedSiteName,
                Error = "CoordinateX/Y could not be parsed as float."
            });

        // Step 2: find nearest active AmbientStation by Haversine
        var latS = lat.ToString("F6", System.Globalization.CultureInfo.InvariantCulture);
        var lngS = lng.ToString("F6", System.Globalization.CultureInfo.InvariantCulture);

        var ambSql = $"""
            SELECT TOP 5
                amb.StationName  AS AmbientStation,
                amb.Latitude,
                amb.Longitude,
                amb.Status,
                ROUND(SQRT(
                    (({lngS} - amb.Longitude) * COS(RADIANS((amb.Latitude + {latS}) / 2.0)) * 111.32) *
                    (({lngS} - amb.Longitude) * COS(RADIANS((amb.Latitude + {latS}) / 2.0)) * 111.32) +
                    ((amb.Latitude - {latS}) * 111.32) *
                    ((amb.Latitude - {latS}) * 111.32)
                ), 2) AS Distance_km
            FROM AmbientStations amb
            WHERE amb.Status = 'Active'
              AND amb.Latitude  IS NOT NULL
              AND amb.Longitude IS NOT NULL
            ORDER BY SQRT(
                (({lngS} - amb.Longitude) * COS(RADIANS((amb.Latitude + {latS}) / 2.0)) * 111.32) *
                (({lngS} - amb.Longitude) * COS(RADIANS((amb.Latitude + {latS}) / 2.0)) * 111.32) +
                ((amb.Latitude - {latS}) * 111.32) *
                ((amb.Latitude - {latS}) * 111.32)
            )
            """;

        var ambResult = await _sql.ExecuteAsync(ambSql, new Dictionary<string, object>(), ct);

        var candidates = ambResult.Rows.Select(r => new AmbientCandidate
        {
            StationName = r["AmbientStation"]?.ToString(),
            Latitude    = double.TryParse(r["Latitude"]?.ToString(),   out var alat) ? alat : null,
            Longitude   = double.TryParse(r["Longitude"]?.ToString(),  out var alng) ? alng : null,
            Status      = r["Status"]?.ToString(),
            Distance_km = double.TryParse(r["Distance_km"]?.ToString(), out var dist) ? dist : null,
        }).ToList();

        _log.Information("nearest-ambient debug: site='{Site}' lat={Lat} lng={Lng} → nearest='{Nearest}' dist={Dist}km",
            resolvedSiteName, lat, lng,
            candidates.FirstOrDefault()?.StationName ?? "(none)",
            candidates.FirstOrDefault()?.Distance_km);

        return Ok(new NearestAmbientResponse
        {
            IndoorSite   = resolvedSiteName,
            IndoorLat    = lat,
            IndoorLng    = lng,
            TopCandidates = candidates
        });
    }

    /// <summary>
    /// Forces an immediate cache refresh from GetAllSiteData.
    /// </summary>
    [HttpPost("refresh")]
    [ProducesResponseType(typeof(SiteCacheRefreshResponse), 200)]
    [ProducesResponseType(typeof(SiteCacheRefreshResponse), 502)]
    public async Task<IActionResult> ForceRefresh(CancellationToken ct)
    {
        _log.Information("SiteCache: manual refresh triggered by user {User}",
            User.Identity?.Name ?? "unknown");

        var success = await _cache.ForceRefreshAsync(ct);
        var info    = _cache.GetCacheInfo();

        var response = new SiteCacheRefreshResponse
        {
            Success         = success,
            SiteCount       = info.SiteCount,
            DeviceCount     = info.DeviceCount,
            LastRefreshedAt = info.LastRefreshedAt,
            Error           = info.LastError
        };

        return success ? Ok(response) : StatusCode(502, response);
    }
}

/// <summary>Cache status response payload.</summary>
public sealed class SiteCacheStatusResponse
{
    public bool IsPopulated { get; init; }
    public int SiteCount { get; init; }
    public int DeviceCount { get; init; }
    public DateTime? LastRefreshedAt { get; init; }
    public DateTime? NextRefreshAt { get; init; }
    public int RefreshCount { get; init; }
    public string? LastError { get; init; }
}

/// <summary>Force-refresh response payload.</summary>
public sealed class SiteCacheRefreshResponse
{
    public bool Success { get; init; }
    public int SiteCount { get; init; }
    public int DeviceCount { get; init; }
    public DateTime? LastRefreshedAt { get; init; }
    public string? Error { get; init; }
}

/// <summary>Nearest ambient station debug response.</summary>
public sealed class NearestAmbientResponse
{
    public string? IndoorSite { get; init; }
    public double? IndoorLat { get; init; }
    public double? IndoorLng { get; init; }
    public List<AmbientCandidate> TopCandidates { get; init; } = [];
    public string? Error { get; init; }
}

/// <summary>One ambient station candidate with distance.</summary>
public sealed class AmbientCandidate
{
    public string? StationName { get; set; }
    public double? Latitude { get; set; }
    public double? Longitude { get; set; }
    public string? Status { get; set; }
    public double? Distance_km { get; set; }
}
