using HawaqmAI.Api.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Serilog;

namespace HawaqmAI.Api.Controllers;

/// <summary>
/// Diagnostic endpoint to inspect and manage the site data cache.
/// GET  /api/sitecache        — returns cache status + metadata
/// GET  /api/sitecache/data   — returns the raw cached JSON from GetAllSiteData
/// POST /api/sitecache/refresh — forces an immediate cache refresh
/// </summary>
[ApiController]
[Route("api/sitecache")]
[AllowAnonymous]
public sealed class SiteCacheController : ControllerBase
{
    private static readonly Serilog.ILogger _log = Log.ForContext<SiteCacheController>();
    private readonly ISiteDataCacheService _cache;

    public SiteCacheController(ISiteDataCacheService cache)
    {
        _cache = cache;
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
