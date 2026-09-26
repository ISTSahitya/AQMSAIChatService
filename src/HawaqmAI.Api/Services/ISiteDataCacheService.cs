namespace HawaqmAI.Api.Services;

/// <summary>
/// Singleton cache for GetAllSiteData API responses.
/// Populated on the first real user request and kept warm every 5 minutes
/// if AqmsApi:ServiceToken is configured, otherwise refreshed opportunistically
/// from user requests via UpdateCache().
/// </summary>
public interface ISiteDataCacheService
{
    /// <summary>
    /// Returns the most recently cached raw JSON string from GetAllSiteData.
    /// Returns null if the cache has not been populated yet.
    /// </summary>
    string? GetRawJson();

    /// <summary>
    /// Seeds or updates the cache with fresh JSON obtained from a live user request.
    /// Call this whenever ExternalApiService fetches GetAllSiteData successfully.
    /// </summary>
    void UpdateCache(string json);

    /// <summary>
    /// Returns cache metadata for monitoring/diagnostic purposes.
    /// </summary>
    SiteDataCacheInfo GetCacheInfo();

    /// <summary>
    /// Forces an immediate cache refresh (requires AqmsApi:ServiceToken to be set).
    /// Returns true if the refresh succeeded.
    /// </summary>
    Task<bool> ForceRefreshAsync(CancellationToken ct = default);
}

/// <summary>Metadata about the current state of the site data cache.</summary>
public sealed class SiteDataCacheInfo
{
    /// <summary>UTC timestamp of the last successful refresh.</summary>
    public DateTime? LastRefreshedAt { get; init; }

    /// <summary>UTC timestamp when the next automatic refresh is scheduled.</summary>
    public DateTime? NextRefreshAt { get; init; }

    /// <summary>Number of sites (station groups) currently in cache.</summary>
    public int SiteCount { get; init; }

    /// <summary>Number of device entries currently in cache.</summary>
    public int DeviceCount { get; init; }

    /// <summary>Whether the cache currently holds data.</summary>
    public bool IsPopulated { get; init; }

    /// <summary>Error message from the last refresh attempt, if any.</summary>
    public string? LastError { get; init; }

    /// <summary>Total number of refresh attempts since startup.</summary>
    public int RefreshCount { get; init; }
}
