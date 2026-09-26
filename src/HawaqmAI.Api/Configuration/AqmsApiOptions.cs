namespace HawaqmAI.Api.Configuration;

/// <summary>
/// Options for calling the AQMS Web API (AirQuality endpoints).
/// Bound from appsettings.json "AqmsApi" section.
/// </summary>
public sealed class AqmsApiOptions
{
    /// <summary>Base URL of the AQMS Web API, e.g. http://localhost:5000</summary>
    public string BaseUrl { get; set; } = string.Empty;

    /// <summary>Timeout in seconds for API calls (default 30).</summary>
    public int TimeoutSeconds { get; set; } = 30;

    /// <summary>
    /// How often (in minutes) the site data cache refreshes from GetAllSiteData (default 5).
    /// </summary>
    public int CacheRefreshIntervalMinutes { get; set; } = 5;

    /// <summary>
    /// Optional Bearer token for background (non-user-request) calls to the AQMS API.
    /// Used by SiteDataCacheService which has no user cookie context.
    /// Set via AqmsApi:ServiceToken in appsettings or user-secrets.
    /// </summary>
    public string ServiceToken { get; set; } = string.Empty;
}
