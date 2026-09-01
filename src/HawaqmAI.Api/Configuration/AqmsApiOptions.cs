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
}
