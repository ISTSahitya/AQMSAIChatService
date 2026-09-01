namespace HawaqmAI.Api.Configuration;

/// <summary>
/// Database connection string options.
/// Bound from appsettings.json "Database" section.
/// </summary>
public sealed class DatabaseOptions
{
    /// <summary>
    /// Read-only connection for air quality data tables.
    /// Uses hawaqm_ai_readonly SQL user.
    /// </summary>
    public string AirQualityConnection { get; set; } = string.Empty;

    /// <summary>
    /// Read-write connection for chat session/message persistence.
    /// Uses hawaqm_ai_chat SQL user.
    /// </summary>
    public string ChatHistoryConnection { get; set; } = string.Empty;
}
