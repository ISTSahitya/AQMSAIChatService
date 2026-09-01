namespace HawaqmAI.Api.Configuration;

/// <summary>
/// Operational limits for chat sessions, query execution, and data retention.
/// Bound from appsettings.json "Chat" section.
/// </summary>
public sealed class ChatOptions
{
    /// <summary>Maximum number of conversation turns kept in context per session.</summary>
    public int MaxConversationTurns { get; set; } = 15;

    /// <summary>Minutes of inactivity before a session is considered idle.</summary>
    public int SessionTimeoutMinutes { get; set; } = 1440;

    /// <summary>Number of days to retain chat history in the database.</summary>
    public int HistoryRetentionDays { get; set; } = 30;

    /// <summary>Default row limit applied to query results when not specified.</summary>
    public int DefaultRowLimit { get; set; } = 50;

    /// <summary>Absolute maximum rows any single query may return.</summary>
    public int MaxRowLimit { get; set; } = 5000;

    /// <summary>SQL query execution timeout in seconds.</summary>
    public int QueryTimeoutSeconds { get; set; } = 30;

    /// <summary>Number of LLM call retries before returning an error.</summary>
    public int MaxRetryAttempts { get; set; } = 2;

    /// <summary>
    /// Minimum query router confidence score to use a template directly
    /// without sending to LLM for confirmation (0.0–1.0).
    /// </summary>
    public double HighConfidenceThreshold { get; set; } = 0.85;

    /// <summary>
    /// Minimum score to present top-N templates to the LLM for selection.
    /// Below this, the chatbot asks for clarification.
    /// </summary>
    public double LowConfidenceThreshold { get; set; } = 0.50;

    /// <summary>Standard disclaimer appended to every AI response.</summary>
    public string Disclaimer { get; set; } =
        "AI-generated analysis. Verify data before use in official reporting.";
}
