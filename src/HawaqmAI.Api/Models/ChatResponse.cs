namespace HawaqmAI.Api.Models;

/// <summary>Full chat response sent back to the frontend.</summary>
public sealed class ChatResponse
{
    /// <summary>Whether the request completed successfully.</summary>
    public bool Success { get; set; }

    /// <summary>Natural language summary or answer (always present).</summary>
    public string Message { get; set; } = string.Empty;

    /// <summary>The validated SQL that was executed (shown to user for transparency).</summary>
    public string? Sql { get; set; }

    /// <summary>How the response should be rendered: "text", "table", or "chart".</summary>
    public string ResponseType { get; set; } = "text";

    /// <summary>Chart type when ResponseType == "chart": "line", "bar", "area", "scatter".</summary>
    public string? ChartType { get; set; }

    /// <summary>Tabular or chart data rows as an array of key-value dictionaries.</summary>
    public List<Dictionary<string, object?>>? Data { get; set; }

    /// <summary>Column headers for table display (derived from Data keys if null).</summary>
    public List<string>? Columns { get; set; }

    /// <summary>Response metadata: source, timing, model, disclaimer.</summary>
    public ResponseMetadata? Metadata { get; set; }

    /// <summary>Session ID (new or existing) for the frontend to track continuity.</summary>
    public Guid SessionId { get; set; }

    /// <summary>Message ID of this assistant turn (for pinning and feedback).</summary>
    public Guid MessageId { get; set; }

    /// <summary>Error detail when Success == false.</summary>
    public string? Error { get; set; }

    /// <summary>Error code for frontend handling (e.g. "CLARIFY", "UNAUTHORIZED").</summary>
    public string? ErrorCode { get; set; }

    /// <summary>Clarification prompt when the query was ambiguous.</summary>
    public string? ClarificationPrompt { get; set; }
}

/// <summary>Metadata attached to every AI response (satisfies NFR-13).</summary>
public sealed class ResponseMetadata
{
    /// <summary>Database table(s) queried.</summary>
    public string? DataSource { get; set; }

    /// <summary>Human-readable date range covered (e.g. "Jul 1 – Jul 31, 2026").</summary>
    public string? DateRange { get; set; }

    /// <summary>Number of rows returned.</summary>
    public int RowCount { get; set; }

    /// <summary>End-to-end query execution time in milliseconds.</summary>
    public int ExecutionTimeMs { get; set; }

    /// <summary>LLM model name used to generate the response.</summary>
    public string? ModelUsed { get; set; }

    /// <summary>Total LLM token usage for this response.</summary>
    public int TokenCount { get; set; }

    /// <summary>Approved query template ID that was selected.</summary>
    public string? QueryTemplateId { get; set; }

    /// <summary>Standard AI disclaimer (always present, per NFR-13).</summary>
    public string Disclaimer { get; set; } =
        "AI-generated analysis. Verify data before use in official reporting.";
}
