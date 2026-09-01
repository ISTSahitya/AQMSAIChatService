using HawaqmAI.Api.Models;

namespace HawaqmAI.Api.Services;

/// <summary>
/// Formats raw SQL results and LLM summaries into the final <see cref="ChatResponse"/>.
/// Always includes metadata: data source, date range, disclaimer (per NFR-13).
/// </summary>
public interface IResponseFormatterService
{
    /// <summary>
    /// Builds a complete ChatResponse from all pipeline outputs.
    /// </summary>
    ChatResponse Format(
        Guid sessionId,
        Guid messageId,
        ApprovedQuery template,
        string llmSummary,
        List<Dictionary<string, object?>> rows,
        int rowCount,
        int executionTimeMs,
        ResolvedScope scope,
        string? modelUsed,
        int tokenCount);

    /// <summary>Builds an error response (e.g. for clarification requests or auth failures).</summary>
    ChatResponse FormatError(
        Guid sessionId,
        string errorMessage,
        string errorCode,
        string? clarificationPrompt = null);

    /// <summary>
    /// Classifies an AQI value into a MOCCAE category label.
    /// </summary>
    string ClassifyAqi(double aqiValue);
}
