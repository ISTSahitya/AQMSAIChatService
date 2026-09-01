using HawaqmAI.Api.Models;

namespace HawaqmAI.Api.Services;

/// <summary>
/// Calls the AQMS Web API on behalf of the chat service,
/// forwarding the user's JWT token and flattening the response
/// into the standard row format used by the rest of the pipeline.
/// </summary>
public interface IExternalApiService
{
    /// <summary>
    /// Calls the AQMS API endpoint defined in <paramref name="apiCall"/>,
    /// substituting LLM parameters and scope values into the path,
    /// and returns flattened rows for the formatter.
    /// </summary>
    Task<ApiCallResult> CallAsync(
        ApiCallDefinition apiCall,
        Dictionary<string, string> llmParams,
        ResolvedScope scope,
        string bearerToken,
        UserContext? user = null,
        string? templateId = null,
        CancellationToken ct = default);
}

/// <summary>Result of an external API call, parallel to <see cref="SqlExecutionResult"/>.</summary>
public sealed class ApiCallResult
{
    public bool Success { get; init; }
    public string? Error { get; init; }
    public List<Dictionary<string, object?>> Rows { get; init; } = [];
    public long ExecutionTimeMs { get; init; }
}
