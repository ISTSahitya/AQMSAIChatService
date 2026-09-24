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

    /// <summary>
    /// Calls RawDataAnalysisReport for the given device, returning time-series rows for all pollutants.
    /// interval: minutes — 60=1H, 480=8H, 1440=24H, 43200=Monthly, 525600=Yearly.
    /// Date format: MM/dd/yyyy as expected by the API.
    /// Returns rows: DeviceName, ParameterName, ParameterValue, Timestamp.
    /// </summary>
    Task<ApiCallResult> GetAQIGraphDataAsync(
        string deviceId,
        string stationId,
        string deviceName,
        string criteria,
        DateTime fromDate,
        DateTime toDate,
        string? parameterNameFilter,
        string bearerToken,
        CancellationToken ct = default);

    /// <summary>
    /// Calls GetDeviceLatestData for the given device, returning the most recent reading for all pollutants.
    /// Returns rows: DeviceName, ParameterName, ParameterValue, UnitName, LastMeasured.
    /// If parameterNameFilter is non-null, only that pollutant is returned.
    /// </summary>
    Task<ApiCallResult> GetDeviceLatestDataAsync(
        string deviceId,
        string deviceName,
        string? parameterNameFilter,
        string bearerToken,
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
