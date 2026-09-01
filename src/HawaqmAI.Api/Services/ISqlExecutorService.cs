namespace HawaqmAI.Api.Services;

/// <summary>
/// Executes read-only, pre-validated SQL queries against the air quality database.
/// Uses the hawaqm_ai_readonly connection string via Dapper.
/// </summary>
public interface ISqlExecutorService
{
    /// <summary>
    /// Executes a validated SELECT query and returns results as a list of row dictionaries.
    /// </summary>
    /// <param name="sql">Validated SQL string (already passed SqlValidator).</param>
    /// <param name="parameters">Dapper parameter object or dictionary.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>List of rows, each row a dictionary of column→value.</returns>
    Task<SqlExecutionResult> ExecuteAsync(
        string sql,
        Dictionary<string, object> parameters,
        CancellationToken ct = default);
}

/// <summary>Result of a SQL execution including timing and row data.</summary>
public sealed class SqlExecutionResult
{
    public bool Success { get; init; }
    public List<Dictionary<string, object?>> Rows { get; init; } = [];
    public int RowCount => Rows.Count;
    public int ExecutionTimeMs { get; init; }
    public string? Error { get; init; }
}
