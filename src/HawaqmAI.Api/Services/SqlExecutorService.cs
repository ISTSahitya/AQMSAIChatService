using System.Diagnostics;
using Dapper;
using HawaqmAI.Api.Configuration;
using Microsoft.Data.SqlClient;
using Microsoft.Extensions.Options;
using Serilog;

namespace HawaqmAI.Api.Services;

/// <summary>
/// Executes read-only queries via Dapper against the air quality database.
/// Never writes to the air quality DB — uses a SELECT-only SQL user.
/// </summary>
public sealed class SqlExecutorService : ISqlExecutorService
{
    private static readonly Serilog.ILogger _log = Log.ForContext<SqlExecutorService>();
    private readonly DatabaseOptions _dbOptions;
    private readonly ChatOptions _chatOptions;

    public SqlExecutorService(
        IOptions<DatabaseOptions> dbOptions,
        IOptions<ChatOptions> chatOptions)
    {
        _dbOptions = dbOptions.Value;
        _chatOptions = chatOptions.Value;
    }

    /// <inheritdoc/>
    public async Task<SqlExecutionResult> ExecuteAsync(
        string sql,
        Dictionary<string, object> parameters,
        CancellationToken ct = default)
    {
        var sw = Stopwatch.StartNew();

        try
        {
            await using var conn = new SqlConnection(_dbOptions.AirQualityConnection);
            await conn.OpenAsync(ct);

            var command = new CommandDefinition(
                commandText: sql,
                parameters: ConvertToDapperParams(parameters),
                commandTimeout: _chatOptions.QueryTimeoutSeconds,
                cancellationToken: ct);

            var rawResults = await conn.QueryAsync(command);
            var rows = rawResults
                .Select(row => ((IDictionary<string, object>)row)
                    .ToDictionary(
                        k => k.Key,
                        k => k.Value is DBNull ? null : (object?)k.Value))
                .ToList();

            sw.Stop();
            _log.Information(
                "SQL executed: {Rows} rows in {Ms}ms | SQL={Sql}",
                rows.Count,
                sw.ElapsedMilliseconds,
                sql[..Math.Min(200, sql.Length)]);

            return new SqlExecutionResult
            {
                Success = true,
                Rows = rows,
                ExecutionTimeMs = (int)sw.ElapsedMilliseconds
            };
        }
        catch (SqlException ex) when (ex.Number == -2)
        {
            // Timeout
            sw.Stop();
            _log.Error(ex, "SQL query timed out after {Timeout}s | SQL={Sql}",
                _chatOptions.QueryTimeoutSeconds, sql[..Math.Min(200, sql.Length)]);
            return new SqlExecutionResult
            {
                Success = false,
                Error = "Query timed out. Please narrow your date range or contact support.",
                ExecutionTimeMs = (int)sw.ElapsedMilliseconds
            };
        }
        catch (Exception ex)
        {
            sw.Stop();
            _log.Error(ex, "SQL execution failed | SQL={Sql}", sql[..Math.Min(200, sql.Length)]);
            return new SqlExecutionResult
            {
                Success = false,
                Error = "Data retrieval failed. Please try again or rephrase your question.",
                ExecutionTimeMs = (int)sw.ElapsedMilliseconds
            };
        }
    }

    private static DynamicParameters ConvertToDapperParams(Dictionary<string, object> parameters)
    {
        var dp = new DynamicParameters();
        foreach (var (key, value) in parameters)
        {
            // Dapper handles List<int> as a table-valued parameter for IN clauses
            dp.Add(key, value);
        }
        return dp;
    }
}
