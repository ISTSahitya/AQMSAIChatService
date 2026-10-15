using System.Text.RegularExpressions;
using HawaqmAI.Api.Configuration;
using Serilog;

namespace HawaqmAI.Api.Validators;

/// <summary>
/// Validates that a SQL string is a safe, read-only SELECT before execution.
/// Guards against SQL injection and privilege escalation.
/// </summary>
public static class SqlValidator
{
    private static readonly Serilog.ILogger _log = Log.ForContext(typeof(SqlValidator));

    private static readonly string[] AllowedTables =
    [
        "ParameterReadings",
        "ParameterAverages",
        "ParameterAveragesMonth",
        "ParameterAveragesYear",
        "DMN_Stations",
        "DMN_Devices",
        "DMN_Parameters",
        "DMN_Flags",
        "DeviceCompliance",
        "Parameters_Excedence_Values",
        "ReportedUnits",
        "Sectors",
        "SubSectors",
        "Regions",
        "DMN_StationGrouping",
        "SiteParameterAverages",
        "DeviceSiteHistory",
        // Ambient (outdoor) tables
        "AmbientData",
        "AmbientStations",
        "AmbientParameters",
        // Chat tables (read-only access granted to readonly user)
        "chat_sessions",
        "chat_messages",
        "chat_feedback",
        "user_pins"
    ];

    private static readonly string[] ForbiddenKeywords =
        ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
         "EXEC", "EXECUTE", "TRUNCATE", "MERGE", "GRANT", "REVOKE",
         "XP_", "SP_", "OPENROWSET", "OPENQUERY", "BULK", "BACKUP", "RESTORE"];

    // Matches comment-hidden forbidden keywords
    private static readonly Regex CommentStripper = new(
        @"/\*.*?\*/|--[^\r\n]*",
        RegexOptions.Singleline | RegexOptions.Compiled);

    // Matches parameter placeholders that could be injection vectors if unparameterised
    private static readonly Regex StringLiteralCheck = new(
        @"'[^']*'",
        RegexOptions.Compiled);

    /// <summary>
    /// Validates the SQL string.
    /// Returns (isValid: true, error: null) on success.
    /// Returns (isValid: false, error: message) on failure.
    /// </summary>
    public static (bool IsValid, string? Error) Validate(string sql)
    {
        if (string.IsNullOrWhiteSpace(sql))
            return (false, "SQL query is empty.");

        // Strip comments to check true content
        var stripped = CommentStripper.Replace(sql, " ").Trim();
        var upper = stripped.ToUpperInvariant();

        // Must start with SELECT or WITH (CTEs use WITH ... AS ( SELECT ... ))
        var trimmedUpper = upper.TrimStart();
        if (!trimmedUpper.StartsWith("SELECT") && !trimmedUpper.StartsWith("WITH"))
        {
            _log.Warning("SQL rejected: does not start with SELECT or WITH. SQL={Sql}", sql[..Math.Min(200, sql.Length)]);
            return (false, "Only SELECT queries are permitted.");
        }

        // Check for forbidden keywords
        foreach (var keyword in ForbiddenKeywords)
        {
            // Use word boundary awareness: look for keyword not adjacent to alphanumeric
            var pattern = $@"(?<![A-Z0-9_]){Regex.Escape(keyword)}(?![A-Z0-9_])";
            if (Regex.IsMatch(upper, pattern))
            {
                _log.Warning("SQL rejected: forbidden keyword '{Keyword}'. SQL={Sql}", keyword, sql[..Math.Min(200, sql.Length)]);
                return (false, $"Query contains a forbidden keyword: {keyword}");
            }
        }

        // Extract CTE names defined in WITH clause so they are not mistaken for real tables.
        // Pattern: WITH CteOne AS (, CteTwo AS ( ...
        var cteNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (Match cteMatch in Regex.Matches(upper, @"\b(\w+)\s+AS\s*\("))
            cteNames.Add(cteMatch.Groups[1].Value);

        // Check all referenced tables are in the allowlist (skip CTE aliases)
        var tableMatches = Regex.Matches(upper,
            @"(?:FROM|JOIN)\s+(\w+)",
            RegexOptions.IgnoreCase);

        foreach (Match m in tableMatches)
        {
            var tableName = m.Groups[1].Value;
            // Skip CTE names — they are not real tables
            if (cteNames.Contains(tableName))
                continue;
            if (!AllowedTables.Contains(tableName, StringComparer.OrdinalIgnoreCase))
            {
                _log.Warning("SQL rejected: table '{Table}' not in allowlist. SQL={Sql}", tableName, sql[..Math.Min(200, sql.Length)]);
                return (false, $"Table '{tableName}' is not permitted.");
            }
        }

        // Reject excessively complex queries (simple check).
        // CTEs legitimately use multiple SELECT statements — allow up to 10.
        var subqueryCount = Regex.Matches(upper, @"\bSELECT\b").Count;
        if (subqueryCount > 10)
        {
            return (false, "Query complexity too high (too many nested SELECT statements).");
        }

        return (true, null);
    }

    /// <summary>
    /// Appends TOP @limit clause if no TOP or OFFSET/FETCH is present.
    /// Prevents runaway result sets.
    /// </summary>
    public static string EnsureTopClause(string sql, int limit)
    {
        var upper = sql.ToUpperInvariant();
        // CTEs (WITH ...) already contain TOP inside their inner SELECTs — skip injection
        if (upper.TrimStart().StartsWith("WITH"))
            return sql;
        if (upper.Contains(" TOP ") || upper.Contains("FETCH NEXT") || upper.Contains("OFFSET "))
            return sql;

        // Insert TOP after SELECT
        return Regex.Replace(sql,
            @"(?i)^\s*SELECT\s+",
            $"SELECT TOP {limit} ",
            RegexOptions.None);
    }

    /// <summary>
    /// Validates that a column name is in the approved pollutant allowlist.
    /// Prevents column injection when column names come from LLM output.
    /// </summary>
    public static bool IsAllowedColumn(string columnName, IEnumerable<string> allowlist)
    {
        return allowlist.Contains(columnName, StringComparer.OrdinalIgnoreCase);
    }
}
