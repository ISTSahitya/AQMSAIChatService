using System.Text.RegularExpressions;
using HawaqmAI.Api.Configuration;
using HawaqmAI.Api.Models;
using HawaqmAI.Api.Validators;
using Microsoft.Extensions.Options;
using Serilog;

namespace HawaqmAI.Api.Services;

/// <summary>
/// Three-level RBAC enforcement implementation.
/// Prevents privilege escalation, data leakage across sites, and runaway queries.
/// </summary>
public sealed class RbacEngine : IRbacEngine
{
    private static readonly Serilog.ILogger _log = Log.ForContext<RbacEngine>();
    private readonly ChatOptions _chatOptions;

    public RbacEngine(IOptions<ChatOptions> chatOptions)
    {
        _chatOptions = chatOptions.Value;
    }

    /// <inheritdoc/>
    // Role hierarchy: higher index = more permissions
    private static readonly List<string> RoleHierarchy = ["viewer", "regulator", "admin"];

    private static bool UserMeetsRequiredRole(UserContext user, string requiredRole)
    {
        var userIdx     = RoleHierarchy.IndexOf(user.Role.ToLowerInvariant());
        var requiredIdx = RoleHierarchy.IndexOf(requiredRole.ToLowerInvariant());
        // Unknown roles default to lowest index (-1 → treated as 0)
        return userIdx >= requiredIdx;
    }

    public IReadOnlyList<ApprovedQuery> FilterTemplates(
        IReadOnlyList<ApprovedQuery> allTemplates,
        UserContext user)
    {
        return allTemplates
            .Where(t => user.PermittedCategories.Contains(t.Category)
                     && UserMeetsRequiredRole(user, t.RequiredRole))
            .ToList();
    }

    /// <inheritdoc/>
    public (bool IsValid, string? Reason) ValidateSelection(ApprovedQuery template, UserContext user)
    {
        if (!user.PermittedCategories.Contains(template.Category))
        {
            _log.Warning(
                "User {UserId} role {Role} attempted to use template {TemplateId} in category {Category} — DENIED",
                user.UserId, user.Role, template.Id, template.Category);
            return (false, $"Your role does not have access to the '{template.Category}' query category.");
        }

        if (!UserMeetsRequiredRole(user, template.RequiredRole))
        {
            _log.Warning(
                "User {UserId} role {Role} attempted to use template {TemplateId} requiring role {Required} — DENIED",
                user.UserId, user.Role, template.Id, template.RequiredRole);
            return (false, $"This query requires '{template.RequiredRole}' role or above. Your current role is '{user.Role}'.");
        }

        return (true, null);
    }

    /// <inheritdoc/>
    public (string Sql, Dictionary<string, object> Parameters) BuildSafeQuery(
        ApprovedQuery template,
        Dictionary<string, string> llmParameters,
        UserContext user,
        ResolvedScope scope)
    {
        var parameters = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);
        var sql = template.Sql;

        // ── Step 1: Substitute column name placeholders ───────────────────────
        sql = SubstituteColumnPlaceholders(sql, template, llmParameters);

        // ── Step 2: Bind typed parameters ────────────────────────────────────
        foreach (var param in template.Params)
        {
            if (param.Type == "column") continue; // already substituted above

            var rawValue = ResolveParamValue(param, llmParameters, scope);
            if (rawValue is null)
            {
                if (!param.Optional)
                    _log.Warning("Missing required parameter '{Name}' for template '{Id}'", param.Name, template.Id);
                continue;
            }

            var typedValue = ConvertParam(param, rawValue);
            if (typedValue is not null)
            {
                parameters[param.Name] = typedValue;
                _log.Information("RBAC param bound: template='{Id}' param='{Name}' value='{Value}'",
                    template.Id, param.Name, typedValue);

                // Auto-inject @stationNameNormalized whenever @stationName is bound —
                // used by queries that need space-stripped fuzzy matching in SQL Server.
                if (param.Name.Equals("stationName", StringComparison.OrdinalIgnoreCase)
                    && sql.Contains("@stationNameNormalized", StringComparison.OrdinalIgnoreCase))
                {
                    parameters["stationNameNormalized"] = rawValue.ToLowerInvariant().Replace(" ", "");
                }
            }
        }

        // ── Step 2b: Remove WHERE conditions for optional params that weren't resolved ──
        // If the SQL references @paramName but it wasn't bound, the query would throw.
        // For optional params, strip the condition so RBAC's site filter takes over.
        foreach (var param in template.Params.Where(p => p.Optional && p.Type != "column"))
        {
            if (!parameters.ContainsKey(param.Name) && sql.Contains($"@{param.Name}", StringComparison.OrdinalIgnoreCase))
            {
                sql = RemoveParamCondition(sql, param.Name);
                _log.Debug("RBAC: removed optional param condition '@{Param}' from SQL (not resolved)", param.Name);
            }
        }

        // ── Step 3: Inject site_id restriction ───────────────────────────────
        var effectiveSiteIds = scope.SiteIds.Count > 0
            ? scope.SiteIds
            : user.PermittedSiteIds;

        if (!template.SkipSiteFilter && effectiveSiteIds.Count > 0 && !user.HasAllSitesAccess)
        {
            // Add site_id IN (...) as Dapper parameter
            sql = InjectSiteIdFilter(sql, effectiveSiteIds, parameters);
        }
        else if (scope.SiteIds.Count == 1 && sql.Contains("@siteId") && !parameters.ContainsKey("siteId"))
        {
            parameters["siteId"] = scope.SiteIds[0];
        }

        // ── Step 4: Inject date range if scope has it and SQL uses @startDate/@endDate ──
        if (scope.StartDate.HasValue && !parameters.ContainsKey("startDate"))
            parameters["startDate"] = scope.StartDate.Value.ToString("yyyy-MM-dd");
        if (scope.EndDate.HasValue && !parameters.ContainsKey("endDate"))
            parameters["endDate"] = scope.EndDate.Value.ToString("yyyy-MM-dd");

        // If SQL still references @startDate/@endDate but they weren't resolved from scope or LLM,
        // default to the current calendar month so the query doesn't fail with an unbound variable.
        if (sql.Contains("@startDate", StringComparison.OrdinalIgnoreCase) && !parameters.ContainsKey("startDate"))
        {
            var today = DateTime.UtcNow;
            parameters["startDate"] = new DateTime(today.Year, today.Month, 1).ToString("yyyy-MM-dd");
            _log.Debug("RBAC: defaulted @startDate to start of current month");
        }
        if (sql.Contains("@endDate", StringComparison.OrdinalIgnoreCase) && !parameters.ContainsKey("endDate"))
        {
            var today = DateTime.UtcNow;
            parameters["endDate"] = today.ToString("yyyy-MM-dd");
            _log.Debug("RBAC: defaulted @endDate to today");
        }

        // ── Step 5: Ensure row limit ──────────────────────────────────────────
        var userLimit = Math.Min(
            user.Role == "viewer" ? _chatOptions.DefaultRowLimit : _chatOptions.MaxRowLimit,
            _chatOptions.MaxRowLimit);

        sql = SqlValidator.EnsureTopClause(sql, userLimit);

        _log.Debug("RBAC: built safe query for template '{Id}', sites={Sites}, params={ParamCount}",
            template.Id, string.Join(",", effectiveSiteIds), parameters.Count);

        return (sql, parameters);
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    private static string SubstituteColumnPlaceholders(
        string sql,
        ApprovedQuery template,
        Dictionary<string, string> llmParameters)
    {
        foreach (var param in template.Params.Where(p => p.Type == "column"))
        {
            if (!llmParameters.TryGetValue(param.Name, out var colValue))
                continue;

            // Validate against allowlist before substitution
            if (param.Allowed is { Count: > 0 } && !SqlValidator.IsAllowedColumn(colValue, param.Allowed))
            {
                _log.Warning("Column '{Col}' not in allowlist for param '{Param}' — rejected", colValue, param.Name);
                continue;
            }

            var safeName = Regex.Replace(colValue, @"[^a-z0-9_]", "", RegexOptions.IgnoreCase).ToLowerInvariant();
            sql = sql.Replace($"{{{param.Name}}}", safeName, StringComparison.OrdinalIgnoreCase);
        }

        return sql;
    }

    private static string? ResolveParamValue(
        QueryParam param,
        Dictionary<string, string> llmParameters,
        ResolvedScope scope)
    {
        // Try LLM-extracted value first
        if (llmParameters.TryGetValue(param.Name, out var llmVal) && !string.IsNullOrWhiteSpace(llmVal))
            return llmVal;

        // Fall back to scope bar values
        return param.Name.ToLowerInvariant() switch
        {
            "siteid" or "siteid1" or "stationid" => scope.SiteIds.Count > 0 ? scope.SiteIds[0].ToString() : null,
            "startdate" => scope.StartDate?.ToString("yyyy-MM-dd"),
            "enddate" => scope.EndDate?.ToString("yyyy-MM-dd"),
            "targetdate" => scope.StartDate?.ToString("yyyy-MM-dd"),
            "regionname" => !string.IsNullOrWhiteSpace(scope.Region) ? scope.Region : null,
            _ => null
        };
    }

    // DB canonical region spellings (as stored in the Regions table)
    private static readonly Dictionary<string, string> _regionAliases = new(StringComparer.OrdinalIgnoreCase)
    {
        ["Abu Dhabi"]  = "Abudhabi",
        ["AbuDhabi"]   = "Abudhabi",
        ["Abudhabi"]   = "Abudhabi",
        ["Al Ain"]     = "AL Ain",
        ["AlAin"]      = "AL Ain",
        ["AL Ain"]     = "AL Ain",
        ["Al Dhafra"]  = "AlDhafra",
        ["AlDhafra"]   = "AlDhafra",
        ["Dhafra"]     = "AlDhafra",
    };

    // DB canonical parameter names (chemical Unicode subscripts as stored in DMN_Parameters)
    private static readonly Dictionary<string, string> _parameterAliases = new(StringComparer.OrdinalIgnoreCase)
    {
        ["CO2"]    = "CO\u2082",   // CO₂
        ["CO₂"]    = "CO\u2082",
        ["O3"]     = "O\u2083",    // O₃
        ["O₃"]     = "O\u2083",
        ["NO2"]    = "NO\u2082",   // NO₂
        ["NO₂"]    = "NO\u2082",
        ["CH2O"]   = "CH\u2082O",  // CH₂O
        ["CH₂O"]   = "CH\u2082O",
        ["SO2"]    = "SO\u2082",   // SO₂
        ["SO₂"]    = "SO\u2082",
        // Plain names that already match DB — map to themselves so allowed-list passes
        ["PM2.5"]       = "PM2.5",
        ["PM10"]        = "PM10",
        ["CO"]          = "CO",
        ["TVOC"]        = "TVOC",
        ["VOC"]         = "VOC",
        ["Temperature"] = "Temperature",
        ["Humidity"]    = "Humidity",
        ["AQI Index"]   = "AQI Index",
        ["Noise"]       = "Noise",
    };

    private static object? ConvertParam(QueryParam param, string rawValue)
    {
        if (param.Type == "string" && param.Name.Equals("regionName", StringComparison.OrdinalIgnoreCase))
        {
            // Canonicalize region name to match DB spelling (e.g. "Abu Dhabi" → "Abudhabi")
            if (_regionAliases.TryGetValue(rawValue.Trim(), out var canonical))
                rawValue = canonical;
        }

        if (param.Type == "string" && param.Name.Equals("parameterName", StringComparison.OrdinalIgnoreCase))
        {
            // Canonicalize parameter name to DB Unicode subscript form (e.g. "CO2" → "CO₂")
            if (_parameterAliases.TryGetValue(rawValue.Trim(), out var canonical))
                rawValue = canonical;
        }

        return param.Type switch
        {
            "int" => int.TryParse(rawValue, out var i) ? i : null as object,
            "date" => rawValue,  // passed as string to SQL Server — it handles date conversion
            "string" => rawValue,
            _ => rawValue
        };
    }

    /// <summary>
    /// Removes a WHERE/AND condition containing @paramName from the SQL.
    /// Handles: "WHERE col = @param AND ...", "AND col = @param AND ...", "AND col = @param"
    /// </summary>
    private static string RemoveParamCondition(string sql, string paramName)
    {
        // Match: optional leading AND/WHERE whitespace, then any condition containing @paramName, then optional trailing AND
        // Strategy: find the token @paramName, walk back to find the condition start, remove it
        var token = $"@{paramName}";
        var idx = sql.IndexOf(token, StringComparison.OrdinalIgnoreCase);
        if (idx < 0) return sql;

        // Walk back to find start of this condition (after WHERE or AND)
        var before = sql[..idx];
        var after = sql[idx..];

        // Find last AND or WHERE before @param
        var andIdx = before.LastIndexOf(" AND ", StringComparison.OrdinalIgnoreCase);
        var whereIdx = before.LastIndexOf("WHERE ", StringComparison.OrdinalIgnoreCase);

        // Find end of @param token in after (skip past the token itself)
        var paramEnd = after.IndexOf(' ');
        var rest = paramEnd >= 0 ? after[paramEnd..] : "";

        if (andIdx > whereIdx)
        {
            // Remove " AND <condition>"
            sql = before[..andIdx] + rest;
        }
        else if (whereIdx >= 0)
        {
            // This is the only/first condition after WHERE — check if there's an AND after
            var andAfterIdx = rest.IndexOf(" AND ", StringComparison.OrdinalIgnoreCase);
            if (andAfterIdx >= 0)
            {
                // Replace WHERE <condition> AND with WHERE
                sql = before[..(whereIdx + 6)] + rest[(andAfterIdx + 5)..];
            }
            else
            {
                // Only condition — remove entire WHERE clause
                sql = before[..whereIdx].TrimEnd() + " " + rest.TrimStart();
            }
        }

        return sql;
    }

    private static string InjectSiteIdFilter(
        string sql,
        List<int> siteIds,
        Dictionary<string, object> parameters)
    {
        // If SQL already has a @siteId or @stationId parameter (exact word boundary), just bind it
        if (Regex.IsMatch(sql, @"@siteId\b", RegexOptions.IgnoreCase) && siteIds.Count == 1)
        {
            parameters["siteId"] = siteIds[0];
            return sql;
        }
        if (Regex.IsMatch(sql, @"@stationId\b", RegexOptions.IgnoreCase) && siteIds.Count == 1)
        {
            if (!parameters.ContainsKey("stationId"))
                parameters["stationId"] = siteIds[0];
            return sql;
        }
        // If SQL filters by station name string, the site is already constrained — skip ID injection
        if (Regex.IsMatch(sql, @"@stationName\b", RegexOptions.IgnoreCase))
            return sql;

        // Inject WHERE/AND clause with IN list (parameterised via Dapper tvp workaround)
        // For Dapper, we pass siteIds as a List<int> bound to @siteIds
        if (!Regex.IsMatch(sql, @"@siteIds\b", RegexOptions.IgnoreCase))
        {
            // Determine the correct column to filter on based on what the SQL references:
            // - Queries on DMN_Stations directly filter on s.ID (station is the primary table)
            // - Queries with aliased tables (e.g. ParameterReadings pr JOIN DMN_Parameters p) use qualified alias
            // - All other queries filter on StationID (foreign key in data tables)
            var upperSql = sql.ToUpperInvariant();
            string siteColumn;
            if (upperSql.Contains("FROM DMN_STATIONS") && !upperSql.Contains("JOIN DMN_STATIONS"))
                siteColumn = "s.ID";
            else if (upperSql.Contains("FROM PARAMETERREADINGS PR") || upperSql.Contains("FROM PARAMETERREADINGS AS PR"))
                siteColumn = "pr.StationID";
            else if (upperSql.Contains("FROM PARAMETERAVERAGES PA") || upperSql.Contains("FROM PARAMETERAVERAGES AS PA"))
                siteColumn = "pa.StationID";
            else if (upperSql.Contains("FROM PARAMETERAVERAGESMONTH PAM") || upperSql.Contains("FROM PARAMETERAVERAGESMONTH AS PAM"))
                siteColumn = "pam.StationID";
            else if (upperSql.Contains("JOIN DMN_STATIONS S ") || upperSql.Contains("JOIN DMN_STATIONS S\r") || upperSql.Contains("JOIN DMN_STATIONS S\n"))
                siteColumn = "s.ID";   // joined DMN_Stations aliased as 's' — e.g. schools_latest_pollutant
            else if (upperSql.Contains("FROM DMN_DEVICES D") && upperSql.Contains("JOIN DMN_STATIONS S"))
                siteColumn = "d.StationID";  // list_devices: filter on devices' station FK
            else
                siteColumn = "StationID";

            if (upperSql.Contains("WHERE"))
            {
                var insertPos = sql.IndexOf("WHERE", StringComparison.OrdinalIgnoreCase) + 5;
                sql = sql[..insertPos] + $" {siteColumn} IN @siteIds AND " + sql[insertPos..];
            }
            else
            {
                // Insert WHERE before GROUP BY (aggregate queries) or ORDER BY, whichever comes first
                var groupByIdx = sql.IndexOf("GROUP BY", StringComparison.OrdinalIgnoreCase);
                var orderByIdx = sql.IndexOf("ORDER BY", StringComparison.OrdinalIgnoreCase);
                var insertIdx = groupByIdx >= 0 ? groupByIdx
                              : orderByIdx >= 0 ? orderByIdx
                              : -1;
                if (insertIdx >= 0)
                    sql = sql[..insertIdx] + $" WHERE {siteColumn} IN @siteIds " + sql[insertIdx..];
                else
                    sql += $" WHERE {siteColumn} IN @siteIds";
            }
        }

        parameters["siteIds"] = siteIds;
        return sql;
    }
}
