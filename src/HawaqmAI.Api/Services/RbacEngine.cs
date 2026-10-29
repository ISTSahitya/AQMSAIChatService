using System.Globalization;
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

                // Auto-inject @paddedCanonical whenever @deviceName is bound —
                // resolves partial device names like "BA 1" → "BA0001", "SEI100M 14" → "SEI100M0014".
                // Real device format: "BA 0001"–"BA 0010", "SEI100M 0014"–"SEI100M 0148".
                if (param.Name.Equals("deviceName", StringComparison.OrdinalIgnoreCase)
                    && sql.Contains("@paddedCanonical", StringComparison.OrdinalIgnoreCase))
                {
                    var norm = rawValue.Replace(" ", "").ToUpperInvariant();
                    var m = System.Text.RegularExpressions.Regex.Match(norm, @"^([A-Z]+(?:\d+[A-Z]+)*)(\d+)$");
                    parameters["paddedCanonical"] = m.Success
                        ? m.Groups[1].Value + m.Groups[2].Value.PadLeft(4, '0')
                        : norm;
                }
            }
        }

        // ── Step 2b: Handle unresolved optional params ───────────────────────
        // If the SQL uses "(@param IS NULL OR ...)" pattern → bind null so SQL Server
        // evaluates the IS NULL branch and skips the filter (no condition removed).
        // Otherwise → remove the condition entirely (legacy behaviour for simple AND conditions).
        foreach (var param in template.Params.Where(p => p.Optional && p.Type != "column"))
        {
            if (!parameters.ContainsKey(param.Name) && sql.Contains($"@{param.Name}", StringComparison.OrdinalIgnoreCase))
            {
                // Detect "@param IS NULL OR ..." pattern (the param appears BEFORE IS NULL in SQL)
                var isNullOrPattern = sql.Contains($"@{param.Name} IS NULL", StringComparison.OrdinalIgnoreCase);

                if (isNullOrPattern)
                {
                    // Bind C# null (not DBNull.Value) — Dapper maps null to SQL NULL correctly
                    parameters[param.Name] = null!;
                    _log.Debug("RBAC: bound null for optional param '@{Param}' (IS NULL OR pattern)", param.Name);
                }
                else
                {
                    sql = RemoveParamCondition(sql, param.Name);
                    _log.Debug("RBAC: removed optional param condition '@{Param}' from SQL (not resolved)", param.Name);
                }
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

        // ── Step 3b: Inject sector filter if scope has one and SQL references @sectorName ──
        // Skip injection for aggregate/summary templates (skipSiteFilter=true) and any template
        // that already groups by sector — injecting a WHERE filter would break those queries.
        var skipSectorInjection = template.SkipSiteFilter
            || template.Id is "sites_count_by_sector" or "sites_count_by_region"
                or "sector_latest_pollutant"
                or "count_sites_in_region" or "sites_by_filter";
        if (!skipSectorInjection && !string.IsNullOrWhiteSpace(scope.Sector))
        {
            if (sql.Contains("@sectorName", StringComparison.OrdinalIgnoreCase) && !parameters.ContainsKey("sectorName"))
            {
                parameters["sectorName"] = scope.Sector;
                _log.Debug("RBAC: injected sector filter '{Sector}' from scope bar", scope.Sector);
            }
            else if (!sql.Contains("@sectorName", StringComparison.OrdinalIgnoreCase))
            {
                // SQL doesn't have a @sectorName param — inject a WHERE/AND condition on SectorName
                sql = InjectSectorFilter(sql, scope.Sector, parameters);
            }
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
            // If startDate is known, default endDate to the same date (single-day query).
            // This handles "from YYYY-MM-DD" with no explicit end — the user meant that date only.
            if (parameters.TryGetValue("startDate", out var sd) && !string.IsNullOrWhiteSpace(sd?.ToString()))
            {
                parameters["endDate"] = sd!.ToString()!;
                _log.Debug("RBAC: defaulted @endDate to startDate = {EndDate}", parameters["endDate"]);
            }
            else
            {
                parameters["endDate"] = DateTime.UtcNow.Date.ToString("yyyy-MM-dd");
                _log.Debug("RBAC: defaulted @endDate to today");
            }
        }

        // ── Step 5: Ensure row limit ──────────────────────────────────────────
        var userLimit = Math.Min(
            user.Role == "viewer" ? _chatOptions.DefaultRowLimit : _chatOptions.MaxRowLimit,
            _chatOptions.MaxRowLimit);

        sql = SqlValidator.EnsureTopClause(sql, userLimit);

        _log.Debug("RBAC: built safe query for template '{Id}', sites={Sites}, params={Params}",
            template.Id, string.Join(",", effectiveSiteIds),
            string.Join(", ", parameters.Select(kv => $"{kv.Key}={kv.Value ?? "NULL"}")));

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
        var scopeValue = param.Name.ToLowerInvariant() switch
        {
            "siteid" or "siteid1" or "stationid" => scope.SiteIds.Count > 0 ? scope.SiteIds[0].ToString() : null,
            "startdate" => scope.StartDate?.ToString("yyyy-MM-dd"),
            "enddate" => scope.EndDate?.ToString("yyyy-MM-dd"),
            "targetdate" => scope.StartDate?.ToString("yyyy-MM-dd"),
            "regionname" => !string.IsNullOrWhiteSpace(scope.Region) ? scope.Region : null,
            "sectorname" => !string.IsNullOrWhiteSpace(scope.Sector) ? scope.Sector : null,
            _ => null
        };
        if (scopeValue is not null)
            return scopeValue;

        // Fall back to param default value (e.g. parameterName defaults to "PM10")
        return !string.IsNullOrWhiteSpace(param.Default) ? param.Default : null;
    }

    // DB canonical region spellings (as stored in the Regions table)
    private static readonly Dictionary<string, string> _regionAliases = new(StringComparer.OrdinalIgnoreCase)
    {
        ["Abu Dhabi"]  = "Abu Dhabi",
        ["AbuDhabi"]   = "Abu Dhabi",
        ["Abudhabi"]   = "Abu Dhabi",
        ["abudhabi"]   = "Abu Dhabi",
        ["Al Ain"]     = "Al Ain",
        ["AlAin"]      = "Al Ain",
        ["AL Ain"]     = "Al Ain",
        ["alain"]      = "Al Ain",
        ["Al Dhafra"]  = "Al Dhafra",
        ["AlDhafra"]   = "Al Dhafra",
        ["Dhafra"]     = "Al Dhafra",
        ["aldhafra"]   = "Al Dhafra",
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
            "date" => TryParseDate(rawValue, out var dt) ? (object)dt : rawValue,
            "string" => rawValue,
            _ => rawValue
        };
    }

    // All date formats the LLM or user might supply, tried in order (most specific first).
    private static readonly string[] _dateFormats =
    [
        "yyyy-MM-dd",          // ISO 8601 — LLM default, e.g. 2026-03-20
        "yyyy-MM-ddTHH:mm:ss", // ISO 8601 with time
        "yyyy-MM-ddTHH:mm",
        "yyyy/MM/dd",
        "dd/MM/yyyy",          // UAE/UK locale, e.g. 20/03/2026
        "dd-MM-yyyy",          // e.g. 20-03-2026
        "dd.MM.yyyy",
        "MM/dd/yyyy",          // US locale
        "MM-dd-yyyy",
        "d/M/yyyy",
        "d-M-yyyy",
        "yyyy-M-d",
        "MMMM d, yyyy",        // e.g. March 20, 2026
        "MMMM dd, yyyy",
        "d MMMM yyyy",         // e.g. 20 March 2026
        "dd MMMM yyyy",
        "MMM d, yyyy",         // e.g. Mar 20, 2026
        "MMM dd, yyyy",
        "d MMM yyyy",
        "dd MMM yyyy",
    ];

    private static bool TryParseDate(string raw, out DateTime result)
    {
        var s = raw?.Trim() ?? string.Empty;

        // Try all explicit formats first (InvariantCulture — no ambiguity)
        if (DateTime.TryParseExact(s, _dateFormats,
                CultureInfo.InvariantCulture,
                DateTimeStyles.None, out result))
        {
            result = result.Date;
            return true;
        }

        // Fallback: let the runtime try with invariant culture
        if (DateTime.TryParse(s, CultureInfo.InvariantCulture,
                DateTimeStyles.None, out result))
        {
            result = result.Date;
            return true;
        }

        result = default;
        return false;
    }

    /// <summary>
    /// Removes a WHERE/AND condition containing @paramName from the SQL.
    /// Handles params inside function calls like LOWER(@param) or REPLACE(LOWER(@param),' ','').
    /// Finds the full condition boundary by looking for the next AND/ORDER BY/GROUP BY after the param.
    /// </summary>
    private static string RemoveParamCondition(string sql, string paramName)
    {
        var token = $"@{paramName}";
        var idx = sql.IndexOf(token, StringComparison.OrdinalIgnoreCase);
        if (idx < 0) return sql;

        var before = sql[..idx];
        var after = sql[idx..];

        // Find the end of this condition: next " AND " or end-of-clause keyword
        // This correctly handles params wrapped in functions like REPLACE(LOWER(@param),' ','')
        var endMarkers = new[] { " AND ", " ORDER BY ", " GROUP BY ", " HAVING " };
        var condEnd = -1;
        var condEndLen = 0;
        foreach (var marker in endMarkers)
        {
            var pos = after.IndexOf(marker, StringComparison.OrdinalIgnoreCase);
            if (pos >= 0 && (condEnd < 0 || pos < condEnd))
            {
                condEnd = pos;
                condEndLen = marker.Length;
            }
        }

        // "rest" is everything after this condition (trim the AND separator if present)
        var rest = condEnd >= 0 ? after[(condEnd + condEndLen)..] : "";
        // If the next marker was ORDER BY / GROUP BY / HAVING, keep it
        var nextKeyword = condEnd >= 0 ? after[condEnd..].TrimStart() : "";
        var isAndMarker = condEnd >= 0 && after[condEnd..].TrimStart().StartsWith("AND ", StringComparison.OrdinalIgnoreCase);
        if (!isAndMarker && condEnd >= 0)
            rest = after[condEnd..]; // keep ORDER BY / GROUP BY in rest

        // Find the condition start: last AND or WHERE before @param
        var andIdx   = before.LastIndexOf(" AND ", StringComparison.OrdinalIgnoreCase);
        var whereIdx = before.LastIndexOf("WHERE ", StringComparison.OrdinalIgnoreCase);

        if (andIdx > whereIdx)
        {
            // Remove " AND <condition>".
            // When the next token is also a condition (isAndMarker), re-add the AND separator
            // so the remaining conditions stay properly joined.
            // When the next token is ORDER BY / GROUP BY / HAVING, keep it as-is (no AND needed).
            var suffix = rest.Length > 0
                ? (isAndMarker ? " AND " + rest.TrimStart() : " " + rest.TrimStart())
                : "";
            sql = before[..andIdx] + suffix;
        }
        else if (whereIdx >= 0)
        {
            // First/only condition after WHERE
            var andAfterIdx = rest.IndexOf(" AND ", StringComparison.OrdinalIgnoreCase);
            if (andAfterIdx >= 0)
                sql = before[..(whereIdx + 6)] + rest[(andAfterIdx + 5)..];
            else
                sql = before[..whereIdx].TrimEnd() + (rest.Length > 0 ? " " + rest.TrimStart() : "");
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
        // If SQL filters by station name string AND a non-null stationName was actually provided,
        // the site is already constrained by the LIKE filter — skip ID injection.
        // If stationName is null (optional param not provided), we still need RBAC site injection.
        if (Regex.IsMatch(sql, @"@stationName\b", RegexOptions.IgnoreCase)
            && parameters.TryGetValue("stationName", out var snVal) && snVal is not null)
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

            // Use the LAST WHERE clause — queries with subqueries have an inner WHERE inside
            // the subquery and an outer WHERE after it. We must inject into the outer WHERE.
            var lastWhereIdx = upperSql.LastIndexOf("WHERE");
            if (lastWhereIdx >= 0)
            {
                var insertPos = lastWhereIdx + 5; // skip past "WHERE"
                sql = sql[..insertPos] + $" {siteColumn} IN @siteIds AND " + sql[insertPos..];
            }
            else
            {
                // No WHERE at all — insert before GROUP BY or ORDER BY
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

    /// <summary>
    /// Injects a sector filter (SectorName LIKE @sectorName) into the SQL WHERE clause.
    /// Only applied when the SQL does not already handle @sectorName explicitly.
    /// </summary>
    private static string InjectSectorFilter(
        string sql,
        string sector,
        Dictionary<string, object> parameters)
    {
        // Determine the sector column based on tables referenced.
        // SectorName lives in the Sectors table (aliased as 'sec' in most queries).
        // Only fall back to a bare column when neither join pattern is found.
        var upperSql = sql.ToUpperInvariant();
        string sectorColumn;
        if (upperSql.Contains("JOIN SECTORS SEC") || upperSql.Contains("LEFT JOIN SECTORS SEC"))
            sectorColumn = "sec.SectorName";
        else if (upperSql.Contains("JOIN SECTORS S ") || upperSql.Contains("LEFT JOIN SECTORS S "))
            sectorColumn = "s.SectorName";
        else if (upperSql.Contains("FROM SECTORS"))
            sectorColumn = "SectorName";
        else
            sectorColumn = "sec.SectorName"; // default — most templates join Sectors as 'sec'

        var condition = $"{sectorColumn} LIKE @sectorName";
        parameters["sectorName"] = $"%{sector}%";

        // Use the LAST WHERE — subqueries have inner WHEREs before the outer one
        var lastWhereIdx = upperSql.LastIndexOf("WHERE");
        if (lastWhereIdx >= 0)
        {
            var insertPos = lastWhereIdx + 5;
            sql = sql[..insertPos] + $" {condition} AND " + sql[insertPos..];
        }
        else
        {
            var groupByIdx = sql.IndexOf("GROUP BY", StringComparison.OrdinalIgnoreCase);
            var orderByIdx = sql.IndexOf("ORDER BY", StringComparison.OrdinalIgnoreCase);
            var insertIdx = groupByIdx >= 0 ? groupByIdx : orderByIdx >= 0 ? orderByIdx : -1;
            if (insertIdx >= 0)
                sql = sql[..insertIdx] + $" WHERE {condition} " + sql[insertIdx..];
            else
                sql += $" WHERE {condition}";
        }

        return sql;
    }
}
