using System.Text.Json;
using HawaqmAI.Api.Models;
using HawaqmAI.Api.Services;
using HawaqmAI.Api.Validators;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Serilog;

namespace HawaqmAI.Api.Controllers;

/// <summary>
/// Main AI chat endpoint. Orchestrates the full pipeline:
/// scope resolution → query routing → LLM parameter fill → RBAC → SQL → format → respond.
/// POST /api/chat
/// </summary>
[ApiController]
[Route("api/[controller]")]
[Authorize]
public sealed class ChatController : ControllerBase
{
    private static readonly Serilog.ILogger _log = Log.ForContext<ChatController>();

    private readonly IQueryRouterService _queryRouter;
    private readonly IAzureAIService _azureAI;
    private readonly IRbacEngine _rbac;
    private readonly ISqlExecutorService _sqlExecutor;
    private readonly ISessionService _sessionService;
    private readonly IResponseFormatterService _formatter;
    private readonly IExternalApiService _externalApi;

    public ChatController(
        IQueryRouterService queryRouter,
        IAzureAIService azureAI,
        IRbacEngine rbac,
        ISqlExecutorService sqlExecutor,
        ISessionService sessionService,
        IResponseFormatterService formatter,
        IExternalApiService externalApi)
    {
        _queryRouter = queryRouter;
        _azureAI = azureAI;
        _rbac = rbac;
        _sqlExecutor = sqlExecutor;
        _sessionService = sessionService;
        _formatter = formatter;
        _externalApi = externalApi;
    }

    /// <summary>
    /// Send a natural language question and receive an AI-generated answer.
    /// </summary>
    [HttpPost]
    [ProducesResponseType(typeof(ChatResponse), 200)]
    [ProducesResponseType(typeof(ChatResponse), 400)]
    [ProducesResponseType(401)]
    public async Task<IActionResult> Chat([FromBody] ChatRequest request, CancellationToken ct)
    {
        if (!ModelState.IsValid)
            return BadRequest(ModelState);

        var user = GetUserContext();
        if (user is null)
            return Unauthorized();

        _log.Information("Chat request received from user {UserId}", user.UserId);

        // ── 1. Resolve and validate scope ───────────────────────────────────
        var (scope, scopeError) = ScopeValidator.Resolve(request.Scope, user);
        if (scopeError is not null)
        {
            _log.Warning("Scope validation failed for user {UserId}: {Error}", user.UserId, scopeError);
            var session0 = await _sessionService.GetOrCreateSessionAsync(request.SessionId, user.UserId, ct);
            return Ok(_formatter.FormatError(session0.SessionId, scopeError, "INVALID_SCOPE"));
        }

        // ── 2. Get or create session ────────────────────────────────────────
        var session = await _sessionService.GetOrCreateSessionAsync(request.SessionId, user.UserId, ct);
        var history = await _sessionService.GetConversationHistoryAsync(session.SessionId, ct: ct);

        // ── 3. Route the question to an approved template ───────────────────
        var routeResult = await _queryRouter.RouteAsync(request.Message, user, history, ct);

        if (routeResult.NeedsClarification)
        {
            await _sessionService.SaveUserMessageAsync(session.SessionId, request.Message, ct);
            var clarifyMsg = routeResult.ClarificationPrompt
                ?? "I'm not sure I understand that question. Could you rephrase it or select a different topic?";
            return Ok(_formatter.FormatError(
                session.SessionId,
                clarifyMsg,
                "CLARIFY",
                "Try asking about: AQI at a site, readings for a device (e.g. BA 0001), or AQI for a region (Abudhabi, AL Ain, AlDhafra)."));
        }

        // ── 4. LLM fills parameters (and optionally selects template) ──────
        ApprovedQuery template;
        Dictionary<string, string> llmParams;
        int tokensUsed = 0;
        string? modelUsed = null;

        if (routeResult.NeedsLlmConfirmation)
        {
            // Medium confidence or follow-up: ask LLM to pick from candidates
            var selection = await _azureAI.SelectTemplateAsync(
                request.Message, routeResult.Candidates, history, scope, routeResult.PreviousTemplateId, ct);

            tokensUsed = selection.TokensUsed;
            modelUsed = selection.ModelUsed;

            var selected = routeResult.Candidates.FirstOrDefault(t => t.Id == selection.SelectedQueryId);
            if (selected is null)
            {
                return Ok(_formatter.FormatError(session.SessionId,
                    "I couldn't match your question to an approved query. Please try rephrasing.",
                    "CLARIFY"));
            }

            template = selected;
            llmParams = selection.Parameters;
        }
        else
        {
            // High confidence: template already known, just fill parameters
            template = routeResult.Template!;
            var paramResult = await _azureAI.FillParametersAsync(
                request.Message, template, history, scope, ct);

            tokensUsed = paramResult.TokensUsed;
            modelUsed = paramResult.ModelUsed;
            llmParams = paramResult.Parameters;
        }

        // ── 5. RBAC: validate selection ─────────────────────────────────────
        var (rbacValid, rbacReason) = _rbac.ValidateSelection(template, user);
        if (!rbacValid)
        {
            _log.Warning("RBAC denied template {TemplateId} for user {UserId}", template.Id, user.UserId);
            return Ok(_formatter.FormatError(session.SessionId, rbacReason!, "UNAUTHORIZED"));
        }

        // ── 6. FAQ short-circuit — answer directly from knowledge base ─────
        if (template.IsFaq)
        {
            var faqPath = Path.Combine(AppContext.BaseDirectory, "Knowledge", "faq.json");
            var faqContext = System.IO.File.Exists(faqPath) ? await System.IO.File.ReadAllTextAsync(faqPath, ct) : string.Empty;
            var faqAnswer = await _azureAI.AnswerFaqAsync(request.Message, faqContext, ct);

            await _sessionService.SaveUserMessageAsync(session.SessionId, request.Message, ct);
            var faqMessageId = Guid.NewGuid();
            await _sessionService.SaveAssistantMessageAsync(
                sessionId: session.SessionId,
                messageId: faqMessageId,
                content: faqAnswer,
                sql: null,
                responseType: "text",
                chartType: null,
                chartDataJson: null,
                dataSource: "FAQ",
                dateRange: null,
                modelUsed: modelUsed,
                executionTimeMs: 0,
                tokenCount: tokensUsed,
                ct: ct);

            if (session.MessageCount <= 1)
                await _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, ct);

            return Ok(_formatter.Format(
                sessionId: session.SessionId,
                messageId: faqMessageId,
                template: template,
                llmSummary: faqAnswer,
                rows: [],
                rowCount: 0,
                executionTimeMs: 0,
                scope: scope,
                modelUsed: modelUsed,
                tokenCount: tokensUsed));
        }

        // ── 7. Execute: API call OR SQL ─────────────────────────────────────
        List<Dictionary<string, object?>> resultRows;
        long executionTimeMs;
        string dataSourceLabel;

        if (template.ApiCall is not null)
        {
            // API path — forward the user's JWT token
            var bearerToken = Request.Headers.Authorization.ToString().Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
            var apiResult = await _externalApi.CallAsync(template.ApiCall, llmParams, scope, bearerToken, user, template.Id, ct);
            if (!apiResult.Success)
                return Ok(_formatter.FormatError(session.SessionId, apiResult.Error!, "API_ERROR"));

            resultRows = apiResult.Rows;
            executionTimeMs = apiResult.ExecutionTimeMs;
            dataSourceLabel = template.TableUsed; // descriptive label e.g. "AirQuality API"
        }
        else
        {
            // ── Resolve stationName → stationId when SQL template needs @stationId ──
            // Some SQL templates (e.g. average_at_station, pollutant_trend_hourly) use
            // @stationId (int) but the LLM always extracts a station name string.
            // Resolve the name to a numeric ID via DB lookup before building the safe query.
            var needsStationId = template.Params.Any(p =>
                p.Name.Equals("stationId", StringComparison.OrdinalIgnoreCase) ||
                p.Name.Equals("stationId1", StringComparison.OrdinalIgnoreCase) ||
                p.Name.Equals("stationId2", StringComparison.OrdinalIgnoreCase));

            if (needsStationId && llmParams.TryGetValue("stationName", out var nameForLookup)
                && !string.IsNullOrWhiteSpace(nameForLookup)
                && !llmParams.ContainsKey("stationId"))
            {
                var idLookup = await _sqlExecutor.ExecuteAsync(
                    "SELECT TOP 1 ID FROM DMN_Stations WHERE REPLACE(LOWER(StationName),' ','') LIKE '%' + REPLACE(LOWER(@name),' ','') + '%' AND Status = 1 ORDER BY LEN(StationName) ASC",
                    new Dictionary<string, object> { ["name"] = nameForLookup.Trim() }, ct);

                if (idLookup.Success && idLookup.Rows.Count > 0)
                {
                    var resolvedId = idLookup.Rows[0]["ID"]?.ToString() ?? "";
                    if (!string.IsNullOrWhiteSpace(resolvedId))
                    {
                        llmParams["stationId"] = resolvedId;
                        _log.Information("ChatController: resolved stationName='{Name}' → stationId={Id}", nameForLookup, resolvedId);
                    }
                }
                else
                {
                    _log.Warning("ChatController: could not resolve stationName='{Name}' to a stationId", nameForLookup);
                    return Ok(_formatter.FormatError(session.SessionId,
                        $"I couldn't find a site matching '{nameForLookup}'. Please check the site name and try again.",
                        "SITE_NOT_FOUND"));
                }
            }

            // ── Multi-site handling for devices_with_readings ──────────────────
            // When stationName contains multiple site names separated by "and"/commas,
            // run one query per site and merge the results.
            if (template.Id == "devices_with_readings"
                && llmParams.TryGetValue("stationName", out var multiSiteName)
                && !string.IsNullOrWhiteSpace(multiSiteName))
            {
                var siteNames = multiSiteName
                    .Split(new[] { " and ", " & ", ", " }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                    .Where(s => !string.IsNullOrWhiteSpace(s))
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .ToList();

                if (siteNames.Count > 1)
                {
                    var mergedRows = new List<Dictionary<string, object?>>();
                    var seenKeys = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                    long totalMs = 0;

                    foreach (var siteName in siteNames)
                    {
                        var perSiteParams = new Dictionary<string, string>(llmParams, StringComparer.OrdinalIgnoreCase)
                        {
                            ["stationName"] = siteName.Trim()
                        };

                        var (perSafeSql, perSqlParams) = _rbac.BuildSafeQuery(template, perSiteParams, user, scope);
                        var (perValid, perError) = SqlValidator.Validate(perSafeSql);
                        if (!perValid)
                        {
                            _log.Error("SQL validation failed (multi-site) for site '{Site}': {Error}", siteName, perError);
                            continue;
                        }

                        var perResult = await _sqlExecutor.ExecuteAsync(perSafeSql, perSqlParams, ct);
                        totalMs += perResult.ExecutionTimeMs;

                        if (perResult.Success)
                        {
                            foreach (var row in perResult.Rows)
                            {
                                var key = $"{row.GetValueOrDefault("DeviceName")}|{row.GetValueOrDefault("StationName")}";
                                if (seenKeys.Add(key))
                                    mergedRows.Add(row);
                            }
                        }
                        else
                        {
                            _log.Warning("SQL_ERROR (multi-site) for site '{Site}': {Error}", siteName, perResult.Error);
                        }
                    }

                    resultRows = mergedRows;
                    executionTimeMs = totalMs;
                    dataSourceLabel = template.TableUsed;
                    goto afterSqlExecution;
                }
            }

            // SQL path — build safe query and validate
            {
            var (safeSql, sqlParams) = _rbac.BuildSafeQuery(template, llmParams, user, scope);

            var (sqlValid, sqlError) = SqlValidator.Validate(safeSql);
            if (!sqlValid)
            {
                _log.Error("SQL validation failed for template {TemplateId}: {Error}", template.Id, sqlError);
                return Ok(_formatter.FormatError(session.SessionId,
                    "The query could not be safely constructed. Please try a different question.",
                    "SQL_INVALID"));
            }

            var sqlResult = await _sqlExecutor.ExecuteAsync(safeSql, sqlParams, ct);
            if (!sqlResult.Success)
            {
                // Translate technical SQL errors into user-friendly messages
                var userFriendlyError = sqlResult.Error!.Contains("scalar variable", StringComparison.OrdinalIgnoreCase)
                    ? "I couldn't gather enough information from your question to run that query. Could you be more specific? For example: 'Show me commercial sites in Abu Dhabi' or 'What is the CO2 at Al Saad Indian School?'"
                    : "I was unable to retrieve data at this time. Please try rephrasing your question or try again shortly.";

                _log.Warning("SQL_ERROR for template {TemplateId}: {Error}", template.Id, sqlResult.Error);
                return Ok(_formatter.FormatError(session.SessionId, userFriendlyError, "SQL_ERROR"));
            }

            resultRows = sqlResult.Rows;
            executionTimeMs = sqlResult.ExecutionTimeMs;
            dataSourceLabel = template.TableUsed;
            }
            afterSqlExecution:;
        }

        // ── 8. Generate natural language summary ────────────────────────────
        var summary = await _azureAI.SummarizeResultsAsync(
            request.Message, template, resultRows, scope, ct);

        // ── 9. Persist conversation ─────────────────────────────────────────
        await _sessionService.SaveUserMessageAsync(session.SessionId, request.Message, ct);

        var messageId = Guid.NewGuid();
        var chartDataJson = resultRows.Count > 0 && template.ResponseType == "chart"
            ? System.Text.Json.JsonSerializer.Serialize(resultRows)
            : null;

        await _sessionService.SaveAssistantMessageAsync(
            sessionId: session.SessionId,
            messageId: messageId,
            content: summary,
            sql: template.ApiCall is not null ? $"API:{template.ApiCall.Path}" : null,
            responseType: template.ResponseType,
            chartType: template.ChartType,
            chartDataJson: chartDataJson,
            dataSource: dataSourceLabel,
            dateRange: scope.DateRangeLabel,
            modelUsed: modelUsed,
            executionTimeMs: (int)Math.Min(executionTimeMs, int.MaxValue),
            tokenCount: tokensUsed,
            ct: ct);

        // Auto-title session from first message
        if (session.MessageCount <= 1)
            await _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, ct);

        // ── 10. Format and return response ──────────────────────────────────
        var response = _formatter.Format(
            sessionId: session.SessionId,
            messageId: messageId,
            template: template,
            llmSummary: summary,
            rows: resultRows,
            rowCount: resultRows.Count,
            executionTimeMs: (int)Math.Min(executionTimeMs, int.MaxValue),
            scope: scope,
            modelUsed: modelUsed,
            tokenCount: tokensUsed);

        return Ok(response);
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    private UserContext? GetUserContext()
        => HttpContext.Items.TryGetValue("UserContext", out var ctx) ? ctx as UserContext : null;
}
