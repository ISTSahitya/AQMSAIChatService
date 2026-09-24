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

        // ── 2b. Hard pre-check: AQI range/definition questions must always go to FAQ ──
        // These questions ask WHAT a category means (numeric range), not which sites are in it.
        // Pattern scorer can be confused by words like "good", "moderate", "aqi" matching
        // site_aqi_by_category patterns, so we intercept here before routing.
        var lowerMsg = request.Message.ToLowerInvariant();
        var aqiRangeDefinitionKeywords = new[] {
            // range-based phrasings
            "what aqi range", "aqi range is considered", "range considered good", "range considered moderate",
            "range considered unhealthy", "range considered hazardous", "range considered very unhealthy",
            "range of aqi", "range for good", "range for moderate", "range for unhealthy", "range for hazardous",
            "aqi good range", "aqi moderate range", "aqi unhealthy range", "aqi hazardous range",
            "what aqi range is", "aqi range for",
            // "considered" phrasings — "what aqi value/range/level is considered X"
            "what aqi is considered", "aqi is considered good", "aqi is considered moderate",
            "aqi is considered unhealthy", "aqi is considered hazardous",
            "aqi value is considered", "aqi level is considered",
            "considered good", "considered moderate", "considered unhealthy",
            "considered hazardous", "considered very unhealthy",
            // "what aqi is X" phrasings
            "what is good aqi", "what is moderate aqi", "what is unhealthy aqi", "what is hazardous aqi",
            "what aqi is good", "what aqi is moderate", "what aqi is unhealthy", "what aqi is hazardous",
            "what aqi is very unhealthy", "what aqi is hazardous",
            // "what value is X" phrasings
            "what value is good", "what value is moderate", "what value is unhealthy", "what value is hazardous",
            "what aqi value is good", "what aqi value is moderate", "what aqi value is unhealthy",
            "what aqi value is hazardous", "what aqi value is very unhealthy",
            // calculation / general
            "how is aqi calculated", "how is site aqi calculated",
            "aqi categories", "aqi levels", "aqi classification", "aqi category breakdown",
            "what are the aqi", "what are aqi categories", "what are aqi levels",
            "aqi scale", "aqi meaning", "what does aqi mean",
            // advisory / operational FAQ phrases — these are generic knowledge questions,
            // not requests for live device/site data
            "which offline devices should be checked first",
            "offline devices should be checked",
            "which devices should be checked first",
            "should be checked first",
            "how should i compare pollutant exceedances",
            "compare pollutant exceedances across",
            "averaging periods",
            "can i use a 1-hour threshold for pm10",
            "1-hour threshold for pm10",
            "can i compare a 5-minute",
            "5-minute pm2.5 reading directly",
            "mold test results be combined with aqi",
            "mold-only site",
            "site has no device but has a mold",
            "data success rate is below 75",
            "dsr is below 75",
            "can i still use the aqi results",
            "estimate the missing aqi",
            "missing aqi values",
            "can you show me radon",
            "radon levels",
            "pollen count",
            "how often do hawaqm devices record",
            "when is a device considered offline",
            "when is a site shown as inactive",
            "how is the site aqi calculated",
            "which air-quality parameters are monitored",
            "what parameters are monitored",
            "does hawaqm monitor temperature",
            "can i export hawaqm data",
            "can i view both indoor and ambient",
            "what happens if a site has no geographic",
            "what is moccae",
            "what are the total regions",
            "which regions are covered",
            "what data am i able to see",
            "what is the current data success rate",
            "what does data success rate mean",
            "why do the total device",
            "why is a site missing from the geographical map",
            "why is a site not showing on the geographical map",
            "can i get the last 24 hours of readings for a device",
            "can i use the statistical report",
            "can i use the exceedance report",
            "which averaging intervals should be used",
            "i need six months of exceedance",
            "why can the site aqi look acceptable",
            "site average be hiding a problem",
            "can you prepare a one-year raw-data export",
            "can mold test results be combined",
            "can you tell me tomorrow",
            "weather forecast",
            "can you prove the ventilation",
            "ventilation system caused",
            "nearest ambient station and indoor site have different update",
            "if the nearest ambient station"
        };
        // Safety/health questions about a NAMED site must NOT go to FAQ — they need live data.
        // Detect if the question is a site-specific safety question by looking for safety words + site indicators.
        var safetyWords = new[] { "safe", "safety", "attend", "suitable", "healthy", "unhealthy for", "good for children", "good for elderly", "good for disabled", "can children", "should children", "can people", "should people" };
        var siteWords = new[] { "school", "residential", "commercial", "institutional", "site", "reyada", "saad", "naeem", "bateen", "khalifa", "kaltham", "gems" };
        bool isSiteSafetyQuestion = safetyWords.Any(s => lowerMsg.Contains(s)) && siteWords.Any(s => lowerMsg.Contains(s));

        if (!isSiteSafetyQuestion && aqiRangeDefinitionKeywords.Any(k => lowerMsg.Contains(k)))
        {
            _log.Debug("ChatController: AQI range-definition pre-check fired → forcing faq_answer");
            var faqTemplatePre = _queryRouter.GetPermittedTemplates(user).FirstOrDefault(t => t.Id == "faq_answer");
            if (faqTemplatePre is not null)
            {
                var faqPathPre = Path.Combine(AppContext.BaseDirectory, "Knowledge", "faq.json");
                var faqContextPre = System.IO.File.Exists(faqPathPre) ? await System.IO.File.ReadAllTextAsync(faqPathPre, ct) : string.Empty;
                var faqAnswerPre = await _azureAI.AnswerFaqAsync(request.Message, faqContextPre, ct);
                await _sessionService.SaveUserMessageAsync(session.SessionId, request.Message, ct);
                var faqPreId = Guid.NewGuid();
                await _sessionService.SaveAssistantMessageAsync(
                    sessionId: session.SessionId,
                    messageId: faqPreId,
                    content: faqAnswerPre,
                    sql: null,
                    responseType: "text",
                    chartType: null,
                    chartDataJson: null,
                    dataSource: "FAQ",
                    dateRange: null,
                    modelUsed: null,
                    executionTimeMs: 0,
                    tokenCount: 0,
                    ct: ct);
                if (session.MessageCount <= 1)
                    await _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, ct);
                return Ok(_formatter.Format(
                    sessionId: session.SessionId,
                    messageId: faqPreId,
                    template: faqTemplatePre,
                    llmSummary: faqAnswerPre,
                    rows: [],
                    rowCount: 0,
                    executionTimeMs: 0,
                    scope: scope,
                    modelUsed: null,
                    tokenCount: 0));
            }
        }

        // ── 2c-pre. Generic AQI question with no site/region/device → ask for clarification ──
        // "Show me the AQI." / "What is the AQI?" with no site, region, sector, or device specified.
        var genericAqiTriggers = new[] { "show me the aqi", "show the aqi", "what is the aqi", "display aqi", "view aqi", "give me the aqi", "tell me the aqi", "show aqi", "current aqi", "aqi now" };
        var hasRegionWord = new[] { "abu dhabi", "abudhabi", "al ain", "alain", "al dhafra", "aldhafra" }.Any(r => lowerMsg.Contains(r));
        var hasSiteWord = new[] { "school", "site", "station", "building", "facility", "reyada", "saad", "naeem", "bateen", "kaltham", "gems", "residential", "commercial" }.Any(s => lowerMsg.Contains(s));
        var hasSectorWord = new[] { "sector", "commercial", "residential", "public", "govt", "government" }.Any(s => lowerMsg.Contains(s));
        bool hasDeviceName = System.Text.RegularExpressions.Regex.IsMatch(lowerMsg, @"\b(ba\s*\d{1,4}|sei\s*100\s*m\s*\d{1,4})\b");
        bool isGenericAqiRequest = genericAqiTriggers.Any(t => lowerMsg.Contains(t)) && !hasRegionWord && !hasSiteWord && !hasSectorWord && !hasDeviceName;

        if (isGenericAqiRequest)
        {
            await _sessionService.SaveUserMessageAsync(session.SessionId, request.Message, ct);
            return Ok(_formatter.FormatError(
                session.SessionId,
                "Happy to — which site or region, and for what period? For example: 'current AQI at Al Reyada School' or 'AQI for the Abu Dhabi region today'.",
                "CLARIFY"));
        }

        // ── 2c. Hard pre-check: safety/health questions about a named site → force site_aqi_single ──
        // These questions need live AQI data, not the generic FAQ "consult health authority" entry.
        if (isSiteSafetyQuestion)
        {
            _log.Debug("ChatController: site safety pre-check fired → forcing site_aqi_single");
            var safetyTemplate = _queryRouter.GetPermittedTemplates(user).FirstOrDefault(t => t.Id == "site_aqi_single");
            if (safetyTemplate is not null && safetyTemplate.ApiCall is not null)
            {
                var safetyBearerToken = Request.Headers.Authorization.ToString().Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
                var safetyParamResult = await _azureAI.FillParametersAsync(request.Message, safetyTemplate, history, scope, ct);
                var safetyApiResult = await _externalApi.CallAsync(safetyTemplate.ApiCall, safetyParamResult.Parameters, scope, safetyBearerToken, user, safetyTemplate.Id, ct);
                if (safetyApiResult.Success)
                {
                    await _sessionService.SaveUserMessageAsync(session.SessionId, request.Message, ct);
                    var safetyMsgId = Guid.NewGuid();
                    var safetySummary = await _azureAI.SummarizeResultsAsync(
                        request.Message, safetyTemplate, safetyApiResult.Rows, scope, ct);
                    await _sessionService.SaveAssistantMessageAsync(
                        sessionId: session.SessionId,
                        messageId: safetyMsgId,
                        content: safetySummary,
                        sql: null,
                        responseType: "text",
                        chartType: null,
                        chartDataJson: null,
                        dataSource: "API",
                        dateRange: null,
                        modelUsed: safetyParamResult.ModelUsed,
                        executionTimeMs: (int)safetyApiResult.ExecutionTimeMs,
                        tokenCount: safetyParamResult.TokensUsed,
                        ct: ct);
                    if (session.MessageCount <= 1)
                        await _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, ct);
                    return Ok(_formatter.Format(
                        sessionId: session.SessionId,
                        messageId: safetyMsgId,
                        template: safetyTemplate,
                        llmSummary: safetySummary,
                        rows: safetyApiResult.Rows,
                        rowCount: safetyApiResult.Rows.Count,
                        executionTimeMs: (int)safetyApiResult.ExecutionTimeMs,
                        scope: scope,
                        modelUsed: safetyParamResult.ModelUsed,
                        tokenCount: safetyParamResult.TokensUsed));
                }
            }
        }

        // ── 3. Route the question to an approved template ───────────────────
        var routeResult = await _queryRouter.RouteAsync(request.Message, user, history, ct);

        if (routeResult.NeedsClarification)
        {
            await _sessionService.SaveUserMessageAsync(session.SessionId, request.Message, ct);
            return Ok(_formatter.FormatError(
                session.SessionId,
                "I'm sorry — that's outside my scope. I can only help with HAWAQM air-quality data. If it's useful, you could ask me: 'Which devices are offline right now?' or 'What is the current AQI at my school?'",
                "CLARIFY"));
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
                    "I'm sorry — that's outside my scope. I can only help with HAWAQM air-quality data. If it's useful, you could ask me: 'Which devices are offline right now?' or 'What is the current AQI at my school?'",
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

        // Extract bearer token once — used by both API paths (template.ApiCall and device_reading_history)
        var bearerToken = Request.Headers.Authorization.ToString().Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);

        if (template.ApiCall is not null)
        {
            // API path — forward the user's JWT token
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
                    "SELECT TOP 1 ID FROM DMN_Stations WHERE REPLACE(LOWER(StationName),' ','') LIKE '%' + REPLACE(LOWER(@name),' ','') + '%' AND Status = 1 AND ISNULL(IsDeleted, 0) = 0 ORDER BY LEN(StationName) ASC",
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

            // ── Multi-region handling for region-filtered templates ────────────────
            // When regionName contains multiple regions (e.g. "Al Ain and Al Dhafra"),
            // run one query per region and merge the results.
            var multiRegionTemplates = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "sites_by_region", "count_sites_in_region", "sites_by_sector",
                "devices_by_filter", "schools_latest_pollutant", "sector_latest_pollutant"
            };

            if (multiRegionTemplates.Contains(template.Id)
                && llmParams.TryGetValue("regionName", out var multiRegionName)
                && !string.IsNullOrWhiteSpace(multiRegionName))
            {
                var regionNames = multiRegionName
                    .Split(new[] { " and ", " or ", " & ", ", " }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                    .Where(r => !string.IsNullOrWhiteSpace(r))
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .ToList();

                if (regionNames.Count > 1)
                {
                    var mergedRows = new List<Dictionary<string, object?>>();
                    var seenKeys = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                    long totalMs = 0;

                    foreach (var regionName in regionNames)
                    {
                        var perRegionParams = new Dictionary<string, string>(llmParams, StringComparer.OrdinalIgnoreCase)
                        {
                            ["regionName"] = regionName.Trim()
                        };

                        var (perSafeSql, perSqlParams) = _rbac.BuildSafeQuery(template, perRegionParams, user, scope);
                        var (perValid, perError) = SqlValidator.Validate(perSafeSql);
                        if (!perValid)
                        {
                            _log.Error("SQL validation failed (multi-region) for region '{Region}': {Error}", regionName, perError);
                            continue;
                        }

                        var perResult = await _sqlExecutor.ExecuteAsync(perSafeSql, perSqlParams, ct);
                        totalMs += perResult.ExecutionTimeMs;

                        if (perResult.Success)
                        {
                            foreach (var row in perResult.Rows)
                            {
                                // Deduplicate by StationName — same site can't appear in two regions
                                var key = row.GetValueOrDefault("StationName")?.ToString() ?? Guid.NewGuid().ToString();
                                if (seenKeys.Add(key))
                                    mergedRows.Add(row);
                            }
                        }
                        else
                        {
                            _log.Warning("SQL_ERROR (multi-region) for region '{Region}': {Error}", regionName, perResult.Error);
                        }
                    }

                    resultRows = mergedRows;
                    executionTimeMs = totalMs;
                    dataSourceLabel = template.TableUsed;
                    goto afterSqlExecution;
                }
            }

            // ── site_device_summary: inject default minDeviceCount/maxDeviceCount when not provided ──
            // The HAVING clause uses both params; if the LLM didn't extract them (no count filter in question),
            // default to 0..9999 so all sites are returned.
            if (template.Id == "site_device_summary")
            {
                if (!llmParams.ContainsKey("minDeviceCount"))
                    llmParams["minDeviceCount"] = "0";
                if (!llmParams.ContainsKey("maxDeviceCount"))
                    llmParams["maxDeviceCount"] = "9999";
            }

            // ── device_last_reading: resolve deviceName → deviceId, call GetDeviceLatestData API ──
            if (template.Id == "device_last_reading")
            {
                var devName = llmParams.GetValueOrDefault("deviceName") ?? "";
                if (string.IsNullOrWhiteSpace(devName))
                {
                    return Ok(_formatter.FormatError(session.SessionId,
                        "Please specify a device name (e.g. 'BA 0010', 'SEI100M 0014').", "MISSING_PARAM"));
                }

                var normDev2 = devName.Replace(" ", "").ToUpperInvariant();
                var paddedDev2 = normDev2;
                var prefixM2 = System.Text.RegularExpressions.Regex.Match(normDev2, @"^([A-Z]+(?:\d+[A-Z]+)*)(\d+)$");
                string canonicalSpacedDev2 = devName;
                if (prefixM2.Success)
                {
                    var pfx2 = prefixM2.Groups[1].Value;
                    var digs2 = prefixM2.Groups[2].Value.PadLeft(4, '0');
                    paddedDev2 = pfx2 + digs2;
                    canonicalSpacedDev2 = pfx2 + " " + digs2;
                }

                var devLookup2 = await _sqlExecutor.ExecuteAsync(
                    """
                    SELECT TOP 1 d.DeviceId, d.DeviceName, d.StationID
                    FROM DMN_Devices d
                    WHERE d.Status = 1
                      AND ISNULL(d.IsDeleted, 0) = 0
                      AND (d.DeviceName = @deviceName
                        OR d.DeviceName = @canonicalSpaced
                        OR REPLACE(UPPER(d.DeviceName),' ','') = @normalised
                        OR REPLACE(UPPER(d.DeviceName),' ','') = @paddedCanonical)
                    ORDER BY CASE WHEN d.DeviceName = @deviceName THEN 0
                                  WHEN d.DeviceName = @canonicalSpaced THEN 1
                                  WHEN REPLACE(UPPER(d.DeviceName),' ','') = @normalised THEN 2
                                  ELSE 3 END, d.DeviceName
                    """,
                    new Dictionary<string, object>
                    {
                        ["deviceName"]      = devName,
                        ["canonicalSpaced"] = canonicalSpacedDev2,
                        ["normalised"]      = normDev2,
                        ["paddedCanonical"] = paddedDev2
                    }, ct);

                if (!devLookup2.Success || devLookup2.Rows.Count == 0)
                {
                    return Ok(_formatter.FormatError(session.SessionId,
                        $"No device found matching '{devName}'. Please check the device name (e.g. 'BA 0010', 'SEI100M 0014').", "DEVICE_NOT_FOUND"));
                }

                var latestDeviceId   = devLookup2.Rows[0]["DeviceId"]?.ToString() ?? "";
                var latestDeviceName = devLookup2.Rows[0]["DeviceName"]?.ToString() ?? devName;
                var latestStationId  = devLookup2.Rows[0]["StationID"]?.ToString() ?? "";

                // RBAC: check station access
                if (!user.HasAllSitesAccess && user.PermittedSiteIds.Count > 0)
                {
                    var stationIdInt2 = devLookup2.Rows[0]["StationID"] is int sid3 ? sid3
                                      : int.TryParse(latestStationId, out var ps3) ? ps3 : 0;
                    if (stationIdInt2 == 0 || !user.PermittedSiteIds.Contains(stationIdInt2))
                    {
                        return Ok(_formatter.FormatError(session.SessionId,
                            $"You do not have access to device '{latestDeviceName}'.", "UNAUTHORIZED"));
                    }
                }

                var latestParamName = llmParams.GetValueOrDefault("parameterName");

                _log.Information("ChatController: device_last_reading deviceId={Id} deviceName='{Name}' param={Param}",
                    latestDeviceId, latestDeviceName, latestParamName ?? "all");

                var latestApiResult = await _externalApi.GetDeviceLatestDataAsync(
                    latestDeviceId, latestDeviceName, latestParamName, bearerToken, ct);

                if (!latestApiResult.Success)
                {
                    _log.Warning("device_last_reading API_ERROR: {Error}", latestApiResult.Error);
                    return Ok(_formatter.FormatError(session.SessionId,
                        "Could not retrieve the latest data for that device. Please try again.", "API_ERROR"));
                }

                resultRows      = latestApiResult.Rows;
                executionTimeMs = latestApiResult.ExecutionTimeMs;
                dataSourceLabel = "GetDeviceLatestData";
                goto afterSqlExecution;
            }

            // ── device_reading_history: resolve deviceName → deviceId, pick interval table ──
            if (template.Id == "device_reading_history")
            {
                var devName = llmParams.GetValueOrDefault("deviceName") ?? "";
                if (string.IsNullOrWhiteSpace(devName))
                {
                    return Ok(_formatter.FormatError(session.SessionId,
                        "Please specify a device name (e.g. 'BA 0010', 'SEI100M 0014').", "MISSING_PARAM"));
                }

                // Zero-pad and normalise the device name (same logic as ExternalApiService)
                var normDev = devName.Replace(" ", "").ToUpperInvariant();
                var paddedDev = normDev;
                var prefixM = System.Text.RegularExpressions.Regex.Match(normDev, @"^([A-Z]+(?:\d+[A-Z]+)*)(\d+)$");
                string canonicalSpacedDev = devName;
                if (prefixM.Success)
                {
                    var pfx = prefixM.Groups[1].Value;
                    var digs = prefixM.Groups[2].Value.PadLeft(4, '0');
                    paddedDev = pfx + digs;
                    canonicalSpacedDev = pfx + " " + digs;
                }

                var devLookup = await _sqlExecutor.ExecuteAsync(
                    """
                    SELECT TOP 1 d.DeviceId, d.DeviceName, d.StationID
                    FROM DMN_Devices d
                    WHERE d.Status = 1
                      AND ISNULL(d.IsDeleted, 0) = 0
                      AND (d.DeviceName = @deviceName
                        OR d.DeviceName = @canonicalSpaced
                        OR REPLACE(UPPER(d.DeviceName),' ','') = @normalised
                        OR REPLACE(UPPER(d.DeviceName),' ','') = @paddedCanonical)
                    ORDER BY CASE WHEN d.DeviceName = @deviceName THEN 0
                                  WHEN d.DeviceName = @canonicalSpaced THEN 1
                                  WHEN REPLACE(UPPER(d.DeviceName),' ','') = @normalised THEN 2
                                  ELSE 3 END, d.DeviceName
                    """,
                    new Dictionary<string, object>
                    {
                        ["deviceName"]      = devName,
                        ["canonicalSpaced"] = canonicalSpacedDev,
                        ["normalised"]      = normDev,
                        ["paddedCanonical"] = paddedDev
                    }, ct);

                if (!devLookup.Success || devLookup.Rows.Count == 0)
                {
                    return Ok(_formatter.FormatError(session.SessionId,
                        $"No device found matching '{devName}'. Please check the device name (e.g. 'BA 0010', 'SEI100M 0014').", "DEVICE_NOT_FOUND"));
                }

                var resolvedDeviceId   = devLookup.Rows[0]["DeviceId"]?.ToString() ?? "";
                var resolvedDeviceName = devLookup.Rows[0]["DeviceName"]?.ToString() ?? devName;
                var resolvedStationId  = devLookup.Rows[0]["StationID"]?.ToString() ?? "";

                // RBAC: check station access
                if (!user.HasAllSitesAccess && user.PermittedSiteIds.Count > 0)
                {
                    var stationIdInt = devLookup.Rows[0]["StationID"] is int sid2 ? sid2
                                     : int.TryParse(resolvedStationId, out var ps2) ? ps2 : 0;
                    if (stationIdInt == 0 || !user.PermittedSiteIds.Contains(stationIdInt))
                    {
                        return Ok(_formatter.FormatError(session.SessionId,
                            $"You do not have access to device '{resolvedDeviceName}'.", "UNAUTHORIZED"));
                    }
                }

                llmParams["deviceId"] = resolvedDeviceId;

                // Pick the right SQL based on interval
                var interval = llmParams.GetValueOrDefault("interval") ?? "5min";
                var paramName = llmParams.GetValueOrDefault("parameterName");
                var paramFilter = string.IsNullOrWhiteSpace(paramName)
                    ? "" : " AND p.ParameterName = @parameterName";

                // ── targetTime snapping ──────────────────────────────────────────────────
                // If the user asked for a specific clock time (e.g. "at 4AM", "at 3:03AM",
                // "at 4:30PM"), snap it to the correct boundary for the chosen interval:
                //   5min table  → floor to nearest 5-minute mark  (3:03 → 3:00, 2:59 → 2:55)
                //   1H/8H/24H  → floor to nearest whole hour       (3:05PM → 3:00PM, 4:30PM → 4:00PM)
                // The anchor date (day) comes from startDate if already set, else "yesterday"
                // (since users asking "yesterday at 4AM" trigger this path most often).
                var rawTargetTime = llmParams.GetValueOrDefault("targetTime");
                if (!string.IsNullOrWhiteSpace(rawTargetTime))
                {
                    // Parse the time portion — accept both 12-hour (3:03AM) and 24-hour (13:00)
                    static DateTime? TryParseTime(string s)
                    {
                        var formats = new[]
                        {
                            "h:mmtt", "h:mmTT", "htt", "hTT",
                            "h:mm tt", "h:mm TT", "h tt", "h TT",
                            "H:mm", "HH:mm", "H:mm:ss", "HH:mm:ss"
                        };
                        s = s.Trim().Replace(" ", "");
                        foreach (var fmt in formats)
                            if (DateTime.TryParseExact(s, fmt,
                                System.Globalization.CultureInfo.InvariantCulture,
                                System.Globalization.DateTimeStyles.None, out var dt))
                                return dt;
                        // Fallback: general parse
                        return DateTime.TryParse(s,
                            System.Globalization.CultureInfo.InvariantCulture,
                            System.Globalization.DateTimeStyles.NoCurrentDateDefault, out var dtg) ? dtg : null;
                    }

                    var parsedTime = TryParseTime(rawTargetTime);
                    if (parsedTime.HasValue)
                    {
                        // Determine anchor date (from existing startDate or yesterday UTC)
                        DateTime anchorDate;
                        if (llmParams.TryGetValue("startDate", out var existingStart) &&
                            DateTime.TryParse(existingStart, out var parsedAnchor))
                            anchorDate = parsedAnchor.Date;
                        else
                            anchorDate = DateTime.UtcNow.Date.AddDays(-1); // default: yesterday

                        var h = parsedTime.Value.Hour;
                        var m = parsedTime.Value.Minute;

                        DateTime snappedStart, snappedEnd;
                        if (interval == "5min" || interval == "5min" || (interval != "1hour" && interval != "8hour" && interval != "24hour"))
                        {
                            // Floor to nearest 5-minute boundary
                            var snappedMinute = (m / 5) * 5;
                            snappedStart = anchorDate.AddHours(h).AddMinutes(snappedMinute);
                            snappedEnd   = snappedStart.AddMinutes(5);
                        }
                        else
                        {
                            // Floor to nearest whole hour
                            snappedStart = anchorDate.AddHours(h);
                            snappedEnd   = snappedStart.AddHours(1);
                        }

                        llmParams["startDate"] = snappedStart.ToString("yyyy-MM-dd HH:mm:ss");
                        llmParams["endDate"]   = snappedEnd.ToString("yyyy-MM-dd HH:mm:ss");

                        _log.Information(
                            "device_reading_history: targetTime='{Raw}' snapped to {Start}–{End} (interval={Interval})",
                            rawTargetTime, llmParams["startDate"], llmParams["endDate"], interval);
                    }
                    else
                    {
                        _log.Warning("device_reading_history: could not parse targetTime '{Raw}'", rawTargetTime);
                    }
                }

                // Determine date range: default based on interval when not specified
                if (!llmParams.ContainsKey("startDate"))
                {
                    var defaultEnd = DateTime.UtcNow.Date;
                    var defaultStart = interval switch
                    {
                        "monthly" => defaultEnd.AddMonths(-12),
                        "yearly"  => defaultEnd.AddYears(-5),
                        "8hour"   => defaultEnd.AddDays(-7),
                        // 1H (Daily API): default to today — API returns all 24 hourly rows for that day
                        // 24H (Weekly API): default to last 7 days — API returns one row per day
                        "24hour"  => defaultEnd.AddDays(-6),
                        "1hour"   => defaultEnd,
                        _         => defaultEnd.AddDays(-1)  // 5min: last 24h
                    };
                    llmParams["startDate"] = defaultStart.ToString("yyyy-MM-dd HH:mm:ss");
                    llmParams["endDate"]   = defaultEnd.ToString("yyyy-MM-dd HH:mm:ss");
                }

                // ── Route to GetAQIGraphData API or SQL based on interval ────────────
                // GetAQIGraphData API criteria mapping:
                //   1hour   → criteria=Daily   — returns 24 hourly rows for a single day
                //   24hour  → criteria=Weekly  — returns 7 daily rows for the supplied date range
                //   monthly → criteria=Monthly — returns monthly summary rows
                //   yearly  → criteria=Yearly  — returns yearly summary rows
                //   8hour   → SQL only (no supported criteria)
                //   5min    → SQL only (raw ParameterReadings)
                var apiCriteria = interval switch
                {
                    "1hour"   => "Daily",
                    "24hour"  => "Weekly",
                    "monthly" => "Monthly",
                    "yearly"  => "Yearly",
                    _         => null  // 5min and 8hour use SQL
                };

                _log.Information("ChatController: device_reading_history deviceId={Id} interval={Interval} criteria={Criteria} param={Param}",
                    resolvedDeviceId, interval, apiCriteria ?? "SQL", paramName ?? "all");

                if (apiCriteria != null)
                {
                    DateTime.TryParse(llmParams.GetValueOrDefault("startDate"), out var apiFrom);
                    DateTime.TryParse(llmParams.GetValueOrDefault("endDate"),   out var apiTo);

                    // ── Adjust date window per API behaviour ────────────────────────────
                    // criteria=Daily (1H): API returns all 24 hourly rows for a single calendar day.
                    //   → Send the target day as both Fromdate and Todate.
                    //   → If user gave a targetTime (e.g. "at 8AM"), apiFrom.Date is already that day.
                    //   → The existing timestamp-window filter in the controller then picks the right hour.
                    //
                    // criteria=Weekly (24H): API returns one row per day for the supplied date range.
                    //   → Send a 7-day window ending on the target day.
                    //   → The existing date filter in the controller then picks the specific day requested.
                    if (apiCriteria == "Daily")
                    {
                        // Both dates = the target day (ignore time component)
                        var targetDay = apiFrom == default ? DateTime.UtcNow.Date : apiFrom.Date;
                        apiFrom = targetDay;
                        apiTo   = targetDay;
                    }
                    else if (apiCriteria == "Weekly")
                    {
                        // End on the requested day; start 6 days earlier to get a 7-day window
                        if (apiTo == default) apiTo = DateTime.UtcNow.Date;
                        apiFrom = apiTo.Date.AddDays(-6);
                        // Store the adjusted range back so the timestamp-filter below uses the right window
                        llmParams["startDate"] = apiFrom.ToString("yyyy-MM-dd HH:mm:ss");
                        llmParams["endDate"]   = apiTo.Date.AddDays(1).AddSeconds(-1).ToString("yyyy-MM-dd HH:mm:ss");
                    }
                    else
                    {
                        if (apiFrom == default) apiFrom = interval switch
                        {
                            "yearly"  => DateTime.UtcNow.AddYears(-5),
                            "monthly" => DateTime.UtcNow.AddMonths(-12),
                            _         => DateTime.UtcNow.AddDays(-7)
                        };
                        if (apiTo == default) apiTo = DateTime.UtcNow;
                    }

                    var apiResult = await _externalApi.GetAQIGraphDataAsync(
                        resolvedDeviceId, resolvedStationId, resolvedDeviceName, apiCriteria,
                        apiFrom, apiTo, paramName, bearerToken, ct);

                    if (!apiResult.Success)
                    {
                        _log.Warning("device_reading_history API_ERROR: {Error}", apiResult.Error);
                        return Ok(_formatter.FormatError(session.SessionId,
                            "Could not retrieve historical data for that device. Please try again.", "API_ERROR"));
                    }

                    var filteredRows = apiResult.Rows;

                    // ── 1H (Daily API): filter to the specific hour the user asked for ───
                    // criteria=Daily returns all 24 hourly rows for the day.
                    // When the user gave a targetTime (e.g. "at 8AM"), startDate/endDate were
                    // already snapped to the [8:00, 9:00) window — filter to that window.
                    // When no targetTime was given, return the full day (all rows).
                    //
                    // Timestamp comparison uses the time-of-day component only (hour match),
                    // ignoring timezone offset so "8AM" matches "8:00:00" regardless of +04:00.
                    if (apiCriteria == "Daily" && !string.IsNullOrWhiteSpace(rawTargetTime)
                        && DateTime.TryParse(llmParams.GetValueOrDefault("startDate"), out var windowStart1H)
                        && DateTime.TryParse(llmParams.GetValueOrDefault("endDate"),   out var windowEnd1H))
                    {
                        var targetHour = windowStart1H.Hour; // e.g. 8 for "8AM"

                        // Log all timestamps we received to aid diagnosis
                        _log.Information("device_reading_history (1H): targetHour={H}, available timestamps: {Ts}",
                            targetHour,
                            string.Join(", ", filteredRows
                                .Select(r => r.GetValueOrDefault("Timestamp")?.ToString())
                                .Distinct()
                                .Take(10)));

                        filteredRows = filteredRows.Where(r =>
                        {
                            var tsStr = r.GetValueOrDefault("Timestamp")?.ToString();
                            if (string.IsNullOrWhiteSpace(tsStr)) return false;
                            // Parse with DateTimeStyles.RoundtripKind to preserve any offset info
                            if (!DateTime.TryParse(tsStr,
                                    System.Globalization.CultureInfo.InvariantCulture,
                                    System.Globalization.DateTimeStyles.RoundtripKind, out var ts))
                                return false;
                            // Match on hour only — the stored time-of-day is what the user asked for
                            return ts.Hour == targetHour;
                        }).ToList();

                        _log.Information(
                            "device_reading_history (1H): filtered to hour {H}, {Count} rows remain",
                            targetHour, filteredRows.Count);

                        // Fallback: if exact hour not found, pick the nearest row per pollutant
                        if (filteredRows.Count == 0)
                        {
                            filteredRows = apiResult.Rows
                                .Where(r => DateTime.TryParse(r.GetValueOrDefault("Timestamp")?.ToString(),
                                    System.Globalization.CultureInfo.InvariantCulture,
                                    System.Globalization.DateTimeStyles.RoundtripKind, out _))
                                .GroupBy(r => r.GetValueOrDefault("ParameterName")?.ToString() ?? "")
                                .Select(g => g
                                    .OrderBy(r =>
                                    {
                                        DateTime.TryParse(r.GetValueOrDefault("Timestamp")?.ToString(),
                                            System.Globalization.CultureInfo.InvariantCulture,
                                            System.Globalization.DateTimeStyles.RoundtripKind, out var ts);
                                        return Math.Abs(ts.Hour - targetHour);
                                    })
                                    .First())
                                .ToList();

                            _log.Information(
                                "device_reading_history (1H): no exact hour match, nearest row per pollutant, {Count} rows",
                                filteredRows.Count);
                        }
                    }

                    // ── 24H (Weekly API): filter to the specific day the user asked for ──
                    // criteria=Weekly returns one row per day for up to 7 days.
                    // The user asked about a specific day (e.g. "yesterday", "last Monday") —
                    // that day's date is in startDate after the Weekly window adjustment above.
                    // Filter to rows whose timestamp falls on that calendar day.
                    if (apiCriteria == "Weekly")
                    {
                        // The target day the user asked about is the original endDate before we widened the window.
                        // Re-parse from the stored llmParams to get the target day's date.
                        DateTime.TryParse(llmParams.GetValueOrDefault("endDate"), out var weeklyEndBoundary);
                        var targetDay24H = weeklyEndBoundary == default
                            ? DateTime.UtcNow.Date
                            : weeklyEndBoundary.Date;

                        // When LLM gave a specific startDate (the user named a date), that is the target day
                        if (DateTime.TryParse(llmParams.GetValueOrDefault("startDate"), out var llmStart)
                            && llmStart.Date > DateTime.MinValue.Date)
                        {
                            // The stored startDate was set to apiFrom (window start, 6 days back).
                            // The target day is apiTo — stored as endDate before we added AddDays(1)-1sec.
                            // Use the endDate midnight boundary as the target day.
                            targetDay24H = weeklyEndBoundary.Date;
                        }

                        var dayStart = targetDay24H.Date;
                        var dayEnd   = dayStart.AddDays(1);

                        var dayFilteredRows = filteredRows.Where(r =>
                        {
                            var tsStr = r.GetValueOrDefault("Timestamp")?.ToString();
                            if (!DateTime.TryParse(tsStr, out var ts)) return false;
                            return ts >= dayStart && ts < dayEnd;
                        }).ToList();

                        _log.Information(
                            "device_reading_history (24H): filtered to day {Day}, {Count} rows remain (from {Total} total)",
                            dayStart.ToString("yyyy-MM-dd"), dayFilteredRows.Count, filteredRows.Count);

                        // If the target day has no row, fall back to the most recent available day
                        filteredRows = dayFilteredRows.Count > 0
                            ? dayFilteredRows
                            : filteredRows
                                .Where(r => DateTime.TryParse(r.GetValueOrDefault("Timestamp")?.ToString(), out _))
                                .GroupBy(r => r.GetValueOrDefault("ParameterName")?.ToString() ?? "")
                                .Select(g => g.OrderByDescending(r => r.GetValueOrDefault("Timestamp")?.ToString()).First())
                                .ToList();
                    }

                    // ── Filter by pollutant name when user specified one ───────────────
                    if (!string.IsNullOrWhiteSpace(paramName))
                    {
                        filteredRows = filteredRows.Where(r =>
                            string.Equals(
                                r.GetValueOrDefault("ParameterName")?.ToString(),
                                paramName,
                                StringComparison.OrdinalIgnoreCase))
                            .ToList();
                    }

                    resultRows      = filteredRows;
                    executionTimeMs = (int)apiResult.ExecutionTimeMs;
                    dataSourceLabel = interval switch
                    {
                        "monthly" => "ParameterAveragesMonth",
                        "yearly"  => "ParameterAveragesYear",
                        _         => "ParameterAverages"
                    };
                    goto afterSqlExecution;
                }

                // SQL path for 5min and 8hour intervals
                // When targetTime is active (point-in-time lookup), use nearest-record query
                // with a ±15 min window rather than a strict range, ordered by proximity.
                bool isPointInTime = !string.IsNullOrWhiteSpace(rawTargetTime)
                                     && llmParams.ContainsKey("startDate");

                string historySql;
                var historyParams = new Dictionary<string, object>
                {
                    ["deviceId"]           = resolvedDeviceId,
                    ["resolvedDeviceName"] = resolvedDeviceName,
                    ["startDate"]          = llmParams["startDate"],
                    ["endDate"]            = llmParams.GetValueOrDefault("endDate") ?? DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss")
                };
                if (!string.IsNullOrWhiteSpace(paramName))
                    historyParams["parameterName"] = paramName;

                if (interval == "8hour")
                {
                    historySql = $"""
                        SELECT TOP 200
                            @resolvedDeviceName AS DeviceName,
                            p.ParameterName,
                            ROUND(pa.Parametervalue, 2) AS ParameterValue,
                            u.UnitName,
                            pa.Interval AS Timestamp
                        FROM ParameterAverages pa
                        JOIN DMN_Parameters p ON pa.ParameterID = p.ID
                        JOIN ReportedUnits u ON p.UnitID = u.ID
                        WHERE p.DeviceID = @deviceId AND pa.Type = 8
                          AND pa.Status = 1{paramFilter}
                          AND pa.Interval >= @startDate AND pa.Interval <= @endDate
                        ORDER BY pa.Interval DESC
                        """;
                }
                else if (isPointInTime)
                {
                    // Point-in-time: widen to ±15 min and pick closest records per parameter
                    var targetDt = llmParams["startDate"]; // already snapped to 5-min boundary
                    historyParams["targetDateTime"] = targetDt;
                    historyParams["windowStart"]    = DateTime.Parse(targetDt).AddMinutes(-15).ToString("yyyy-MM-dd HH:mm:ss");
                    historyParams["windowEnd"]      = DateTime.Parse(targetDt).AddMinutes(15).ToString("yyyy-MM-dd HH:mm:ss");

                    historySql = $"""
                        SELECT
                            @resolvedDeviceName AS DeviceName,
                            p.ParameterName,
                            ROUND(pr.Parametervalue, 2) AS ParameterValue,
                            u.UnitName,
                            pr.CreatedTime AS Timestamp
                        FROM ParameterReadings pr
                        JOIN DMN_Parameters p ON pr.ParameterID = p.ID
                        JOIN ReportedUnits u ON p.UnitID = u.ID
                        WHERE p.DeviceID = @deviceId
                          AND pr.Status = 1{paramFilter}
                          AND pr.CreatedTime >= @windowStart AND pr.CreatedTime <= @windowEnd
                        ORDER BY ABS(DATEDIFF(SECOND, CAST(@targetDateTime AS DATETIME), pr.CreatedTime)),
                                 p.ParameterName
                        """;
                }
                else
                {
                    // Normal range query — last 24h or LLM-supplied date range
                    historySql = $"""
                        SELECT TOP 288
                            @resolvedDeviceName AS DeviceName,
                            p.ParameterName,
                            ROUND(pr.Parametervalue, 2) AS ParameterValue,
                            u.UnitName,
                            pr.CreatedTime AS Timestamp
                        FROM ParameterReadings pr
                        JOIN DMN_Parameters p ON pr.ParameterID = p.ID
                        JOIN ReportedUnits u ON p.UnitID = u.ID
                        WHERE p.DeviceID = @deviceId
                          AND pr.Status = 1{paramFilter}
                          AND pr.CreatedTime >= @startDate AND pr.CreatedTime <= @endDate
                        ORDER BY pr.CreatedTime DESC
                        """;
                }

                var histResult = await _sqlExecutor.ExecuteAsync(historySql, historyParams, ct);
                if (!histResult.Success)
                {
                    _log.Warning("device_reading_history SQL_ERROR: {Error}", histResult.Error);
                    return Ok(_formatter.FormatError(session.SessionId,
                        "Could not retrieve historical data for that device. Please try again.", "SQL_ERROR"));
                }

                // For point-in-time: deduplicate to one row per parameter (the closest timestamp)
                var rawHistRows = histResult.Rows;
                if (isPointInTime && rawHistRows.Count > 0)
                {
                    rawHistRows = rawHistRows
                        .GroupBy(r => r.GetValueOrDefault("ParameterName")?.ToString() ?? "")
                        .Select(g => g.First()) // already sorted by proximity ASC
                        .OrderBy(r => r.GetValueOrDefault("ParameterName")?.ToString())
                        .ToList();
                }

                resultRows      = rawHistRows;
                executionTimeMs = histResult.ExecutionTimeMs;
                dataSourceLabel = interval == "8hour" ? "ParameterAverages" : "ParameterReadings";
                goto afterSqlExecution;
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

            // Detect any @param references still in the SQL that were not bound (missing required params)
            var unboundParams = System.Text.RegularExpressions.Regex.Matches(safeSql, @"@([a-zA-Z]\w*)")
                .Select(m => m.Groups[1].Value)
                .Where(p => !sqlParams.ContainsKey(p))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();
            if (unboundParams.Count > 0)
            {
                var missing = template.Params
                    .Where(p => unboundParams.Contains(p.Name, StringComparer.OrdinalIgnoreCase) && !p.Optional)
                    .Select(p => p.Name)
                    .ToList();
                if (missing.Count > 0)
                {
                    _log.Warning("Missing required params {Params} for template {TemplateId} — asking for clarification",
                        string.Join(", ", missing), template.Id);
                    var clarifyMsg = (missing.Contains("stationId1", StringComparer.OrdinalIgnoreCase) || missing.Contains("stationId2", StringComparer.OrdinalIgnoreCase))
                        ? "Please specify the two site names you want to compare (e.g. 'Compare Al Reyada School and Al Saad Indian School')."
                        : missing.Contains("stationName", StringComparer.OrdinalIgnoreCase)
                        ? "Please specify the site name you are asking about (e.g. 'Al Reyada School', 'Abu Dhabi Residential', 'Al Saad Indian School')."
                        : missing.Contains("parameterName", StringComparer.OrdinalIgnoreCase)
                        ? "Please specify which pollutant you want (e.g. PM2.5, CO₂, Temperature, Humidity)."
                        : $"I need a bit more detail — please include: {string.Join(", ", missing)}.";
                    return Ok(_formatter.FormatError(session.SessionId, clarifyMsg, "MISSING_PARAM"));
                }
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

        // ── 7b. Physical / Chemical group filter for device queries ─────────
        // Physical: PM2.5, PM10, Temperature, Humidity, Noise
        // Chemical: CO, CO2 (CO₂), NO2 (NO₂), SO2 (SO₂), O3 (O₃), CH2O (CH₂O), VOC/TVOC
        if (template.Id is "device_last_reading" or "device_reading_history")
        {
            var lowerMsgDevice = request.Message.ToLowerInvariant();
            var wantsPhysical = lowerMsgDevice.Contains("physical");
            var wantsChemical = lowerMsgDevice.Contains("chemical");

            if (wantsPhysical && !wantsChemical)
            {
                var physicalParams = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
                    { "PM2.5", "PM10", "Temperature", "Humidity", "Noise" };
                resultRows = resultRows.Where(r =>
                    physicalParams.Contains(r.GetValueOrDefault("ParameterName")?.ToString() ?? ""))
                    .ToList();
            }
            else if (wantsChemical && !wantsPhysical)
            {
                // Match both plain and Unicode subscript forms stored in DB
                var chemicalParams = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
                    { "CO", "CO2", "CO\u2082", "NO2", "NO\u2082", "SO2", "SO\u2082",
                      "O3", "O\u2083", "CH2O", "CH\u2082O", "VOC", "TVOC" };
                resultRows = resultRows.Where(r =>
                    chemicalParams.Contains(r.GetValueOrDefault("ParameterName")?.ToString() ?? ""))
                    .ToList();
            }
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
