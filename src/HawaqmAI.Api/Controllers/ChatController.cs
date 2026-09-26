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

        // ── 2a. Enrich message with scope-bar context for routing and LLM ──
        // When the user has a sector selected in the filter bar, append it to the message
        // so the query router picks sector-aware templates (e.g. sites_by_sector) even when
        // the user's phrasing doesn't mention the sector explicitly.
        var enrichedMessage = request.Message;
        if (!string.IsNullOrWhiteSpace(scope.Sector))
        {
            var sectorLower = scope.Sector.ToLowerInvariant();
            if (!enrichedMessage.ToLowerInvariant().Contains(sectorLower))
                enrichedMessage = $"{enrichedMessage} in the {scope.Sector} sector";
        }
        if (!string.IsNullOrWhiteSpace(scope.Region))
        {
            var regionLower = scope.Region.ToLowerInvariant();
            if (!enrichedMessage.ToLowerInvariant().Contains(regionLower))
                enrichedMessage = $"{enrichedMessage} in {scope.Region}";
        }

        // ── 2a-ii. Follow-up reference resolution ──────────────────────────
        // When the user uses pronouns or positional references ("the above site",
        // "that device", "it", "the previous one", "same site", "next", "that school")
        // without naming the entity, extract the entity from the last user message
        // and inject it into the current question so routing and param extraction work correctly.
        var followUpReferenceWords = new[]
        {
            "above site", "above school", "above device", "above station",
            "that site", "that school", "that device", "that station",
            "this site", "this school", "this device", "this station",
            "the site", "the school", "the device", "the station",
            "previous site", "previous school", "previous device",
            "same site", "same school", "same device",
            "above", "that one", "this one", "it ", " it?", "the same"
        };
        var msgLower = enrichedMessage.ToLowerInvariant();
        bool isReferentialFollowUp = followUpReferenceWords.Any(w => msgLower.Contains(w))
            && history.Count > 0;

        if (isReferentialFollowUp)
        {
            // Find the last user message that contained a named entity
            var lastUserMessage = history.LastOrDefault(h => h.Role == "user")?.Content ?? "";

            // Extract the most likely entity: look for known site/device patterns in the previous message
            // Device: BA/SEI100M followed by digits
            var deviceMatch = System.Text.RegularExpressions.Regex.Match(
                lastUserMessage, @"\b(BA\s*\d{1,4}|SEI100M\s*\d{1,4})\b",
                System.Text.RegularExpressions.RegexOptions.IgnoreCase);

            if (deviceMatch.Success)
            {
                var deviceName = deviceMatch.Value.Trim();
                if (!enrichedMessage.Contains(deviceName, StringComparison.OrdinalIgnoreCase))
                {
                    enrichedMessage = $"{enrichedMessage} for device {deviceName}";
                    _log.Debug("Follow-up enriched with device '{Device}' from history", deviceName);
                }
            }
            else if (!string.IsNullOrWhiteSpace(lastUserMessage))
            {
                // For sites: strip common preposition prefixes to isolate the name portion.
                // Pattern: "for the site X", "at X", "for X", "at site X" — extract X.
                var siteMatch = System.Text.RegularExpressions.Regex.Match(
                    lastUserMessage,
                    @"(?:for|at|of|about|on)\s+(?:the\s+)?(?:site|school|station|device)?\s*([A-Z][A-Za-z0-9 \-'\.]{3,60}?)(?:\?|$|\s+in\s|\s+region|\s+aqi|\s+sector)",
                    System.Text.RegularExpressions.RegexOptions.IgnoreCase);

                if (siteMatch.Success)
                {
                    var siteName = siteMatch.Groups[1].Value.Trim().TrimEnd('?', '.', ',');
                    if (siteName.Length > 3 && !enrichedMessage.Contains(siteName, StringComparison.OrdinalIgnoreCase))
                    {
                        enrichedMessage = $"{enrichedMessage} for {siteName}";
                        _log.Debug("Follow-up enriched with site '{Site}' from history", siteName);
                    }
                }
                else
                {
                    // Fallback: append the entire previous user message as context hint
                    enrichedMessage = $"{enrichedMessage} (referring to: {lastUserMessage})";
                    _log.Debug("Follow-up enriched with full previous message as context");
                }
            }
        }

        // ── 2b. Hard pre-check: AQI range/definition questions must always go to FAQ ──
        // These questions ask WHAT a category means (numeric range), not which sites are in it.
        // Pattern scorer can be confused by words like "good", "moderate", "aqi" matching
        // site_aqi_by_category patterns, so we intercept here before routing.
        var lowerMsg = enrichedMessage.ToLowerInvariant();
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
                var faqPreId = Guid.NewGuid();
                await _sessionService.SaveBothMessagesAsync(
                    sessionId: session.SessionId,
                    userContent: request.Message,
                    assistantMessageId: faqPreId,
                    assistantContent: faqAnswerPre,
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
                    _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
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
                    var safetyMsgId = Guid.NewGuid();
                    var safetySummary = await _azureAI.SummarizeResultsAsync(
                        request.Message, safetyTemplate, safetyApiResult.Rows, scope, ct);
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId,
                        userContent: request.Message,
                        assistantMessageId: safetyMsgId,
                        assistantContent: safetySummary,
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
                        _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
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
        var routeResult = await _queryRouter.RouteAsync(enrichedMessage, user, history, ct);

        if (routeResult.NeedsClarification)
        {
            await _sessionService.SaveUserMessageAsync(session.SessionId, request.Message, ct);
            return Ok(_formatter.FormatError(
                session.SessionId,
                "I'm sorry — that's outside my scope. I can only help with HAWAQM air-quality data. If it's useful, you could ask me: 'Which devices are offline right now?' or 'What is the current AQI at my school?'",
                "CLARIFY"));
        }

        // ── 3b. Hard override: device name + date/time reference → device_reading_history ──
        // The pattern scorer matches "reading" → device_last_reading with high confidence,
        // but any date/time reference makes it historical. Override BEFORE LLM param fill.
        bool msgHasDeviceCode = System.Text.RegularExpressions.Regex.IsMatch(
            lowerMsg, @"(ba\s*\d{1,4}|sei\s*100\s*m\s*\d{1,4})");
        bool msgHasDateWord =
            lowerMsg.Contains("yesterday") || lowerMsg.Contains("today") ||
            lowerMsg.Contains("last hour") || lowerMsg.Contains("last day") ||
            lowerMsg.Contains("last week") || lowerMsg.Contains("last month") ||
            lowerMsg.Contains("last year") || lowerMsg.Contains("last 24") ||
            lowerMsg.Contains("this morning") || lowerMsg.Contains("this week") ||
            lowerMsg.Contains("this month") ||
            System.Text.RegularExpressions.Regex.IsMatch(lowerMsg,
                @"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)" +
                @"|\d{1,2}(st|nd|rd|th)|\d{4}-\d{2}-\d{2}|on\s+\d");

        _log.Debug("ChatController: deviceCode={HasDevice} dateWord={HasDate} routedTo={Template}",
            msgHasDeviceCode, msgHasDateWord, routeResult.Template?.Id ?? "candidates");

        if (msgHasDeviceCode && msgHasDateWord
            && routeResult.Template?.Id != "device_reading_history")
        {
            var histTemplate = _queryRouter.GetPermittedTemplates(user)
                .FirstOrDefault(t => t.Id == "device_reading_history");
            if (histTemplate is not null)
            {
                _log.Information("ChatController: overriding '{From}' → device_reading_history (device+date detected)",
                    routeResult.Template?.Id ?? "candidates");
                routeResult = new QueryRouterResult
                {
                    Template = histTemplate,
                    Score = 1.0,
                    Candidates = [],
                    PreviousTemplateId = routeResult.PreviousTemplateId
                };
            }
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
                enrichedMessage, routeResult.Candidates, history, scope, routeResult.PreviousTemplateId, ct);

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
                enrichedMessage, template, history, scope, ct);

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

            var faqMessageId = Guid.NewGuid();
            await _sessionService.SaveBothMessagesAsync(
                sessionId: session.SessionId,
                userContent: request.Message,
                assistantMessageId: faqMessageId,
                assistantContent: faqAnswer,
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
                _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);

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
                    SELECT TOP 1 d.ID AS DeviceId, d.DeviceName, d.StationID
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

                // ── Smart interval defaulting ──────────────────────────────────────
                // If the LLM did not extract an explicit interval, infer one from the
                // date range the user requested:
                //   < 2 hours requested  → 5min  (ParameterReadings — raw granularity)
                //   < 2 days requested   → 1hour (ParameterAverages Type=1)
                //   < 60 days requested  → 24hour (ParameterAverages Type=24)
                //   < 2 years requested  → monthly (ParameterAveragesMonth)
                //   otherwise            → yearly  (ParameterAveragesYear)
                var rawInterval = llmParams.GetValueOrDefault("interval");
                string interval;
                bool isSpecificPastDay = false;
                DateTime iStart = default;

                // Normalise whatever the LLM extracted to our canonical interval tokens.
                // The LLM sometimes returns "1H", "1h", "1 hour", "hourly", "daily", "24H" etc.
                static string NormaliseInterval(string raw)
                {
                    var r = raw.Trim().ToLowerInvariant().Replace(" ", "").Replace("-", "");
                    return r switch
                    {
                        "5min" or "5m" or "5mins" or "5minute" or "5minutes" or "fiveminute" => "5min",
                        "1h" or "1hour" or "1hours" or "hourly" or "onehour" => "1hour",
                        "8h" or "8hour" or "8hours" or "eighthour" => "8hour",
                        "24h" or "24hour" or "24hours" or "daily" or "1day" or "day" or "twentyfourhour" => "24hour",
                        "monthly" or "1month" or "month" or "1mo" => "monthly",
                        "yearly" or "annual" or "annually" or "1year" or "year" or "1y" => "yearly",
                        _ => ""   // unknown — fall through to span-based inference
                    };
                }

                var normalisedInterval = string.IsNullOrWhiteSpace(rawInterval) ? "" : NormaliseInterval(rawInterval);
                if (!string.IsNullOrWhiteSpace(normalisedInterval))
                {
                    interval = normalisedInterval;
                }
                else
                {
                    // Parse the date range the LLM extracted to determine span
                    DateTime.TryParse(llmParams.GetValueOrDefault("startDate"), out iStart);
                    DateTime.TryParse(llmParams.GetValueOrDefault("endDate"),   out var iEnd);

                    // For "yesterday", "last 2 days" etc. the LLM gives startDate but no interval.
                    // Compute span in hours; use endDate=now if not set.
                    var spanEnd   = iEnd   == default ? DateTime.UtcNow : iEnd;
                    var spanStart = iStart == default ? spanEnd.AddDays(-1) : iStart;
                    var spanHours = (spanEnd - spanStart).TotalHours;

                    // Special case: user asked for a specific past day (e.g. "Sep 19th", "on Monday")
                    // where startDate is a calendar day >24h ago and endDate = startDate+1 or unset.
                    // The span would be 0–24h, but the correct interval is 24hour (daily average),
                    // not 1hour — because the user wants the full-day summary, not hourly breakdown.
                    isSpecificPastDay = iStart != default
                        && (DateTime.UtcNow - iStart).TotalHours > 24   // the day is >24h in the past
                        && spanHours <= 25;                               // span is one calendar day

                    interval = isSpecificPastDay ? "24hour" : spanHours switch
                    {
                        < 2    => "5min",
                        < 48   => "1hour",   // up to 2 days → 1H averages
                        < 1440 => "24hour",  // up to 60 days → daily averages
                        < 8760 => "monthly", // up to 1 year → monthly averages
                        _      => "yearly"
                    };

                    _log.Information(
                        "device_reading_history: no interval extracted, span={Hours:F1}h pastDay={PastDay} → defaulted to '{Interval}'",
                        spanHours, isSpecificPastDay, interval);
                }

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
                    var defaultEnd = DateTime.UtcNow;
                    var defaultStart = interval switch
                    {
                        "yearly"  => new DateTime(defaultEnd.Year - 5, 1, 1),
                        "monthly" => defaultEnd.AddMonths(-12),
                        "24hour"  => defaultEnd.AddDays(-30),
                        "8hour"   => defaultEnd.AddDays(-7),
                        "1hour"   => defaultEnd.AddDays(-1),  // last 24h of 1H rows
                        _         => defaultEnd.AddDays(-1)   // 5min: last 24h
                    };
                    llmParams["startDate"] = defaultStart.ToString("yyyy-MM-dd HH:mm:ss");
                    llmParams["endDate"]   = defaultEnd.ToString("yyyy-MM-dd HH:mm:ss");
                }
                // When a specific past day was requested, ensure endDate covers the full day
                // (LLM often sets endDate = startDate with no time, so the window is 0 seconds).
                else if (isSpecificPastDay && iStart != default)
                {
                    llmParams["startDate"] = iStart.Date.ToString("yyyy-MM-dd HH:mm:ss");
                    llmParams["endDate"]   = iStart.Date.AddDays(1).AddSeconds(-1).ToString("yyyy-MM-dd HH:mm:ss");
                }

                // ── Route to correct SQL table based on interval ──────────────────
                // 5min    → ParameterReadings          (raw 5-minute readings)
                // 1hour   → ParameterAverages Type=1   (1-hour averages)
                // 8hour   → ParameterAverages Type=8   (8-hour averages)
                // 24hour  → ParameterAverages Type=24  (24-hour / daily averages)
                // monthly → ParameterAveragesMonth     (monthly averages)
                // yearly  → ParameterAveragesYear      (yearly averages)

                _log.Information("ChatController: device_reading_history deviceId={Id} interval={Interval} param={Param}",
                    resolvedDeviceId, interval, paramName ?? "all");

                // Point-in-time lookup: used for 5min when targetTime is supplied
                bool isPointInTime = !string.IsNullOrWhiteSpace(rawTargetTime)
                                     && llmParams.ContainsKey("startDate")
                                     && interval == "5min";

                // Validate startDate/endDate are actual dates — LLM can put interval strings like "1H" in these fields.
                // If invalid, fall back to sensible defaults so the SQL doesn't throw a conversion error.
                static DateTime SafeParseDate(string? raw, DateTime fallback)
                    => DateTime.TryParse(raw, out var dt) ? dt : fallback;

                var safeStartDate = SafeParseDate(
                    llmParams.GetValueOrDefault("startDate"),
                    DateTime.UtcNow.AddDays(-1));
                var safeEndDate = SafeParseDate(
                    llmParams.GetValueOrDefault("endDate"),
                    DateTime.UtcNow);

                // If start > end (LLM got them backwards) or they're equal (0-span), fix end.
                if (safeEndDate <= safeStartDate)
                    safeEndDate = safeStartDate.Date.AddDays(1).AddSeconds(-1);

                // ParameterAverages.DeviceID and DMN_Parameters.DeviceID both have dirty nvarchar
                // data (e.g. '1H') that causes conversion errors when compared with an int param.
                // Safest approach: pre-fetch the list of DMN_Parameters.ID for this device using
                // TRY_CAST to skip bad rows, then filter averages/readings by ParameterID IN (...).
                var paramIdLookup = await _sqlExecutor.ExecuteAsync(
                    "SELECT p.ID FROM DMN_Parameters p WHERE TRY_CAST(p.DeviceID AS INT) = @deviceIntId AND p.Status = 1",
                    new Dictionary<string, object> { ["deviceIntId"] = resolvedDeviceId }, ct);

                var parameterIds = paramIdLookup.Success
                    ? paramIdLookup.Rows
                        .Select(r => r.GetValueOrDefault("ID")?.ToString() ?? "")
                        .Where(id => !string.IsNullOrWhiteSpace(id))
                        .ToList()
                    : new List<string>();

                if (parameterIds.Count == 0)
                {
                    _log.Warning("device_reading_history: no parameters found for deviceId={Id}", resolvedDeviceId);
                    resultRows = [];
                    executionTimeMs = 0;
                    dataSourceLabel = "ParameterAverages";
                    goto afterSqlExecution;
                }

                // Build an inline IN list (safe — all values are int IDs from DB, not user input)
                var paramIdList = string.Join(",", parameterIds);
                var paramNameFilter2 = string.IsNullOrWhiteSpace(paramName)
                    ? "" : " AND p.ParameterName = @parameterName";

                string historySql;
                var historyParams = new Dictionary<string, object>
                {
                    ["resolvedDeviceName"] = resolvedDeviceName,
                    ["startDate"]          = safeStartDate.ToString("yyyy-MM-dd HH:mm:ss"),
                    ["endDate"]            = safeEndDate.ToString("yyyy-MM-dd HH:mm:ss")
                };
                if (!string.IsNullOrWhiteSpace(paramName))
                    historyParams["parameterName"] = paramName;

                switch (interval)
                {
                    case "1hour":
                        historySql = $"""
                            SELECT TOP 500
                                @resolvedDeviceName AS DeviceName,
                                p.ParameterName,
                                ROUND(pa.Parametervalue, 2) AS ParameterValue,
                                u.UnitName,
                                pa.Interval AS Timestamp
                            FROM ParameterAverages pa
                            JOIN DMN_Parameters p ON pa.ParameterID = p.ID
                            JOIN ReportedUnits u ON p.UnitID = u.ID
                            WHERE pa.ParameterID IN ({paramIdList}) AND pa.Type = 1
                              AND pa.Status = 1{paramNameFilter2}
                              AND pa.Interval >= @startDate AND pa.Interval <= @endDate
                            ORDER BY pa.Interval DESC
                            """;
                        dataSourceLabel = "ParameterAverages";
                        break;

                    case "8hour":
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
                            WHERE pa.ParameterID IN ({paramIdList}) AND pa.Type = 8
                              AND pa.Status = 1{paramNameFilter2}
                              AND pa.Interval >= @startDate AND pa.Interval <= @endDate
                            ORDER BY pa.Interval DESC
                            """;
                        dataSourceLabel = "ParameterAverages";
                        break;

                    case "24hour":
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
                            WHERE pa.ParameterID IN ({paramIdList}) AND pa.Type = 24
                              AND pa.Status = 1{paramNameFilter2}
                              AND pa.Interval >= @startDate AND pa.Interval <= @endDate
                            ORDER BY pa.Interval DESC
                            """;
                        dataSourceLabel = "ParameterAverages";
                        break;

                    case "monthly":
                        historySql = $"""
                            SELECT TOP 60
                                @resolvedDeviceName AS DeviceName,
                                p.ParameterName,
                                ROUND(pam.Parametervalue, 2) AS ParameterValue,
                                u.UnitName,
                                pam.Interval AS Timestamp
                            FROM ParameterAveragesMonth pam
                            JOIN DMN_Parameters p ON pam.ParameterID = p.ID
                            JOIN ReportedUnits u ON p.UnitID = u.ID
                            WHERE pam.ParameterID IN ({paramIdList})
                              AND pam.Status = 1{paramNameFilter2}
                              AND pam.Interval >= @startDate AND pam.Interval <= @endDate
                            ORDER BY pam.Interval DESC
                            """;
                        dataSourceLabel = "ParameterAveragesMonth";
                        break;

                    case "yearly":
                        historySql = $"""
                            SELECT TOP 20
                                @resolvedDeviceName AS DeviceName,
                                p.ParameterName,
                                ROUND(pay.Parametervalue, 2) AS ParameterValue,
                                u.UnitName,
                                pay.Interval AS Timestamp
                            FROM ParameterAveragesYear pay
                            JOIN DMN_Parameters p ON pay.ParameterID = p.ID
                            JOIN ReportedUnits u ON p.UnitID = u.ID
                            WHERE pay.ParameterID IN ({paramIdList})
                              AND pay.Status = 1{paramNameFilter2}
                              AND pay.Interval >= @startDate AND pay.Interval <= @endDate
                            ORDER BY pay.Interval DESC
                            """;
                        dataSourceLabel = "ParameterAveragesYear";
                        break;

                    default: // 5min — ParameterReadings
                        if (isPointInTime)
                        {
                            var targetDt = safeStartDate.ToString("yyyy-MM-dd HH:mm:ss");
                            historyParams["targetDateTime"] = targetDt;
                            historyParams["windowStart"]    = safeStartDate.AddMinutes(-15).ToString("yyyy-MM-dd HH:mm:ss");
                            historyParams["windowEnd"]      = safeStartDate.AddMinutes(15).ToString("yyyy-MM-dd HH:mm:ss");

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
                                WHERE pr.ParameterID IN ({paramIdList})
                                  AND pr.Status = 1{paramNameFilter2}
                                  AND pr.CreatedTime >= @windowStart AND pr.CreatedTime <= @windowEnd
                                ORDER BY ABS(DATEDIFF(SECOND, CAST(@targetDateTime AS DATETIME), pr.CreatedTime)),
                                         p.ParameterName
                                """;
                        }
                        else
                        {
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
                                WHERE pr.ParameterID IN ({paramIdList})
                                  AND pr.Status = 1{paramNameFilter2}
                                  AND pr.CreatedTime >= @startDate AND pr.CreatedTime <= @endDate
                                ORDER BY pr.CreatedTime DESC
                                """;
                        }
                        dataSourceLabel = "ParameterReadings";
                        break;
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
        // When no data is returned, short-circuit with a polite message instead of
        // letting the LLM hallucinate an empty markdown table in the summary.
        if (resultRows.Count == 0 && template.ResponseType != "text")
        {
            var noDataMessage = BuildNoDataMessage(template.Id, scope);

            var noDataMsgId = Guid.NewGuid();
            await _sessionService.SaveBothMessagesAsync(
                sessionId: session.SessionId,
                userContent: request.Message,
                assistantMessageId: noDataMsgId,
                assistantContent: noDataMessage,
                sql: null,
                responseType: "text",
                chartType: null,
                chartDataJson: null,
                dataSource: dataSourceLabel,
                dateRange: scope.DateRangeLabel,
                modelUsed: modelUsed,
                executionTimeMs: (int)Math.Min(executionTimeMs, int.MaxValue),
                tokenCount: tokensUsed,
                ct: ct);

            if (session.MessageCount <= 1)
                _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);

            return Ok(_formatter.FormatError(session.SessionId, noDataMessage, "NO_DATA"));
        }

        // For table/chart responses the data is rendered visually — skip LLM summarization.
        // Generate a zero-cost summary string instead (~0ms vs ~10s).
        // Only call SummarizeResultsAsync for text responses where the answer IS the summary.
        string summary;
        if (template.ResponseType is "table" or "chart")
        {
            summary = resultRows.Count == 1
                ? "1 result found."
                : $"{resultRows.Count} results found.";
            tokensUsed = 0;
        }
        else
        {
            summary = await _azureAI.SummarizeResultsAsync(
                enrichedMessage, template, resultRows, scope, ct);
        }

        // ── 9. Persist conversation (single DB round-trip) ──────────────────
        var messageId = Guid.NewGuid();
        var chartDataJson = resultRows.Count > 0 && template.ResponseType == "chart"
            ? System.Text.Json.JsonSerializer.Serialize(resultRows)
            : null;

        await _sessionService.SaveBothMessagesAsync(
            sessionId: session.SessionId,
            userContent: request.Message,
            assistantMessageId: messageId,
            assistantContent: summary,
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

        // Auto-title session from first message — fire-and-forget (does not block response)
        if (session.MessageCount <= 1)
            _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);

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

    private static string BuildNoDataMessage(string templateId, ResolvedScope scope)
    {
        var region = string.IsNullOrWhiteSpace(scope.Region) ? "the selected region" : scope.Region;
        var date = scope.DateRangeLabel ?? "the selected date";

        return templateId switch
        {
            "current_all_stations" or "site_aqi_all" =>
                $"No air quality readings are currently available for {region}. This may be because devices are still syncing or no active sites have reported data yet. Please try again shortly.",

            "sites_by_region" or "count_sites_in_region" =>
                $"No sites were found for {region}. Please check that the region name is correct or try a different filter.",

            "sites_outside_safe_range" or "site_aqi_by_category" =>
                $"Great news — no sites in {region} are currently outside the safe air quality range.",

            "critical_sites" or "critical_sites_today" =>
                $"No critical sites were found in {region} for {date}. All monitored sites appear to be within acceptable limits.",

            "offline_devices" or "devices_offline" =>
                $"All devices in {region} appear to be online. No offline devices were detected at this time.",

            "top_polluted_sites" =>
                $"No pollution data is available for {region} at the moment. Devices may still be syncing. Please try again shortly.",

            _ =>
                $"No data was found for your query in {region} for {date}. The devices may be syncing or no readings match the selected filters. Please try adjusting your filters or check back shortly."
        };
    }
}
