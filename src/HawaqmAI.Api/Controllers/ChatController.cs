using System.Text.Json;
using HawaqmAI.Api.Configuration;
using HawaqmAI.Api.Models;
using HawaqmAI.Api.Services;
using HawaqmAI.Api.Validators;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;
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
    private readonly ChatOptions _chatOptions;

    public ChatController(
        IQueryRouterService queryRouter,
        IAzureAIService azureAI,
        IRbacEngine rbac,
        ISqlExecutorService sqlExecutor,
        ISessionService sessionService,
        IResponseFormatterService formatter,
        IExternalApiService externalApi,
        IOptions<ChatOptions> chatOptions)
    {
        _queryRouter = queryRouter;
        _azureAI = azureAI;
        _rbac = rbac;
        _sqlExecutor = sqlExecutor;
        _sessionService = sessionService;
        _formatter = formatter;
        _externalApi = externalApi;
        _chatOptions = chatOptions.Value;
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
        try
        {

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

        // ── 2a-pre. Daily token limit enforcement ───────────────────────────
        if (_chatOptions.DailyTokenLimitPerUser > 0)
        {
            var todayUsage = await _sessionService.GetDailyTokenUsageAsync(user.UserId, ct);
            if (todayUsage >= _chatOptions.DailyTokenLimitPerUser)
            {
                _log.Warning("Daily token limit reached for user {UserId}: {Used}/{Limit}",
                    user.UserId, todayUsage, _chatOptions.DailyTokenLimitPerUser);
                await _sessionService.SaveUserMessageAsync(session.SessionId, request.Message, ct);
                return Ok(_formatter.FormatError(session.SessionId,
                    $"You have reached your daily usage limit ({_chatOptions.DailyTokenLimitPerUser:N0} tokens). Your limit resets at midnight UTC. Please try again tomorrow.",
                    "DAILY_LIMIT_REACHED"));
            }
            _log.Debug("Daily token usage for user {UserId}: {Used}/{Limit}", user.UserId, todayUsage, _chatOptions.DailyTokenLimitPerUser);
        }

        // ── 2a. Message enrichment ──────────────────────────────────────────
        // Do NOT append scope filter-bar values to the message — the filter bar
        // selections (region, sector) are only applied when the user explicitly
        // mentions them in their question. Broad questions ("chart of all schools")
        // should return all data regardless of what the filter bar shows.
        var enrichedMessage = request.Message;

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
            // Device: letter-prefix (1-8 chars, optionally mixed with digits) followed by 1-4 digit suffix.
            // Generic pattern works for current models (BA, SEI100M) and any future device naming convention.
            var deviceMatch = System.Text.RegularExpressions.Regex.Match(
                lastUserMessage, @"\b([A-Z]{1,8}(?:\d+[A-Z]+)*\s*\d{1,4})\b",
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

        // ── 2a-dsr. Dynamic DSR interpretation pre-check ──
        // Handles "What does a X% Data Success Rate indicate for YEAR?" and variants.
        // Extracts DSR value and year from the question and returns a computed interpretation.
        var lowerMsg = enrichedMessage.ToLowerInvariant();
        // Normalise curly/smart apostrophes and quotes → straight so keyword phrases match regardless
        // of how the user's keyboard or browser encodes them (e.g. "can't" vs "can\u2019t").
        lowerMsg = lowerMsg.Replace('\u2019', '\'').Replace('\u2018', '\'')
                           .Replace('\u201c', '"').Replace('\u201d', '"');
        // Normalise "residence" / "residence sector" → "residential" so all sector checks work uniformly.
        // "residence" is a natural alias users type for the Residential sector.
        lowerMsg = System.Text.RegularExpressions.Regex.Replace(
            lowerMsg, @"\bresidence\b", "residential", System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        {
            // DSR question detection: question must mention "data success rate", "data rate", or "dsr"
            // AND contain a percentage number somewhere in the message.
            bool hasDsrKeyword = lowerMsg.Contains("data success rate") || lowerMsg.Contains("dsr")
                || (lowerMsg.Contains("data rate") && (lowerMsg.Contains("%") || System.Text.RegularExpressions.Regex.IsMatch(lowerMsg, @"\d")));

            // Match number adjacent to "data success rate / dsr" (tight match)
            var dsrMatch = System.Text.RegularExpressions.Regex.Match(
                lowerMsg,
                @"(?:(\d{1,3}(?:\.\d+)?)\s*%?\s*(?:data success rate|dsr))|(?:(?:data success rate|dsr)\s*(?:of\s*)?(\d{1,3}(?:\.\d+)?)\s*%?)",
                System.Text.RegularExpressions.RegexOptions.IgnoreCase);
            // Fallback: any percentage number in the message when DSR keyword present
            var anyPercentMatch = hasDsrKeyword
                ? System.Text.RegularExpressions.Regex.Match(lowerMsg, @"(\d{1,3}(?:\.\d+)?)\s*%")
                : System.Text.RegularExpressions.Match.Empty;

            string dsrRawValue = dsrMatch.Success
                ? (dsrMatch.Groups[1].Success ? dsrMatch.Groups[1].Value : dsrMatch.Groups[2].Value)
                : (anyPercentMatch.Success ? anyPercentMatch.Groups[1].Value : string.Empty);

            var dsrInterpretTriggers = new[] {
                "indicate", "mean", "what does", "what is", "tell me about", "explain",
                "interpret", "how should i", "how do i", "how to interpret", "understand",
                "what can i", "implication", "significance", "represent", "reflect",
                "good data rate", "good dsr", "acceptable data rate", "acceptable dsr",
                "is it good", "is this good", "is that good", "good rate", "bad rate",
                "what does it mean", "what does this mean"
            };
            bool isDsrInterpretation = hasDsrKeyword && !string.IsNullOrEmpty(dsrRawValue)
                && dsrInterpretTriggers.Any(t => lowerMsg.Contains(t));

            if (isDsrInterpretation && double.TryParse(dsrRawValue, out double dsrValue))
            {
                // Extract year from question (any 4-digit year), default to current year
                var yearMatch = System.Text.RegularExpressions.Regex.Match(lowerMsg, @"\b(\d{4})\b");
                int currentYear = DateTime.Now.Year;
                // Filter out years that look like percentages or nonsense — accept 1990–2999
                int targetYear = currentYear;
                if (yearMatch.Success && int.TryParse(yearMatch.Groups[1].Value, out int parsedYear)
                    && parsedYear >= 1990 && parsedYear <= 2999
                    && parsedYear != (int)dsrValue) // don't confuse DSR value for year
                    targetYear = parsedYear;
                bool isCurrentYear = targetYear == currentYear;
                string ytdSuffix = isCurrentYear ? " so far this year (year-to-date)" : $" for {targetYear}";

                double target = 75.0;
                double gap = Math.Round(Math.Abs(dsrValue - target), 1);
                string aboveOrBelow = dsrValue >= target ? "above" : "below";

                string interpretationLine;
                if (dsrValue >= 90)
                    interpretationLine = "This is an excellent result — data coverage is strong and readings can be used with high confidence for official reporting.";
                else if (dsrValue >= 75)
                    interpretationLine = "Data coverage meets the HAWAQM minimum standard. Results are reliable for reporting, though any gap from 100% means some readings were missed.";
                else if (dsrValue >= 50)
                    interpretationLine = "Data coverage is below the acceptable threshold. Use results with caution and quote the DSR alongside any figures, as averages are built on incomplete data.";
                else
                    interpretationLine = "Data coverage is significantly below the minimum standard. Reliability is low — findings should not be used for official reporting without investigation into the missing readings.";

                var dsrAnswer = $"A Data Success Rate of {dsrValue}%{ytdSuffix} means that {dsrValue}% of the readings expected from the devices during that period actually arrived. " +
                    $"Devices report every 5 minutes, so the expected count for {targetYear} is known exactly. " +
                    $"The HAWAQM target is at least 75%, and {dsrValue}% is {aboveOrBelow} the target by {gap} percentage points. " +
                    interpretationLine;

                var dsrTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "faq_answer");
                if (dsrTemplate is not null)
                {
                    var dsrMsgId = Guid.NewGuid();
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId,
                        userContent: request.Message,
                        assistantMessageId: dsrMsgId,
                        assistantContent: dsrAnswer,
                        sql: null,
                        responseType: "text",
                        chartType: null,
                        chartDataJson: null,
                        dataSource: "DSR_CALC",
                        dateRange: null,
                        modelUsed: "rule",
                        executionTimeMs: 0,
                        tokenCount: 0,
                        ct: ct);
                    if (session.MessageCount <= 1)
                        _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                    return Ok(_formatter.Format(
                        sessionId: session.SessionId,
                        messageId: dsrMsgId,
                        template: dsrTemplate,
                        llmSummary: dsrAnswer,
                        rows: [],
                        rowCount: 0,
                        executionTimeMs: 0,
                        scope: scope,
                        modelUsed: "rule",
                        tokenCount: 0));
                }
            }
        }

        // ── 2b. Hard pre-check: AQI range/definition questions must always go to FAQ ──
        // These questions ask WHAT a category means (numeric range), not which sites are in it.
        // Pattern scorer can be confused by words like "good", "moderate", "aqi" matching
        // site_aqi_by_category patterns, so we intercept here before routing.
        var aqiRangeDefinitionKeywords = new[] {
            // range-based phrasings
            "what aqi range", "aqi range is considered", "range considered good", "range considered moderate",
            "range considered unhealthy", "range considered hazardous", "range considered very unhealthy",
            "range of aqi", "range for good", "range for moderate", "range for unhealthy", "range for hazardous",
            "range for very unhealthy", "range for good air", "range for good air quality",
            "what's the range for", "whats the range for",
            "aqi good range", "aqi moderate range", "aqi unhealthy range", "aqi hazardous range",
            "what aqi range is", "aqi range for",
            "when is aqi considered good", "when is aqi considered moderate",
            "when is aqi considered unhealthy", "when is aqi considered hazardous",
            "when is aqi considered very unhealthy",
            "which aqi values are classified", "what aqi values are classified",
            "which aqi values fall", "what aqi values fall",
            "aqi values classified as", "aqi value for hazardous", "aqi value for good",
            "aqi value for moderate", "aqi value for unhealthy", "aqi value for very unhealthy",
            "hazardous aqi value", "very unhealthy aqi value", "good aqi value",
            "moderate aqi value", "unhealthy aqi value",
            "aqi for sensitive groups", "unhealthy for sensitive groups", "aqi for sensitive groups",
            "classified as unhealthy", "classified as good", "classified as moderate",
            "classified as hazardous", "classified as very unhealthy",
            "fall within the moderate", "fall within the good", "fall within the unhealthy",
            "fall within the hazardous", "fall under the moderate", "fall under the good",
            "fall under the unhealthy", "fall under the hazardous",
            "aqi numbers mean", "what aqi numbers mean", "aqi counts as",
            "what aqi counts as", "aqi is good", "acceptable aqi range",
            "what is acceptable aqi", "aqi mean good", "aqi mean moderate",
            "acceptable aqi range for", "aqi range for the good",
            "aqi range for the moderate", "aqi range for the unhealthy",
            "aqi range for the hazardous", "aqi range for the very unhealthy",
            "aqi range for the sensitive", "what is the acceptable aqi",
            // "considered" phrasings — "what aqi value/range/level is considered X"
            // NOTE: use specific phrases only — "considered good" alone is too broad and blocks
            // valid site-filter questions like "show me sites considered hazardous"
            "what aqi is considered", "aqi is considered good", "aqi is considered moderate",
            "aqi is considered unhealthy", "aqi is considered hazardous",
            "aqi value is considered", "aqi level is considered",
            "aqi range is considered good", "aqi range is considered moderate",
            "aqi range is considered unhealthy", "aqi range is considered hazardous",
            "range is considered good", "range is considered moderate",
            "range is considered unhealthy", "range is considered hazardous",
            // "what air quality is considered X" phrasings
            "what air quality is considered", "air quality is considered unhealthy",
            "air quality is considered good", "air quality is considered moderate",
            "air quality is considered hazardous", "air quality is considered very unhealthy",
            "what is considered unhealthy air", "what is considered good air quality",
            "what is considered moderate air quality", "what is considered hazardous air quality",
            "considered unhealthy air quality", "considered good air quality",
            // "what aqi is X" phrasings
            "what is good aqi", "what is moderate aqi", "what is unhealthy aqi", "what is hazardous aqi",
            "what aqi is good", "what aqi is moderate", "what aqi is unhealthy", "what aqi is hazardous",
            "what aqi is very unhealthy",
            // "what value is X" phrasings
            "what value is good", "what value is moderate", "what value is unhealthy", "what value is hazardous",
            "what aqi value is good", "what aqi value is moderate", "what aqi value is unhealthy",
            "what aqi value is hazardous", "what aqi value is very unhealthy",
            // calculation / general — only definition questions, NOT "show sites by aqi category"
            "how is aqi calculated", "how is site aqi calculated",
            "aqi classification", "aqi category breakdown",
            "what are the aqi categories", "what are the aqi levels", "what are aqi categories", "what are aqi levels",
            "aqi scale", "aqi meaning", "what does aqi mean",
            "explain aqi", "explain the aqi", "aqi explained",
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
            "data recording frequency",
            "recording frequency",
            "capture frequency",
            "at what intervals do",
            "at what interval do",
            "what intervals do hawaqm",
            "capture data",
            "intervals do devices capture",
            "intervals do hawaqm devices",
            "recording interval",
            "data interval",
            "how frequently do",
            "how frequently does hawaqm",
            "frequency of data recording",
            "frequency of recording",
            "how often is data recorded",
            "when is a device considered offline",
            "when is a site shown as inactive",
            "how is the site aqi calculated",
            "which air-quality parameters are monitored",
            "what parameters are monitored",
            "does hawaqm monitor temperature",
            "does hawaqm measure temperature",
            "does hawaqm measure temperature and humidity",
            "does hawaqm track temperature",
            "are temperature, humidity and noise monitored",
            "are temperature and humidity monitored",
            "does it track noise",
            "does hawaqm track noise",
            "does hawaqm monitor noise",
            "does hawaqm measure noise",
            "besides pollutants, does hawaqm",
            "besides pollutants does hawaqm",
            "does hawaqm also measure temperature",
            "does hawaqm track environmental conditions",
            "does hawaqm also monitor temperature",
            "does hawaqm track humidity",
            "does hawaqm measure humidity",
            "how does the system calculate the overall aqi for a site",
            "how is the overall aqi for a site calculated",
            "how is the site aqi calculated for multiple devices",
            "how does site aqi work with more than one device",
            "site with more than one device",
            "site with multiple devices",
            "overall aqi for a site with",
            "calculate the overall aqi",
            "can i export hawaqm data",
            "can i view both indoor and ambient",
            "can i see indoor and ambient",
            "does the map show both indoor and ambient",
            "does hawaqm map show indoor and ambient",
            "does the map show indoor",
            "does the map show ambient",
            "indoor and ambient air-quality locations on the map",
            "indoor and ambient locations on the map",
            "indoor & ambient air-quality locations on map",
            "indoor & ambient air-quality locations",
            "indoor & ambient locations",
            "indoor & ambient",
            "map show both indoor and ambient",
            "show indoor and outdoor on the map",
            "see indoor and ambient air-quality",
            "indoor outdoor map",
            "ambient stations on the map",
            "indoor and outdoor air quality",
            "indoor and outdoor aqi",
            "indoor and outdoor air quality on map",
            "indoor and outdoor aqi on map",
            "indoor and outdoor air quality or aqi",
            "indoor and outdoor air quality or aqi locations on map",
            "show me indoor and outdoor",
            "show indoor and outdoor air quality",
            "show indoor and outdoor aqi",
            "aqi locations on map",
            "air quality locations on map",
            "show aqi locations on map",
            "show air quality locations on map",
            "outdoor air quality on map",
            "outdoor aqi on map",
            "view indoor and outdoor",
            "see indoor and outdoor air quality",
            "see indoor and outdoor aqi",
            "what happens if a site has no geographic",
            "what happens when a site has no geographic",
            "what happens when a site has no location",
            "site has no location coordinates",
            "site has no geographic coordinates",
            "site with no geographic coordinates",
            "site with no location coordinates",
            "site without coordinates",
            "no geographic coordinates",
            "no location coordinates",
            "can a site without geographic coordinates",
            "can a site without coordinates",
            "site without geographic coordinates still",
            "shown on the map without coordinates",
            "site without coordinates",
            "what is moccae",
            "what are the total regions",
            "which regions are covered",
            "which areas are covered by hawaqm",
            "which areas does hawaqm cover",
            "what areas are covered by hawaqm",
            "areas covered by hawaqm",
            "which regions does hawaqm cover",
            "what regions does hawaqm cover",
            "hawaqm covered regions",
            "hawaqm regions",
            "regions in hawaqm",
            "regions available in hawaqm",
            "can you tell me the regions",
            "what data am i able to see",
            "what kind of data can i see",
            "what can i view in hawaqm",
            "what can i see in hawaqm",
            "what data can i see",
            "what data can i access",
            "what information is available in hawaqm",
            "can you tell me what information is available",
            "what information can i access in hawaqm",
            "information available in hawaqm",
            "hawaqm data available",
            "older year device count unchanged",
            "older year — device count unchanged",
            "older year - device count unchanged",
            "device count unchanged older year",
            "device count unchanged when selecting older year",
            "device count same for older year",
            "device count does not change for older year",
            "why is device count unchanged for older year",
            "should the device numbers change when i select a different year",
            "should device numbers change when i select a different year",
            "why do the device numbers not change",
            "why don't the device numbers change",
            "why dont the device numbers change",
            "device numbers stay the same when i change the year",
            "device count doesn't change when i select a year",
            "device count does not change when i select a year",
            "why does the device count not change when i change the year",
            "total devices not changing when i select year",
            "active devices not changing when i select year",
            "why do device numbers stay the same",
            "device numbers change with year",
            "why can't i see all the sites",
            "why cant i see all the sites",
            "why can't i see all sites",
            "why cant i see all sites",
            "why don't i see all the sites",
            "why dont i see all the sites",
            "why don't i see all sites",
            "why dont i see all sites",
            "i can't see all sites",
            "i cant see all sites",
            "i can't see all the sites",
            "i cant see all the sites",
            "not seeing all sites",
            "cannot see all sites",
            "can't see all sites",
            "missing sites from my view",
            "some sites are missing",
            "sites not showing in my account",
            "why are some sites not visible",
            "why are some sites missing",
            "not all sites are visible",
            "not all sites showing",
            "i don't see all the sites",
            "i dont see all the sites",
            "data available in hawaqm",
            "data available hawaqm",
            "what is the current data success rate",
            "what does data success rate mean",
            "why do the total device",
            "why is a site missing from the geographical map",
            "why is a site not showing on the geographical map",
            "can i get the last 24 hours of readings for a device",
            "can i see the readings from the last 24 hours",
            "can i see the last 24 hours",
            "can i view the last 24 hours",
            "show readings from the last 24 hours for a device",
            "readings from the last 24 hours for a device",
            "last 24 hours for a device",
            "24 hours of readings for a device",
            "see the readings from the last 24",
            "get the last 24 hours for a device",
            "device readings — last 24 hours",
            "device readings - last 24 hours",
            "device readings last 24 hours",
            "device readings — last 24",
            "device readings - last 24",
            "can i use the statistical report",
            "can i use the exceedance report",
            "which averaging intervals should be used",
            "six-month exceedance analysis",
            "six month exceedance analysis",
            "6 month exceedance analysis",
            "6-month exceedance analysis",
            "exceedance analysis for six months",
            "exceedance analysis for 6 months",
            "six months of exceedance analysis",
            "6 months of exceedance analysis",
            "i need six months of exceedance",
            "why can the site aqi look acceptable",
            "site average be hiding a problem",
            "how can one device have a high reading",
            "one device have a high reading",
            "one device high reading while",
            "one device shows high",
            "device have a high reading while",
            "device high reading while the overall",
            "device high reading while the site",
            "high reading while the overall site",
            "high reading while the site aqi",
            "device reading high but site",
            "device aqi high but overall",
            "overall site aqi is still acceptable",
            "site aqi is still acceptable",
            "overall aqi is still acceptable",
            "site aqi still acceptable when",
            "site aqi still good when",
            "could one device have poor readings even if the overall site aqi looks okay",
            "one device have poor readings even if the overall site aqi",
            "one device have poor readings even if",
            "could one device have poor readings",
            "can one device have poor readings",
            "one device poor readings site aqi",
            "device poor readings overall site",
            "poor readings even if the overall",
            "device reading poor but site aqi",
            "one device runs high offset by another",
            "device running high offset",
            "site aqi looks okay but device",
            "site aqi looks fine but device",
            "overall site aqi looks okay",
            "overall site looks okay but device",
            "can you prepare a one-year raw-data export",
            "can mold test results be combined",
            "can you tell me tomorrow",
            "weather forecast",
            "can you prove the ventilation",
            "ventilation system caused",
            "ventilation system cause",
            "did the ventilation",
            "cause the aqi spike",
            "cause yesterday's aqi",
            "cause the spike",
            "caused the aqi spike",
            "caused yesterday's aqi",
            // Causation questions — HAWAQM cannot prove what caused a spike/reading
            "what caused the spike",
            "what caused the aqi",
            "what caused the pollution",
            "what caused the reading",
            "what caused the high",
            "why did the aqi spike",
            "why was the aqi high",
            "why is the aqi high",
            "what caused the increase",
            "what caused the rise",
            "reason for the spike",
            "reason for the aqi",
            "nearest ambient station and indoor site have different update",
            "if the nearest ambient station",
            // General AQI information/level questions
            "what is aqi", "tell me about aqi", "about aqi", "aqi information",
            "aqi overview", "aqi basics", "aqi guide", "aqi intro",
            "what is the aqi", "what does aqi stand for", "aqi stands for",
            "what does an aqi", "what does an aqi value", "what does an aqi of",
            "what does aqi of", "aqi of 50", "aqi of 100", "aqi of 150", "aqi of 200", "aqi of 300",
            "what is a good aqi level", "what is a bad aqi level", "what is a safe aqi level",
            "what aqi level is safe", "what aqi level is good", "what aqi level is bad",
            "what aqi level is dangerous", "what aqi level is unhealthy",
            "aqi level meaning", "aqi level information", "aqi levels explained",
            "aqi categories explained", "aqi category meaning", "aqi category information",
            "explain aqi levels", "explain aqi categories", "explain the aqi levels",
            "tell me about aqi levels", "tell me about aqi categories",
            "what level of aqi", "which level of aqi", "what aqi level",
            // "air quality index" variants — user may say "air quality index" instead of "aqi"
            "what is air quality index", "air quality index range", "air quality index level",
            "what air quality index is good", "what air quality index is unhealthy",
            "what air quality index is moderate", "what air quality index is hazardous",
            "what air quality index is very unhealthy",
            "air quality index considered good", "air quality index considered unhealthy",
            "air quality index considered moderate", "air quality index considered hazardous",
            "what is a good air quality index", "what is a bad air quality index",
            "air quality index categories", "air quality index levels", "air quality index scale",
            "air quality index classification", "air quality index meaning",
            "what does air quality index mean", "explain air quality index",
            // "air quality" range/level/category variants
            "what is good air quality", "what is bad air quality", "what is poor air quality",
            "what is unhealthy air quality", "what is moderate air quality", "what is hazardous air quality",
            "what is very unhealthy air quality", "what is safe air quality",
            "good air quality range", "moderate air quality range", "unhealthy air quality range",
            "hazardous air quality range", "very unhealthy air quality range",
            "what range is good air quality", "what range is unhealthy air quality",
            "what range is moderate air quality", "what range is hazardous air quality",
            "air quality range good", "air quality range moderate", "air quality range unhealthy",
            "air quality range hazardous", "air quality range very unhealthy",
            "what is the air quality range for good", "what is the air quality range for moderate",
            "what is the air quality range for unhealthy", "what is the air quality range for hazardous",
            "what is the air quality range for very unhealthy",
            "air quality levels explained", "air quality categories", "air quality classification",
            "air quality scale", "air quality categories explained", "air quality category meaning",
            "what are the air quality levels", "what are the air quality categories",
            "explain air quality levels", "explain air quality categories",
            // "quality of the air" variants
            "quality of the air", "quality of air is considered", "quality of the air is considered",
            "what quality of air is good", "what quality of air is unhealthy",
            "what quality of air is moderate", "what quality of air is hazardous",
            "quality of air considered good", "quality of air considered unhealthy",
            "quality of air considered moderate", "quality of air considered hazardous",
            // "AQI reading" variants
            "what aqi reading is good", "what aqi reading is unhealthy", "what aqi reading is moderate",
            "what aqi reading is hazardous", "what aqi reading is very unhealthy",
            "aqi reading considered good", "aqi reading considered unhealthy",
            "aqi reading considered moderate", "aqi reading considered hazardous",
            "aqi reading for good", "aqi reading for unhealthy", "aqi reading for moderate",
            "aqi reading for hazardous", "aqi reading for very unhealthy",
            "what does an aqi reading of", "aqi reading of 50", "aqi reading of 100",
            "aqi reading of 150", "aqi reading of 200", "aqi reading of 300",
            // IAQ-I definition phrases — always FAQ, never site-listing
            "what is iaq-i", "what does iaq-i stand for", "iaq-i stand for", "iaq-i stands for",
            "iaq-i full form", "full form of iaq-i", "what is indoor air quality index",
            "explain iaq-i", "what does iaq stand for", "iaq-i meaning", "what is iaqi",
            "iaq i stands for", "iaq-i definition", "define iaq-i"
        };
        // Safety/health questions about a NAMED site must NOT go to FAQ — they need live data.
        // Detect if the question is a site-specific safety/quality question by looking for:
        //   (a) safety or quality-judgement words, AND
        //   (b) a site reference — either a known site-type word OR the pattern "at <Name>" (e.g. "at Latifa Mohammed")
        var safetyWords = new[]
        {
            // explicit safety/attendance
            "safe", "safety", "attend", "suitable",
            "can children", "should children", "can people", "should people",
            "good for children", "good for elderly", "good for disabled",
            // quality judgement — must be specific to a SITE, not a region
            // NOTE: "how is the air" alone is too broad — it matches region questions like
            // "How is the air quality in the Abu Dhabi region today?" — use more specific phrases only.
            "good or bad", "is it good", "is it bad", "is the air good", "is the air bad",
            "is the aqi good", "is the aqi bad", "is it healthy", "is it unhealthy",
            "is the air ok", "is it ok to", "is it fine", "is it safe",
            "how is the air quality at", // specific to a named site ("how is the air quality at Al Reyada")
            // NOTE: "how is the air", "how is the quality", "how is the air quality today" removed —
            // they falsely match region-level questions. Keep only the "at <site>" variant above.
            // NOTE: "unhealthy for sensitive groups" is an AQI definition phrase, not a site safety phrase.
            // Only match "unhealthy for" when followed by people-words, not "sensitive groups".
            "unhealthy for children", "unhealthy for elderly", "unhealthy for people",
            "unhealthy for pregnant", "unhealthy for patients", "unhealthy for visitors",
            // AQI category questions about a specific site — "categorize the air quality at X",
            // "what category does the air quality at X fall under", "tell me the air quality at X"
            "categorize the air", "categorise the air", "what category does",
            // NOTE: "fall under" removed — it falsely matches "which AQI values fall under the Unhealthy..."
            "air quality today for"
        };
        // Region-level questions are never site safety questions — guard explicitly.
        var knownRegions = new[] { "abu dhabi", "al ain", "al dhafra", "dhafra" };
        // Site-name indicators — proper nouns / words that only appear in named-site questions.
        // "Al Ain" or "Abu Dhabi" alone with none of these words is a region question.
        var siteNameIndicatorsForRegionGuard = new[]
        {
            "building", "facility", "centre", "center", "hospital", "clinic",
            "office", "park", "mall", "tower", "villa", "compound", "camp",
            "indian", "british", "american", "international", "national",
            "bateen", "saad", "naeem", "naim", "khalifa", "zayed", "hamdan",
            "mushrif", "mussafah", "baniyas", "karama", "shakhbout", "shahama",
            "madinat", "khalidiyah", "corniche", "mangrove", "reem", "yas",
            "wahda", "rowdah", "muroor", "electra", "hameem", "liwa", "gayathi",
            "reyada", "reayada", "kaltham", "obeidli", "gems", "shawamekh", "areej"
        };
        bool containsRegionName = knownRegions.Any(r => lowerMsg.Contains(r));
        bool msgHasSiteNameIndicator = siteNameIndicatorsForRegionGuard.Any(w => lowerMsg.Contains(w));
        // A question is a region question when it names a known region AND either:
        //   (a) explicitly says "region" / "in <region>" — clear region intent, OR
        //   (b) contains no site-name indicators — bare "Al Ain" / "Abu Dhabi" alone is a region, not a site.
        bool isRegionQuestion = containsRegionName
            && (lowerMsg.Contains("region") || lowerMsg.Contains("in abu dhabi") || lowerMsg.Contains("in al ain") || lowerMsg.Contains("in al dhafra") || lowerMsg.Contains("in dhafra")
                || !msgHasSiteNameIndicator);
        var siteTypeWords = new[] { "school", "residential", "commercial", "institutional", "site", "station",
                                     "reyada", "saad", "naeem", "bateen", "khalifa", "kaltham", "gems" };
        // Detect a site name reference using any common preposition or phrasing.
        // Covers: "at X", "in X", "for X", "of X", "about X", "near X", "under X",
        //         "the device X belongs to", "X site", "belongs to X", "located at X", etc.
        // Uses original-case message so capitalised proper nouns are detected reliably.
        bool hasAtSitePattern = System.Text.RegularExpressions.Regex.IsMatch(
            request.Message,
            @"\b(?:at|in|for|of|about|near|under|from|regarding|within|inside)\s+(?:the\s+)?(?:site\s+|school\s+|station\s+|building\s+)?[A-Z][a-zA-Z]{1,}(?:\s+[A-Za-z]{2,}){0,5}")
            || System.Text.RegularExpressions.Regex.IsMatch(
            request.Message,
            @"\bthe\s+[A-Z][a-zA-Z]{1,}(?:\s+[A-Za-z]{2,}){0,4}\s+(?:site|school|station|building|facility)\b",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase)
            || System.Text.RegularExpressions.Regex.IsMatch(
            request.Message,
            @"(?:belongs\s+to|located\s+at|deployed\s+at|installed\s+at)\s+[A-Z][a-zA-Z]{1,}(?:\s+[A-Za-z]{2,}){0,5}",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        // AQI definition questions are NEVER site safety questions — guard before safetyWords check.
        bool isDefinitelyAqiDefinition = aqiRangeDefinitionKeywords.Any(k => lowerMsg.Contains(k));
        // Multi-site list questions ("which sites are outside the safe range") are NOT single-site safety questions.
        // "all sites" in access/visibility questions ("can't see all sites") must not be treated as a data list request.
        bool isAllSitesVisibilityQuestion = isDefinitelyAqiDefinition
            || lowerMsg.Contains("can't see all") || lowerMsg.Contains("cant see all")
            || lowerMsg.Contains("cannot see all") || lowerMsg.Contains("don't see all")
            || lowerMsg.Contains("dont see all") || lowerMsg.Contains("not seeing all")
            || lowerMsg.Contains("i don't see all") || lowerMsg.Contains("i dont see all")
            || lowerMsg.Contains("missing from my view") || lowerMsg.Contains("sites are missing")
            || lowerMsg.Contains("sites not showing") || lowerMsg.Contains("sites not visible");
        bool isMultiSiteListQuestion = lowerMsg.Contains("which sites") || lowerMsg.Contains("what sites")
            || (!isAllSitesVisibilityQuestion && lowerMsg.Contains("all sites")) || lowerMsg.Contains("sites outside")
            || lowerMsg.Contains("sites currently") || lowerMsg.Contains("sites are outside")
            || lowerMsg.Contains("sites exceed") || lowerMsg.Contains("sites with")
            || lowerMsg.Contains("list of sites") || lowerMsg.Contains("show sites")
            // Sector-qualified all-sites queries — "all residential sites", "all commercial sites"
            || lowerMsg.Contains("all residential") || lowerMsg.Contains("all commercial")
            || lowerMsg.Contains("residential sites") || lowerMsg.Contains("commercial sites")
            // "Which residential in [region] have [AQI]?" — sector+AQI category filter questions
            || (lowerMsg.Contains("which residential") && (lowerMsg.Contains("aqi") || lowerMsg.Contains("moderate") || lowerMsg.Contains("good") || lowerMsg.Contains("unhealthy") || lowerMsg.Contains("hazardous")))
            || (lowerMsg.Contains("which commercial") && (lowerMsg.Contains("aqi") || lowerMsg.Contains("moderate") || lowerMsg.Contains("good") || lowerMsg.Contains("unhealthy") || lowerMsg.Contains("hazardous")))
            || (lowerMsg.Contains("residential in") && (lowerMsg.Contains("aqi") || lowerMsg.Contains("moderate") || lowerMsg.Contains("good") || lowerMsg.Contains("unhealthy") || lowerMsg.Contains("hazardous")))
            || (lowerMsg.Contains("commercial in") && (lowerMsg.Contains("aqi") || lowerMsg.Contains("moderate") || lowerMsg.Contains("good") || lowerMsg.Contains("unhealthy") || lowerMsg.Contains("hazardous")))
            // Numeric AQI threshold questions about multiple sites/schools
            || (lowerMsg.Contains("aqi") && (lowerMsg.Contains("above") || lowerMsg.Contains("below")
                || lowerMsg.Contains("more than") || lowerMsg.Contains("less than")
                || lowerMsg.Contains("greater than") || lowerMsg.Contains("higher than")
                || lowerMsg.Contains("lower than") || lowerMsg.Contains("between")
                // Symbol-style: "AQI > 50", "AQI < 120"
                || System.Text.RegularExpressions.Regex.IsMatch(lowerMsg, @"aqi\s*[><]=?\s*\d+")))
            || lowerMsg.Contains("how many schools") || lowerMsg.Contains("how many sites")
            || lowerMsg.Contains("unhealthy for sensitive groups")
            // Relational AQI category phrases
            || (lowerMsg.Contains("worse than") && (lowerMsg.Contains("good") || lowerMsg.Contains("moderate") || lowerMsg.Contains("unhealthy")))
            || (lowerMsg.Contains("better than") && (lowerMsg.Contains("moderate") || lowerMsg.Contains("unhealthy") || lowerMsg.Contains("hazardous")))
            || lowerMsg.Contains("schools where air quality") || lowerMsg.Contains("sites where air quality");
        bool isSiteSafetyQuestion = !isDefinitelyAqiDefinition
            && !isRegionQuestion
            && !isMultiSiteListQuestion
            && safetyWords.Any(s => lowerMsg.Contains(s))
            && (siteTypeWords.Any(s => lowerMsg.Contains(s)) || hasAtSitePattern);

        // Guard: if the user is asking for a LIST of sites filtered by AQI category/range,
        // do NOT force FAQ even if a definition keyword matched. Site-list indicators override.
        var siteListIndicators = new[] {
            "show me sites", "show sites", "list sites", "find sites", "which sites",
            "sites with good", "sites with moderate", "sites with unhealthy", "sites with hazardous",
            "sites with very unhealthy", "sites where aqi", "sites in good", "sites in moderate",
            "sites in unhealthy", "sites in hazardous", "sites in very unhealthy",
            "sites for good", "sites for moderate", "sites for unhealthy", "sites for hazardous",
            "sites currently", "sites that are good", "sites that are moderate",
            "sites that are unhealthy", "sites that are hazardous", "sites that are very unhealthy",
            "good air quality sites", "moderate air quality sites", "hazardous air quality sites",
            "sites in the good", "sites in the moderate", "sites in the unhealthy",
            "sites in the hazardous", "critical sites", "hotspot sites",
            "schools with good", "schools with moderate", "schools with unhealthy", "schools where aqi",
            // Numeric AQI threshold indicators — "schools with AQI above 100", "sites with AQI below 50"
            "sites with aqi above", "sites with aqi below", "sites with aqi more than", "sites with aqi less than",
            "schools with aqi above", "schools with aqi below", "schools with aqi more than",
            "how many schools have aqi", "how many sites have aqi", "how many schools with aqi",
            "how many sites with aqi", "sites where aqi is above", "sites where aqi is below",
            "schools where aqi is above", "schools where aqi is below",
            "aqi above", "aqi below", "aqi more than", "aqi less than", "aqi greater than",
            "aqi higher than", "aqi lower than", "aqi between", "aqi over", "aqi under",
            // Symbol-style AQI filters — "AQI > 50", "AQI < 120", "AQI >= 100"
            "aqi >", "aqi <", "aqi >=", "aqi <=",
            // Sector-prefixed all-sites queries — "all residential sites", "all commercial sites"
            "all residential sites", "all commercial sites", "all school sites", "all active sites",
            "residential sites with aqi", "commercial sites with aqi", "residential sites where",
            // Relational / comparative AQI category phrases
            "worse than moderate", "worse than good", "worse than unhealthy", "worse than hazardous",
            "better than moderate", "better than unhealthy", "better than hazardous",
            "air quality is worse", "air quality worse than", "worse air quality",
            "schools where air quality", "sites where air quality", "all schools where",
            "not good", "not in good", "not safe", "poor air quality",
            // "how many schools/sites" in AQI category — always a data query, never a definition
            "how many schools", "how many sites", "how many moe", "how many government schools",
            "how many public", "how many residential", "how many commercial",
            // "Unhealthy for Sensitive Groups" as a category filter — data query, not definition
            "unhealthy for sensitive groups", "in unhealthy for sensitive", "unhealthy for sensitive",
            "sensitive groups aqi", "sensitive group aqi",
            // "schools in [AQI category]" — data queries
            "schools in good", "schools in moderate", "schools in unhealthy", "schools in hazardous",
            "schools in very unhealthy", "schools in sensitive"
        };
        bool isSiteListAqiQuestion = siteListIndicators.Any(k => lowerMsg.Contains(k));
        if (!isSiteSafetyQuestion && !isSiteListAqiQuestion && aqiRangeDefinitionKeywords.Any(k => lowerMsg.Contains(k)))
        {
            _log.Debug("ChatController: AQI range-definition pre-check fired → forcing faq_answer");
            var faqTemplatePre = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "faq_answer");
            if (faqTemplatePre is not null)
            {
                var faqPathPre = Path.Combine(AppContext.BaseDirectory, "Knowledge", "faq.json");
                var faqContextPre = System.IO.File.Exists(faqPathPre) ? await System.IO.File.ReadAllTextAsync(faqPathPre, ct) : string.Empty;
                var faqAnswerPre = SubstituteFaqPlaceholders(
                    await _azureAI.AnswerFaqAsync(request.Message, faqContextPre, ct),
                    request.Message);
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

        // ── 2b-1b. Hard pre-check: site access/visibility questions → fixed answer ──
        // "Why can't I see all the sites?" is an access question, not a data query.
        // Return the fixed answer directly without hitting the DB or LLM routing.
        var siteVisibilityPhrases = new[] {
            "why can't i see all the sites", "why cant i see all the sites",
            "why can't i see all sites", "why cant i see all sites",
            "why don't i see all the sites", "why dont i see all the sites",
            "why don't i see all sites", "why dont i see all sites",
            "i can't see all sites", "i cant see all sites",
            "i can't see all the sites", "i cant see all the sites",
            "not seeing all sites", "cannot see all sites", "can't see all sites",
            "missing sites from my view", "some sites are missing",
            "sites not showing in my account", "why are some sites not visible",
            "why are some sites missing", "not all sites are visible", "not all sites showing",
            "i don't see all the sites", "i dont see all the sites",
            "why is a site missing", "site not visible in my account",
            "site not in my list", "site not showing in my list"
        };
        // Guard: "why is a site missing from the geographical map" is a geo-coordinates FAQ, not an access question.
        // Let the aqiRangeDefinitionKeywords block handle it with the correct faq.json lookup.
        bool isGeoMapMissingQuestion = lowerMsg.Contains("geographical map") || lowerMsg.Contains("geographical analysis")
            || (lowerMsg.Contains("geographic") && lowerMsg.Contains("missing"))
            || (lowerMsg.Contains("missing") && lowerMsg.Contains("map"))
            || (lowerMsg.Contains("not showing") && lowerMsg.Contains("map"));
        if (!isGeoMapMissingQuestion && siteVisibilityPhrases.Any(p => lowerMsg.Contains(p)))
        {
            _log.Debug("ChatController: site-visibility pre-check fired → returning fixed access answer");
            const string visibilityAnswer = "Your view of HAWAQM depends on the access given to your account. " +
                "Access can be set at a single site, a sector, a region, or the full system for administrators, " +
                "so the data you see is always within your permitted scope. This also means two people can ask " +
                "the same question and get different totals, depending on their access. If you need to see more " +
                "sites or regions, please contact your HAWAQM administrator.";
            var visibilityMsgId = Guid.NewGuid();
            var faqTemplateVis = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "faq_answer");
            await _sessionService.SaveBothMessagesAsync(
                sessionId: session.SessionId,
                userContent: request.Message,
                assistantMessageId: visibilityMsgId,
                assistantContent: visibilityAnswer,
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
                messageId: visibilityMsgId,
                template: faqTemplateVis,
                llmSummary: visibilityAnswer,
                rows: [],
                rowCount: 0,
                executionTimeMs: 0,
                scope: scope,
                modelUsed: null,
                tokenCount: 0));
        }

        // ── 2b-2. Hard pre-check: "last 24 hours of data for <device>" → FAQ ──
        // Questions like "Last 24 hours of data for BA 0001?" are capability/FAQ questions,
        // not data requests. They should always return the FAQ answer about 5-minute records.
        bool hasDeviceNameInMsg = System.Text.RegularExpressions.Regex.IsMatch(lowerMsg, @"\b(ba\s*\d{1,4}|sei\s*100\s*m\s*\d{1,4})\b");
        bool isLast24HoursFaqMsg =
            hasDeviceNameInMsg
            && (lowerMsg.Contains("last 24 hours") || lowerMsg.Contains("24 hours of data")
                || lowerMsg.Contains("last 24h") || lowerMsg.Contains("past 24 hours"))
            && !lowerMsg.Contains("show me") && !lowerMsg.Contains("get me")
            && !lowerMsg.Contains("5 min") && !lowerMsg.Contains("5min") && !lowerMsg.Contains("5-min")
            && !lowerMsg.Contains("1 hour") && !lowerMsg.Contains("1h ") && !lowerMsg.Contains("8 hour") && !lowerMsg.Contains("8h ");
        if (isLast24HoursFaqMsg)
        {
            _log.Debug("ChatController: last-24-hours-for-device pre-check fired → forcing faq_answer");
            var faq24hTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "faq_answer");
            if (faq24hTemplate is not null)
            {
                var faq24hPath = Path.Combine(AppContext.BaseDirectory, "Knowledge", "faq.json");
                var faq24hContext = System.IO.File.Exists(faq24hPath) ? await System.IO.File.ReadAllTextAsync(faq24hPath, ct) : string.Empty;
                var faq24hAnswer = SubstituteFaqPlaceholders(
                    await _azureAI.AnswerFaqAsync(request.Message, faq24hContext, ct),
                    request.Message);
                var faq24hMsgId = Guid.NewGuid();
                await _sessionService.SaveBothMessagesAsync(
                    sessionId: session.SessionId,
                    userContent: request.Message,
                    assistantMessageId: faq24hMsgId,
                    assistantContent: faq24hAnswer,
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
                    messageId: faq24hMsgId,
                    template: faq24hTemplate,
                    llmSummary: faq24hAnswer,
                    rows: [],
                    rowCount: 0,
                    executionTimeMs: 0,
                    scope: scope,
                    modelUsed: null,
                    tokenCount: 0));
            }
        }

        // ── 2b-3. Hard pre-check: "Site Name AQI — YYYY" shorthand → site_yearly_averages ──
        // Handles concise patterns like "Al Falahiyya School AQI — 2019" where a site name
        // is followed by AQI and a 4-digit year, separated by a dash/em-dash.
        var yearlyAqiShorthandMatch = System.Text.RegularExpressions.Regex.Match(
            request.Message,
            @"^(.+?)\s*(?:aqi|air quality|iaq)\s*[-–—]+\s*((19|20)\d{2})\s*$",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        if (yearlyAqiShorthandMatch.Success && !hasDeviceNameInMsg)
        {
            var yearlyShorthandSite = yearlyAqiShorthandMatch.Groups[1].Value.Trim();
            var yearlyShorthandYear = yearlyAqiShorthandMatch.Groups[2].Value;
            var yearlyShorthandTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct))
                .FirstOrDefault(t => t.Id == "site_yearly_averages");
            if (yearlyShorthandTemplate is not null && !string.IsNullOrWhiteSpace(yearlyShorthandSite))
            {
                _log.Information("ChatController: yearly-aqi-shorthand pre-check fired → site_yearly_averages, site='{Site}', year={Year}",
                    yearlyShorthandSite, yearlyShorthandYear);
                var yearlyShorthandParams = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                {
                    ["stationName"] = yearlyShorthandSite,
                    ["year"] = yearlyShorthandYear
                };
                var (yearlyShorthandSql, yearlyShorthandSqlParams) = _rbac.BuildSafeQuery(yearlyShorthandTemplate, yearlyShorthandParams, user, scope);
                var (yearlyShorthandValid, _) = SqlValidator.Validate(yearlyShorthandSql);
                if (yearlyShorthandValid)
                {
                    var yearlyShorthandResult = await _sqlExecutor.ExecuteAsync(yearlyShorthandSql, yearlyShorthandSqlParams, ct);
                    var yearlyShorthandMsgId = Guid.NewGuid();
                    string yearlyShorthandSummary;
                    if (!yearlyShorthandResult.Success || yearlyShorthandResult.Rows.Count == 0)
                    {
                        yearlyShorthandSummary = $"I couldn't find any yearly average data for {yearlyShorthandSite} in {yearlyShorthandYear}. HAWAQM yearly averages are only available from the years the system has been recording data — historical data before the system's deployment is not available.";
                        yearlyShorthandTemplate.ResponseType = "text";
                    }
                    else
                    {
                        yearlyShorthandSummary = $"Here are the yearly averages for {yearlyShorthandSite} in {yearlyShorthandYear}. The table below shows all available parameters.";
                        foreach (var row in yearlyShorthandResult.Rows) row.Remove("StationName");
                    }
                    await _sessionService.SaveBothMessagesAsync(session.SessionId, request.Message, yearlyShorthandMsgId, yearlyShorthandSummary, yearlyShorthandSql, yearlyShorthandTemplate.ResponseType, null, null, yearlyShorthandTemplate.TableUsed, null, null, (int)yearlyShorthandResult.ExecutionTimeMs, 0, ct);
                    if (session.MessageCount <= 1)
                        _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                    return Ok(_formatter.Format(session.SessionId, yearlyShorthandMsgId, yearlyShorthandTemplate, yearlyShorthandSummary, yearlyShorthandResult.Rows, yearlyShorthandResult.Rows.Count, (int)yearlyShorthandResult.ExecutionTimeMs, scope, null, 0));
                }
            }
        }

        // ── 2c-pre. Generic AQI question with no site/region/device → ask for clarification ──
        // "Show me the AQI." / "What is the AQI?" with no site, region, sector, or device specified.
        var genericAqiTriggers = new[] { "show me the aqi", "show the aqi", "what is the aqi", "display aqi", "view aqi", "give me the aqi", "tell me the aqi", "show aqi", "current aqi", "aqi now" };
        var hasRegionWord = new[] { "abu dhabi", "abudhabi", "al ain", "alain", "al dhafra", "aldhafra" }.Any(r => lowerMsg.Contains(r));
        var hasSiteWord = new[] { "school", "site", "station", "building", "facility", "reyada", "saad", "naeem", "bateen", "kaltham", "gems", "residential", "commercial" }.Any(s => lowerMsg.Contains(s));
        var hasSectorWord = new[] { "sector", "commercial", "residential", "public", "govt", "government" }.Any(s => lowerMsg.Contains(s));
        bool hasDeviceName = hasDeviceNameInMsg;
        bool isGenericAqiRequest = genericAqiTriggers.Any(t => lowerMsg.Contains(t)) && !hasRegionWord && !hasSiteWord && !hasSectorWord && !hasDeviceName && !hasAtSitePattern;

        if (isGenericAqiRequest)
        {
            await _sessionService.SaveUserMessageAsync(session.SessionId, request.Message, ct);
            return Ok(_formatter.FormatError(
                session.SessionId,
                "Happy to — which site or region, and for what period? For example: 'current AQI at Al Reyada School' or 'AQI for the Abu Dhabi region today'.",
                "CLARIFY"));
        }

        // ── 2c-pre2. Hard pre-check: region AQI question → force region_aqi_geographical ──
        // "What is the AQI at Al Ain?", "current air quality in Al Dhafra region?" etc.
        // These match region keywords + AQI/air-quality signal with no site-name indicators —
        // the LLM sometimes picks faq_answer for "at <Region>" phrasing instead of the live API template.
        // Guard: sector-filter questions ("which residential aldhafra sites has Moderate AQI?") contain
        // a region name but are asking about sites within a sector — must NOT route to region AQI.
        bool msgHasSectorWord = lowerMsg.Contains("residential") || lowerMsg.Contains("commercial")
            || lowerMsg.Contains("institutional") || lowerMsg.Contains("school") || lowerMsg.Contains("sector");
        bool msgHasExplicitChartRequest = lowerMsg.Contains("pie chart") || lowerMsg.Contains("pie graph") || lowerMsg.Contains("in pie")
            || lowerMsg.Contains("bar chart") || lowerMsg.Contains("bar graph");
        if (isRegionQuestion && (lowerMsg.Contains("aqi") || lowerMsg.Contains("iaq") || lowerMsg.Contains("air quality")) && !msgHasSiteNameIndicator && !msgHasSectorWord && !msgHasExplicitChartRequest)
        {
            _log.Debug("ChatController: region-AQI pre-check fired → forcing region_aqi_geographical");
            var regionAqiTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "region_aqi_geographical");
            if (regionAqiTemplate is not null && regionAqiTemplate.ApiCall is not null)
            {
                var regionAqiBearerToken = Request.Headers.Authorization.ToString().Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
                var regionAqiFill = await _azureAI.FillParametersAsync(request.Message, regionAqiTemplate, history, scope, ct);

                // Ensure regionName is injected if LLM missed it
                if (!regionAqiFill.Parameters.ContainsKey("regionName") || string.IsNullOrWhiteSpace(regionAqiFill.Parameters.GetValueOrDefault("regionName")))
                {
                    if (lowerMsg.Contains("abu dhabi") || lowerMsg.Contains("abudhabi"))
                        regionAqiFill.Parameters["regionName"] = "Abu Dhabi";
                    else if (lowerMsg.Contains("al ain") || lowerMsg.Contains("alain"))
                        regionAqiFill.Parameters["regionName"] = "Al Ain";
                    else if (lowerMsg.Contains("al dhafra") || lowerMsg.Contains("aldhafra") || lowerMsg.Contains("dhafra"))
                        regionAqiFill.Parameters["regionName"] = "Al Dhafra";
                }

                var regionAqiResult = await _externalApi.CallAsync(regionAqiTemplate.ApiCall, regionAqiFill.Parameters, scope, regionAqiBearerToken, user, regionAqiTemplate.Id, ct);
                if (regionAqiResult.Success)
                {
                    var regionAqiMsgId = Guid.NewGuid();
                    var regionAqiSummary = await _azureAI.SummarizeResultsAsync(request.Message, regionAqiTemplate, regionAqiResult.Rows, scope, ct);
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId,
                        userContent: request.Message,
                        assistantMessageId: regionAqiMsgId,
                        assistantContent: regionAqiSummary,
                        sql: null,
                        responseType: regionAqiTemplate.ResponseType,
                        chartType: null,
                        chartDataJson: null,
                        dataSource: regionAqiTemplate.TableUsed,
                        dateRange: null,
                        modelUsed: regionAqiFill.ModelUsed,
                        executionTimeMs: (int)regionAqiResult.ExecutionTimeMs,
                        tokenCount: regionAqiFill.TokensUsed,
                        ct: ct);
                    if (session.MessageCount <= 1)
                        _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                    return Ok(_formatter.Format(
                        sessionId: session.SessionId,
                        messageId: regionAqiMsgId,
                        template: regionAqiTemplate,
                        llmSummary: regionAqiSummary,
                        rows: regionAqiResult.Rows,
                        rowCount: regionAqiResult.Rows.Count,
                        executionTimeMs: (int)regionAqiResult.ExecutionTimeMs,
                        scope: scope,
                        modelUsed: regionAqiFill.ModelUsed,
                        tokenCount: regionAqiFill.TokensUsed));
                }
                _log.Warning("ChatController: region-AQI pre-check API failed, falling through to routing");
            }
        }

        // ── 2c. Hard pre-check: safety/health/quality questions about a named site → force site_aqi_single ──
        // These questions need live AQI data. This block always returns — never falls through to routing.
        if (isSiteSafetyQuestion)
        {
            _log.Debug("ChatController: site safety/quality pre-check fired → forcing site_aqi_single");
            var safetyTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "site_aqi_single");
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

                // API call failed — return user-friendly error, do NOT fall through to routing
                _log.Warning("ChatController: site safety pre-check API call failed for message: {Msg}", request.Message);
                await _sessionService.SaveUserMessageAsync(session.SessionId, request.Message, ct);
                string safetyErrMsg;
                if (!string.IsNullOrWhiteSpace(safetyApiResult.Error)
                    && safetyApiResult.Error.Contains("do not have access", StringComparison.OrdinalIgnoreCase))
                {
                    var askedSiteName = safetyParamResult.Parameters.GetValueOrDefault("stationName")
                                     ?? safetyParamResult.Parameters.GetValueOrDefault("siteName")
                                     ?? safetyParamResult.Parameters.GetValueOrDefault("station")
                                     ?? "That site";
                    safetyErrMsg = BuildAccessDeniedMessage(askedSiteName, user, scope);
                }
                else if (!string.IsNullOrWhiteSpace(safetyApiResult.Error)
                    && safetyApiResult.Error.Contains("doesn't match any known site", StringComparison.OrdinalIgnoreCase))
                {
                    var askedSiteName = safetyParamResult.Parameters.GetValueOrDefault("stationName")
                                     ?? safetyParamResult.Parameters.GetValueOrDefault("siteName")
                                     ?? safetyParamResult.Parameters.GetValueOrDefault("station");
                    safetyErrMsg = string.IsNullOrWhiteSpace(askedSiteName)
                        ? "There is no site with that name in the system. Please check the name and try again. You can ask \"Show me all sites\" to see the full list."
                        : $"There is no site called \"{askedSiteName}\" in the system. Please check the name and try again. You can ask \"Show me all sites\" to see the full list.";
                }
                else
                {
                    safetyErrMsg = "I wasn't able to retrieve current air quality data for that site. The device may be offline. Please try again shortly.";
                }
                return Ok(_formatter.FormatError(session.SessionId, safetyErrMsg, "API_ERROR"));
            }

            // Template not found — return error, do NOT fall through to routing
            _log.Warning("ChatController: site_aqi_single template not found or has no ApiCall for safety pre-check");
            await _sessionService.SaveUserMessageAsync(session.SessionId, request.Message, ct);
            return Ok(_formatter.FormatError(session.SessionId,
                "I wasn't able to retrieve current air quality data for that site. Please try again shortly.",
                "API_ERROR"));
        }

        // ── 2d. Hard pre-check: parameter reading at a named site → force site_aqi_single ──
        // Questions like "What is the latest PM2.5 at AL Saad Indian School?" must never
        // be routed to faq_answer. Detect: pollutant/parameter keyword + site reference (at <Name>).
        var parameterKeywords = new[]
        {
            "pm2.5", "pm 2.5", "pm10", "pm 10", "co2", "carbon dioxide", "carbon monoxide",
            "no2", "nitrogen dioxide", "o3", "ozone", "so2", "sulfur dioxide", "sulphur dioxide",
            "tvoc", "voc", "ch2o", "formaldehyde", "particulate matter", "fine particles", "coarse particles",
            "temperature", "temp", "humidity", "noise", "aqi", "air quality index",
            "air quality", "air-quality", "air quality category", "air quality level",
            "reading", "readings", "level", "levels", "value", "concentration"
        };
        bool hasParameterKeyword = parameterKeywords.Any(p => lowerMsg.Contains(p));
        bool hasAtSiteRef = hasAtSitePattern                          // "at <CapitalName>"
            || siteTypeWords.Any(s => lowerMsg.Contains(s));         // or "school", "site", etc.
        // Exclude distribution/chart questions — "pie chart of schools by AQI category" is NOT a site reading lookup
        var chartDistributionKeywords = new[] { "pie chart", "pie graph", "bar chart", "bar graph", "by aqi category", "by category", "distribution", "breakdown", "chart of schools", "chart of sites", "chart of devices" };
        bool isChartDistribution = chartDistributionKeywords.Any(k => lowerMsg.Contains(k));
        // Detect whether user asked for a bar or pie chart so pre-checks can honour the request
        bool userWantsBar = lowerMsg.Contains("bar chart") || lowerMsg.Contains("bar graph");
        bool userWantsPie = lowerMsg.Contains("pie chart") || lowerMsg.Contains("pie graph") || lowerMsg.Contains("plot a pie") || lowerMsg.Contains("plot pie");
        // Schools AQI pie: only force schools_aqi_pie when the user EXPLICITLY asked for a chart/pie/bar
        // AND the question is about schools/sites broken down BY AQI category.
        // "MOE schools by AQI category region-wise" (no chart word) must NOT trigger this — route to table instead.
        var aqiCategoryWords = new[] { "aqi category", "aqi level", "aqi categories", "aqi levels", "by category", "by aqi" };
        var schoolSiteWords  = new[] { "school", "schools", "sites", "site" };
        var explicitChartWords = new[] { "pie chart", "pie graph", "bar chart", "bar graph", "plot a pie", "plot pie", "donut chart", "doughnut chart", "chart of schools", "chart of sites" };
        bool userAskedForChart = explicitChartWords.Any(k => lowerMsg.Contains(k));
        // Only treat as a schools-specific AQI pie when the user explicitly mentioned "school" or "schools".
        // Generic "sites" queries (e.g. "show me the sites AQI categories in pie chart") must NOT route here.
        bool mentionsSchoolExplicitly = lowerMsg.Contains("school") || lowerMsg.Contains("schools");
        bool isSchoolsAqiPie = userAskedForChart
            && (aqiCategoryWords.Any(k => lowerMsg.Contains(k)) || lowerMsg.Contains("aqi") || lowerMsg.Contains("air quality"))
            && schoolSiteWords.Any(k => lowerMsg.Contains(k))
            && mentionsSchoolExplicitly;
        // Exclude device-count / device-summary questions — "total number of devices per site" is NOT a parameter reading lookup
        var deviceCountKeywords = new[] { "devices per site", "device per site", "devices at each site", "number of devices", "device count", "total devices", "devices in the application", "devices in the system" };
        bool isDeviceCountQuestion = deviceCountKeywords.Any(k => lowerMsg.Contains(k));
        // Detect a specific date reference (YYYY-MM-DD, D/M/YYYY, "on March 20", "on the 15th")
        bool msgHasSpecificDateRef =
            System.Text.RegularExpressions.Regex.IsMatch(lowerMsg,
                @"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}") ||
            System.Text.RegularExpressions.Regex.IsMatch(lowerMsg,
                @"\bon\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}|\bon\s+\d{1,2}(st|nd|rd|th)?\b",
                System.Text.RegularExpressions.RegexOptions.IgnoreCase);

        // Detect a month-only reference (with or without year): "in March", "for January 2025",
        // "last month", "this month", "March 2025", "in September"
        bool msgHasMonthRef =
            System.Text.RegularExpressions.Regex.IsMatch(lowerMsg,
                @"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
                System.Text.RegularExpressions.RegexOptions.IgnoreCase) ||
            lowerMsg.Contains("last month") || lowerMsg.Contains("previous month") ||
            lowerMsg.Contains("this month") || lowerMsg.Contains("current month");
        // A month ref paired with a 4-digit year is still a month question, NOT a year question.
        // msgHasYearRef is defined later; month takes priority when both match.

        // Exclude past-date site questions — "AQI at Al Reyada School yesterday" must route to site_daily_averages, NOT site_aqi_single
        bool msgHasPastDate =
            lowerMsg.Contains("yesterday") || lowerMsg.Contains("last monday") || lowerMsg.Contains("last tuesday") ||
            lowerMsg.Contains("last wednesday") || lowerMsg.Contains("last thursday") || lowerMsg.Contains("last friday") ||
            lowerMsg.Contains("last saturday") || lowerMsg.Contains("last sunday") ||
            lowerMsg.Contains("previous day") || lowerMsg.Contains("day before") || lowerMsg.Contains("last night") ||
            msgHasSpecificDateRef;
        // Exclude sector-filter questions — "all Abu Dhabi sites under Residential sector with their latest CO2"
        // is a sector-wide listing, NOT a single-site parameter lookup. These must route to sector_latest_pollutant.
        var sectorFilterPhrases = new[] {
            "sites under residential", "sites under commercial", "sites under institutional",
            "all sites under", "residential sector", "commercial sector", "institutional sector",
            "residential sites", "commercial sites", "institutional sites",
            "under residential", "under commercial", "under institutional",
            // "which residential / commercial / institutional [in region]" — sector AQI category queries
            "which residential", "which commercial", "which institutional",
            "residential in abu dhabi", "residential in al ain", "residential in al dhafra", "residential in dhafra",
            "commercial in abu dhabi", "commercial in al ain", "commercial in al dhafra", "commercial in dhafra",
            "institutional in abu dhabi", "institutional in al ain", "institutional in al dhafra"
        };
        // No "all"/"show"/"list" required — a bare sector+parameter query like
        // "Abu Dhabi residential latest CO2 readings" is also a sector filter question.
        bool isSectorFilterQuestion = sectorFilterPhrases.Any(p => lowerMsg.Contains(p));
        // Exclude generic school questions — "all Abu Dhabi schools with their latest CO2" is a schools-sector
        // listing, NOT a single-site lookup. Must route to schools_latest_pollutant, not site_aqi_single.
        var genericSchoolPhrases = new[] {
            "all schools", "schools in", "schools with", "abu dhabi schools", "al ain schools",
            "al dhafra schools", "show me schools", "table of schools", "list schools",
            "all abu dhabi schools", "all al ain schools", "all al dhafra schools",
            "schools with their", "schools latest", "school sites", "school co2", "school aqi"
        };
        bool isGenericSchoolMsg = genericSchoolPhrases.Any(p => lowerMsg.Contains(p))
            || (lowerMsg.Contains("schools") && (lowerMsg.Contains("all") || lowerMsg.Contains("table") || lowerMsg.Contains("list") || lowerMsg.Contains("show")));

        // Detect indoor vs ambient compare — must never be treated as a generic school query
        var ambientComparePhrases = new[] {
            "ambient station", "nearest ambient", "nearest outdoor station",
            "indoor vs outdoor", "indoor and outdoor", "compare indoor aqi",
            "indoor aqi with ambient", "indoor aqi vs ambient", "indoor outdoor aqi",
            "compare with ambient", "compare with outdoor", "current indoor aqi",
            "with the nearest ambient", "higher or lower than the nearest ambient",
            "nearest ambient station at", "what is the nearest ambient",
            "which ambient station", "what ambient station", "nearest outdoor station at"
        };
        bool isAmbientCompareMsg = ambientComparePhrases.Any(p => lowerMsg.Contains(p));

        // ── 2d-pre-dsr. DSR signals — detect before routing so LLM can't mis-select a live template ──

        bool hasDsrKeywordQ =
            lowerMsg.Contains("data success rate") || lowerMsg.Contains(" dsr ") ||
            lowerMsg.EndsWith(" dsr") || lowerMsg.StartsWith("dsr ") || lowerMsg == "dsr";

        // "schools that repeatedly exceeded AQI 100 but still had Data Success Rate of at least 75%"
        bool hasDsrExceedanceSignal =
            hasDsrKeywordQ
            && (lowerMsg.Contains("exceeded") || lowerMsg.Contains("exceedance") || lowerMsg.Contains("exceed aqi")
                || lowerMsg.Contains("above aqi") || lowerMsg.Contains("aqi above") || lowerMsg.Contains("crossed aqi")
                || lowerMsg.Contains("went above") || lowerMsg.Contains("aqi exceedance") || lowerMsg.Contains("aqi exceeded"));

        // "show me all school sites with a Data Success Rate below 75%"
        // "which sites have DSR below 80%?" / "schools with DSR above 90%"
        bool hasDsrSiteListSignal =
            hasDsrKeywordQ
            && !hasDsrExceedanceSignal
            && (lowerMsg.Contains("show") || lowerMsg.Contains("list") || lowerMsg.Contains("which") ||
                lowerMsg.Contains("sites with") || lowerMsg.Contains("schools with") || lowerMsg.Contains("sites that") ||
                lowerMsg.Contains("schools that") || lowerMsg.Contains("all sites") || lowerMsg.Contains("all schools") ||
                lowerMsg.Contains("school sites") || lowerMsg.Contains("give me") || lowerMsg.Contains("find"))
            && (lowerMsg.Contains("below") || lowerMsg.Contains("above") || lowerMsg.Contains("less than") ||
                lowerMsg.Contains("more than") || lowerMsg.Contains("greater than") || lowerMsg.Contains("higher than") ||
                lowerMsg.Contains("lower than") || lowerMsg.Contains("under") || lowerMsg.Contains("over") ||
                lowerMsg.Contains("at least") || lowerMsg.Contains("at most") || lowerMsg.Contains("poor") ||
                lowerMsg.Contains("low") || lowerMsg.Contains("high") || lowerMsg.Contains("good") ||
                lowerMsg.Contains("bad") || lowerMsg.Contains("failing") || lowerMsg.Contains("failed"));

        // ── 2d-pre0. Hard pre-check: numeric AQI threshold query → force site_aqi_by_category ──
        // Fires for questions like:
        //   "how many schools have AQI above 100 in Abu Dhabi?"
        //   "sites with AQI below 50 in Al Ain"
        //   "show me schools where AQI is more than 150 in the Commercial sector"
        //   "how many sites have AQI between 100 and 200 in Abu Dhabi?"
        // Must run BEFORE 2d-pre (schools_latest_pollutant) so "schools + AQI above N" doesn't
        // get mis-routed to schools_latest_pollutant which doesn't do numeric AQI filtering.
        var aqiThresholdRegex = new System.Text.RegularExpressions.Regex(
            @"aqi\s*(?:is\s+|are\s+)?(?:above|over|more\s+than|greater\s+than|higher\s+than|below|under|less\s+than|lower\s+than|equal\s+to|equals?|=|between|>|>=|<|<=)\s*\d+",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        // Also match symbol notation: "AQI > 50", "AQI >= 50", "AQI < 120", "AQI <= 120"
        var aqiSymbolRegex = new System.Text.RegularExpressions.Regex(
            @"aqi\s*(?:>=?|<=?|>|<)\s*\d+",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        var numericAqiThreshold = aqiThresholdRegex.IsMatch(lowerMsg)
            || aqiSymbolRegex.IsMatch(lowerMsg)
            || (System.Text.RegularExpressions.Regex.IsMatch(lowerMsg, @"\b(above|below|more than|less than|greater than|higher than|lower than|over|under)\s+\d+")
                && lowerMsg.Contains("aqi"));
        bool isBetweenQuery = System.Text.RegularExpressions.Regex.IsMatch(lowerMsg,
            @"\baqi\s+between\s+\d+\s+and\s+\d+|\bbetween\s+\d+\s+and\s+\d+.*aqi|aqi.*between\s+\d+\s+and\s+\d+",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase)
            // "AQI > 50 and < 120" / "AQI > 50 and AQI < 120"
            || System.Text.RegularExpressions.Regex.IsMatch(lowerMsg,
                @"aqi\s*[><]=?\s*\d+\s*(?:and|&&|,)\s*(?:aqi\s*)?[><]=?\s*\d+",
                System.Text.RegularExpressions.RegexOptions.IgnoreCase);

        if (numericAqiThreshold || isBetweenQuery)
        {
            _log.Debug("ChatController: numeric-AQI-threshold pre-check fired → forcing site_aqi_by_category");
            var aqiThreshTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "site_aqi_by_category");
            if (aqiThreshTemplate is not null && aqiThreshTemplate.ApiCall is not null)
            {
                var aqiThreshBearer = Request.Headers.Authorization.ToString().Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
                var aqiThreshFill = await _azureAI.FillParametersAsync(enrichedMessage, aqiThreshTemplate, history, scope, ct);

                // --- Extract minValue / maxValue from message if LLM didn't ---
                if (!aqiThreshFill.Parameters.ContainsKey("minValue") && !aqiThreshFill.Parameters.ContainsKey("maxValue"))
                {
                    // "between N and M" → minValue=N, maxValue=M
                    var betweenMatch = System.Text.RegularExpressions.Regex.Match(lowerMsg,
                        @"between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)");
                    if (betweenMatch.Success)
                    {
                        aqiThreshFill.Parameters["minValue"] = betweenMatch.Groups[1].Value;
                        aqiThreshFill.Parameters["maxValue"] = betweenMatch.Groups[2].Value;
                    }
                    else
                    {
                        // Symbol compound: "AQI > 50 and < 120" / "AQI > 50 and AQI < 120"
                        // Capture all symbol comparisons and resolve to min/max
                        var allSymbolMatches = System.Text.RegularExpressions.Regex.Matches(lowerMsg,
                            @"(?:aqi\s*)?(?<op>>=?|<=?)\s*(?<val>\d+(?:\.\d+)?)",
                            System.Text.RegularExpressions.RegexOptions.IgnoreCase);
                        foreach (System.Text.RegularExpressions.Match sm in allSymbolMatches)
                        {
                            var op  = sm.Groups["op"].Value.Trim();
                            var val = sm.Groups["val"].Value.Trim();
                            if (op is ">" or ">=")
                            {
                                if (!aqiThreshFill.Parameters.ContainsKey("minValue"))
                                    aqiThreshFill.Parameters["minValue"] = val;
                            }
                            else if (op is "<" or "<=")
                            {
                                if (!aqiThreshFill.Parameters.ContainsKey("maxValue"))
                                    aqiThreshFill.Parameters["maxValue"] = val;
                            }
                        }

                        // Word-based fallbacks (only if symbol extraction didn't already set them)
                        // "above/over/more than/greater than/higher than N" → minValue=N
                        var aboveMatch = System.Text.RegularExpressions.Regex.Match(lowerMsg,
                            @"(?:above|over|more\s+than|greater\s+than|higher\s+than)\s+(\d+(?:\.\d+)?)");
                        if (aboveMatch.Success && !aqiThreshFill.Parameters.ContainsKey("minValue"))
                            aqiThreshFill.Parameters["minValue"] = aboveMatch.Groups[1].Value;

                        // "below/under/less than/lower than N" → maxValue=N
                        var belowMatch = System.Text.RegularExpressions.Regex.Match(lowerMsg,
                            @"(?:below|under|less\s+than|lower\s+than)\s+(\d+(?:\.\d+)?)");
                        if (belowMatch.Success && !aqiThreshFill.Parameters.ContainsKey("maxValue"))
                            aqiThreshFill.Parameters["maxValue"] = belowMatch.Groups[1].Value;

                        // "equal to N" / "equals N" → set both min and max to N-1 and N for exact match
                        var equalMatch = System.Text.RegularExpressions.Regex.Match(lowerMsg,
                            @"(?:equal\s+to|equals?|=)\s+(\d+(?:\.\d+)?)");
                        if (equalMatch.Success && !aboveMatch.Success && !belowMatch.Success
                            && !aqiThreshFill.Parameters.ContainsKey("minValue")
                            && !aqiThreshFill.Parameters.ContainsKey("maxValue"))
                        {
                            var eqVal = equalMatch.Groups[1].Value;
                            aqiThreshFill.Parameters["minValue"] = (double.Parse(eqVal) - 1).ToString();
                            aqiThreshFill.Parameters["maxValue"] = eqVal;
                        }
                    }
                }

                // --- Force sector from keyword if LLM missed it ---
                if (!aqiThreshFill.Parameters.ContainsKey("sectorName") || string.IsNullOrWhiteSpace(aqiThreshFill.Parameters.GetValueOrDefault("sectorName")))
                {
                    if (lowerMsg.Contains("school") || lowerMsg.Contains("moe") || lowerMsg.Contains("gov"))
                        aqiThreshFill.Parameters["sectorName"] = "Public & Govt-School";
                    else if (lowerMsg.Contains("commercial"))
                        aqiThreshFill.Parameters["sectorName"] = "Commercial";
                    else if (lowerMsg.Contains("residential"))
                        aqiThreshFill.Parameters["sectorName"] = "Residential";
                }

                // --- Force regionName from keywords if LLM missed it ---
                if (!aqiThreshFill.Parameters.ContainsKey("regionName") || string.IsNullOrWhiteSpace(aqiThreshFill.Parameters.GetValueOrDefault("regionName")))
                {
                    if (lowerMsg.Contains("abu dhabi"))
                        aqiThreshFill.Parameters["regionName"] = "Abu Dhabi";
                    else if (lowerMsg.Contains("al ain"))
                        aqiThreshFill.Parameters["regionName"] = "Al Ain";
                    else if (lowerMsg.Contains("al dhafra") || lowerMsg.Contains("dhafra"))
                        aqiThreshFill.Parameters["regionName"] = "Al Dhafra";
                }

                _log.Information("ChatController: numeric-AQI-threshold params — min={Min} max={Max} sector={Sec} region={Reg}",
                    aqiThreshFill.Parameters.GetValueOrDefault("minValue") ?? "(none)",
                    aqiThreshFill.Parameters.GetValueOrDefault("maxValue") ?? "(none)",
                    aqiThreshFill.Parameters.GetValueOrDefault("sectorName") ?? "(none)",
                    aqiThreshFill.Parameters.GetValueOrDefault("regionName") ?? "(none)");

                var aqiThreshResult = await _externalApi.CallAsync(aqiThreshTemplate.ApiCall, aqiThreshFill.Parameters, scope, aqiThreshBearer, user, aqiThreshTemplate.Id, ct);
                if (aqiThreshResult.Success)
                {
                    if (aqiThreshResult.Rows.Count == 0)
                    {
                        var aqiThreshSector  = aqiThreshFill.Parameters.GetValueOrDefault("sectorName") ?? "";
                        var aqiThreshRegion  = aqiThreshFill.Parameters.GetValueOrDefault("regionName") ?? "";
                        var aqiThreshMin     = aqiThreshFill.Parameters.GetValueOrDefault("minValue") ?? "";
                        var aqiThreshMax     = aqiThreshFill.Parameters.GetValueOrDefault("maxValue") ?? "";
                        var aqiThreshScope   = (!string.IsNullOrWhiteSpace(aqiThreshSector) && !string.IsNullOrWhiteSpace(aqiThreshRegion))
                            ? $"{aqiThreshSector} sites in {aqiThreshRegion}"
                            : !string.IsNullOrWhiteSpace(aqiThreshSector) ? $"{aqiThreshSector} sites"
                            : !string.IsNullOrWhiteSpace(aqiThreshRegion) ? $"sites in {aqiThreshRegion}"
                            : "sites";
                        var aqiThreshRange   = !string.IsNullOrWhiteSpace(aqiThreshMin) && !string.IsNullOrWhiteSpace(aqiThreshMax)
                            ? $"between {aqiThreshMin} and {aqiThreshMax}"
                            : !string.IsNullOrWhiteSpace(aqiThreshMin) ? $"above {aqiThreshMin}"
                            : !string.IsNullOrWhiteSpace(aqiThreshMax) ? $"below {aqiThreshMax}"
                            : "in that range";
                        var aqiThreshNoDataMsg = $"No {aqiThreshScope} were found with an AQI {aqiThreshRange} at this time.";
                        var aqiThreshNoDataId = Guid.NewGuid();
                        await _sessionService.SaveBothMessagesAsync(session.SessionId, request.Message, aqiThreshNoDataId, aqiThreshNoDataMsg, null, "text", null, null, aqiThreshTemplate.TableUsed, null, aqiThreshFill.ModelUsed, (int)aqiThreshResult.ExecutionTimeMs, aqiThreshFill.TokensUsed, ct);
                        var savedType = aqiThreshTemplate.ResponseType;
                        aqiThreshTemplate.ResponseType = "text";
                        var noDataResp = _formatter.Format(session.SessionId, aqiThreshNoDataId, aqiThreshTemplate, aqiThreshNoDataMsg, [], 0, (int)aqiThreshResult.ExecutionTimeMs, scope, aqiThreshFill.ModelUsed, aqiThreshFill.TokensUsed);
                        aqiThreshTemplate.ResponseType = savedType;
                        return Ok(noDataResp);
                    }
                    var aqiThreshSummary = await _azureAI.SummarizeResultsAsync(
                        request.Message, aqiThreshTemplate, aqiThreshResult.Rows, scope, ct);
                    var aqiThreshMsgId = Guid.NewGuid();
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId,
                        userContent: request.Message,
                        assistantMessageId: aqiThreshMsgId,
                        assistantContent: aqiThreshSummary,
                        sql: null,
                        responseType: aqiThreshTemplate.ResponseType,
                        chartType: null,
                        chartDataJson: SerializeRowsForHistory(aqiThreshTemplate.ResponseType, aqiThreshResult.Rows),
                        dataSource: aqiThreshTemplate.TableUsed,
                        dateRange: null,
                        modelUsed: aqiThreshFill.ModelUsed,
                        executionTimeMs: (int)aqiThreshResult.ExecutionTimeMs,
                        tokenCount: aqiThreshFill.TokensUsed,
                        ct: ct);
                    return Ok(_formatter.Format(session.SessionId, aqiThreshMsgId, aqiThreshTemplate,
                        aqiThreshSummary, aqiThreshResult.Rows, aqiThreshResult.Rows.Count,
                        (int)aqiThreshResult.ExecutionTimeMs, scope, aqiThreshFill.ModelUsed, aqiThreshFill.TokensUsed));
                }
                _log.Warning("ChatController: numeric-AQI-threshold pre-check API failed, falling through to routing");
            }
        }

        // ── 2d-pre-mostpolluted. Single-site "worst / highest / most polluted" → highest_aqi_today ──
        // "most polluted", "worst AQI", "highest AQI" all mean THE ONE peak site right now.
        // List indicators ("list", "all", "show me", "schools") override to isWorstAirPhrase below.
        var singleSiteWorstPhrases = new[]
        {
            // "most polluted" variants
            "most polluted site", "most polluted aqi", "most polluted right now", "most polluted currently",
            "most polluted today", "which site is most polluted", "which site is the most polluted",
            "most polluted site name", "most polluted site today", "most polluted location",
            "most polluted area", "most polluted aqi site", "most polluted name",
            "what is the most polluted", "which is the most polluted",
            "name of most polluted", "name of the most polluted",
            // "worst polluted" variants
            "worst polluted site", "worst polluted location", "worst polluted area",
            "what is the worst polluted", "which is the worst polluted",
            // "highest polluted" variants
            "highest polluted site", "highest polluted location", "highest polluted area",
            "highest polluted", "what is the highest polluted", "which is the highest polluted",
            "name of highest polluted",
            // "largest polluted" / "largest pollution" variants
            "largest polluted site", "largest polluted", "largest pollution site", "largest pollution",
            // "worst AQI" variants
            "worst aqi site", "worst aqi today", "worst aqi right now", "worst aqi currently",
            "site with the worst aqi", "site with worst aqi",
            "which site has the worst aqi", "which site has worst aqi",
            // "highest AQI" variants
            "highest aqi site", "highest aqi today", "highest aqi right now", "highest aqi currently",
            "highest aqi site name", "highest aqi name",
            "site with the highest aqi", "site with highest aqi",
            "which site has the highest aqi", "which site has highest aqi",
            // "highest pollution" variants
            "highest pollution site", "highest pollution",
            // "most hazardous" variants
            "most hazardous site", "most hazardous site today", "most hazardous right now",
            // "most pollution" / bare variants
            "most pollution", "which site is polluted",
        };
        var listOverrideWords = new[] { "list", "all", "show me", "schools", "how many", "which sites", "what sites" };
        bool isMostPollutedSingleSiteQ =
            singleSiteWorstPhrases.Any(p => lowerMsg.Contains(p))
            && !listOverrideWords.Any(w => lowerMsg.Contains(w));
        if (isMostPollutedSingleSiteQ)
        {
            _log.Debug("ChatController: most-polluted-single-site pre-check fired → forcing highest_aqi_today");
            var peakTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "highest_aqi_today");
            if (peakTemplate is not null && !string.IsNullOrWhiteSpace(peakTemplate.Sql))
            {
                // highest_aqi_today is SQL-only (no ApiCall) — execute via _sqlExecutor directly.
                var peakSqlResult = await _sqlExecutor.ExecuteAsync(
                    peakTemplate.Sql,
                    new Dictionary<string, object> { ["now"] = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss") },
                    ct);
                if (peakSqlResult.Success)
                {
                    var peakMsgId = Guid.NewGuid();
                    var peakSummary = await _azureAI.SummarizeResultsAsync(request.Message, peakTemplate, peakSqlResult.Rows, scope, ct);
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId, userContent: request.Message,
                        assistantMessageId: peakMsgId, assistantContent: peakSummary,
                        sql: peakTemplate.Sql, responseType: peakTemplate.ResponseType,
                        chartType: null, chartDataJson: SerializeRowsForHistory(peakTemplate.ResponseType, peakSqlResult.Rows),
                        dataSource: peakTemplate.TableUsed, dateRange: null, modelUsed: null,
                        executionTimeMs: (int)peakSqlResult.ExecutionTimeMs, tokenCount: 0, ct: ct);
                    if (session.MessageCount <= 1)
                        _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                    return Ok(_formatter.Format(session.SessionId, peakMsgId, peakTemplate,
                        peakSummary, peakSqlResult.Rows, peakSqlResult.Rows.Count,
                        (int)peakSqlResult.ExecutionTimeMs, scope, null, 0));
                }
                _log.Warning("ChatController: most-polluted-single-site SQL failed: {Err}, falling through", peakSqlResult.Error);
            }
            else if (peakTemplate is not null && peakTemplate.ApiCall is not null)
            {
                var peakBearer = Request.Headers.Authorization.ToString().Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
                var peakFill = await _azureAI.FillParametersAsync(enrichedMessage, peakTemplate, history, scope, ct);
                var peakResult = await _externalApi.CallAsync(peakTemplate.ApiCall, peakFill.Parameters, scope, peakBearer, user, peakTemplate.Id, ct);
                if (peakResult.Success)
                {
                    var peakSummary = await _azureAI.SummarizeResultsAsync(request.Message, peakTemplate, peakResult.Rows, scope, ct);
                    var peakMsgId = Guid.NewGuid();
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId, userContent: request.Message,
                        assistantMessageId: peakMsgId, assistantContent: peakSummary,
                        sql: null, responseType: peakTemplate.ResponseType,
                        chartType: null, chartDataJson: SerializeRowsForHistory(peakTemplate.ResponseType, peakResult.Rows),
                        dataSource: peakTemplate.TableUsed, dateRange: null, modelUsed: peakFill.ModelUsed,
                        executionTimeMs: (int)peakResult.ExecutionTimeMs, tokenCount: peakFill.TokensUsed, ct: ct);
                    if (session.MessageCount <= 1)
                        _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                    return Ok(_formatter.Format(session.SessionId, peakMsgId, peakTemplate,
                        peakSummary, peakResult.Rows, peakResult.Rows.Count,
                        (int)peakResult.ExecutionTimeMs, scope, peakFill.ModelUsed, peakFill.TokensUsed));
                }
                _log.Warning("ChatController: most-polluted-single-site pre-check API failed, falling through");
            }
        }

        // ── 2d-pre-worstair. Hard pre-check: "worst air quality" → site_aqi_by_category (no category filter, sorted desc) ──
        // "list of Abu Dhabi schools with worst air quality" must route to site_aqi_by_category
        // with NO category filter — returns all sites sorted by AQI descending so the highest AQI
        // sites appear first. Without this guard the LLM picks schools_latest_pollutant which shows
        // Parameter/Value/Unit columns — not AQI.
        // isWorstAirPhrase handles LIST requests ("show schools with worst air quality").
        // Single-site peak questions are already handled by isMostPollutedSingleSiteQ above.
        bool isWorstAirPhrase = !isMostPollutedSingleSiteQ
            && (lowerMsg.Contains("worst air") || lowerMsg.Contains("most polluted")
            || lowerMsg.Contains("highest aqi") || lowerMsg.Contains("highest pollution")
            || lowerMsg.Contains("highest polluted") || lowerMsg.Contains("most hazardous"));
        if (isWorstAirPhrase)
        {
            _log.Debug("ChatController: worst-air pre-check fired → forcing site_aqi_by_category(no filter, desc)");
            var worstTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "site_aqi_by_category");
            if (worstTemplate is not null && worstTemplate.ApiCall is not null)
            {
                var worstBearer = Request.Headers.Authorization.ToString().Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
                var worstFill = await _azureAI.FillParametersAsync(enrichedMessage, worstTemplate, history, scope, ct);

                // "Worst" = Hazardous range (AQI > 300), per project AQI legend.
                worstFill.Parameters.Remove("aqiCategory");
                worstFill.Parameters["minValue"] = "300";
                worstFill.Parameters.Remove("maxValue");

                // Inject sector from keywords if LLM missed it
                if (!worstFill.Parameters.ContainsKey("sectorName") || string.IsNullOrWhiteSpace(worstFill.Parameters.GetValueOrDefault("sectorName")))
                {
                    if (lowerMsg.Contains("school") || lowerMsg.Contains("moe") || lowerMsg.Contains("gov"))
                        worstFill.Parameters["sectorName"] = "Public & Govt-School";
                    else if (lowerMsg.Contains("commercial")) worstFill.Parameters["sectorName"] = "Commercial";
                    else if (lowerMsg.Contains("residential")) worstFill.Parameters["sectorName"] = "Residential";
                }

                // Inject region from keywords if LLM missed it
                if (!worstFill.Parameters.ContainsKey("regionName") || string.IsNullOrWhiteSpace(worstFill.Parameters.GetValueOrDefault("regionName")))
                {
                    if (lowerMsg.Contains("abu dhabi"))      worstFill.Parameters["regionName"] = "Abu Dhabi";
                    else if (lowerMsg.Contains("al ain"))    worstFill.Parameters["regionName"] = "Al Ain";
                    else if (lowerMsg.Contains("al dhafra") || lowerMsg.Contains("dhafra"))
                        worstFill.Parameters["regionName"] = "Al Dhafra";
                }

                _log.Information("ChatController: worst-air params — no category filter, sector={Sec} region={Reg}",
                    worstFill.Parameters.GetValueOrDefault("sectorName") ?? "(none)",
                    worstFill.Parameters.GetValueOrDefault("regionName") ?? "(none)");

                var worstResult = await _externalApi.CallAsync(worstTemplate.ApiCall, worstFill.Parameters, scope, worstBearer, user, worstTemplate.Id, ct);
                if (worstResult.Success)
                {
                    // Reverse to worst-first (BuildAqiCategoryRows returns ascending by AQI)
                    worstResult.Rows.Reverse();

                    // Build summary directly — do not use LLM summarizer which would just describe
                    // the top AQI category without explaining this is ranked worst-to-best.
                    string worstSiteLabel = lowerMsg.Contains("school") || lowerMsg.Contains("moe") || lowerMsg.Contains("gov")
                        ? "school site(s)" : lowerMsg.Contains("commercial") ? "commercial site(s)"
                        : lowerMsg.Contains("residential") ? "residential site(s)" : "site(s)";
                    string worstRegion = lowerMsg.Contains("abu dhabi") ? " in Abu Dhabi"
                        : lowerMsg.Contains("al ain") ? " in Al Ain"
                        : lowerMsg.Contains("al dhafra") || lowerMsg.Contains("dhafra") ? " in Al Dhafra" : "";
                    var worstSummary = worstResult.Rows.Count == 0
                        ? $"No {worstSiteLabel}{worstRegion} currently have Hazardous air quality (AQI above 300)."
                        : $"Here are {worstResult.Rows.Count} {worstSiteLabel}{worstRegion} with Hazardous air quality (AQI above 300), ranked from worst to best.";
                    var worstMsgId = Guid.NewGuid();
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId, userContent: request.Message,
                        assistantMessageId: worstMsgId, assistantContent: worstSummary,
                        sql: null, responseType: worstTemplate.ResponseType,
                        chartType: null, chartDataJson: SerializeRowsForHistory(worstTemplate.ResponseType, worstResult.Rows),
                        dataSource: worstTemplate.TableUsed, dateRange: null, modelUsed: worstFill.ModelUsed,
                        executionTimeMs: (int)worstResult.ExecutionTimeMs, tokenCount: worstFill.TokensUsed, ct: ct);
                    return Ok(_formatter.Format(session.SessionId, worstMsgId, worstTemplate,
                        worstSummary, worstResult.Rows, worstResult.Rows.Count,
                        (int)worstResult.ExecutionTimeMs, scope, worstFill.ModelUsed, worstFill.TokensUsed));
                }
                _log.Warning("ChatController: worst-air pre-check API failed, falling through to routing");
            }
        }

        // ── 2d-pre-badair. Hard pre-check: "bad air" / "poor air quality" / "unhealthy air" phrases ──
        // "How many schools in Abu Dhabi have bad air right now?" must route to site_aqi_by_category
        // with minValue=150 (AQI > 150 = Unhealthy, per project definition). Without this guard the
        // LLM router picks list_stations (site_list_with_status) which has no AQI column.
        bool isBadAirPhrase = lowerMsg.Contains("bad air") || lowerMsg.Contains("poor air quality")
            || lowerMsg.Contains("unhealthy air");
        bool isBadAirCriticalNow = new[] { "hotspot", "critical site", "critical air quality",
            "any critical", "total critical", "number of critical", "how many critical",
            "how many hotspot", "classified as critical", "currently critical",
            "critical sites", "critical locations", "sites are critical" }.Any(p => lowerMsg.Contains(p));
        if (isBadAirPhrase && !isBadAirCriticalNow)
        {
            _log.Debug("ChatController: bad-air pre-check fired → forcing site_aqi_by_category(minValue=150)");
            var badAirTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "site_aqi_by_category");
            if (badAirTemplate is not null && badAirTemplate.ApiCall is not null)
            {
                var badAirBearer = Request.Headers.Authorization.ToString().Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
                var badAirFill = await _azureAI.FillParametersAsync(enrichedMessage, badAirTemplate, history, scope, ct);

                // Force minValue=150 (AQI > 150 by project definition of "bad air")
                badAirFill.Parameters["minValue"] = "150";
                badAirFill.Parameters.Remove("aqiCategory");

                // Inject sector from keywords if LLM missed it
                if (!badAirFill.Parameters.ContainsKey("sectorName") || string.IsNullOrWhiteSpace(badAirFill.Parameters.GetValueOrDefault("sectorName")))
                {
                    if (lowerMsg.Contains("school") || lowerMsg.Contains("moe") || lowerMsg.Contains("gov"))
                        badAirFill.Parameters["sectorName"] = "Public & Govt-School";
                    else if (lowerMsg.Contains("commercial")) badAirFill.Parameters["sectorName"] = "Commercial";
                    else if (lowerMsg.Contains("residential")) badAirFill.Parameters["sectorName"] = "Residential";
                }

                // Inject region from keywords if LLM missed it
                if (!badAirFill.Parameters.ContainsKey("regionName") || string.IsNullOrWhiteSpace(badAirFill.Parameters.GetValueOrDefault("regionName")))
                {
                    if (lowerMsg.Contains("abu dhabi"))      badAirFill.Parameters["regionName"] = "Abu Dhabi";
                    else if (lowerMsg.Contains("al ain"))    badAirFill.Parameters["regionName"] = "Al Ain";
                    else if (lowerMsg.Contains("al dhafra") || lowerMsg.Contains("dhafra"))
                        badAirFill.Parameters["regionName"] = "Al Dhafra";
                }

                _log.Information("ChatController: bad-air params — min=150 sector={Sec} region={Reg}",
                    badAirFill.Parameters.GetValueOrDefault("sectorName") ?? "(none)",
                    badAirFill.Parameters.GetValueOrDefault("regionName") ?? "(none)");

                var badAirResult = await _externalApi.CallAsync(badAirTemplate.ApiCall, badAirFill.Parameters, scope, badAirBearer, user, badAirTemplate.Id, ct);
                if (badAirResult.Success)
                {
                    var badAirSummary = await _azureAI.SummarizeResultsAsync(request.Message, badAirTemplate, badAirResult.Rows, scope, ct);
                    var badAirMsgId = Guid.NewGuid();
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId, userContent: request.Message,
                        assistantMessageId: badAirMsgId, assistantContent: badAirSummary,
                        sql: null, responseType: badAirTemplate.ResponseType,
                        chartType: null, chartDataJson: SerializeRowsForHistory(badAirTemplate.ResponseType, badAirResult.Rows),
                        dataSource: badAirTemplate.TableUsed, dateRange: null, modelUsed: badAirFill.ModelUsed,
                        executionTimeMs: (int)badAirResult.ExecutionTimeMs, tokenCount: badAirFill.TokensUsed, ct: ct);
                    return Ok(_formatter.Format(session.SessionId, badAirMsgId, badAirTemplate,
                        badAirSummary, badAirResult.Rows, badAirResult.Rows.Count,
                        (int)badAirResult.ExecutionTimeMs, scope, badAirFill.ModelUsed, badAirFill.TokensUsed));
                }
                _log.Warning("ChatController: bad-air pre-check API failed, falling through to routing");
            }
        }

        // ── 2d-pre-0. Hard pre-check: schools + "main pollutant" → force schools_main_pollutant ──
        // "Show me Abu Dhabi region schools where CO2 is the main pollutant" must route to
        // schools_main_pollutant, which uses 1H exceedance-ratio SQL logic — not schools_latest_pollutant.
        var mainPollutantSchoolPhrases = new[] {
            "main pollutant", "primary pollutant", "is the main", "as the main", "as main pollutant",
            "as primary pollutant", "main indoor", "identified as the main", "main air quality concern"
        };
        // Also catches "which site has PM2.5 as their main pollutant" (no school word)
        bool isGenericSiteMainPollutant = hasParameterKeyword
            && mainPollutantSchoolPhrases.Any(p => lowerMsg.Contains(p))
            && (lowerMsg.Contains("site") || lowerMsg.Contains("school") || lowerMsg.Contains("location") || lowerMsg.Contains("station"));

        // Exclude messages that explicitly mention a non-school sector — "residential schools with O3 as main pollutant"
        // should NOT route to schools_main_pollutant (which is hardcoded to Public & Govt-School sector).
        bool hasNonSchoolSector = lowerMsg.Contains("residential") || lowerMsg.Contains("commercial");
        bool isMainPollutantSchoolQuery = (isGenericSchoolMsg || isGenericSiteMainPollutant) && hasParameterKeyword
            && mainPollutantSchoolPhrases.Any(p => lowerMsg.Contains(p))
            && !hasNonSchoolSector;

        if (isMainPollutantSchoolQuery)
        {
            _log.Debug("ChatController: schools-main-pollutant pre-check fired → forcing schools_main_pollutant");
            var mainPollTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "schools_main_pollutant");
            if (mainPollTemplate is not null && mainPollTemplate.ApiCall is not null)
            {
                var mainPollBearerToken = Request.Headers.Authorization.ToString().Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
                var mainPollFillResult = await _azureAI.FillParametersAsync(request.Message, mainPollTemplate, history, scope, ct);
                var mainPollApiResult = await _externalApi.CallAsync(mainPollTemplate.ApiCall, mainPollFillResult.Parameters, scope, mainPollBearerToken, user, mainPollTemplate.Id, ct);
                if (mainPollApiResult.Success)
                {
                    var mainPollMsgId = Guid.NewGuid();
                    var mainPollSummary = await _azureAI.SummarizeResultsAsync(request.Message, mainPollTemplate, mainPollApiResult.Rows, scope, ct);
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId,
                        userContent: request.Message,
                        assistantMessageId: mainPollMsgId,
                        assistantContent: mainPollSummary,
                        sql: null,
                        responseType: mainPollTemplate.ResponseType,
                        chartType: null,
                        chartDataJson: SerializeRowsForHistory(mainPollTemplate.ResponseType, mainPollApiResult.Rows),
                        dataSource: mainPollTemplate.TableUsed,
                        dateRange: null,
                        modelUsed: null,
                        executionTimeMs: (int)mainPollApiResult.ExecutionTimeMs,
                        tokenCount: 0,
                        ct: ct);
                    if (session.MessageCount <= 1)
                        _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                    return Ok(_formatter.Format(
                        sessionId: session.SessionId,
                        messageId: mainPollMsgId,
                        template: mainPollTemplate,
                        llmSummary: mainPollSummary,
                        rows: mainPollApiResult.Rows,
                        rowCount: mainPollApiResult.Rows.Count,
                        executionTimeMs: (int)mainPollApiResult.ExecutionTimeMs,
                        scope: scope,
                        modelUsed: null,
                        tokenCount: 0));
                }
                _log.Warning("ChatController: schools_main_pollutant pre-check API call failed: {Error}", mainPollApiResult.Error);
            }
            // Fall through to normal routing if template missing or API failed
        }

        // ── 2d-pre. Hard pre-check: generic schools + pollutant → force schools_latest_pollutant ──
        // "Show me a table of all Abu Dhabi schools with their latest CO2 reading" must never
        // resolve to site_aqi_single. Force schools_latest_pollutant directly.
        // EXCEPTION: if the question asks about an AQI quality level (Moderate, Good, Unhealthy, etc.)
        // it must route to site_aqi_by_category — schools_latest_pollutant cannot filter by category.
        var aqiQualityCategoryWords = new[] {
            "moderate", "good", "unhealthy", "hazardous", "very unhealthy",
            "sensitive groups", "critical", "hotspot"
        };
        bool hasAqiQualityCategoryWord = aqiQualityCategoryWords.Any(k => lowerMsg.Contains(k));
        if (isGenericSchoolMsg && hasParameterKeyword && !isChartDistribution && !hasAqiQualityCategoryWord
            && !isMainPollutantSchoolQuery && !isAmbientCompareMsg && !numericAqiThreshold && !isBetweenQuery)
        {
            _log.Debug("ChatController: generic-school-pollutant pre-check fired → forcing schools_latest_pollutant");
            var schoolPollTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "schools_latest_pollutant");
            if (schoolPollTemplate is not null && schoolPollTemplate.ApiCall is not null)
            {
                var schoolBearerToken = Request.Headers.Authorization.ToString().Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
                var schoolFillResult = await _azureAI.FillParametersAsync(request.Message, schoolPollTemplate, history, scope, ct);
                var schoolApiResult = await _externalApi.CallAsync(schoolPollTemplate.ApiCall, schoolFillResult.Parameters, scope, schoolBearerToken, user, schoolPollTemplate.Id, ct);
                if (schoolApiResult.Success)
                {
                    var schoolMsgId = Guid.NewGuid();
                    var schoolSummary = BuildSchoolPollutantSummary(schoolApiResult.Rows, schoolFillResult.Parameters);
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId,
                        userContent: request.Message,
                        assistantMessageId: schoolMsgId,
                        assistantContent: schoolSummary,
                        sql: null,
                        responseType: schoolPollTemplate.ResponseType,
                        chartType: null,
                        chartDataJson: SerializeRowsForHistory(schoolPollTemplate.ResponseType, schoolApiResult.Rows),
                        dataSource: schoolPollTemplate.TableUsed,
                        dateRange: null,
                        modelUsed: null,
                        executionTimeMs: (int)schoolApiResult.ExecutionTimeMs,
                        tokenCount: 0,
                        ct: ct);
                    if (session.MessageCount <= 1)
                        _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                    return Ok(_formatter.Format(
                        sessionId: session.SessionId,
                        messageId: schoolMsgId,
                        template: schoolPollTemplate,
                        llmSummary: schoolSummary,
                        rows: schoolApiResult.Rows,
                        rowCount: schoolApiResult.Rows.Count,
                        executionTimeMs: (int)schoolApiResult.ExecutionTimeMs,
                        scope: scope,
                        modelUsed: null,
                        tokenCount: 0));
                }
                _log.Warning("ChatController: schools_latest_pollutant pre-check API call failed: {Error}", schoolApiResult.Error);
            }
            // Fall through to normal routing if template missing or API failed
        }

        // ── 2d-pre-sector. Hard pre-check: sector (non-school) + pollutant → force sector_latest_pollutant ──
        // Catches queries like "Abu Dhabi residential latest CO2 readings",
        // "residential CO2", "commercial NO2 in Al Ain", "residential sector PM2.5".
        // Must NOT fire when it's a schools question (handled above by schools_latest_pollutant),
        // or an AQI category/range filter question.
        bool isSectorPollutantQuestion = hasParameterKeyword
            && !isGenericSchoolMsg
            && !hasAqiQualityCategoryWord
            && !isChartDistribution
            && !numericAqiThreshold
            && !isBetweenQuery
            && (lowerMsg.Contains("residential") || lowerMsg.Contains("commercial"))
            && !lowerMsg.Contains("aqi >") && !lowerMsg.Contains("aqi <");

        if (isSectorPollutantQuestion)
        {
            _log.Debug("ChatController: sector-pollutant pre-check fired → forcing sector_latest_pollutant");
            var sectorPollTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct))
                .FirstOrDefault(t => t.Id == "sector_latest_pollutant");
            if (sectorPollTemplate is not null && sectorPollTemplate.ApiCall is not null)
            {
                var sectorPollBearer = Request.Headers.Authorization.ToString()
                    .Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
                var sectorPollFill = await _azureAI.FillParametersAsync(enrichedMessage, sectorPollTemplate, history, scope, ct);

                // Force sector from keyword if LLM missed it
                if (!sectorPollFill.Parameters.ContainsKey("sectorName") || string.IsNullOrWhiteSpace(sectorPollFill.Parameters.GetValueOrDefault("sectorName")))
                {
                    if (lowerMsg.Contains("residential"))
                        sectorPollFill.Parameters["sectorName"] = "Residential";
                    else if (lowerMsg.Contains("commercial"))
                        sectorPollFill.Parameters["sectorName"] = "Commercial";
                }

                // Force regionName from keyword if LLM missed it
                if (!sectorPollFill.Parameters.ContainsKey("regionName") || string.IsNullOrWhiteSpace(sectorPollFill.Parameters.GetValueOrDefault("regionName")))
                {
                    if (lowerMsg.Contains("abu dhabi") || lowerMsg.Contains("abudhabi"))
                        sectorPollFill.Parameters["regionName"] = "Abu Dhabi";
                    else if (lowerMsg.Contains("al ain") || lowerMsg.Contains("alain"))
                        sectorPollFill.Parameters["regionName"] = "Al Ain";
                    else if (lowerMsg.Contains("al dhafra") || lowerMsg.Contains("dhafra"))
                        sectorPollFill.Parameters["regionName"] = "Al Dhafra";
                }

                _log.Information("ChatController: sector-pollutant params — param={P} sector={S} region={R}",
                    sectorPollFill.Parameters.GetValueOrDefault("parameterName") ?? "(none)",
                    sectorPollFill.Parameters.GetValueOrDefault("sectorName") ?? "(none)",
                    sectorPollFill.Parameters.GetValueOrDefault("regionName") ?? "(none)");

                var sectorPollResult = await _externalApi.CallAsync(
                    sectorPollTemplate.ApiCall, sectorPollFill.Parameters, scope, sectorPollBearer, user, sectorPollTemplate.Id, ct);
                if (sectorPollResult.Success)
                {
                    var sectorPollMsgId = Guid.NewGuid();
                    var sectorPollSummary = BuildSchoolPollutantSummary(sectorPollResult.Rows, sectorPollFill.Parameters);
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId,
                        userContent: request.Message,
                        assistantMessageId: sectorPollMsgId,
                        assistantContent: sectorPollSummary,
                        sql: null,
                        responseType: sectorPollTemplate.ResponseType,
                        chartType: null,
                        chartDataJson: SerializeRowsForHistory(sectorPollTemplate.ResponseType, sectorPollResult.Rows),
                        dataSource: sectorPollTemplate.TableUsed,
                        dateRange: null,
                        modelUsed: null,
                        executionTimeMs: (int)sectorPollResult.ExecutionTimeMs,
                        tokenCount: 0,
                        ct: ct);
                    if (session.MessageCount <= 1)
                        _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                    return Ok(_formatter.Format(
                        sessionId: session.SessionId,
                        messageId: sectorPollMsgId,
                        template: sectorPollTemplate,
                        llmSummary: sectorPollSummary,
                        rows: sectorPollResult.Rows,
                        rowCount: sectorPollResult.Rows.Count,
                        executionTimeMs: (int)sectorPollResult.ExecutionTimeMs,
                        scope: scope,
                        modelUsed: null,
                        tokenCount: 0));
                }
                _log.Warning("ChatController: sector-pollutant pre-check API call failed: {Error}", sectorPollResult.Error);
            }
            // Fall through to normal routing if template missing or API failed
        }

        bool isAmbientAllStationsQuery = (lowerMsg.Contains("list of outdoor") || lowerMsg.Contains("list outdoor")
            || lowerMsg.Contains("list ambient") || lowerMsg.Contains("list of ambient")
            || lowerMsg.Contains("all outdoor stations") || lowerMsg.Contains("all ambient stations")
            || lowerMsg.Contains("show outdoor stations") || lowerMsg.Contains("show ambient stations")
            || lowerMsg.Contains("outdoor stations list") || lowerMsg.Contains("ambient stations list")
            || lowerMsg.Contains("show me outdoor stations") || lowerMsg.Contains("show me ambient stations")
            || lowerMsg.Contains("what are the outdoor stations") || lowerMsg.Contains("what are the ambient stations")
            || lowerMsg.Contains("which outdoor stations"));

        // Exclude indoor vs ambient comparison queries — "compare the current indoor AQI ... with the nearest ambient station"
        // is an indoor_ambient_compare query, not an ambient_station_readings lookup.
        // Also catches "Al Reyada School AQI vs nearest ambient station" (no explicit "indoor" word).
        // "what is the nearest ambient station to X" / "identify nearest ambient station for X"
        // are identity queries that use indoor_ambient_compare (not ambient_station_readings).
        bool isNearestAmbientIdentityQuery =
            (lowerMsg.Contains("nearest ambient") || lowerMsg.Contains("nearest outdoor station"))
            && (lowerMsg.Contains("what is") || lowerMsg.Contains("which") || lowerMsg.Contains("identify")
                || lowerMsg.Contains("find") || lowerMsg.Contains("tell me") || lowerMsg.Contains("show me")
                || lowerMsg.Contains("nearest ambient station to") || lowerMsg.Contains("nearest ambient station for")
                || lowerMsg.Contains("nearest ambient station at") || lowerMsg.Contains("nearest ambient station near"));

        bool isIndoorAmbientCompareQuery =
            isNearestAmbientIdentityQuery
            || ((lowerMsg.Contains("compare") || lowerMsg.Contains("comparison") || lowerMsg.Contains("vs") || lowerMsg.Contains("versus"))
                && (lowerMsg.Contains("ambient") || lowerMsg.Contains("outdoor"))
                && (lowerMsg.Contains("indoor") || lowerMsg.Contains("nearest ambient") || lowerMsg.Contains("nearest outdoor")
                    || lowerMsg.Contains("ambient station") || lowerMsg.Contains("outdoor station")
                    || (hasAtSiteRef && (lowerMsg.Contains("ambient aqi") || lowerMsg.Contains("outdoor aqi")))));
        bool isAmbientStationQuery = !isAmbientAllStationsQuery
            && !isIndoorAmbientCompareQuery
            && (lowerMsg.Contains("outdoor station") || lowerMsg.Contains("ambient station"))
            && hasAtSiteRef;

        bool msgHasYearRef = System.Text.RegularExpressions.Regex.IsMatch(lowerMsg, @"\bin\s+(19|20)\d{2}\b|\bfor\s+(19|20)\d{2}\b|\b(19|20)\d{2}\b")
            && !System.Text.RegularExpressions.Regex.IsMatch(lowerMsg, @"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}")
            && !msgHasMonthRef;  // "March 2025" is a month question, not a year question

        // Detect "site reading" / "reading of <site>" / "all readings" / "pollutant readings" — user wants the full pollutant table.
        // These must route to site_readings_all, NOT site_aqi_single.
        bool isSiteReadingsQuestion = hasAtSiteRef && !msgHasPastDate
            && (lowerMsg.Contains("site reading") || lowerMsg.Contains("site readings")
                || lowerMsg.Contains("reading of") || lowerMsg.Contains("readings of")
                || lowerMsg.Contains("all readings") || lowerMsg.Contains("all parameters")
                || lowerMsg.Contains("all pollutants") || lowerMsg.Contains("complete readings")
                || lowerMsg.Contains("full readings") || lowerMsg.Contains("show readings")
                || lowerMsg.Contains("show me readings") || lowerMsg.Contains("current readings")
                || lowerMsg.Contains("latest readings") || lowerMsg.Contains("live readings")
                || lowerMsg.Contains("live data") || lowerMsg.Contains("show data")
                || lowerMsg.Contains("pollutant reading") || lowerMsg.Contains("pollutant readings")
                || lowerMsg.Contains("pollutants reading") || lowerMsg.Contains("pollutants readings")
                || lowerMsg.Contains("parameter reading") || lowerMsg.Contains("parameter readings")
                || lowerMsg.Contains("parameters reading") || lowerMsg.Contains("parameters readings"));

        if (isSiteReadingsQuestion)
        {
            _log.Debug("ChatController: site-readings pre-check fired → forcing site_readings_all");
            var readingsTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "site_readings_all");
            if (readingsTemplate is not null && readingsTemplate.ApiCall is not null)
            {
                var readingsBearerToken = Request.Headers.Authorization.ToString().Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
                var readingsFillResult = await _azureAI.FillParametersAsync(request.Message, readingsTemplate, history, scope, ct);
                var readingsApiResult = await _externalApi.CallAsync(readingsTemplate.ApiCall, readingsFillResult.Parameters, scope, readingsBearerToken, user, readingsTemplate.Id, ct);
                if (readingsApiResult.Success)
                {
                    var readingsMsgId = Guid.NewGuid();
                    var readingsSummary = await _azureAI.SummarizeResultsAsync(request.Message, readingsTemplate, readingsApiResult.Rows, scope, ct);
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId,
                        userContent: request.Message,
                        assistantMessageId: readingsMsgId,
                        assistantContent: readingsSummary,
                        sql: null,
                        responseType: readingsTemplate.ResponseType,
                        chartType: null,
                        chartDataJson: SerializeRowsForHistory(readingsTemplate.ResponseType, readingsApiResult.Rows),
                        dataSource: readingsTemplate.TableUsed,
                        dateRange: null,
                        modelUsed: readingsFillResult.ModelUsed,
                        executionTimeMs: (int)readingsApiResult.ExecutionTimeMs,
                        tokenCount: readingsFillResult.TokensUsed,
                        ct: ct);
                    if (session.MessageCount <= 1)
                        _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                    return Ok(_formatter.Format(
                        sessionId: session.SessionId,
                        messageId: readingsMsgId,
                        template: readingsTemplate,
                        llmSummary: readingsSummary,
                        rows: readingsApiResult.Rows,
                        rowCount: readingsApiResult.Rows.Count,
                        executionTimeMs: (int)readingsApiResult.ExecutionTimeMs,
                        scope: scope,
                        modelUsed: readingsFillResult.ModelUsed,
                        tokenCount: readingsFillResult.TokensUsed));
                }
                _log.Warning("ChatController: site-readings pre-check API failed, falling through to routing");
            }
        }

        bool isParameterAtSiteQuestion = hasParameterKeyword && hasAtSiteRef && !isSiteSafetyQuestion && !isRegionQuestion && !isChartDistribution && !isDeviceCountQuestion && !msgHasPastDate && !isSectorFilterQuestion && !isGenericSchoolMsg && !msgHasYearRef && !msgHasMonthRef && !isIndoorAmbientCompareQuery && !isAmbientCompareMsg && !isSiteReadingsQuestion;

        if (isParameterAtSiteQuestion)
        {
            _log.Debug("ChatController: parameter-at-site pre-check fired → forcing site_aqi_single");
            var paramTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "site_aqi_single");
            if (paramTemplate is not null && paramTemplate.ApiCall is not null)
            {
                var paramBearerToken = Request.Headers.Authorization.ToString().Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
                var paramFillResult = await _azureAI.FillParametersAsync(request.Message, paramTemplate, history, scope, ct);
                var paramApiResult = await _externalApi.CallAsync(paramTemplate.ApiCall, paramFillResult.Parameters, scope, paramBearerToken, user, paramTemplate.Id, ct);
                if (paramApiResult.Success)
                {
                    var paramMsgId = Guid.NewGuid();
                    var paramSummary = await _azureAI.SummarizeResultsAsync(request.Message, paramTemplate, paramApiResult.Rows, scope, ct);
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId,
                        userContent: request.Message,
                        assistantMessageId: paramMsgId,
                        assistantContent: paramSummary,
                        sql: null,
                        responseType: "text",
                        chartType: null,
                        chartDataJson: null,
                        dataSource: "API",
                        dateRange: null,
                        modelUsed: paramFillResult.ModelUsed,
                        executionTimeMs: (int)paramApiResult.ExecutionTimeMs,
                        tokenCount: paramFillResult.TokensUsed,
                        ct: ct);
                    if (session.MessageCount <= 1)
                        _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                    return Ok(_formatter.Format(
                        sessionId: session.SessionId,
                        messageId: paramMsgId,
                        template: paramTemplate,
                        llmSummary: paramSummary,
                        rows: paramApiResult.Rows,
                        rowCount: paramApiResult.Rows.Count,
                        executionTimeMs: (int)paramApiResult.ExecutionTimeMs,
                        scope: scope,
                        modelUsed: paramFillResult.ModelUsed,
                        tokenCount: paramFillResult.TokensUsed));
                }
                // API call failed — check for access-denied or unknown-site before falling through
                _log.Warning("ChatController: parameter-at-site pre-check API call failed for message: {Msg}", request.Message);
                if (!string.IsNullOrWhiteSpace(paramApiResult.Error)
                    && paramApiResult.Error.Contains("do not have access", StringComparison.OrdinalIgnoreCase))
                {
                    var paramAskedSite = paramFillResult.Parameters.GetValueOrDefault("stationName")
                                      ?? paramFillResult.Parameters.GetValueOrDefault("siteName")
                                      ?? paramFillResult.Parameters.GetValueOrDefault("station")
                                      ?? "That site";
                    await _sessionService.SaveUserMessageAsync(session.SessionId, request.Message, ct);
                    return Ok(_formatter.FormatError(session.SessionId,
                        BuildAccessDeniedMessage(paramAskedSite, user, scope), "ACCESS_DENIED"));
                }
                else if (!string.IsNullOrWhiteSpace(paramApiResult.Error)
                    && paramApiResult.Error.Contains("doesn't match any known site", StringComparison.OrdinalIgnoreCase))
                {
                    var paramAskedSite = paramFillResult.Parameters.GetValueOrDefault("stationName")
                                      ?? paramFillResult.Parameters.GetValueOrDefault("siteName")
                                      ?? paramFillResult.Parameters.GetValueOrDefault("station");
                    var notFoundMsg = string.IsNullOrWhiteSpace(paramAskedSite)
                        ? "There is no site with that name in the system. Please check the name and try again. You can ask \"Show me all sites\" to see the full list."
                        : $"There is no site called \"{paramAskedSite}\" in the system. Please check the name and try again. You can ask \"Show me all sites\" to see the full list.";
                    await _sessionService.SaveUserMessageAsync(session.SessionId, request.Message, ct);
                    return Ok(_formatter.FormatError(session.SessionId, notFoundMsg, "NOT_FOUND"));
                }
                // Other failure — fall through to normal routing so user still gets a response
            }
        }

        // ── 2e-pre. Hard pre-check: schools-by-AQI-category pie → force schools_aqi_pie ──
        // "Create a pie chart of schools by AQI category" scores 1.0 on schools_latest_pollutant
        // because of the word "schools". Only intercept when BOTH a school/site word AND an AQI
        // category word are present — generic pie charts (by region, sector, count) fall through
        // to normal routing so the LLM can pick the right template.
        if (isSchoolsAqiPie)
        {
            _log.Debug("ChatController: chart-distribution pre-check fired → forcing schools_aqi_pie");
            var pieTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "schools_aqi_pie");
            _log.Debug("ChatController: schools_aqi_pie template={Found} apiCall={HasApiCall} chartType={ChartType}",
                pieTemplate?.Id ?? "(null)", pieTemplate?.ApiCall?.ResponseShape ?? "(null)", pieTemplate?.ChartType ?? "(null)");
            if (pieTemplate is not null && pieTemplate.ApiCall is not null)
            {
                // Honour the user's chart-type preference (bar vs pie)
                var effectiveSchoolChartType = userWantsBar ? "bar" : "pie";
                pieTemplate.ChartType = effectiveSchoolChartType;

                var pieBearerToken = Request.Headers.Authorization.ToString().Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
                var pieFillResult = await _azureAI.FillParametersAsync(request.Message, pieTemplate, history, scope, ct);

                // Force region if LLM missed it
                if (!pieFillResult.Parameters.ContainsKey("regionName") || string.IsNullOrWhiteSpace(pieFillResult.Parameters.GetValueOrDefault("regionName")))
                {
                    if (lowerMsg.Contains("al ain") || lowerMsg.Contains("alain"))
                        pieFillResult.Parameters["regionName"] = "Al Ain";
                    else if (lowerMsg.Contains("abu dhabi") || lowerMsg.Contains("abudhabi"))
                        pieFillResult.Parameters["regionName"] = "Abu Dhabi";
                    else if (lowerMsg.Contains("al dhafra") || lowerMsg.Contains("dhafra"))
                        pieFillResult.Parameters["regionName"] = "Al Dhafra";
                }
                // Force sectorName for school queries
                if (!pieFillResult.Parameters.ContainsKey("sectorName") || string.IsNullOrWhiteSpace(pieFillResult.Parameters.GetValueOrDefault("sectorName")))
                {
                    if (lowerMsg.Contains("school") || lowerMsg.Contains("schools"))
                        pieFillResult.Parameters["sectorName"] = "Public & Govt-School";
                }

                _log.Information("ChatController: schools_aqi_pie params — sector={S} region={R}",
                    pieFillResult.Parameters.GetValueOrDefault("sectorName") ?? "(none)",
                    pieFillResult.Parameters.GetValueOrDefault("regionName") ?? "(none)");

                var pieApiResult = await _externalApi.CallAsync(pieTemplate.ApiCall, pieFillResult.Parameters, scope, pieBearerToken, user, pieTemplate.Id, ct);
                if (pieApiResult.Success && pieApiResult.Rows.Count == 0)
                {
                    // No AQI data available — return a friendly message instead of a blank chart
                    var regionLabel = pieFillResult.Parameters.GetValueOrDefault("regionName");
                    var noDataMsg = string.IsNullOrWhiteSpace(regionLabel)
                        ? "No current AQI data is available for school sites at this time. Sites may not have active devices reporting readings."
                        : $"No current AQI data is available for school sites in {regionLabel}. The sites in this region may not have active devices reporting readings right now.";
                    await _sessionService.SaveUserMessageAsync(session.SessionId, request.Message, ct);
                    return Ok(_formatter.FormatError(session.SessionId, noDataMsg, "NO_DATA"));
                }
                if (pieApiResult.Success)
                {
                    var pieMsgId = Guid.NewGuid();
                    // Prefer DB total (all sector sites) over pie-row sum (only sites with live AQI readings)
                    // so the count matches the actual application site count.
                    var pieTotalSites = pieApiResult.TotalSiteCount
                        ?? pieApiResult.Rows.Sum(r => r.TryGetValue("Count", out var cv) && cv is int ci ? ci : (int.TryParse(cv?.ToString(), out var cp) ? cp : 0));
                    var pieSectorLabel = pieFillResult.Parameters.GetValueOrDefault("sectorName")
                                     ?? (!string.IsNullOrWhiteSpace(scope.Sector) ? scope.Sector : "MOE (Public & Government)");
                    var pieRegionLabel = pieFillResult.Parameters.GetValueOrDefault("regionName");
                    var pieSummary = string.IsNullOrWhiteSpace(pieRegionLabel)
                        ? $"There are a total of {pieTotalSites} {pieSectorLabel} school sites across all AQI categories. The chart below shows the breakdown."
                        : $"There are a total of {pieTotalSites} {pieSectorLabel} school sites in {pieRegionLabel} across all AQI categories. The chart below shows the breakdown.";
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId,
                        userContent: request.Message,
                        assistantMessageId: pieMsgId,
                        assistantContent: pieSummary,
                        sql: null,
                        responseType: pieTemplate.ResponseType,
                        chartType: effectiveSchoolChartType,
                        chartDataJson: SerializeRowsForHistory(pieTemplate.ResponseType, pieApiResult.Rows),
                        dataSource: pieTemplate.TableUsed,
                        dateRange: null,
                        modelUsed: pieFillResult.ModelUsed,
                        executionTimeMs: (int)pieApiResult.ExecutionTimeMs,
                        tokenCount: pieFillResult.TokensUsed,
                        ct: ct);
                    if (session.MessageCount <= 1)
                        _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                    return Ok(_formatter.Format(
                        sessionId: session.SessionId,
                        messageId: pieMsgId,
                        template: pieTemplate,
                        llmSummary: pieSummary,
                        rows: pieApiResult.Rows,
                        rowCount: pieApiResult.Rows.Count,
                        executionTimeMs: (int)pieApiResult.ExecutionTimeMs,
                        scope: scope,
                        modelUsed: pieFillResult.ModelUsed,
                        tokenCount: pieFillResult.TokensUsed));
                }
                _log.Warning("ChatController: schools_aqi_pie API call failed (error='{Err}'), falling through to routing", pieApiResult.Error ?? "(null)");
            }
            else if (pieTemplate is null)
            {
                _log.Warning("ChatController: schools_aqi_pie template not found in permitted templates — falling through to routing");
            }
        }

        // ── 2e-pre2. Hard pre-check: "pie/bar chart of sites by region" → force sites_by_region_chart ──
        // "number of sites in each region" scores 1.0 on count_sites_in_region which requires regionName.
        // Detect explicit chart request about sites grouped by region, with no AQI/category content.
        var chartByRegionKeywords = new[] { "sites in each region", "sites per region", "sites by region", "number of sites in each region", "count of sites per region", "site count by region", "sites across regions", "breakdown of sites by region", "distribution of sites by region", "region bar graph", "region bar chart", "by region bar", "region pie chart", "region pie graph" };
        bool isSitesByRegionChart = isChartDistribution && chartByRegionKeywords.Any(k => lowerMsg.Contains(k));

        if (isSitesByRegionChart)
        {
            var effectiveChartType = userWantsBar ? "bar" : "pie";
            _log.Debug("ChatController: sites-by-region chart pre-check fired → forcing sites_by_region_chart (chartType={Type})", effectiveChartType);
            var regionChartTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "sites_by_region_chart");
            if (regionChartTemplate is not null && regionChartTemplate.ApiCall is not null)
            {
                regionChartTemplate.ChartType = effectiveChartType;
                var regionBearerToken = Request.Headers.Authorization.ToString().Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
                var regionApiResult = await _externalApi.CallAsync(regionChartTemplate.ApiCall, new Dictionary<string, string>(), scope, regionBearerToken, user, regionChartTemplate.Id, ct);
                if (regionApiResult.Success)
                {
                    var regionMsgId = Guid.NewGuid();
                    var regionSummary = await _azureAI.SummarizeResultsAsync(request.Message, regionChartTemplate, regionApiResult.Rows, scope, ct);
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId,
                        userContent: request.Message,
                        assistantMessageId: regionMsgId,
                        assistantContent: regionSummary,
                        sql: null,
                        responseType: regionChartTemplate.ResponseType,
                        chartType: effectiveChartType,
                        chartDataJson: SerializeRowsForHistory(regionChartTemplate.ResponseType, regionApiResult.Rows),
                        dataSource: regionChartTemplate.TableUsed,
                        dateRange: null,
                        modelUsed: null,
                        executionTimeMs: (int)regionApiResult.ExecutionTimeMs,
                        tokenCount: 0,
                        ct: ct);
                    if (session.MessageCount <= 1)
                        _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                    return Ok(_formatter.Format(
                        sessionId: session.SessionId,
                        messageId: regionMsgId,
                        template: regionChartTemplate,
                        llmSummary: regionSummary,
                        rows: regionApiResult.Rows,
                        rowCount: regionApiResult.Rows.Count,
                        executionTimeMs: (int)regionApiResult.ExecutionTimeMs,
                        scope: scope,
                        modelUsed: null,
                        tokenCount: 0));
                }
                _log.Warning("ChatController: sites_by_region_chart API call failed, falling through to routing");
            }
        }

        // ── 2e-pre2z. Hard pre-check: "which/list schools in [region]" → force sites_by_filter ──
        // Questions like "Which public and government schools are in Al Ain?" must never route
        // to schools_latest_pollutant (which requires a pollutant reading). They are pure site-listing
        // questions and must always use sites_by_filter with sectorName="Public & Govt-School".
        var schoolListingKeywords = new[] {
            "which schools", "which public schools", "which government schools",
            "which public and government schools", "which govt schools",
            "list schools", "list the schools", "list public schools", "list government schools",
            "list public and government schools", "list govt schools", "list the government schools",
            "list the public schools", "list the public and government schools",
            "what schools are in", "what public schools are in", "what government schools are in",
            "show me schools in", "show me public schools in", "show me government schools in",
            "show me public and government schools in",
            "schools in al ain", "schools in abu dhabi", "schools in al dhafra",
            "public schools in al ain", "public schools in abu dhabi", "public schools in al dhafra",
            "government schools in al ain", "government schools in abu dhabi", "government schools in al dhafra",
            "public and government schools in", "public and government school sites in",
            "government school sites in", "public school sites in",
            "can you list the public and government school", "public and government schools —"
        };
        bool isSchoolListingQuestion = schoolListingKeywords.Any(k => lowerMsg.Contains(k))
            && !hasParameterKeyword   // not a pollutant-reading question
            && !isChartDistribution;  // not a chart question

        if (isSchoolListingQuestion)
        {
            _log.Debug("ChatController: school-listing pre-check fired → forcing sites_by_filter");
            var schoolListTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "sites_by_filter");
            if (schoolListTemplate is not null)
            {
                var schoolListParams = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                {
                    ["sectorName"] = "Public & Govt-School"
                };
                // Extract region from message
                if (lowerMsg.Contains("al ain"))           schoolListParams["regionName"] = "Al Ain";
                else if (lowerMsg.Contains("abu dhabi"))   schoolListParams["regionName"] = "Abu Dhabi";
                else if (lowerMsg.Contains("al dhafra") || lowerMsg.Contains("dhafra"))
                                                           schoolListParams["regionName"] = "Al Dhafra";
                var (schoolListSql, schoolListSqlParams) = _rbac.BuildSafeQuery(schoolListTemplate, schoolListParams, user, scope);
                var (schoolListValid, _) = SqlValidator.Validate(schoolListSql);
                if (schoolListValid)
                {
                    var schoolListResult = await _sqlExecutor.ExecuteAsync(schoolListSql, schoolListSqlParams, ct);
                    if (schoolListResult.Success)
                    {
                        var schoolListMsgId = Guid.NewGuid();
                        string schoolListSummary;
                        if (schoolListResult.Rows.Count == 0)
                        {
                            var regionLabel = schoolListParams.GetValueOrDefault("regionName");
                            schoolListSummary = string.IsNullOrWhiteSpace(regionLabel)
                                ? "No Public & Government School sites were found."
                                : $"No Public & Government School sites were found in the {regionLabel} region.";
                            schoolListTemplate.ResponseType = "text";
                        }
                        else
                        {
                            var regionLabel = schoolListParams.GetValueOrDefault("regionName");
                            schoolListSummary = string.IsNullOrWhiteSpace(regionLabel)
                                ? $"Found {schoolListResult.Rows.Count} Public & Government School site(s)."
                                : $"Found {schoolListResult.Rows.Count} Public & Government School site(s) in {regionLabel}.";
                        }
                        await _sessionService.SaveBothMessagesAsync(
                            sessionId: session.SessionId,
                            userContent: request.Message,
                            assistantMessageId: schoolListMsgId,
                            assistantContent: schoolListSummary,
                            sql: schoolListSql,
                            responseType: schoolListTemplate.ResponseType,
                            chartType: null,
                            chartDataJson: SerializeRowsForHistory(schoolListTemplate.ResponseType, schoolListResult.Rows),
                            dataSource: schoolListTemplate.TableUsed,
                            dateRange: null,
                            modelUsed: null,
                            executionTimeMs: (int)schoolListResult.ExecutionTimeMs,
                            tokenCount: 0,
                            ct: ct);
                        if (session.MessageCount <= 1)
                            _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                        return Ok(_formatter.Format(
                            sessionId: session.SessionId,
                            messageId: schoolListMsgId,
                            template: schoolListTemplate,
                            llmSummary: schoolListSummary,
                            rows: schoolListResult.Rows,
                            rowCount: schoolListResult.Rows.Count,
                            executionTimeMs: (int)schoolListResult.ExecutionTimeMs,
                            scope: scope,
                            modelUsed: null,
                            tokenCount: 0));
                    }
                }
                _log.Warning("ChatController: school-listing pre-check SQL failed, falling through to routing");
            }
        }

        // ── 2e-pre3. Hard pre-check: "pie/bar chart of sites by sector" → force sites_by_filter_chart ──
        var chartBySectorKeywords = new[] { "sites in each sector", "sites per sector", "sites by sector", "number of sites in each sector", "count of sites per sector", "site count by sector", "breakdown of sites by sector", "distribution of sites by sector", "sector bar graph", "sector bar chart", "by sector bar", "sector pie chart", "sector pie graph" };
        bool isSitesBySectorChart = isChartDistribution && chartBySectorKeywords.Any(k => lowerMsg.Contains(k));

        if (isSitesBySectorChart)
        {
            _log.Debug("ChatController: sites-by-sector chart pre-check fired → forcing sites_by_filter_chart");
            var sectorChartTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "sites_by_filter_chart");
            if (sectorChartTemplate is not null && sectorChartTemplate.ApiCall is not null)
            {
                var sectorBearerToken = Request.Headers.Authorization.ToString().Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
                var sectorApiResult = await _externalApi.CallAsync(sectorChartTemplate.ApiCall, new Dictionary<string, string>(), scope, sectorBearerToken, user, sectorChartTemplate.Id, ct);
                if (sectorApiResult.Success)
                {
                    var sectorMsgId = Guid.NewGuid();
                    var sectorSummary = await _azureAI.SummarizeResultsAsync(request.Message, sectorChartTemplate, sectorApiResult.Rows, scope, ct);
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId,
                        userContent: request.Message,
                        assistantMessageId: sectorMsgId,
                        assistantContent: sectorSummary,
                        sql: null,
                        responseType: sectorChartTemplate.ResponseType,
                        chartType: sectorChartTemplate.ChartType,
                        chartDataJson: SerializeRowsForHistory(sectorChartTemplate.ResponseType, sectorApiResult.Rows),
                        dataSource: sectorChartTemplate.TableUsed,
                        dateRange: null,
                        modelUsed: null,
                        executionTimeMs: (int)sectorApiResult.ExecutionTimeMs,
                        tokenCount: 0,
                        ct: ct);
                    if (session.MessageCount <= 1)
                        _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                    return Ok(_formatter.Format(
                        sessionId: session.SessionId,
                        messageId: sectorMsgId,
                        template: sectorChartTemplate,
                        llmSummary: sectorSummary,
                        rows: sectorApiResult.Rows,
                        rowCount: sectorApiResult.Rows.Count,
                        executionTimeMs: (int)sectorApiResult.ExecutionTimeMs,
                        scope: scope,
                        modelUsed: null,
                        tokenCount: 0));
                }
                _log.Warning("ChatController: sites_by_filter_chart API call failed, falling through to routing");
            }
        }

        // ── 2e-pre3b. Hard pre-check: "sites for each sector" (non-chart) → force site_count ──
        // "Show me sites for each sector" must route to site_count (sector distribution sentence),
        // NOT to sites_by_filter which returns individual site rows.
        var sectorCountKeywords = new[] {
            "sites for each sector", "sites in each sector", "sites per sector", "site count by sector",
            "site count per sector", "sites by sector", "breakdown of sites by sector",
            "show me sites for each sector", "show sites for each sector",
            "distribution of sites by sector", "how many sites in each sector",
            "how many sites per sector", "total sites in each sector",
            "sites registered under each sector", "sites under each sector",
            "number of sites per sector", "number of sites in each sector",
            "count of sites per sector", "count of sites by sector"
        };
        bool isSitesBySectorCount = !isChartDistribution && sectorCountKeywords.Any(k => lowerMsg.Contains(k));

        if (isSitesBySectorCount)
        {
            _log.Debug("ChatController: sites-by-sector count pre-check fired → forcing site_count");
            var sectorCountTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "site_count");
            if (sectorCountTemplate is not null)
            {
                // site_count is a SQL template — use SQL executor with groupBy=sector
                var sectorCountParams = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                {
                    ["groupBy"] = "sector"
                };
                var (sectorCountSql, sectorCountSqlParams) = _rbac.BuildSafeQuery(sectorCountTemplate, sectorCountParams, user, scope);
                var (sectorCountValid, _) = SqlValidator.Validate(sectorCountSql);
                if (sectorCountValid)
                {
                    var sectorCountResult = await _sqlExecutor.ExecuteAsync(sectorCountSql, sectorCountSqlParams, ct);
                    if (sectorCountResult.Success && sectorCountResult.Rows.Count > 0)
                    {
                        // Build sector distribution sentence directly without LLM
                        var sectorTotals = sectorCountResult.Rows
                            .GroupBy(r => r.GetValueOrDefault("SectorName")?.ToString() ?? "")
                            .ToDictionary(g => g.Key, g => g.Sum(r => int.TryParse(r.GetValueOrDefault("TotalSites")?.ToString(), out var n) ? n : 0));
                        int schoolCount      = sectorTotals.FirstOrDefault(kv => kv.Key.Contains("School", StringComparison.OrdinalIgnoreCase)).Value;
                        int residentialCount = sectorTotals.GetValueOrDefault("Residential");
                        int commercialCount  = sectorTotals.GetValueOrDefault("Commercial");
                        int industrialCount  = sectorTotals.FirstOrDefault(kv => kv.Key.Contains("Industrial", StringComparison.OrdinalIgnoreCase)).Value;
                        var sectorCountSummary = $"The current sector distribution is Public & Government School: {schoolCount}, Residential: {residentialCount}, Commercial: {commercialCount}, and Industrial: {industrialCount}.";
                        var sectorCountMsgId = Guid.NewGuid();
                        sectorCountTemplate.ResponseType = "text";
                        await _sessionService.SaveBothMessagesAsync(
                            sessionId: session.SessionId,
                            userContent: request.Message,
                            assistantMessageId: sectorCountMsgId,
                            assistantContent: sectorCountSummary,
                            sql: sectorCountSql,
                            responseType: "text",
                            chartType: null,
                            chartDataJson: null,
                            dataSource: sectorCountTemplate.TableUsed,
                            dateRange: null,
                            modelUsed: null,
                            executionTimeMs: (int)sectorCountResult.ExecutionTimeMs,
                            tokenCount: 0,
                            ct: ct);
                        if (session.MessageCount <= 1)
                            _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                        return Ok(_formatter.Format(
                            sessionId: session.SessionId,
                            messageId: sectorCountMsgId,
                            template: sectorCountTemplate,
                            llmSummary: sectorCountSummary,
                            rows: sectorCountResult.Rows,
                            rowCount: sectorCountResult.Rows.Count,
                            executionTimeMs: (int)sectorCountResult.ExecutionTimeMs,
                            scope: scope,
                            modelUsed: null,
                            tokenCount: 0));
                    }
                }
                _log.Warning("ChatController: site_count sector pre-check SQL failed or returned no rows, falling through to routing");
            }
        }

        // ── 2e-pre4. Hard pre-check: "AQI by region chart" → force aqi_by_region_chart ──
        var aqiByRegionKeywords = new[] {
            "aqi by region", "aqi per region", "average aqi by region", "average aqi per region",
            "aqi chart by region", "aqi bar chart by region", "aqi pie chart by region",
            "bar chart of aqi by region", "pie chart of aqi by region",
            "aqi levels by region", "aqi across regions", "region aqi chart",
            "region aqi bar", "region aqi comparison", "compare aqi across regions",
            "aqi distribution by region"
        };
        bool isAqiByRegionChart = isChartDistribution && aqiByRegionKeywords.Any(k => lowerMsg.Contains(k));

        if (isAqiByRegionChart)
        {
            _log.Debug("ChatController: aqi-by-region chart pre-check fired → forcing aqi_by_region_chart");
            var aqiRegionTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "aqi_by_region_chart");
            if (aqiRegionTemplate is not null && aqiRegionTemplate.ApiCall is not null)
            {
                var effectiveAqiRegionType = userWantsBar ? "bar" : (lowerMsg.Contains("pie") ? "pie" : "bar");
                aqiRegionTemplate.ChartType = effectiveAqiRegionType;
                var aqiRegionBearer = Request.Headers.Authorization.ToString().Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
                var aqiRegionResult = await _externalApi.CallAsync(aqiRegionTemplate.ApiCall, new Dictionary<string, string>(), scope, aqiRegionBearer, user, aqiRegionTemplate.Id, ct);
                if (aqiRegionResult.Success)
                {
                    var aqiRegionMsgId = Guid.NewGuid();
                    var aqiRegionSummary = await _azureAI.SummarizeResultsAsync(request.Message, aqiRegionTemplate, aqiRegionResult.Rows, scope, ct);
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId, userContent: request.Message,
                        assistantMessageId: aqiRegionMsgId, assistantContent: aqiRegionSummary,
                        sql: null, responseType: aqiRegionTemplate.ResponseType,
                        chartType: effectiveAqiRegionType, chartDataJson: SerializeRowsForHistory(aqiRegionTemplate.ResponseType, aqiRegionResult.Rows),
                        dataSource: aqiRegionTemplate.TableUsed, dateRange: null,
                        modelUsed: null, executionTimeMs: (int)aqiRegionResult.ExecutionTimeMs,
                        tokenCount: 0, ct: ct);
                    if (session.MessageCount <= 1)
                        _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                    return Ok(_formatter.Format(
                        sessionId: session.SessionId, messageId: aqiRegionMsgId,
                        template: aqiRegionTemplate, llmSummary: aqiRegionSummary,
                        rows: aqiRegionResult.Rows, rowCount: aqiRegionResult.Rows.Count,
                        executionTimeMs: (int)aqiRegionResult.ExecutionTimeMs,
                        scope: scope, modelUsed: null, tokenCount: 0));
                }
                _log.Warning("ChatController: aqi_by_region_chart API call failed, falling through to routing");
            }
        }

        // ── 2e-pre5. Hard pre-check: "sites by status chart" → force sites_by_status_chart ──
        var siteStatusKeywords = new[] {
            "sites by status", "site status chart", "site status pie", "site status bar",
            "active vs offline", "active vs inactive sites", "online vs offline sites",
            "site availability chart", "breakdown of sites by status",
            "distribution of site status", "active and inactive sites chart"
        };
        bool isSitesByStatusChart = isChartDistribution && siteStatusKeywords.Any(k => lowerMsg.Contains(k));

        if (isSitesByStatusChart)
        {
            _log.Debug("ChatController: sites-by-status chart pre-check fired → forcing sites_by_status_chart");
            var statusTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "sites_by_status_chart");
            if (statusTemplate is not null && statusTemplate.ApiCall is not null)
            {
                var effectiveStatusType = userWantsBar ? "bar" : "pie";
                statusTemplate.ChartType = effectiveStatusType;
                var statusBearer = Request.Headers.Authorization.ToString().Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
                var statusResult = await _externalApi.CallAsync(statusTemplate.ApiCall, new Dictionary<string, string>(), scope, statusBearer, user, statusTemplate.Id, ct);
                if (statusResult.Success)
                {
                    var statusMsgId = Guid.NewGuid();
                    var statusSummary = await _azureAI.SummarizeResultsAsync(request.Message, statusTemplate, statusResult.Rows, scope, ct);
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId, userContent: request.Message,
                        assistantMessageId: statusMsgId, assistantContent: statusSummary,
                        sql: null, responseType: statusTemplate.ResponseType,
                        chartType: effectiveStatusType, chartDataJson: SerializeRowsForHistory(statusTemplate.ResponseType, statusResult.Rows),
                        dataSource: statusTemplate.TableUsed, dateRange: null,
                        modelUsed: null, executionTimeMs: (int)statusResult.ExecutionTimeMs,
                        tokenCount: 0, ct: ct);
                    if (session.MessageCount <= 1)
                        _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                    return Ok(_formatter.Format(
                        sessionId: session.SessionId, messageId: statusMsgId,
                        template: statusTemplate, llmSummary: statusSummary,
                        rows: statusResult.Rows, rowCount: statusResult.Rows.Count,
                        executionTimeMs: (int)statusResult.ExecutionTimeMs,
                        scope: scope, modelUsed: null, tokenCount: 0));
                }
                _log.Warning("ChatController: sites_by_status_chart API call failed, falling through to routing");
            }
        }

        // ── 2e-pre6. Hard pre-check: "pollutant by sector chart" → force pollutant_by_sector_chart ──
        var pollBySectorKeywords = new[] {
            "pollutant by sector", "pm2.5 by sector", "pm10 by sector", "aqi by sector",
            "average pollutant by sector", "average pm2.5 by sector", "average aqi by sector",
            "bar chart of pollutant by sector", "bar chart of aqi by sector",
            "pollutant levels by sector", "pollutant across sectors",
            "sector pollutant comparison", "compare pollutant across sectors",
            "aqi per sector chart", "pm2.5 per sector", "sector aqi bar", "sector pollutant chart",
            "sector aqi chart", "aqi chart by sector", "sector pollutant bar"
        };
        bool isPollutantBySectorChart = isChartDistribution && pollBySectorKeywords.Any(k => lowerMsg.Contains(k));

        if (isPollutantBySectorChart)
        {
            _log.Debug("ChatController: pollutant-by-sector chart pre-check fired → forcing pollutant_by_sector_chart");
            var pollSectorTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "pollutant_by_sector_chart");
            if (pollSectorTemplate is not null && pollSectorTemplate.ApiCall is not null)
            {
                var effectivePollType = lowerMsg.Contains("pie") ? "pie" : "bar";
                pollSectorTemplate.ChartType = effectivePollType;
                var pollBearer = Request.Headers.Authorization.ToString().Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
                var pollFillResult = await _azureAI.FillParametersAsync(request.Message, pollSectorTemplate, history, scope, ct);
                var pollResult = await _externalApi.CallAsync(pollSectorTemplate.ApiCall, pollFillResult.Parameters, scope, pollBearer, user, pollSectorTemplate.Id, ct);
                if (pollResult.Success)
                {
                    var pollMsgId = Guid.NewGuid();
                    var pollSummary = await _azureAI.SummarizeResultsAsync(request.Message, pollSectorTemplate, pollResult.Rows, scope, ct);
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId, userContent: request.Message,
                        assistantMessageId: pollMsgId, assistantContent: pollSummary,
                        sql: null, responseType: pollSectorTemplate.ResponseType,
                        chartType: effectivePollType, chartDataJson: SerializeRowsForHistory(pollSectorTemplate.ResponseType, pollResult.Rows),
                        dataSource: pollSectorTemplate.TableUsed, dateRange: null,
                        modelUsed: pollFillResult.ModelUsed, executionTimeMs: (int)pollResult.ExecutionTimeMs,
                        tokenCount: pollFillResult.TokensUsed, ct: ct);
                    if (session.MessageCount <= 1)
                        _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                    return Ok(_formatter.Format(
                        sessionId: session.SessionId, messageId: pollMsgId,
                        template: pollSectorTemplate, llmSummary: pollSummary,
                        rows: pollResult.Rows, rowCount: pollResult.Rows.Count,
                        executionTimeMs: (int)pollResult.ExecutionTimeMs,
                        scope: scope, modelUsed: pollFillResult.ModelUsed, tokenCount: pollFillResult.TokensUsed));
                }
                _log.Warning("ChatController: pollutant_by_sector_chart API call failed, falling through to routing");
            }
        }

        // ── 2e-pre7. Hard pre-check: "devices by model chart" → force devices_by_model_chart ──
        var deviceModelKeywords = new[] {
            "distribution of devices by model", "devices by model", "device model distribution",
            "device model breakdown", "devices based on the model", "based on the model",
            "how many devices of each model", "device model pie", "device model bar",
            "devices grouped by model", "devices by type", "device type distribution",
            "distribution of device"
        };
        bool isDevicesByModelChart = isChartDistribution && deviceModelKeywords.Any(k => lowerMsg.Contains(k));

        if (isDevicesByModelChart)
        {
            _log.Debug("ChatController: devices-by-model chart pre-check fired → forcing devices_by_model_chart");
            var devModelTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "devices_by_model_chart");
            if (devModelTemplate is not null && !string.IsNullOrWhiteSpace(devModelTemplate.Sql))
            {
                var effectiveDevModelType = userWantsBar ? "bar" : "pie";
                devModelTemplate.ChartType = effectiveDevModelType;
                var devModelSqlResult = await _sqlExecutor.ExecuteAsync(devModelTemplate.Sql, new Dictionary<string, object>(), ct);
                if (devModelSqlResult.Success && devModelSqlResult.Rows.Count > 0)
                {
                    var devModelMsgId = Guid.NewGuid();
                    var devModelSummary = await _azureAI.SummarizeResultsAsync(request.Message, devModelTemplate, devModelSqlResult.Rows, scope, ct);
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId, userContent: request.Message,
                        assistantMessageId: devModelMsgId, assistantContent: devModelSummary,
                        sql: devModelTemplate.Sql, responseType: devModelTemplate.ResponseType,
                        chartType: effectiveDevModelType, chartDataJson: System.Text.Json.JsonSerializer.Serialize(devModelSqlResult.Rows),
                        dataSource: devModelTemplate.TableUsed, dateRange: null,
                        modelUsed: null, executionTimeMs: (int)devModelSqlResult.ExecutionTimeMs,
                        tokenCount: 0, ct: ct);
                    if (session.MessageCount <= 1)
                        _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                    return Ok(_formatter.Format(
                        sessionId: session.SessionId, messageId: devModelMsgId,
                        template: devModelTemplate, llmSummary: devModelSummary,
                        rows: devModelSqlResult.Rows, rowCount: devModelSqlResult.Rows.Count,
                        executionTimeMs: (int)devModelSqlResult.ExecutionTimeMs,
                        scope: scope, modelUsed: null, tokenCount: 0));
                }
                _log.Warning("ChatController: devices_by_model_chart SQL failed, falling through to routing");
            }
        }

        // ── 2e-pre8. Hard pre-check: "sites by subsector chart" → force sites_by_subsector_chart ──
        var subsectorChartKeywords = new[] {
            "school subsector", "subsector chart", "type of school", "private and public school",
            "private vs public school", "school type distribution", "subsectors pie", "subsectors bar",
            "pie chart based on school subsector", "pie chart based on type of school",
            "plot pie chart based on school", "based on school subsector", "based on the type of school",
            "school sub sector", "type of school pie", "type of school bar"
        };
        bool isSubsectorChart = isChartDistribution && subsectorChartKeywords.Any(k => lowerMsg.Contains(k));

        if (isSubsectorChart)
        {
            _log.Debug("ChatController: sites-by-subsector chart pre-check fired → forcing sites_by_subsector_chart");
            var subsectorTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "sites_by_subsector_chart");
            if (subsectorTemplate is not null && !string.IsNullOrWhiteSpace(subsectorTemplate.Sql))
            {
                var effectiveSubType = userWantsBar ? "bar" : "pie";
                subsectorTemplate.ChartType = effectiveSubType;
                var subSqlResult = await _sqlExecutor.ExecuteAsync(subsectorTemplate.Sql, new Dictionary<string, object>(), ct);
                if (subSqlResult.Success && subSqlResult.Rows.Count > 0)
                {
                    var subMsgId = Guid.NewGuid();
                    var subSummary = await _azureAI.SummarizeResultsAsync(request.Message, subsectorTemplate, subSqlResult.Rows, scope, ct);
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId, userContent: request.Message,
                        assistantMessageId: subMsgId, assistantContent: subSummary,
                        sql: subsectorTemplate.Sql, responseType: subsectorTemplate.ResponseType,
                        chartType: effectiveSubType, chartDataJson: System.Text.Json.JsonSerializer.Serialize(subSqlResult.Rows),
                        dataSource: subsectorTemplate.TableUsed, dateRange: null,
                        modelUsed: null, executionTimeMs: (int)subSqlResult.ExecutionTimeMs,
                        tokenCount: 0, ct: ct);
                    if (session.MessageCount <= 1)
                        _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                    return Ok(_formatter.Format(
                        sessionId: session.SessionId, messageId: subMsgId,
                        template: subsectorTemplate, llmSummary: subSummary,
                        rows: subSqlResult.Rows, rowCount: subSqlResult.Rows.Count,
                        executionTimeMs: (int)subSqlResult.ExecutionTimeMs,
                        scope: scope, modelUsed: null, tokenCount: 0));
                }
                _log.Warning("ChatController: sites_by_subsector_chart SQL failed, falling through to routing");
            }
        }

        // ── 2e-pre9. Hard pre-check: "sites by AQI level chart" → force sites_aqi_level_chart ──
        var aqiLevelChartKeywords = new[] {
            "sites based on aqi level", "aqi level distribution", "sites by aqi level",
            "aqi level chart", "aqi level pie", "aqi level bar", "sites aqi level",
            "aqi levels chart", "based on aqi level", "sites based on aqi",
            "sites aqi categories", "aqi categories", "sites aqi category",
            "aqi category pie", "aqi category chart", "aqi category breakdown",
            "show me the sites aqi", "sites aqi",
            "aqi distribution", "sites aqi distribution", "aqi category distribution",
            "sites by aqi", "aqi breakdown"
        };
        // Also catch "pie chart aqi ..." and "... aqi in pie chart" patterns (must have explicit chart word to avoid grabbing plain AQI queries)
        bool hasPieOrBarWord = lowerMsg.Contains("pie chart") || lowerMsg.Contains("pie graph") || lowerMsg.Contains("in pie") || lowerMsg.Contains("bar chart") || lowerMsg.Contains("bar graph");
        bool isAqiLevelChart = aqiLevelChartKeywords.Any(k => lowerMsg.Contains(k))
            || (hasPieOrBarWord && lowerMsg.Contains("aqi") && !lowerMsg.Contains("school") && !lowerMsg.Contains("moe"));

        if (isAqiLevelChart)
        {
            _log.Debug("ChatController: sites-by-aqi-level chart pre-check fired → forcing sites_aqi_level_chart");
            var aqiLvlTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "sites_aqi_level_chart");
            if (aqiLvlTemplate is not null && aqiLvlTemplate.ApiCall is not null)
            {
                var effectiveAqiLvlType = userWantsBar ? "bar" : "pie";
                aqiLvlTemplate.ChartType = effectiveAqiLvlType;
                var aqiLvlBearer = Request.Headers.Authorization.ToString().Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
                // Extract sector/region mentioned inline in the message (e.g. "residential sites AQI distribution")
                // so the ExternalApiService filter picks them up even when scope bar is not set.
                var aqiLvlInlineParams = new Dictionary<string, string>();
                string[] aqiLvlSectors = ["Residential", "Industrial", "Commercial", "Public & Govt-School"];
                string[] aqiLvlRegions = ["Abu Dhabi", "Al Ain", "Al Dhafra"];
                var aqiLvlInlineSector = aqiLvlSectors.FirstOrDefault(s => lowerMsg.Contains(s.ToLowerInvariant()));
                // "school" / "schools" in the message → Public & Govt-School sector
                if (aqiLvlInlineSector is null && (lowerMsg.Contains("school") || lowerMsg.Contains("schools")))
                    aqiLvlInlineSector = "Public & Govt-School";
                var aqiLvlInlineRegion = aqiLvlRegions.FirstOrDefault(r => lowerMsg.Contains(r.ToLowerInvariant()));
                if (aqiLvlInlineSector is not null) aqiLvlInlineParams["sectorName"] = aqiLvlInlineSector;
                if (aqiLvlInlineRegion is not null) aqiLvlInlineParams["regionName"] = aqiLvlInlineRegion;
                var aqiLvlResult = await _externalApi.CallAsync(aqiLvlTemplate.ApiCall, aqiLvlInlineParams, scope, aqiLvlBearer, user, aqiLvlTemplate.Id, ct);
                if (aqiLvlResult.Success)
                {
                    var aqiLvlMsgId = Guid.NewGuid();
                    string aqiLvlSummary;
                    if (aqiLvlResult.Rows.Count == 0)
                    {
                        var noDataSector = aqiLvlInlineParams.GetValueOrDefault("sectorName") ?? scope.Sector;
                        var noDataRegion = aqiLvlInlineParams.GetValueOrDefault("regionName") ?? scope.Region;
                        var noDataParts = new List<string>();
                        if (!string.IsNullOrWhiteSpace(noDataSector)) noDataParts.Add(noDataSector);
                        if (!string.IsNullOrWhiteSpace(noDataRegion)) noDataParts.Add(noDataRegion);
                        var noDataContext = noDataParts.Count > 0 ? $" for {string.Join(", ", noDataParts)}" : "";
                        aqiLvlSummary = $"There are no active sites{noDataContext} with AQI data available at this time.";
                    }
                    else
                        aqiLvlSummary = await _azureAI.SummarizeResultsAsync(request.Message, aqiLvlTemplate, aqiLvlResult.Rows, scope, ct);
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId, userContent: request.Message,
                        assistantMessageId: aqiLvlMsgId, assistantContent: aqiLvlSummary,
                        sql: null, responseType: aqiLvlTemplate.ResponseType,
                        chartType: effectiveAqiLvlType, chartDataJson: SerializeRowsForHistory(aqiLvlTemplate.ResponseType, aqiLvlResult.Rows),
                        dataSource: aqiLvlTemplate.TableUsed, dateRange: null,
                        modelUsed: null, executionTimeMs: (int)aqiLvlResult.ExecutionTimeMs,
                        tokenCount: 0, ct: ct);
                    if (session.MessageCount <= 1)
                        _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                    return Ok(_formatter.Format(
                        sessionId: session.SessionId, messageId: aqiLvlMsgId,
                        template: aqiLvlTemplate, llmSummary: aqiLvlSummary,
                        rows: aqiLvlResult.Rows, rowCount: aqiLvlResult.Rows.Count,
                        executionTimeMs: (int)aqiLvlResult.ExecutionTimeMs,
                        scope: scope, modelUsed: null, tokenCount: 0));
                }
                _log.Warning("ChatController: sites_aqi_level_chart API call failed, falling through to routing");
            }
        }

        // ── 2e-pre9. Hard pre-check: "how many devices does [site] have" → force devices_by_filter with stationName ──
        // "how many devices does Al Bahya Educational Complex have?" — the LLM often fails to extract
        // stationName from this phrasing (site is the subject, not after a preposition like "at").
        // Detect this pattern directly and extract the site name via regex.
        bool isHowManyDevicesAtSite = lowerMsg.Contains("how many devices does") || lowerMsg.Contains("how many devices do");
        if (isHowManyDevicesAtSite)
        {
            // Extract site name: "how many devices does [SITE NAME] have"
            var devSiteMatch = System.Text.RegularExpressions.Regex.Match(
                request.Message,
                @"how many devices (?:does|do)\s+(.+?)\s+(?:have|contain|include|currently have|currently contain)\b",
                System.Text.RegularExpressions.RegexOptions.IgnoreCase);
            if (devSiteMatch.Success)
            {
                var extractedSiteName = devSiteMatch.Groups[1].Value.Trim();
                // Strip common noise words that may appear before the site name
                extractedSiteName = System.Text.RegularExpressions.Regex.Replace(
                    extractedSiteName, @"^(?:the\s+|a\s+|an\s+)", "", System.Text.RegularExpressions.RegexOptions.IgnoreCase).Trim();

                if (!string.IsNullOrWhiteSpace(extractedSiteName))
                {
                    _log.Debug("ChatController: devices-at-site pre-check fired → forcing devices_by_filter (stationName={Site})", extractedSiteName);
                    var devSiteTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "devices_by_filter");
                    if (devSiteTemplate is not null)
                    {
                        var devSiteParams = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                        {
                            ["stationName"] = extractedSiteName
                        };
                        var (devSiteSql, devSiteSqlParams) = _rbac.BuildSafeQuery(devSiteTemplate, devSiteParams, user, scope);
                        var (devSiteValid, _) = SqlValidator.Validate(devSiteSql);
                        if (devSiteValid)
                        {
                            var devSiteResult = await _sqlExecutor.ExecuteAsync(devSiteSql, devSiteSqlParams, ct);
                            if (devSiteResult.Success)
                            {
                                var devSiteSummary = await _azureAI.SummarizeResultsAsync(request.Message, devSiteTemplate, devSiteResult.Rows, scope, ct);
                                var devSiteMsgId = Guid.NewGuid();
                                await _sessionService.SaveBothMessagesAsync(
                                    sessionId: session.SessionId,
                                    userContent: request.Message,
                                    assistantMessageId: devSiteMsgId,
                                    assistantContent: devSiteSummary,
                                    sql: devSiteSql,
                                    responseType: devSiteTemplate.ResponseType,
                                    chartType: null, chartDataJson: SerializeRowsForHistory(devSiteTemplate.ResponseType, devSiteResult.Rows),
                                    dataSource: devSiteTemplate.TableUsed, dateRange: null,
                                    modelUsed: null, executionTimeMs: (int)devSiteResult.ExecutionTimeMs,
                                    tokenCount: 0, ct: ct);
                                if (session.MessageCount <= 1)
                                    _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                                return Ok(_formatter.Format(
                                    sessionId: session.SessionId, messageId: devSiteMsgId,
                                    template: devSiteTemplate, llmSummary: devSiteSummary,
                                    rows: devSiteResult.Rows, rowCount: devSiteResult.Rows.Count,
                                    executionTimeMs: (int)devSiteResult.ExecutionTimeMs,
                                    scope: scope, modelUsed: null, tokenCount: 0));
                            }
                        }
                        _log.Warning("ChatController: devices-at-site pre-check failed (site={Site}), falling through to routing", extractedSiteName);
                    }
                }
            }
        }

        // ── 2e. Hard pre-check: offline/inactive device questions → force offline_devices ──
        // Scope-bar enrichment appends "in Abudhabi" to the message, causing these questions to
        // match devices_by_filter (which requires BOTH regionName AND sectorName → returns nothing).
        // Detect offline-device intent and route directly to the offline_devices template,
        // which runs a standalone SQL with no region/sector requirement.
        var offlineKeywords = new[]
        {
            "offline", "not sending", "not reporting", "no data", "inactive device",
            "inactive devices", "stopped sending", "stopped reporting", "not transmitting",
            "lost connection", "lost connectivity", "disconnected device", "disconnected devices",
            "devices offline", "devices not sending", "devices not reporting",
            "which devices are offline", "which device is offline",
            "how many devices are offline", "list offline", "show offline",
            // "connected" / "not connected" — connectivity language maps to online/offline status
            "not connected", "disconnected", "devices that are not connected",
            "which devices are not connected", "are there any devices that are not connected",
            "inactive sensor", "inactive sensors", "not transmitting",
            "devices not transmitting", "which devices are inactive", "devices that are inactive",
            "devices currently inactive", "show inactive", "list inactive"
        };
        // A specific device code (BA 0001, SEI100M 0009) means the user is asking about ONE device,
        // not requesting a list of all offline devices — exclude those from the offline pre-check.
        bool msgHasSpecificDeviceCode = System.Text.RegularExpressions.Regex.IsMatch(
            lowerMsg, @"(ba\s*\d{1,4}|sei\s*100\s*m\s*\d{1,4})");
        // Count/number intent must NOT be treated as a listing question — route to count template instead
        bool msgIsDeviceCountIntent = (lowerMsg.Contains("count") || lowerMsg.Contains("how many")
            || lowerMsg.Contains("total") || lowerMsg.Contains("number of"))
            && (lowerMsg.Contains("device") || lowerMsg.Contains("sensor"));
        bool isOfflineDeviceQuestion = offlineKeywords.Any(k => lowerMsg.Contains(k))
            && !msgHasSpecificDeviceCode   // single-device status queries must NOT be treated as list queries
            && !msgIsDeviceCountIntent     // count questions must route to active_device_count, not list
            && !lowerMsg.Contains("should be checked first")   // FAQ advisory → already handled above
            && !lowerMsg.Contains("when is a device considered");  // FAQ definition → already handled above

        if (isOfflineDeviceQuestion)
        {
            // ── Sector-filtered offline: "schools with offline devices", "offline devices at schools" etc.
            // When user specifies a sector (school/commercial/residential), use devices_by_filter with
            // statusFilter="Inactive" and the matched sectorName — NOT the unfiltered offline_devices list.
            string? offlineSectorFilter = null;
            if (lowerMsg.Contains("school") || lowerMsg.Contains("govt") || lowerMsg.Contains("government") || lowerMsg.Contains("moe"))
                offlineSectorFilter = "Public & Govt-School";
            else if (lowerMsg.Contains("commercial"))
                offlineSectorFilter = "Commercial";
            else if (lowerMsg.Contains("residential"))
                offlineSectorFilter = "Residential";

            if (offlineSectorFilter is not null)
            {
                _log.Debug("ChatController: offline-device pre-check with sector filter → forcing devices_by_filter (sector={Sector})", offlineSectorFilter);
                var devFilterTemplateOff = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "devices_by_filter");
                if (devFilterTemplateOff is not null)
                {
                    var offSectorParams = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                    {
                        ["sectorName"]    = offlineSectorFilter,
                        ["statusFilter"]  = "Inactive"
                    };
                    // Also inject regionName if user mentioned one
                    if (lowerMsg.Contains("abu dhabi") || lowerMsg.Contains("abudhabi"))
                        offSectorParams["regionName"] = "Abu Dhabi";
                    else if (lowerMsg.Contains("al ain") || lowerMsg.Contains("alain"))
                        offSectorParams["regionName"] = "Al Ain";
                    else if (lowerMsg.Contains("al dhafra") || lowerMsg.Contains("dhafra"))
                        offSectorParams["regionName"] = "Al Dhafra";

                    var (offSectorSql, offSectorSqlParams) = _rbac.BuildSafeQuery(devFilterTemplateOff, offSectorParams, user, scope);
                    var (offSectorValid, _) = SqlValidator.Validate(offSectorSql);
                    if (offSectorValid)
                    {
                        var offSectorResult = await _sqlExecutor.ExecuteAsync(offSectorSql, offSectorSqlParams, ct);
                        if (offSectorResult.Success)
                        {
                            var offSectorSummary = await _azureAI.SummarizeResultsAsync(request.Message, devFilterTemplateOff, offSectorResult.Rows, scope, ct);
                            var offSectorMsgId = Guid.NewGuid();
                            await _sessionService.SaveBothMessagesAsync(
                                sessionId: session.SessionId,
                                userContent: request.Message,
                                assistantMessageId: offSectorMsgId,
                                assistantContent: offSectorSummary,
                                sql: offSectorSql,
                                responseType: devFilterTemplateOff.ResponseType,
                                chartType: null, chartDataJson: SerializeRowsForHistory(devFilterTemplateOff.ResponseType, offSectorResult.Rows),
                                dataSource: devFilterTemplateOff.TableUsed, dateRange: null,
                                modelUsed: null, executionTimeMs: (int)offSectorResult.ExecutionTimeMs,
                                tokenCount: 0, ct: ct);
                            if (session.MessageCount <= 1)
                                _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                            return Ok(_formatter.Format(
                                sessionId: session.SessionId, messageId: offSectorMsgId,
                                template: devFilterTemplateOff, llmSummary: offSectorSummary,
                                rows: offSectorResult.Rows, rowCount: offSectorResult.Rows.Count,
                                executionTimeMs: (int)offSectorResult.ExecutionTimeMs,
                                scope: scope, modelUsed: null, tokenCount: 0));
                        }
                    }
                    _log.Warning("ChatController: sector-filtered offline pre-check failed, falling through to unfiltered offline_devices");
                }
            }

            _log.Debug("ChatController: offline-device pre-check fired → forcing offline_devices");
            var offlineTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "offline_devices");
            if (offlineTemplate is not null && !string.IsNullOrWhiteSpace(offlineTemplate.Sql))
            {
                // No extra params needed — offline_devices SQL uses only the 15-min cutoff
                var emptyParams = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                var (offlineSafeSql, offlineSqlParams) = _rbac.BuildSafeQuery(offlineTemplate, emptyParams, user, scope);
                var (offlineValid, offlineError) = SqlValidator.Validate(offlineSafeSql);
                if (offlineValid)
                {
                    var offlineResult = await _sqlExecutor.ExecuteAsync(offlineSafeSql, offlineSqlParams, ct);
                    if (offlineResult.Success)
                    {
                        var offlineMsgId = Guid.NewGuid();
                        var offlineSummary = await _azureAI.SummarizeResultsAsync(request.Message, offlineTemplate, offlineResult.Rows, scope, ct);
                        await _sessionService.SaveBothMessagesAsync(
                            sessionId: session.SessionId,
                            userContent: request.Message,
                            assistantMessageId: offlineMsgId,
                            assistantContent: offlineSummary,
                            sql: offlineSafeSql,
                            responseType: offlineTemplate.ResponseType,
                            chartType: null,
                            chartDataJson: SerializeRowsForHistory(offlineTemplate.ResponseType, offlineResult.Rows),
                            dataSource: offlineTemplate.TableUsed,
                            dateRange: null,
                            modelUsed: null,
                            executionTimeMs: (int)offlineResult.ExecutionTimeMs,
                            tokenCount: 0,
                            ct: ct);
                        if (session.MessageCount <= 1)
                            _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                        return Ok(_formatter.Format(
                            sessionId: session.SessionId,
                            messageId: offlineMsgId,
                            template: offlineTemplate,
                            llmSummary: offlineSummary,
                            rows: offlineResult.Rows,
                            rowCount: offlineResult.Rows.Count,
                            executionTimeMs: (int)offlineResult.ExecutionTimeMs,
                            scope: scope,
                            modelUsed: null,
                            tokenCount: 0));
                    }
                    _log.Warning("ChatController: offline_devices SQL execution failed: {Error}", offlineResult.Error);
                }
                else
                {
                    _log.Warning("ChatController: offline_devices SQL validation failed: {Error}", offlineError);
                }
            }
            // If template missing or SQL failed, fall through to normal routing
        }

        // ── 2f. Hard pre-check: online/connected device questions → force online_devices ──
        // Same scope-bar issue as 2e — "connected devices in Abudhabi" hits devices_by_filter.
        var onlineKeywords = new[]
        {
            "connected device", "connected devices", "which devices are connected",
            "devices that are connected", "devices currently connected",
            "show connected", "list connected", "which sensors are connected",
            "connected sensors", "devices transmitting", "which devices are transmitting",
            "devices that are transmitting", "are there any devices that are connected",
            "which devices are online right now", "which devices are active right now",
            "devices online right now", "devices active right now"
        };
        bool isOnlineDeviceQuestion = onlineKeywords.Any(k => lowerMsg.Contains(k))
            && !isOfflineDeviceQuestion;  // don't double-fire if somehow both match

        if (isOnlineDeviceQuestion)
        {
            // Sector-filtered online: same logic as offline — redirect to devices_by_filter when user names a sector.
            string? onlineSectorFilter = null;
            if (lowerMsg.Contains("school") || lowerMsg.Contains("govt") || lowerMsg.Contains("government") || lowerMsg.Contains("moe"))
                onlineSectorFilter = "Public & Govt-School";
            else if (lowerMsg.Contains("commercial"))
                onlineSectorFilter = "Commercial";
            else if (lowerMsg.Contains("residential"))
                onlineSectorFilter = "Residential";

            if (onlineSectorFilter is not null)
            {
                _log.Debug("ChatController: online-device pre-check with sector filter → forcing devices_by_filter (sector={Sector})", onlineSectorFilter);
                var devFilterTemplateOn = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "devices_by_filter");
                if (devFilterTemplateOn is not null)
                {
                    var onSectorParams = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                    {
                        ["sectorName"]   = onlineSectorFilter,
                        ["statusFilter"] = "Active"
                    };
                    if (lowerMsg.Contains("abu dhabi") || lowerMsg.Contains("abudhabi"))
                        onSectorParams["regionName"] = "Abu Dhabi";
                    else if (lowerMsg.Contains("al ain") || lowerMsg.Contains("alain"))
                        onSectorParams["regionName"] = "Al Ain";
                    else if (lowerMsg.Contains("al dhafra") || lowerMsg.Contains("dhafra"))
                        onSectorParams["regionName"] = "Al Dhafra";

                    var (onSectorSql, onSectorSqlParams) = _rbac.BuildSafeQuery(devFilterTemplateOn, onSectorParams, user, scope);
                    var (onSectorValid, _) = SqlValidator.Validate(onSectorSql);
                    if (onSectorValid)
                    {
                        var onSectorResult = await _sqlExecutor.ExecuteAsync(onSectorSql, onSectorSqlParams, ct);
                        if (onSectorResult.Success)
                        {
                            var onSectorSummary = await _azureAI.SummarizeResultsAsync(request.Message, devFilterTemplateOn, onSectorResult.Rows, scope, ct);
                            var onSectorMsgId = Guid.NewGuid();
                            await _sessionService.SaveBothMessagesAsync(
                                sessionId: session.SessionId,
                                userContent: request.Message,
                                assistantMessageId: onSectorMsgId,
                                assistantContent: onSectorSummary,
                                sql: onSectorSql,
                                responseType: devFilterTemplateOn.ResponseType,
                                chartType: null, chartDataJson: SerializeRowsForHistory(devFilterTemplateOn.ResponseType, onSectorResult.Rows),
                                dataSource: devFilterTemplateOn.TableUsed, dateRange: null,
                                modelUsed: null, executionTimeMs: (int)onSectorResult.ExecutionTimeMs,
                                tokenCount: 0, ct: ct);
                            if (session.MessageCount <= 1)
                                _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                            return Ok(_formatter.Format(
                                sessionId: session.SessionId, messageId: onSectorMsgId,
                                template: devFilterTemplateOn, llmSummary: onSectorSummary,
                                rows: onSectorResult.Rows, rowCount: onSectorResult.Rows.Count,
                                executionTimeMs: (int)onSectorResult.ExecutionTimeMs,
                                scope: scope, modelUsed: null, tokenCount: 0));
                        }
                    }
                    _log.Warning("ChatController: sector-filtered online pre-check failed, falling through to unfiltered online_devices");
                }
            }

            _log.Debug("ChatController: online-device pre-check fired → forcing online_devices");
            var onlineTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "online_devices");
            if (onlineTemplate is not null && !string.IsNullOrWhiteSpace(onlineTemplate.Sql))
            {
                var emptyParams2 = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                var (onlineSafeSql, onlineSqlParams) = _rbac.BuildSafeQuery(onlineTemplate, emptyParams2, user, scope);
                var (onlineValid, onlineError) = SqlValidator.Validate(onlineSafeSql);
                if (onlineValid)
                {
                    var onlineResult = await _sqlExecutor.ExecuteAsync(onlineSafeSql, onlineSqlParams, ct);
                    if (onlineResult.Success)
                    {
                        var onlineMsgId = Guid.NewGuid();
                        var onlineSummary = await _azureAI.SummarizeResultsAsync(request.Message, onlineTemplate, onlineResult.Rows, scope, ct);
                        await _sessionService.SaveBothMessagesAsync(
                            sessionId: session.SessionId,
                            userContent: request.Message,
                            assistantMessageId: onlineMsgId,
                            assistantContent: onlineSummary,
                            sql: onlineSafeSql,
                            responseType: onlineTemplate.ResponseType,
                            chartType: null,
                            chartDataJson: SerializeRowsForHistory(onlineTemplate.ResponseType, onlineResult.Rows),
                            dataSource: onlineTemplate.TableUsed,
                            dateRange: null,
                            modelUsed: null,
                            executionTimeMs: (int)onlineResult.ExecutionTimeMs,
                            tokenCount: 0,
                            ct: ct);
                        if (session.MessageCount <= 1)
                            _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                        return Ok(_formatter.Format(
                            sessionId: session.SessionId,
                            messageId: onlineMsgId,
                            template: onlineTemplate,
                            llmSummary: onlineSummary,
                            rows: onlineResult.Rows,
                            rowCount: onlineResult.Rows.Count,
                            executionTimeMs: (int)onlineResult.ExecutionTimeMs,
                            scope: scope,
                            modelUsed: null,
                            tokenCount: 0));
                    }
                    _log.Warning("ChatController: online_devices SQL execution failed: {Error}", onlineResult.Error);
                }
                else
                {
                    _log.Warning("ChatController: online_devices SQL validation failed: {Error}", onlineError);
                }
            }
            // If template missing or SQL failed, fall through to normal routing
        }

        // ── 2h. Hard pre-check: "devices under/in region or sector" → force devices_by_filter ──
        // Route all device listing/filter queries to devices_by_filter.
        // Detect region or sector words and force devices_by_filter before LLM routing.
        // NOTE: Only sector-as-category phrases trigger this (e.g. "school sector", "commercial sector").
        //       A site name that contains "school" (e.g. "Mleih Cycle 3 School") should NOT trigger this.
        var devFilterRegionWords  = new[] { "abu dhabi", "abudhabi", "al ain", "alain", "al dhafra", "aldhafra" };
        // Sector phrases that unambiguously mean the sector category, not a site name
        var devFilterSectorPhrases = new[] {
            "commercial sector", "residential sector", "school sector", "industrial sector", "public sector",
            "in commercial", "in residential", "under commercial", "under residential",
            "under industrial", "in industrial",
            "school sites", "school site"
        };
        var devFilterDeviceWords  = new[] { "devices", "sensors", "device", "sensor" };
        bool devHasRegionWord  = devFilterRegionWords.Any(r => lowerMsg.Contains(r));
        bool devHasSectorPhrase = devFilterSectorPhrases.Any(s => lowerMsg.Contains(s));
        bool devHasDeviceWord  = devFilterDeviceWords.Any(d => lowerMsg.Contains(d));
        // Only fire when asking about devices in a region/sector — not for AQI/readings questions
        bool isDevicesByRegionOrSector = devHasDeviceWord && (devHasRegionWord || devHasSectorPhrase)
            && !lowerMsg.Contains("aqi") && !lowerMsg.Contains("air quality")
            && !lowerMsg.Contains("reading") && !lowerMsg.Contains("pm2.5")
            && !lowerMsg.Contains("co2") && !lowerMsg.Contains("temperature");

        if (isDevicesByRegionOrSector)
        {
            _log.Debug("ChatController: devices-by-region/sector pre-check fired → forcing devices_by_filter");
            var devFilterTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "devices_by_filter");
            if (devFilterTemplate is not null)
            {
                var devFilterFill = await _azureAI.FillParametersAsync(enrichedMessage, devFilterTemplate, history, scope, ct);

                // Hard-override sectorName when message mentions school but LLM left it blank
                if (string.IsNullOrWhiteSpace(devFilterFill.Parameters.GetValueOrDefault("sectorName")))
                {
                    if (lowerMsg.Contains("school"))
                        devFilterFill.Parameters["sectorName"] = "Public & Govt-School";
                    else if (lowerMsg.Contains("commercial"))
                        devFilterFill.Parameters["sectorName"] = "Commercial";
                    else if (lowerMsg.Contains("residential"))
                        devFilterFill.Parameters["sectorName"] = "Residential";
                }

                var (devFilterSafeSql, devFilterSqlParams) =
                    _rbac.BuildSafeQuery(devFilterTemplate, devFilterFill.Parameters, user, scope);
                var (devFilterSqlValid, devFilterSqlError) = SqlValidator.Validate(devFilterSafeSql);
                if (devFilterSqlValid)
                {
                    var devFilterResult = await _sqlExecutor.ExecuteAsync(devFilterSafeSql, devFilterSqlParams, ct);
                    if (devFilterResult.Success)
                    {
                        var devFilterSummary = await _azureAI.SummarizeResultsAsync(
                            request.Message, devFilterTemplate, devFilterResult.Rows, scope, ct);
                        var devFilterMsgId = Guid.NewGuid();
                        await _sessionService.SaveBothMessagesAsync(
                            sessionId: session.SessionId,
                            userContent: request.Message,
                            assistantMessageId: devFilterMsgId,
                            assistantContent: devFilterSummary,
                            sql: devFilterSafeSql,
                            responseType: devFilterTemplate.ResponseType,
                            chartType: null,
                            chartDataJson: SerializeRowsForHistory(devFilterTemplate.ResponseType, devFilterResult.Rows),
                            dataSource: devFilterTemplate.TableUsed,
                            dateRange: null,
                            modelUsed: devFilterFill.ModelUsed,
                            executionTimeMs: (int)devFilterResult.ExecutionTimeMs,
                            tokenCount: devFilterFill.TokensUsed,
                            ct: ct);
                        return Ok(_formatter.Format(session.SessionId, devFilterMsgId, devFilterTemplate,
                            devFilterSummary, devFilterResult.Rows, devFilterResult.Rows.Count,
                            (int)devFilterResult.ExecutionTimeMs, scope, devFilterFill.ModelUsed, devFilterFill.TokensUsed));
                    }
                }
                _log.Warning("ChatController: devices_by_filter pre-check failed (valid={V} err='{E}'), falling through to routing",
                    devFilterSqlValid, devFilterSqlError ?? "(none)");
            }
        }

        // ── 2h-pre. Hard pre-check: "what sensors are currently reporting at <site>" → force site_readings_all ──
        // "Tell me what the sensors are currently reporting at Test Site" means: show live readings
        // for all devices at that site — maps to site_readings_all, not devices_by_filter.
        var sensorReportingPhrases = new[] {
            "sensors are currently reporting", "sensors currently reporting",
            "what sensors are reporting", "what are the sensors reporting",
            "sensors report at", "sensors reporting at",
            "what is the sensor reporting", "sensor is currently reporting",
            "sensors are reporting at", "what sensors report at",
            "currently reporting at", "sensors are measuring at",
            "what are the sensors measuring", "sensors currently measuring",
            "what are the devices reporting", "devices currently reporting",
            "devices are currently reporting", "devices are reporting at",
            "what are the sensors showing", "sensors are showing at",
            "what is being reported at", "what is being measured at",
            "readings being reported", "data being reported at",
            "what parameters are being measured", "what is being monitored at",
            "current readings at", "current sensor readings",
            "live readings at", "live sensor data at"
        };
        bool isSensorReportingQuestion = sensorReportingPhrases.Any(p => lowerMsg.Contains(p))
            && !lowerMsg.Contains("yesterday") && !lowerMsg.Contains("last week")
            && !lowerMsg.Contains("last month") && !System.Text.RegularExpressions.Regex.IsMatch(lowerMsg, @"\b(19|20)\d{2}\b");

        if (isSensorReportingQuestion)
        {
            _log.Debug("ChatController: sensor-reporting pre-check fired → forcing site_readings_all");
            var sensorReadTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct))
                .FirstOrDefault(t => t.Id == "site_readings_all");
            if (sensorReadTemplate is not null && sensorReadTemplate.ApiCall is not null)
            {
                var sensorBearer = Request.Headers.Authorization.ToString()
                    .Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
                var sensorFill = await _azureAI.FillParametersAsync(enrichedMessage, sensorReadTemplate, history, scope, ct);
                var sensorResult = await _externalApi.CallAsync(
                    sensorReadTemplate.ApiCall, sensorFill.Parameters, scope, sensorBearer, user, sensorReadTemplate.Id, ct);
                if (sensorResult.Success)
                {
                    var sensorSummary = await _azureAI.SummarizeResultsAsync(
                        request.Message, sensorReadTemplate, sensorResult.Rows, scope, ct);
                    var sensorMsgId = Guid.NewGuid();
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId,
                        userContent: request.Message,
                        assistantMessageId: sensorMsgId,
                        assistantContent: sensorSummary,
                        sql: null,
                        responseType: sensorReadTemplate.ResponseType,
                        chartType: null,
                        chartDataJson: null,
                        dataSource: sensorReadTemplate.TableUsed,
                        dateRange: null,
                        modelUsed: sensorFill.ModelUsed,
                        executionTimeMs: (int)sensorResult.ExecutionTimeMs,
                        tokenCount: sensorFill.TokensUsed,
                        ct: ct);
                    return Ok(_formatter.Format(session.SessionId, sensorMsgId, sensorReadTemplate,
                        sensorSummary, sensorResult.Rows, sensorResult.Rows.Count,
                        (int)sensorResult.ExecutionTimeMs, scope, sensorFill.ModelUsed, sensorFill.TokensUsed));
                }
                _log.Warning("ChatController: sensor-reporting pre-check API failed, falling through to routing");
            }
        }

        // ── 2i. Hard pre-check: "list/count devices for/at <site>" → force devices_by_filter ──
        // Catches queries like "List devices for Mubarak Bin Mohammed School Cycle 2" and
        // "How many devices does Al Ain School have?" which can be mis-routed to device_live_reading
        // when the site name contains number-like tokens (e.g. "Cycle 2") that fool QuestionHasDeviceName.
        var siteDeviceListPhrases = new[] {
            "list devices for", "list devices at", "list devices in",
            "list the devices installed at", "list the devices installed in",
            "list the devices for", "list the devices at",
            "devices for", "devices at", "devices installed at", "devices under",
            "which devices are installed at", "which devices are installed in",
            "which devices are currently available at", "which devices are currently available in",
            "which devices are available at", "which devices are available in",
            "which devices are at", "which devices are in",
            "currently available at", "currently available in",
            "sensors for", "sensors at", "sensors installed at",
            "how many devices does", "how many devices at", "how many devices are at",
            "how many devices in", "how many sensors does", "how many sensors at",
            "what devices are at", "what devices are in", "what sensors are at",
            "device count at", "device count for", "show devices at", "show devices for"
        };
        // Only fire when NO actual device code (BA/SEI100M) is present — site queries don't have codes
        bool hasBaOrSeiCode = System.Text.RegularExpressions.Regex.IsMatch(lowerMsg, @"\b(ba\s*\d{1,4}|sei\s*100\s*m\s*\d{1,4})\b");
        bool isSiteDeviceListQuestion = !hasBaOrSeiCode
            && siteDeviceListPhrases.Any(p => lowerMsg.Contains(p))
            && !lowerMsg.Contains("aqi") && !lowerMsg.Contains("reading")
            && !lowerMsg.Contains("pm2.5") && !lowerMsg.Contains("co2")
            && !lowerMsg.Contains("temperature");

        if (isSiteDeviceListQuestion)
        {
            _log.Debug("ChatController: site-device-list pre-check fired → forcing devices_by_filter");
            var siteDevTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "devices_by_filter");
            if (siteDevTemplate is not null)
            {
                var siteDevFill = await _azureAI.FillParametersAsync(enrichedMessage, siteDevTemplate, history, scope, ct);
                var (siteDevSafeSql, siteDevSqlParams) =
                    _rbac.BuildSafeQuery(siteDevTemplate, siteDevFill.Parameters, user, scope);
                var (siteDevSqlValid, siteDevSqlError) = SqlValidator.Validate(siteDevSafeSql);
                if (siteDevSqlValid)
                {
                    var siteDevResult = await _sqlExecutor.ExecuteAsync(siteDevSafeSql, siteDevSqlParams, ct);
                    if (siteDevResult.Success)
                    {
                        var siteDevSummary = await _azureAI.SummarizeResultsAsync(
                            request.Message, siteDevTemplate, siteDevResult.Rows, scope, ct);
                        var siteDevMsgId = Guid.NewGuid();
                        await _sessionService.SaveBothMessagesAsync(
                            sessionId: session.SessionId,
                            userContent: request.Message,
                            assistantMessageId: siteDevMsgId,
                            assistantContent: siteDevSummary,
                            sql: siteDevSafeSql,
                            responseType: siteDevTemplate.ResponseType,
                            chartType: null,
                            chartDataJson: SerializeRowsForHistory(siteDevTemplate.ResponseType, siteDevResult.Rows),
                            dataSource: siteDevTemplate.TableUsed,
                            dateRange: null,
                            modelUsed: siteDevFill.ModelUsed,
                            executionTimeMs: (int)siteDevResult.ExecutionTimeMs,
                            tokenCount: siteDevFill.TokensUsed,
                            ct: ct);
                        return Ok(_formatter.Format(session.SessionId, siteDevMsgId, siteDevTemplate,
                            siteDevSummary, siteDevResult.Rows, siteDevResult.Rows.Count,
                            (int)siteDevResult.ExecutionTimeMs, scope, siteDevFill.ModelUsed, siteDevFill.TokensUsed));
                    }
                }
                _log.Warning("ChatController: site-device-list pre-check failed (valid={V} err='{E}'), falling through to routing",
                    siteDevSqlValid, siteDevSqlError ?? "(none)");
            }
        }

        // ── 2j. Hard pre-check: "critical/hotspot sites right now / currently" → force site_aqi_by_category ──
        // priority_hotspots uses Compliancestatas (yearly averages, threshold all devices > 150) which
        // almost always returns 0 sites. For "live / right now / currently" questions use the live
        // GetAllSiteData path (site_aqi_by_category with minValue=150, matching the Executive Dashboard
        // definition: a site is a priority hotspot when ALL its devices have IAQ-I > 150).
        var criticalNowPhrases = new[] {
            "right now", "at the moment", "currently", "current number", "at this moment", "live",
            "today", "now", "site names", "show me", "list", "give me"
        };
        var criticalHotspotWords = new[] {
            "hotspot", "critical site", "critical air quality", "critical air-quality",
            "any critical", "total critical", "number of critical", "how many critical",
            "how many hotspot", "classified as critical", "currently critical",
            "critical sites", "critical locations", "sites are critical",
            "sites under critical", "sites in critical"
        };
        bool isCriticalNowQuestion = criticalHotspotWords.Any(p => lowerMsg.Contains(p));

        if (isCriticalNowQuestion)
        {
            _log.Debug("ChatController: critical-sites-live pre-check fired → forcing site_aqi_by_category(critical)");
            var critTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "site_aqi_by_category");
            if (critTemplate is not null && critTemplate.ApiCall is not null)
            {
                var critBearerToken = Request.Headers.Authorization.ToString().Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
                // Force minValue=150 — Executive Dashboard defines "priority hotspot" as IAQ-I > 150
                // (all devices at the site must exceed 150, same threshold as Compliancestatas)
                var critFill = await _azureAI.FillParametersAsync(enrichedMessage, critTemplate, history, scope, ct);
                critFill.Parameters.Remove("aqiCategory");
                critFill.Parameters["minValue"] = "150";

                var critApiResult = await _externalApi.CallAsync(critTemplate.ApiCall, critFill.Parameters, scope, critBearerToken, user, critTemplate.Id, ct);
                if (critApiResult.Success)
                {
                    var critSummary = await _azureAI.SummarizeResultsAsync(
                        request.Message, critTemplate, critApiResult.Rows, scope, ct);
                    var critMsgId = Guid.NewGuid();
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId,
                        userContent: request.Message,
                        assistantMessageId: critMsgId,
                        assistantContent: critSummary,
                        sql: null,
                        responseType: critTemplate.ResponseType,
                        chartType: null,
                        chartDataJson: SerializeRowsForHistory(critTemplate.ResponseType, critApiResult.Rows),
                        dataSource: critTemplate.TableUsed,
                        dateRange: null,
                        modelUsed: critFill.ModelUsed,
                        executionTimeMs: (int)critApiResult.ExecutionTimeMs,
                        tokenCount: critFill.TokensUsed,
                        ct: ct);
                    return Ok(_formatter.Format(session.SessionId, critMsgId, critTemplate,
                        critSummary, critApiResult.Rows, critApiResult.Rows.Count,
                        (int)critApiResult.ExecutionTimeMs, scope, critFill.ModelUsed, critFill.TokensUsed));
                }
                _log.Warning("ChatController: critical-sites-live pre-check API failed, falling through to routing");
            }
        }

        // ── 2k. Hard pre-check: sites filtered by AQI category → force site_aqi_by_category ──
        // Catches "total good sites", "show moderate sites", "sites in hazardous", "unhealthy sites today"
        // before the pattern scorer can route them to site_aqi_all.
        var aqiCategoryTriggerPhrases = new[] {
            // good
            "good sites", "sites with good", "sites in good", "sites where aqi is good",
            "sites that are good", "show good", "list good sites", "good air quality sites",
            "total good", "how many good",
            // moderate
            "moderate sites", "sites with moderate", "sites in moderate", "sites where aqi is moderate",
            "sites that are moderate", "show moderate", "list moderate sites",
            "total moderate", "how many moderate",
            // unhealthy for sensitive groups
            "sensitive groups sites", "sites with sensitive", "sensitive group sites",
            "unhealthy for sensitive groups", "in unhealthy for sensitive", "unhealthy for sensitive",
            "schools in unhealthy for sensitive", "schools with unhealthy for sensitive",
            "sites in unhealthy for sensitive", "sites with unhealthy for sensitive",
            // unhealthy
            "unhealthy sites", "sites with unhealthy", "sites in unhealthy", "sites where aqi is unhealthy",
            "sites that are unhealthy", "show unhealthy", "list unhealthy sites",
            "total unhealthy", "how many unhealthy",
            // very unhealthy
            "very unhealthy sites", "sites with very unhealthy", "sites in very unhealthy",
            "total very unhealthy", "how many very unhealthy",
            // hazardous
            "hazardous sites", "sites with hazardous", "sites in hazardous", "sites where aqi is hazardous",
            "sites that are hazardous", "show hazardous", "list hazardous sites",
            "total hazardous", "how many hazardous",
            // critical / hotspot (without needing a time phrase)
            "critical sites", "sites that are critical", "sites in critical", "sites under critical",
            "total critical sites", "how many critical sites", "list critical sites", "show critical sites",
            // relational / comparative AQI category phrases
            "worse than moderate", "worse than good", "worse than unhealthy", "worse than hazardous",
            "better than moderate", "better than unhealthy", "better than hazardous",
            "not good sites", "not in good", "not in moderate", "not safe sites",
            "air quality is worse", "schools where air quality", "sites where air quality",
            "all schools where", "all sites where aqi",
            // "schools where [the] [current] AQI is [category]" — region may appear anywhere
            "schools where the current aqi is", "schools where aqi is", "schools where the aqi is",
            "schools where current aqi is",
            // "sites where [the] [current] AQI is [category]"
            "sites where the current aqi is", "sites where the aqi is", "sites where current aqi is",
            // sector type + AQI category filter — "which residential in Al Dhafra have Moderate AQI"
            "which residential", "which commercial", "which institutional",
            "residential in abu dhabi", "residential in al ain", "residential in al dhafra", "residential in dhafra",
            "commercial in abu dhabi", "commercial in al ain", "commercial in al dhafra", "commercial in dhafra"
        };
        bool isAqiCategoryListQuestion = aqiCategoryTriggerPhrases.Any(p => lowerMsg.Contains(p));

        // ── 2k-rel. Hard pre-check: relational AQI category phrases → site_aqi_by_category with numeric range ──
        // Handles: "worse than Moderate", "better than Unhealthy", "not Good", "above Good",
        //          "schools where air quality is worse than Moderate", "sites not in Good", etc.
        // AQI ranges (from HAWAQM legend):
        //   Good: 0–50   Moderate: 51–100   Unhealthy for Sensitive Groups: 101–150
        //   Unhealthy: 151–200   Very Unhealthy: 201–300   Hazardous: 301–500
        // Category upper bounds (used as minValue thresholds for "worse than X"):
        static (string? min, string? max) ResolveRelationalAqiRange(string msg)
        {
            // "worse than X" / "above X category" / "beyond X" / "exceeds X" / "not X or better"
            //  → minValue = upper bound of category X
            // "better than X" / "below X category" / "not worse than X"
            //  → maxValue = lower bound of category X (exclusive, i.e. upper of the category below)
            var worsePhrases  = new[] { "worse than", "worst than", "beyond", "above the", "above category",
                                        "not good enough", "exceed", "not in good", "not in moderate",
                                        "not good", "not safe", "poor", "bad air", "unhealthy air",
                                        "air quality is worse", "air quality worse", "worse air" };
            var betterPhrases = new[] { "better than", "below the", "below category", "not worse than",
                                        "not unhealthy", "not hazardous", "acceptable", "ok or better",
                                        "good or better", "within safe", "safe range" };

            // "bad air" / "bad air right now" — AQI > 150 by project definition (no category word needed)
            if (msg.Contains("bad air") || msg.Contains("poor air quality") || msg.Contains("unhealthy air"))
                return ("150", null);

            bool isWorse  = worsePhrases.Any(p => msg.Contains(p));
            bool isBetter = betterPhrases.Any(p => msg.Contains(p));

            // Identify referenced category
            string? cat = msg.Contains("hazardous") ? "Hazardous"
                : msg.Contains("very unhealthy")     ? "Very Unhealthy"
                : msg.Contains("unhealthy for sensitive") ? "Unhealthy for Sensitive Groups"
                : msg.Contains("unhealthy")           ? "Unhealthy"
                : msg.Contains("moderate")            ? "Moderate"
                : msg.Contains("good")                ? "Good"
                : null;

            if (cat is null) return (null, null);

            // Upper bound of each category (AQI ≤ this value = IN category)
            var upperBound = cat switch
            {
                "Good"                           => 50,
                "Moderate"                       => 100,
                "Unhealthy for Sensitive Groups" => 150,
                "Unhealthy"                      => 200,
                "Very Unhealthy"                 => 300,
                "Hazardous"                      => 500,
                _                                => (int?)null
            };
            if (!upperBound.HasValue) return (null, null);

            if (isWorse)
                // Sites WORSE than X → AQI > upper bound of X
                return (upperBound.Value.ToString(), null);   // minValue = upper bound
            if (isBetter)
                // Sites BETTER than X → AQI ≤ lower bound of X (= upper bound of previous category)
                return (null, (upperBound.Value - (cat == "Hazardous" ? 200 : 50)).ToString()); // maxValue

            return (null, null);
        }

        var relationalAqiPhrases = new[] {
            "worse than", "worst than", "better than", "above the good", "above the moderate",
            "above good", "above moderate", "above unhealthy", "above hazardous",
            "not good", "not in good", "not safe", "poor air", "bad air quality", "bad air",
            "air quality is worse", "air quality worse", "worse air quality",
            "not acceptable", "beyond moderate", "beyond good", "beyond unhealthy",
            "all schools where air quality", "all sites where air quality",
            "schools where air quality is", "sites where air quality is"
        };
        // "bad air" phrases are self-contained — no category word needed (AQI > 150 by definition)
        bool isBadAirQuestion = lowerMsg.Contains("bad air") || lowerMsg.Contains("poor air quality")
            || lowerMsg.Contains("unhealthy air") || lowerMsg.Contains("bad air right");
        bool isRelationalAqiQuestion = relationalAqiPhrases.Any(p => lowerMsg.Contains(p))
            && (isBadAirQuestion
                || lowerMsg.Contains("good") || lowerMsg.Contains("moderate") || lowerMsg.Contains("unhealthy")
                || lowerMsg.Contains("hazardous") || lowerMsg.Contains("very unhealthy") || lowerMsg.Contains("safe"));

        if (isRelationalAqiQuestion && !isCriticalNowQuestion)
        {
            _log.Debug("ChatController: relational-AQI pre-check fired for '{Msg}'", lowerMsg);
            var relTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "site_aqi_by_category");
            if (relTemplate is not null && relTemplate.ApiCall is not null)
            {
                var relBearer = Request.Headers.Authorization.ToString().Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
                var relFill = await _azureAI.FillParametersAsync(enrichedMessage, relTemplate, history, scope, ct);

                // Override category param — numeric range handles the filter, not category string
                var (relMin, relMax) = ResolveRelationalAqiRange(lowerMsg);
                if (relMin is not null) { relFill.Parameters["minValue"] = relMin; relFill.Parameters.Remove("aqiCategory"); }
                if (relMax is not null) { relFill.Parameters["maxValue"] = relMax; relFill.Parameters.Remove("aqiCategory"); }

                // Inject sector / region from keywords if LLM missed them
                if (!relFill.Parameters.ContainsKey("sectorName") || string.IsNullOrWhiteSpace(relFill.Parameters.GetValueOrDefault("sectorName")))
                {
                    if (lowerMsg.Contains("school") || lowerMsg.Contains("moe") || lowerMsg.Contains("gov"))
                        relFill.Parameters["sectorName"] = "Public & Govt-School";
                    else if (lowerMsg.Contains("commercial")) relFill.Parameters["sectorName"] = "Commercial";
                    else if (lowerMsg.Contains("residential")) relFill.Parameters["sectorName"] = "Residential";
                }
                if (!relFill.Parameters.ContainsKey("regionName") || string.IsNullOrWhiteSpace(relFill.Parameters.GetValueOrDefault("regionName")))
                {
                    if (lowerMsg.Contains("abu dhabi"))      relFill.Parameters["regionName"] = "Abu Dhabi";
                    else if (lowerMsg.Contains("al ain"))    relFill.Parameters["regionName"] = "Al Ain";
                    else if (lowerMsg.Contains("al dhafra") || lowerMsg.Contains("dhafra"))
                        relFill.Parameters["regionName"] = "Al Dhafra";
                }

                _log.Information("ChatController: relational-AQI params — min={Min} max={Max} sector={Sec} region={Reg}",
                    relFill.Parameters.GetValueOrDefault("minValue") ?? "(none)",
                    relFill.Parameters.GetValueOrDefault("maxValue") ?? "(none)",
                    relFill.Parameters.GetValueOrDefault("sectorName") ?? "(none)",
                    relFill.Parameters.GetValueOrDefault("regionName") ?? "(none)");

                var relResult = await _externalApi.CallAsync(relTemplate.ApiCall, relFill.Parameters, scope, relBearer, user, relTemplate.Id, ct);
                if (relResult.Success)
                {
                    var relSummary = await _azureAI.SummarizeResultsAsync(request.Message, relTemplate, relResult.Rows, scope, ct);
                    var relMsgId = Guid.NewGuid();
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId, userContent: request.Message,
                        assistantMessageId: relMsgId, assistantContent: relSummary,
                        sql: null, responseType: relTemplate.ResponseType,
                        chartType: null, chartDataJson: SerializeRowsForHistory(relTemplate.ResponseType, relResult.Rows), dataSource: relTemplate.TableUsed,
                        dateRange: null, modelUsed: relFill.ModelUsed,
                        executionTimeMs: (int)relResult.ExecutionTimeMs, tokenCount: relFill.TokensUsed, ct: ct);
                    return Ok(_formatter.Format(session.SessionId, relMsgId, relTemplate,
                        relSummary, relResult.Rows, relResult.Rows.Count,
                        (int)relResult.ExecutionTimeMs, scope, relFill.ModelUsed, relFill.TokensUsed));
                }
                _log.Warning("ChatController: relational-AQI pre-check API failed, falling through");
            }
        }

        if (isAqiCategoryListQuestion && !isCriticalNowQuestion)
        {
            _log.Debug("ChatController: AQI-category-list pre-check fired → forcing site_aqi_by_category");
            var catTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "site_aqi_by_category");
            if (catTemplate is not null && catTemplate.ApiCall is not null)
            {
                var catBearerToken = Request.Headers.Authorization.ToString().Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
                var catFill = await _azureAI.FillParametersAsync(enrichedMessage, catTemplate, history, scope, ct);

                // Inject aqiCategory from keywords if LLM missed it
                if (!catFill.Parameters.ContainsKey("aqiCategory") || string.IsNullOrWhiteSpace(catFill.Parameters.GetValueOrDefault("aqiCategory")))
                {
                    if (lowerMsg.Contains("hazardous") || lowerMsg.Contains("critical") || lowerMsg.Contains("hotspot"))
                        catFill.Parameters["aqiCategory"] = "Hazardous";
                    else if (lowerMsg.Contains("very unhealthy"))
                        catFill.Parameters["aqiCategory"] = "Very Unhealthy";
                    else if (lowerMsg.Contains("unhealthy for sensitive"))
                        catFill.Parameters["aqiCategory"] = "Unhealthy for Sensitive Groups";
                    else if (lowerMsg.Contains("unhealthy"))
                        catFill.Parameters["aqiCategory"] = "Unhealthy";
                    else if (lowerMsg.Contains("moderate"))
                        catFill.Parameters["aqiCategory"] = "Moderate";
                    else if (lowerMsg.Contains("good"))
                        catFill.Parameters["aqiCategory"] = "Good";
                }

                // Inject sector / region from keywords if LLM missed them
                if (!catFill.Parameters.ContainsKey("sectorName") || string.IsNullOrWhiteSpace(catFill.Parameters.GetValueOrDefault("sectorName")))
                {
                    if (lowerMsg.Contains("school") || lowerMsg.Contains("moe") || lowerMsg.Contains("gov"))
                        catFill.Parameters["sectorName"] = "Public & Govt-School";
                    else if (lowerMsg.Contains("commercial")) catFill.Parameters["sectorName"] = "Commercial";
                    else if (lowerMsg.Contains("residential")) catFill.Parameters["sectorName"] = "Residential";
                }
                if (!catFill.Parameters.ContainsKey("regionName") || string.IsNullOrWhiteSpace(catFill.Parameters.GetValueOrDefault("regionName")))
                {
                    if (lowerMsg.Contains("abu dhabi"))      catFill.Parameters["regionName"] = "Abu Dhabi";
                    else if (lowerMsg.Contains("al ain"))    catFill.Parameters["regionName"] = "Al Ain";
                    else if (lowerMsg.Contains("al dhafra") || lowerMsg.Contains("dhafra"))
                        catFill.Parameters["regionName"] = "Al Dhafra";
                }

                var catApiResult = await _externalApi.CallAsync(catTemplate.ApiCall, catFill.Parameters, scope, catBearerToken, user, catTemplate.Id, ct);
                if (catApiResult.Success)
                {
                    string catSummary;
                    if (catApiResult.Rows.Count == 0)
                    {
                        catSummary = BuildNoDataMessage(catTemplate.Id, scope,
                            catFill.Parameters.GetValueOrDefault("aqiCategory"),
                            catFill.Parameters.GetValueOrDefault("sectorName"));
                    }
                    else
                    {
                        catSummary = await _azureAI.SummarizeResultsAsync(
                            request.Message, catTemplate, catApiResult.Rows, scope, ct);
                    }
                    var catMsgId = Guid.NewGuid();
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId,
                        userContent: request.Message,
                        assistantMessageId: catMsgId,
                        assistantContent: catSummary,
                        sql: null,
                        responseType: catApiResult.Rows.Count == 0 ? "text" : catTemplate.ResponseType,
                        chartType: null,
                        chartDataJson: SerializeRowsForHistory(catTemplate.ResponseType, catApiResult.Rows),
                        dataSource: catTemplate.TableUsed,
                        dateRange: null,
                        modelUsed: catFill.ModelUsed,
                        executionTimeMs: (int)catApiResult.ExecutionTimeMs,
                        tokenCount: catFill.TokensUsed,
                        ct: ct);
                    return Ok(_formatter.Format(session.SessionId, catMsgId, catTemplate,
                        catSummary, catApiResult.Rows, catApiResult.Rows.Count,
                        (int)catApiResult.ExecutionTimeMs, scope, catFill.ModelUsed, catFill.TokensUsed));
                }
                _log.Warning("ChatController: AQI-category-list pre-check API failed, falling through to routing");
            }
        }

        // ── 2l. Hard pre-check: schools/sector sites by AQI category → force site_aqi_by_category ──
        // Catches "how many schools are in each AQI category", "schools by AQI category by region",
        // "MOE schools in each AQI category" before routing to schools_latest_pollutant.
        var schoolsAqiCategoryPhrases = new[] {
            "schools are in each aqi category", "schools in each aqi category",
            "schools by aqi category", "schools grouped by aqi",
            "how many schools in each aqi", "schools in each aqi",
            "schools are in each aqi", "school sites by aqi category",
            "public and government schools in each aqi", "public & government schools in each aqi",
            "moe schools in each aqi", "govt schools in each aqi", "gov schools in each aqi",
            "government schools in each aqi", "government schools by aqi",
            "schools in aqi category", "schools per aqi category",
            "commercial sites in each aqi", "residential sites in each aqi",
            "sites in each aqi category", "sites by aqi category", "sites per aqi category",
            "sites grouped by aqi category", "sites grouped by aqi level",
            "how many sites in each aqi category", "how many sites are in each aqi"
        };
        bool isSchoolsAqiCategoryQuestion = schoolsAqiCategoryPhrases.Any(p => lowerMsg.Contains(p));

        // Detect "count by AQI category [by region]" intent — transforms raw site rows into a grouped summary table.
        // Fires for: "how many in each AQI category", "by region", "grouped by AQI", "breakdown by AQI"
        var aqiCategoryGroupPhrases = new[] {
            "in each aqi category", "in each aqi level", "by aqi category", "by aqi level",
            "grouped by aqi", "per aqi category", "per aqi level", "aqi category breakdown",
            "aqi level breakdown", "each aqi category", "each aqi level",
            "how many in each", "by region", "by sector"
        };
        bool isAqiCategoryGroupByQuestion = isSchoolsAqiCategoryQuestion
            || (lowerMsg.Contains("aqi") && aqiCategoryGroupPhrases.Any(p => lowerMsg.Contains(p)));

        if (isSchoolsAqiCategoryQuestion)
        {
            _log.Debug("ChatController: schools-by-AQI-category pre-check fired → forcing site_aqi_by_category");
            var sectorCatTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "site_aqi_by_category");
            if (sectorCatTemplate is not null && sectorCatTemplate.ApiCall is not null)
            {
                var sectorCatBearer = Request.Headers.Authorization.ToString().Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
                var sectorCatFill = await _azureAI.FillParametersAsync(enrichedMessage, sectorCatTemplate, history, scope, ct);

                // Ensure sectorName is extracted — force school sector if "school" / "moe" / "govt" words present
                if (!sectorCatFill.Parameters.ContainsKey("sectorName") || string.IsNullOrWhiteSpace(sectorCatFill.Parameters.GetValueOrDefault("sectorName")))
                {
                    var msgLower2 = lowerMsg;
                    if (msgLower2.Contains("school") || msgLower2.Contains("moe") || msgLower2.Contains("gov"))
                        sectorCatFill.Parameters["sectorName"] = "Public & Govt-School";
                    else if (msgLower2.Contains("commercial"))
                        sectorCatFill.Parameters["sectorName"] = "Commercial";
                    else if (msgLower2.Contains("residential"))
                        sectorCatFill.Parameters["sectorName"] = "Residential";
                }

                // Force regionName from keywords if LLM missed it
                if (!sectorCatFill.Parameters.ContainsKey("regionName") || string.IsNullOrWhiteSpace(sectorCatFill.Parameters.GetValueOrDefault("regionName")))
                {
                    if (lowerMsg.Contains("abu dhabi"))
                        sectorCatFill.Parameters["regionName"] = "Abu Dhabi";
                    else if (lowerMsg.Contains("al ain"))
                        sectorCatFill.Parameters["regionName"] = "Al Ain";
                    else if (lowerMsg.Contains("al dhafra") || lowerMsg.Contains("dhafra"))
                        sectorCatFill.Parameters["regionName"] = "Al Dhafra";
                }

                var sectorCatResult = await _externalApi.CallAsync(sectorCatTemplate.ApiCall, sectorCatFill.Parameters, scope, sectorCatBearer, user, sectorCatTemplate.Id, ct);
                if (sectorCatResult.Success)
                {
                    // Group raw site rows into {Region, AQICategory, Count} when user asked for a category breakdown
                    bool wantsByRegion = lowerMsg.Contains("by region") || lowerMsg.Contains("per region")
                        || lowerMsg.Contains("each region") || lowerMsg.Contains("region");
                    var displayRows = isAqiCategoryGroupByQuestion
                        ? GroupAqiCategoryRows(sectorCatResult.Rows, wantsByRegion)
                        : sectorCatResult.Rows;

                    var sectorCatSummary = displayRows.Count == 0
                        ? BuildNoDataMessage(sectorCatTemplate.Id, scope,
                            sectorCatFill.Parameters.GetValueOrDefault("aqiCategory"),
                            sectorCatFill.Parameters.GetValueOrDefault("sectorName"))
                        : BuildAqiCategorySummary(displayRows, request.Message, lowerMsg, isAqiCategoryGroupByQuestion);
                    var sectorCatMsgId = Guid.NewGuid();
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId,
                        userContent: request.Message,
                        assistantMessageId: sectorCatMsgId,
                        assistantContent: sectorCatSummary,
                        sql: null,
                        responseType: displayRows.Count == 0 ? "text" : sectorCatTemplate.ResponseType,
                        chartType: null,
                        chartDataJson: SerializeRowsForHistory(sectorCatTemplate.ResponseType, displayRows),
                        dataSource: sectorCatTemplate.TableUsed,
                        dateRange: null,
                        modelUsed: sectorCatFill.ModelUsed,
                        executionTimeMs: (int)sectorCatResult.ExecutionTimeMs,
                        tokenCount: sectorCatFill.TokensUsed,
                        ct: ct);
                    return Ok(_formatter.Format(session.SessionId, sectorCatMsgId, sectorCatTemplate,
                        sectorCatSummary, displayRows, displayRows.Count,
                        (int)sectorCatResult.ExecutionTimeMs, scope, sectorCatFill.ModelUsed, sectorCatFill.TokensUsed));
                }
                _log.Warning("ChatController: schools-by-AQI-category pre-check API failed, falling through to routing");
            }
        }

        // ── 2n. Hard pre-check: ambient station query → force ambient_station_readings ──
        // Fires when isAmbientStationQuery=true (outdoor/ambient station + site reference).
        // Extracts stationName via LLM, falls back to regex, and guards against missing stationName.
        if (isAmbientStationQuery)
        {
            var ambTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "ambient_station_readings");
            if (ambTemplate is not null && !string.IsNullOrWhiteSpace(ambTemplate.Sql))
            {
                var ambFill = await _azureAI.FillParametersAsync(request.Message, ambTemplate, history, scope, ct);

                // If the LLM couldn't extract stationName, try regex fallback before calling SQL.
                if (!ambFill.Parameters.ContainsKey("stationName") || string.IsNullOrWhiteSpace(ambFill.Parameters.GetValueOrDefault("stationName")))
                {
                    var ambStMatch = System.Text.RegularExpressions.Regex.Match(
                        request.Message,
                        @"\b(?:at|for|of|about|from)\s+(?:the\s+)?([A-Z][a-zA-Z]{1,}(?:\s+[A-Za-z]{2,}){0,5})(?:\s+(?:outdoor|ambient|station|readings?|pm|aqi|now|today|current)|\?|$)",
                        System.Text.RegularExpressions.RegexOptions.IgnoreCase);
                    if (ambStMatch.Success)
                        ambFill.Parameters["stationName"] = ambStMatch.Groups[1].Value.Trim();
                }

                // Guard: if stationName is still missing, fall through to normal routing rather than crash.
                if (!ambFill.Parameters.ContainsKey("stationName") || string.IsNullOrWhiteSpace(ambFill.Parameters.GetValueOrDefault("stationName")))
                {
                    _log.Warning("ChatController: ambient_station_readings skipped — stationName not extracted from '{Msg}'", request.Message);
                    goto skipAmbientStationPreCheck;
                }

                var (ambSafeSql, ambSqlParams) = _rbac.BuildSafeQuery(ambTemplate, ambFill.Parameters, user, scope);
                var (ambSqlValid, ambSqlError) = SqlValidator.Validate(ambSafeSql);
                if (ambSqlValid)
                {
                    var ambResult = await _sqlExecutor.ExecuteAsync(ambSafeSql, ambSqlParams, ct);
                    if (ambResult.Success)
                    {
                        var ambMsgId = Guid.NewGuid();
                        var ambSummary = await _azureAI.SummarizeResultsAsync(request.Message, ambTemplate, ambResult.Rows, scope, ct);
                        await _sessionService.SaveBothMessagesAsync(
                            sessionId: session.SessionId,
                            userContent: request.Message,
                            assistantMessageId: ambMsgId,
                            assistantContent: ambSummary,
                            sql: ambSafeSql,
                            responseType: ambTemplate.ResponseType,
                            chartType: null,
                            chartDataJson: SerializeRowsForHistory(ambTemplate.ResponseType, ambResult.Rows),
                            dataSource: ambTemplate.TableUsed,
                            dateRange: null,
                            modelUsed: ambFill.ModelUsed,
                            executionTimeMs: (int)ambResult.ExecutionTimeMs,
                            tokenCount: ambFill.TokensUsed,
                            ct: ct);
                        if (session.MessageCount <= 1)
                            _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                        return Ok(_formatter.Format(
                            sessionId: session.SessionId,
                            messageId: ambMsgId,
                            template: ambTemplate,
                            llmSummary: ambSummary,
                            rows: ambResult.Rows,
                            rowCount: ambResult.Rows.Count,
                            executionTimeMs: (int)ambResult.ExecutionTimeMs,
                            scope: scope,
                            modelUsed: ambFill.ModelUsed,
                            tokenCount: ambFill.TokensUsed));
                    }
                    _log.Warning("ChatController: ambient_station_readings SQL execution failed: {Error}", ambResult.Error);
                }
                else
                {
                    _log.Warning("ChatController: ambient_station_readings SQL validation failed: {Error}", ambSqlError);
                }
            }
            skipAmbientStationPreCheck: ;
        }

        // ── 2o. Hard pre-check: indoor vs ambient comparison → force indoor_ambient_compare ──
        // "Compare the current indoor AQI at X with the nearest ambient station" was being mis-routed
        // to ambient_station_readings (because "ambient station" matched isAmbientStationQuery).
        // Now caught here and routed correctly before the main router runs.
        if (isIndoorAmbientCompareQuery && hasAtSiteRef)
        {
            _log.Debug("ChatController: indoor-ambient-compare pre-check fired → forcing indoor_ambient_compare");
            var iacTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct)).FirstOrDefault(t => t.Id == "indoor_ambient_compare");
            if (iacTemplate?.ApiCall is not null)
            {
                var iacBearer = Request.Headers.Authorization.ToString().Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);
                var iacFill = await _azureAI.FillParametersAsync(request.Message, iacTemplate, history, scope, ct);
                var iacApiResult = await _externalApi.CallAsync(iacTemplate.ApiCall, iacFill.Parameters, scope, iacBearer, user, iacTemplate.Id, ct);
                if (iacApiResult.Success)
                {
                    var iacSummary = await _azureAI.SummarizeResultsAsync(request.Message, iacTemplate, iacApiResult.Rows, scope, ct);
                    var iacMsgId = Guid.NewGuid();
                    await _sessionService.SaveBothMessagesAsync(
                        sessionId: session.SessionId,
                        userContent: request.Message,
                        assistantMessageId: iacMsgId,
                        assistantContent: iacSummary,
                        sql: null,
                        responseType: iacTemplate.ResponseType,
                        chartType: null,
                        chartDataJson: SerializeRowsForHistory(iacTemplate.ResponseType, iacApiResult.Rows),
                        dataSource: iacTemplate.TableUsed,
                        dateRange: null,
                        modelUsed: iacFill.ModelUsed,
                        executionTimeMs: (int)iacApiResult.ExecutionTimeMs,
                        tokenCount: iacFill.TokensUsed,
                        ct: ct);
                    if (session.MessageCount <= 1)
                        _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                    return Ok(_formatter.Format(
                        sessionId: session.SessionId,
                        messageId: iacMsgId,
                        template: iacTemplate,
                        llmSummary: iacSummary,
                        rows: iacApiResult.Rows,
                        rowCount: iacApiResult.Rows.Count,
                        executionTimeMs: (int)iacApiResult.ExecutionTimeMs,
                        scope: scope,
                        modelUsed: iacFill.ModelUsed,
                        tokenCount: iacFill.TokensUsed));
                }
                _log.Warning("ChatController: indoor_ambient_compare pre-check API failed, falling through to routing");
            }
        }

        // ── 2z. Hard pre-check: compound multi-region AQI-category question ──
        // Detects questions like "Al Ain schools with Unhealthy AQI and Abu Dhabi schools with Good AQI"
        // which contain two separate region+category queries joined by "and".
        // Each sub-question is run independently through site_aqi_by_category and results are combined.
        {
            var regionWords = new[] { "abu dhabi", "abudhabi", "al ain", "alain", "al dhafra", "dhafra" };
            var aqiCategoryKeywords = new[] { "good", "moderate", "unhealthy", "hazardous", "very unhealthy" };
            var schoolSectorWords   = new[] { "school", "schools", "residential", "commercial" };

            int regionCount  = regionWords.Count(r => lowerMsg.Contains(r));
            int categoryCount = aqiCategoryKeywords.Count(c => lowerMsg.Contains(c));
            bool hasSectorContext = schoolSectorWords.Any(s => lowerMsg.Contains(s));

            // Only trigger when two distinct regions AND two distinct AQI categories appear in one message
            bool isCompoundMultiQuery = regionCount >= 2 && categoryCount >= 2 && hasSectorContext
                && (lowerMsg.Contains(" and ") || lowerMsg.Contains("&"));

            if (isCompoundMultiQuery)
            {
                _log.Information("ChatController: compound-multi-query pre-check fired — splitting into sub-queries");

                // Split on " and " to get two sub-questions
                // Use a broad split that handles patterns like "Al Ain schools Unhealthy AQI and Abu Dhabi schools Good AQI"
                var parts = System.Text.RegularExpressions.Regex.Split(
                    enrichedMessage,
                    @"\s+and\s+",
                    System.Text.RegularExpressions.RegexOptions.IgnoreCase);

                if (parts.Length >= 2)
                {
                    var multiTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct))
                        .FirstOrDefault(t => t.Id == "site_aqi_by_category");

                    if (multiTemplate is not null && multiTemplate.ApiCall is not null)
                    {
                        var multiBearer = Request.Headers.Authorization.ToString()
                            .Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);

                        var combinedRows    = new List<Dictionary<string, object?>>();
                        var summaryParts    = new List<string>();
                        int totalExecMs     = 0;
                        int totalTokens     = 0;

                        string? firstModelUsed = null;

                        foreach (var part in parts)
                        {
                            var partLower = part.ToLowerInvariant();
                            // Skip parts that don't contain at least a region or category
                            bool partHasRegion    = regionWords.Any(r => partLower.Contains(r));
                            bool partHasCategory  = aqiCategoryKeywords.Any(c => partLower.Contains(c));
                            if (!partHasRegion && !partHasCategory) continue;

                            // Build a self-contained sub-question for this part
                            // If the part is missing a sector context, inherit from full message
                            string subQuestion = part.Trim();
                            if (!schoolSectorWords.Any(s => partLower.Contains(s)))
                            {
                                // Inherit sector from original message
                                if (lowerMsg.Contains("school"))      subQuestion = "schools " + subQuestion;
                                else if (lowerMsg.Contains("residential")) subQuestion = "residential sites " + subQuestion;
                                else if (lowerMsg.Contains("commercial"))  subQuestion = "commercial sites " + subQuestion;
                            }

                            var partFill = await _azureAI.FillParametersAsync(subQuestion, multiTemplate, null, scope, ct);
                            firstModelUsed ??= partFill.ModelUsed;
                            totalTokens += partFill.TokensUsed;

                            // Force sector from full message if LLM missed it
                            if (!partFill.Parameters.ContainsKey("sectorName") || string.IsNullOrWhiteSpace(partFill.Parameters.GetValueOrDefault("sectorName")))
                            {
                                if (lowerMsg.Contains("school") || lowerMsg.Contains("moe") || lowerMsg.Contains("gov"))
                                    partFill.Parameters["sectorName"] = "Public & Govt-School";
                                else if (lowerMsg.Contains("residential"))
                                    partFill.Parameters["sectorName"] = "Residential";
                                else if (lowerMsg.Contains("commercial"))
                                    partFill.Parameters["sectorName"] = "Commercial";
                            }

                            // Force AQI category from part keywords if LLM missed it
                            if (!partFill.Parameters.ContainsKey("aqiCategory") || string.IsNullOrWhiteSpace(partFill.Parameters.GetValueOrDefault("aqiCategory")))
                            {
                                if (partLower.Contains("very unhealthy"))       partFill.Parameters["aqiCategory"] = "Very Unhealthy";
                                else if (partLower.Contains("unhealthy"))        partFill.Parameters["aqiCategory"] = "Unhealthy";
                                else if (partLower.Contains("hazardous"))        partFill.Parameters["aqiCategory"] = "Hazardous";
                                else if (partLower.Contains("moderate"))         partFill.Parameters["aqiCategory"] = "Moderate";
                                else if (partLower.Contains("good"))             partFill.Parameters["aqiCategory"] = "Good";
                            }

                            // Force region from part keywords if LLM missed it
                            if (!partFill.Parameters.ContainsKey("regionName") || string.IsNullOrWhiteSpace(partFill.Parameters.GetValueOrDefault("regionName")))
                            {
                                if (partLower.Contains("abu dhabi") || partLower.Contains("abudhabi"))
                                    partFill.Parameters["regionName"] = "Abu Dhabi";
                                else if (partLower.Contains("al ain") || partLower.Contains("alain"))
                                    partFill.Parameters["regionName"] = "Al Ain";
                                else if (partLower.Contains("al dhafra") || partLower.Contains("dhafra"))
                                    partFill.Parameters["regionName"] = "Al Dhafra";
                            }

                            var partResult = await _externalApi.CallAsync(multiTemplate.ApiCall, partFill.Parameters, scope, multiBearer, user, multiTemplate.Id, ct);
                            totalExecMs += (int)partResult.ExecutionTimeMs;

                            if (partResult.Success && partResult.Rows.Count > 0)
                            {
                                combinedRows.AddRange(partResult.Rows);
                            }
                            else
                            {
                                var subRegion   = partFill.Parameters.GetValueOrDefault("regionName") ?? "";
                                var subSector   = partFill.Parameters.GetValueOrDefault("sectorName") ?? "sites";
                                var subCategory = partFill.Parameters.GetValueOrDefault("aqiCategory") ?? "";
                                var noResultMsg = !string.IsNullOrWhiteSpace(subRegion)
                                    ? $"No {subSector} in {subRegion} currently have {subCategory} air quality."
                                    : $"No {subSector} currently have {subCategory} air quality.";
                                summaryParts.Add(noResultMsg);
                            }
                        }

                        if (combinedRows.Count > 0 || summaryParts.Count > 0)
                        {
                            string multiSummary;
                            if (combinedRows.Count > 0)
                            {
                                multiSummary = await _azureAI.SummarizeResultsAsync(request.Message, multiTemplate, combinedRows, scope, ct);
                                if (summaryParts.Count > 0)
                                    multiSummary = multiSummary + " " + string.Join(" ", summaryParts);
                            }
                            else
                            {
                                multiSummary = string.Join(" ", summaryParts);
                            }

                            var multiMsgId = Guid.NewGuid();
                            await _sessionService.SaveBothMessagesAsync(
                                sessionId: session.SessionId,
                                userContent: request.Message,
                                assistantMessageId: multiMsgId,
                                assistantContent: multiSummary,
                                sql: null,
                                responseType: multiTemplate.ResponseType,
                                chartType: null,
                                chartDataJson: SerializeRowsForHistory(multiTemplate.ResponseType, combinedRows),
                                dataSource: multiTemplate.TableUsed,
                                dateRange: null,
                                modelUsed: firstModelUsed,
                                executionTimeMs: totalExecMs,
                                tokenCount: totalTokens,
                                ct: ct);
                            if (session.MessageCount <= 1)
                                _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                            return Ok(_formatter.Format(
                                sessionId: session.SessionId,
                                messageId: multiMsgId,
                                template: multiTemplate,
                                llmSummary: multiSummary,
                                rows: combinedRows,
                                rowCount: combinedRows.Count,
                                executionTimeMs: totalExecMs,
                                scope: scope,
                                modelUsed: firstModelUsed,
                                tokenCount: totalTokens));
                        }
                        _log.Warning("ChatController: compound-multi-query pre-check returned no rows — falling through to routing");
                    }
                }
            }
        }

        // ── 2z2. Hard pre-check: compound multi-pollutant highest/lowest question ──
        // Detects questions that ask for highest/lowest for two or more different pollutants in one message,
        // e.g. "which site has the highest PM2.5 and lowest CO?" or "highest PM2.5 and lowest PM10 and highest CO2".
        // Each sub-question is run against highest_readings or lowest_readings independently and results combined.
        {
            var pollutantNames = new[] { "pm2.5", "pm10", "co2", "co ", " co?", "no2", "o3", "so2", "tvoc", "ch2o", "aqi" };
            var highestWords   = new[] { "highest", "worst", "most polluted", "maximum", "top" };
            var lowestWords    = new[] { "lowest", "cleanest", "minimum", "least polluted", "best" };

            int pollutantCount = pollutantNames.Count(p => lowerMsg.Contains(p));
            bool hasHighest    = highestWords.Any(w => lowerMsg.Contains(w));
            bool hasLowest     = lowestWords.Any(w => lowerMsg.Contains(w));

            // Trigger when: 2+ pollutants AND both highest+lowest, OR 2+ pollutants AND one direction
            // AND the message contains "and" (indicating compound question)
            bool isMultiPollutant = pollutantCount >= 2 && (lowerMsg.Contains(" and ") || lowerMsg.Contains("&"))
                && (hasHighest || hasLowest);

            if (isMultiPollutant)
            {
                _log.Information("ChatController: multi-pollutant pre-check fired for message: {Msg}", enrichedMessage);

                // Split on " and " to get sub-questions
                var mpParts = System.Text.RegularExpressions.Regex.Split(
                    enrichedMessage,
                    @"\s+and\s+",
                    System.Text.RegularExpressions.RegexOptions.IgnoreCase);

                var allTemplates = await _queryRouter.GetPermittedTemplatesAsync(user, ct);
                var highTemplate = allTemplates.FirstOrDefault(t => t.Id == "highest_readings");
                var lowTemplate  = allTemplates.FirstOrDefault(t => t.Id == "lowest_readings");

                if (highTemplate?.ApiCall is not null && lowTemplate?.ApiCall is not null)
                {
                    var mpBearer = Request.Headers.Authorization.ToString()
                        .Replace("Bearer ", "", StringComparison.OrdinalIgnoreCase);

                    var mpCombinedRows = new List<Dictionary<string, object?>>();
                    var mpSummaryParts = new List<string>();
                    int mpTotalExecMs  = 0;
                    int mpTotalTokens  = 0;
                    string? mpFirstModel = null;
                    ApprovedQuery? mpLastTemplate = null;

                    foreach (var part in mpParts)
                    {
                        var partLower = part.ToLowerInvariant();
                        bool partHasPollutant = pollutantNames.Any(p => partLower.Contains(p));
                        if (!partHasPollutant) continue;

                        // Determine direction for this part
                        bool partIsHighest = highestWords.Any(w => partLower.Contains(w));
                        bool partIsLowest  = lowestWords.Any(w => partLower.Contains(w));

                        // If the part has no direction keyword, inherit from the full message
                        if (!partIsHighest && !partIsLowest)
                        {
                            partIsHighest = hasHighest && !hasLowest;
                            partIsLowest  = hasLowest && !hasHighest;
                            if (!partIsHighest && !partIsLowest) partIsHighest = true; // default: highest
                        }

                        var chosenTemplate = partIsLowest && !partIsHighest ? lowTemplate : highTemplate;
                        mpLastTemplate = chosenTemplate;

                        // Build sub-question with direction context
                        string subQ = part.Trim();
                        if (!partIsHighest && !partIsLowest) subQ = "which site has the highest " + subQ;

                        var mpFill = await _azureAI.FillParametersAsync(subQ, chosenTemplate, null, scope, ct);
                        mpFirstModel ??= mpFill.ModelUsed;
                        mpTotalTokens += mpFill.TokensUsed;

                        // Force sortOrder based on direction if LLM didn't set it
                        if (!mpFill.Parameters.ContainsKey("sortOrder") || string.IsNullOrWhiteSpace(mpFill.Parameters.GetValueOrDefault("sortOrder")))
                            mpFill.Parameters["sortOrder"] = partIsLowest && !partIsHighest ? "asc" : "desc";

                        // Force topN=1 if not set (single highest/lowest)
                        if (!mpFill.Parameters.ContainsKey("topN") || string.IsNullOrWhiteSpace(mpFill.Parameters.GetValueOrDefault("topN")))
                            mpFill.Parameters["topN"] = "1";

                        // Extract pollutant from the part if LLM missed it
                        if (!mpFill.Parameters.ContainsKey("parameterName") || string.IsNullOrWhiteSpace(mpFill.Parameters.GetValueOrDefault("parameterName")))
                        {
                            foreach (var p in new[] { "PM2.5", "PM10", "CO2", "CO", "NO2", "O3", "SO2", "TVOC", "CH2O", "AQI Index" })
                            {
                                if (partLower.Contains(p.ToLowerInvariant()) || partLower.Contains(p.Replace(".", "").ToLowerInvariant()))
                                {
                                    mpFill.Parameters["parameterName"] = p;
                                    break;
                                }
                            }
                        }

                        var mpResult = await _externalApi.CallAsync(chosenTemplate.ApiCall, mpFill.Parameters, scope, mpBearer, user, chosenTemplate.Id, ct);
                        mpTotalExecMs += (int)mpResult.ExecutionTimeMs;

                        if (mpResult.Success && mpResult.Rows.Count > 0)
                        {
                            // Tag each row with direction context
                            foreach (var row in mpResult.Rows)
                                row["_direction"] = partIsLowest && !partIsHighest ? "lowest" : "highest";
                            mpCombinedRows.AddRange(mpResult.Rows);
                        }
                        else
                        {
                            var mpParam = mpFill.Parameters.GetValueOrDefault("parameterName") ?? "pollutant";
                            mpSummaryParts.Add($"No live data found for {mpParam}.");
                        }
                    }

                    if (mpCombinedRows.Count > 0 || mpSummaryParts.Count > 0)
                    {
                        // Remove internal direction tag before display
                        foreach (var row in mpCombinedRows) row.Remove("_direction");

                        var mpFinalTemplate = mpLastTemplate ?? highTemplate;
                        var mpSummary = mpCombinedRows.Count > 0
                            ? await _azureAI.SummarizeResultsAsync(request.Message, mpFinalTemplate, mpCombinedRows, scope, ct)
                            : string.Join(" ", mpSummaryParts);
                        if (mpSummaryParts.Count > 0 && mpCombinedRows.Count > 0)
                            mpSummary += " " + string.Join(" ", mpSummaryParts);

                        var mpMsgId = Guid.NewGuid();
                        await _sessionService.SaveBothMessagesAsync(
                            sessionId: session.SessionId,
                            userContent: request.Message,
                            assistantMessageId: mpMsgId,
                            assistantContent: mpSummary,
                            sql: null,
                            responseType: mpFinalTemplate.ResponseType,
                            chartType: null,
                            chartDataJson: SerializeRowsForHistory(mpFinalTemplate.ResponseType, mpCombinedRows),
                            dataSource: mpFinalTemplate.TableUsed,
                            dateRange: null,
                            modelUsed: mpFirstModel,
                            executionTimeMs: mpTotalExecMs,
                            tokenCount: mpTotalTokens,
                            ct: ct);
                        if (session.MessageCount <= 1)
                            _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                        return Ok(_formatter.Format(
                            sessionId: session.SessionId,
                            messageId: mpMsgId,
                            template: mpFinalTemplate,
                            llmSummary: mpSummary,
                            rows: mpCombinedRows,
                            rowCount: mpCombinedRows.Count,
                            executionTimeMs: mpTotalExecMs,
                            scope: scope,
                            modelUsed: mpFirstModel,
                            tokenCount: mpTotalTokens));
                    }
                    _log.Warning("ChatController: multi-pollutant pre-check returned no rows — falling through to routing");
                }
            }
        }

        // ── 3. Route the question to an approved template ───────────────────
        var routeResult = await _queryRouter.RouteAsync(enrichedMessage, user, history, ct);

        if (routeResult.NeedsClarification)
        {
            await _sessionService.SaveUserMessageAsync(session.SessionId, request.Message, ct);

            // Detect if the user is asking about a pollutant/parameter with no site/device mentioned.
            // Route to schools_latest_pollutant for all-sites reading instead of returning an error.
            var pollutantNames = new[] {
                "pm2.5", "pm10", "co2", "co", "no2", "so2", "o3", "ozone", "tvoc", "voc", "ch2o",
                "formaldehyde", "temperature", "humidity", "noise", "aqi", "particulate"
            };
            // Map aliases to canonical parameter names the template understands
            var pollutantAliasMap = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["ozone"] = "O3", ["o3"] = "O3", ["pm2.5"] = "PM2.5", ["pm10"] = "PM10",
                ["co2"] = "CO2", ["no2"] = "NO2", ["so2"] = "SO2", ["tvoc"] = "TVOC",
                ["voc"] = "VOC", ["ch2o"] = "CH2O", ["formaldehyde"] = "CH2O",
                ["temperature"] = "Temperature", ["humidity"] = "Humidity",
                ["noise"] = "Noise", ["co"] = "CO", ["aqi"] = "AQI Index"
            };
            bool looksLikePollutantQuestion = pollutantNames.Any(p => lowerMsg.Contains(p))
                && (lowerMsg.Contains("reading") || lowerMsg.Contains("current") || lowerMsg.Contains("level")
                    || lowerMsg.Contains("value") || lowerMsg.Contains("now") || lowerMsg.Contains("latest")
                    || lowerMsg.Contains("all sites") || lowerMsg.Contains("all the sites"));

            if (looksLikePollutantQuestion)
            {
                // Find matched pollutant and map to canonical name
                var matchedAlias = pollutantNames.FirstOrDefault(p => lowerMsg.Contains(p)) ?? "AQI Index";
                var canonicalParam = pollutantAliasMap.TryGetValue(matchedAlias, out var cp) ? cp : "AQI Index";

                var allSitesPollTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct))
                    .FirstOrDefault(t => t.Id == "all_sites_latest_pollutant");
                if (allSitesPollTemplate is not null)
                {
                    var pollParams = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                    {
                        ["parameterName"] = canonicalParam
                    };
                    var (pollSql, pollSqlParams) = _rbac.BuildSafeQuery(allSitesPollTemplate, pollParams, user, scope);
                    var (pollValid, _) = SqlValidator.Validate(pollSql);
                    if (pollValid)
                    {
                        var pollResult = await _sqlExecutor.ExecuteAsync(pollSql, pollSqlParams, ct);
                        if (pollResult.Success && pollResult.Rows.Count > 0)
                        {
                            var pollMsgId = Guid.NewGuid();
                            var pollSummary = await _azureAI.SummarizeResultsAsync(request.Message, allSitesPollTemplate, pollResult.Rows, scope, ct);
                            await _sessionService.SaveBothMessagesAsync(
                                sessionId: session.SessionId,
                                userContent: request.Message,
                                assistantMessageId: pollMsgId,
                                assistantContent: pollSummary,
                                sql: pollSql,
                                responseType: allSitesPollTemplate.ResponseType,
                                chartType: null,
                                chartDataJson: SerializeRowsForHistory(allSitesPollTemplate.ResponseType, pollResult.Rows),
                                dataSource: allSitesPollTemplate.TableUsed,
                                dateRange: null,
                                modelUsed: null,
                                executionTimeMs: (int)pollResult.ExecutionTimeMs,
                                tokenCount: 0,
                                ct: ct);
                            if (session.MessageCount <= 1)
                                _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                            return Ok(_formatter.Format(
                                sessionId: session.SessionId,
                                messageId: pollMsgId,
                                template: allSitesPollTemplate,
                                llmSummary: pollSummary,
                                rows: pollResult.Rows,
                                rowCount: pollResult.Rows.Count,
                                executionTimeMs: (int)pollResult.ExecutionTimeMs,
                                scope: scope,
                                modelUsed: null,
                                tokenCount: 0));
                        }
                    }
                }
            }

            string clarifyMsg = looksLikePollutantQuestion
                ? $"Which site or device are you asking about? For example: 'current {(pollutantNames.FirstOrDefault(p => lowerMsg.Contains(p)) ?? "PM2.5").ToUpperInvariant()} reading at Al Reyada School' or 'current reading for BA 0001'."
                : "I'm sorry — that's outside my scope. I can only help with HAWAQM air-quality data. If it's useful, you could ask me: 'Which devices are offline right now?' or 'What is the current AQI at my school?'";

            return Ok(_formatter.FormatError(
                session.SessionId,
                clarifyMsg,
                "CLARIFY"));
        }

        // ── 3b. Hard override: device name + date/time reference → device_reading_history ──
        // The pattern scorer matches "reading" → device_live_reading with high confidence,
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

        // ── 3b2. Hard override: specific device + status question → device_status ──
        // Catches "is device SEI100M 0009 active or inactive?" being mis-routed to offline_devices.
        var deviceStatusKeywords = new[] { "active or inactive", "inactive or active", "online or offline",
            "offline or online", "is it active", "is it inactive", "is it online", "is it offline",
            "active right now", "inactive right now", "currently active", "currently inactive",
            "currently online", "currently offline", "is active", "is inactive", "is online", "is offline",
            "device status", "status of device" };
        if (msgHasDeviceCode && deviceStatusKeywords.Any(k => lowerMsg.Contains(k))
            && routeResult.Template?.Id is not "device_status")
        {
            var devStatusTemplate = _queryRouter.GetPermittedTemplates(user)
                .FirstOrDefault(t => t.Id == "device_status");
            if (devStatusTemplate is not null)
            {
                _log.Information("ChatController: overriding '{From}' → device_status (device code + status keyword detected)",
                    routeResult.Template?.Id ?? "candidates");
                routeResult = new QueryRouterResult
                {
                    Template = devStatusTemplate,
                    PreviousTemplateId = routeResult.PreviousTemplateId
                };
            }
        }

        // ── 3c. Hard override: named site + past date → site_daily_averages ──
        // Covers cases where pre-checks or the router still selected site_aqi_single despite a past date.
        // Must run AFTER device_reading_history override (3b) so device+date questions are unaffected.
        bool msgHasSiteRef = siteTypeWords.Any(s => lowerMsg.Contains(s)) || hasAtSitePattern;

        // Detect indoor-ambient-historical queries (compare indoor vs outdoor on a past date)
        var indoorHistoricalKeywords = new[]
        {
            "more likely coming from indoors", "more likely coming from outdoor",
            "indoor or outdoor source", "indoor vs outdoor", "indoor outdoor ratio",
            "from indoor sources", "from outdoor", "indoor source", "outdoor source",
            "from indoors or outdoor", "indoors or outdoor", "indoor or outdoor",
            "is it indoor or outdoor", "from outdoor pollution", "from indoor pollution",
            "due to outdoor pollution", "due to indoor sources", "due to outdoor sources",
            "outdoor pollution rather than indoor", "rather than indoor sources",
            "mainly due to outdoor", "mainly due to indoor",
            "outdoor dust contributor", "is outdoor dust",
            "respond during the dust event", "how did indoor air quality"
        };
        var dustKeywords = new[] { "dust event", "dust storm", "sandstorm" };
        bool isIndoorAmbientHistorical = hasAtSiteRef
            && indoorHistoricalKeywords.Any(k => lowerMsg.Contains(k))
            && (msgHasPastDate || dustKeywords.Any(k => lowerMsg.Contains(k)));

        if (msgHasPastDate && msgHasSiteRef && !msgHasDeviceCode && !isIndoorAmbientHistorical && !isAmbientStationQuery
            && routeResult.Template?.Id is "site_aqi_single" or "site_readings_all" or "main_pollutant_at_station"
                or "device_info" or null)
        {
            var dailyTemplate = _queryRouter.GetPermittedTemplates(user)
                .FirstOrDefault(t => t.Id == "site_daily_averages");
            if (dailyTemplate is not null)
            {
                _log.Information("ChatController: overriding '{From}' → site_daily_averages (site+past-date detected)",
                    routeResult.Template?.Id ?? "candidates");
                routeResult = new QueryRouterResult
                {
                    Template = dailyTemplate,
                    Score = 1.0,
                    Candidates = [],
                    PreviousTemplateId = routeResult.PreviousTemplateId
                };
            }
        }

        // ── 3c-month. Hard override: named site + month reference → site_monthly_averages ──
        // Fires for: "AQI at Al Reyada in March 2025", "average for January", "last month at X".
        // Month takes priority over year — must run BEFORE 3c-year.
        string? preExtractedMonth = null;
        string? preExtractedMonthYear = null;
        if (msgHasMonthRef && msgHasSiteRef && !msgHasDeviceCode && !isIndoorAmbientHistorical && !isAmbientStationQuery
            && routeResult.Template?.Id is "site_aqi_single" or "site_daily_averages" or "site_readings_all"
                or "main_pollutant_at_station" or "site_yearly_averages" or "device_info" or null)
        {
            var monthlyTemplate = _queryRouter.GetPermittedTemplates(user).FirstOrDefault(t => t.Id == "site_monthly_averages");
            if (monthlyTemplate is not null)
            {
                _log.Information("ChatController: overriding '{From}' → site_monthly_averages (month ref + site ref detected)",
                    routeResult.Template?.Id ?? "candidates");
                routeResult = new QueryRouterResult
                {
                    Template = monthlyTemplate,
                    Score = 1.0,
                    Candidates = [],
                    PreviousTemplateId = routeResult.PreviousTemplateId
                };

                // Extract month number from name
                var monthNames = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                {
                    ["january"] = "1", ["february"] = "2", ["march"] = "3", ["april"] = "4",
                    ["may"] = "5", ["june"] = "6", ["july"] = "7", ["august"] = "8",
                    ["september"] = "9", ["october"] = "10", ["november"] = "11", ["december"] = "12"
                };
                var monthMatch = System.Text.RegularExpressions.Regex.Match(lowerMsg,
                    @"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
                    System.Text.RegularExpressions.RegexOptions.IgnoreCase);
                if (monthMatch.Success)
                    preExtractedMonth = monthNames[monthMatch.Value.ToLowerInvariant()];

                // "last month" → derive month and year
                if (lowerMsg.Contains("last month") || lowerMsg.Contains("previous month"))
                {
                    var lastMonth = DateTime.Today.AddMonths(-1);
                    preExtractedMonth = lastMonth.Month.ToString();
                    preExtractedMonthYear = lastMonth.Year.ToString();
                }
                else if (lowerMsg.Contains("this month") || lowerMsg.Contains("current month"))
                {
                    preExtractedMonth = DateTime.Today.Month.ToString();
                    preExtractedMonthYear = DateTime.Today.Year.ToString();
                }
                else
                {
                    // Extract year if present alongside month name (e.g. "March 2025")
                    var yearMatch2 = System.Text.RegularExpressions.Regex.Match(lowerMsg, @"\b(19|20)\d{2}\b");
                    if (yearMatch2.Success)
                        preExtractedMonthYear = yearMatch2.Value;
                }
            }
        }

        // ── 3c-year. Hard override: named site + specific year → site_yearly_averages ──
        string? preExtractedYear = null;
        if (msgHasYearRef && msgHasSiteRef && !msgHasDeviceCode && !isIndoorAmbientHistorical
            && routeResult.Template?.Id is not ("site_yearly_averages" or "site_yearly_averages_pay"
                or "site_aqi_trend_yearly" or "worst_month_pollutants" or "top_sites_aqi_yearly"
                or "bottom_sites_aqi_yearly" or "region_aqi_single_year" or "compare_devices_monthly"))
        {
            var yearlyTemplate = _queryRouter.GetPermittedTemplates(user).FirstOrDefault(t => t.Id == "site_yearly_averages");
            if (yearlyTemplate is not null)
            {
                _log.Information("ChatController: overriding '{From}' → site_yearly_averages (year ref + site ref detected)",
                    routeResult.Template?.Id ?? "candidates");
                routeResult = new QueryRouterResult
                {
                    Template = yearlyTemplate,
                    Score = 1.0,
                    Candidates = [],
                    PreviousTemplateId = routeResult.PreviousTemplateId
                };
                var yearMatch = System.Text.RegularExpressions.Regex.Match(lowerMsg, @"\b(19|20)\d{2}\b");
                if (yearMatch.Success && int.TryParse(yearMatch.Value, out var extractedYear))
                    preExtractedYear = extractedYear.ToString();
            }
        }

        // ── 3d. Hard override: indoor/ambient historical query mis-routed → indoor_source_analysis ──
        if (isIndoorAmbientHistorical
            && routeResult.Template?.Id is "site_daily_averages" or "site_aqi_single" or "site_readings_all" or null)
        {
            var sourceTemplate = _queryRouter.GetPermittedTemplates(user)
                .FirstOrDefault(t => t.Id == "indoor_source_analysis");
            if (sourceTemplate is not null)
            {
                _log.Information("ChatController: overriding '{From}' → indoor_source_analysis (indoor-ambient-historical detected)",
                    routeResult.Template?.Id ?? "candidates");
                routeResult = new QueryRouterResult
                {
                    Template = sourceTemplate,
                    Score = 1.0,
                    Candidates = [],
                    PreviousTemplateId = routeResult.PreviousTemplateId
                };
            }
        }

        // ── 3g. Hard override: pollutant + no site name → all_sites_latest_pollutant ──
        // Catches "current CO2 reading", "show me the current O3 reading", "ozone reading" etc.
        // when no specific site or device is mentioned — return readings across all sites.
        {
            var pollutantTokens3g = new[] {
                "pm2.5", "pm10", "co2", "no2", "so2", "o3", "ozone", "tvoc", "voc", "ch2o",
                "formaldehyde", "temperature", "humidity", "noise", "co"
            };
            var pollutantAliasMap3g = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["ozone"] = "O3", ["o3"] = "O3", ["pm2.5"] = "PM2.5", ["pm10"] = "PM10",
                ["co2"] = "CO2", ["no2"] = "NO2", ["so2"] = "SO2", ["tvoc"] = "TVOC",
                ["voc"] = "VOC", ["ch2o"] = "CH2O", ["formaldehyde"] = "CH2O",
                ["temperature"] = "Temperature", ["humidity"] = "Humidity",
                ["noise"] = "Noise", ["co"] = "CO"
            };
            // "co" needs word-boundary check to avoid matching "commercial", "current", etc.
            bool hasPollutantToken = pollutantTokens3g.Where(p => p != "co").Any(p => lowerMsg.Contains(p))
                || System.Text.RegularExpressions.Regex.IsMatch(lowerMsg, @"\bco\b");
            bool hasPollutantReadingWord = lowerMsg.Contains("reading") || lowerMsg.Contains("level")
                || lowerMsg.Contains("value") || lowerMsg.Contains("current") || lowerMsg.Contains("latest")
                || lowerMsg.Contains("show me") || lowerMsg.Contains("what is") || lowerMsg.Contains("what are")
                || lowerMsg.Contains("display") || lowerMsg.Contains("high") || lowerMsg.Contains("list sites where")
                || lowerMsg.Contains("sites where") || lowerMsg.Contains("where") && lowerMsg.Contains("level");
            // Detect region filter in message
            string? detectedRegion3g = null;
            if (lowerMsg.Contains("al ain") || lowerMsg.Contains("alain"))
                detectedRegion3g = "Al Ain";
            else if (lowerMsg.Contains("abu dhabi") || lowerMsg.Contains("abudhabi"))
                detectedRegion3g = "Abu Dhabi";
            else if (lowerMsg.Contains("al dhafra") || lowerMsg.Contains("dhafra"))
                detectedRegion3g = "Al Dhafra";

            // Also allow when query is a pollutant ranking/listing question (e.g. "list sites where CO2 is high")
            var pollutantQualityWords3g = new[] { "high", "elevated", "above", "exceeding", "bad", "poor", "low", "below" };
            bool isPollutantRankingQ = pollutantQualityWords3g.Any(w => lowerMsg.Contains(w))
                && (lowerMsg.Contains("where") || lowerMsg.Contains("list sites") || lowerMsg.Contains("display"));

            // Trigger when pollutant + reading word AND:
            //   • no device code, AND
            //   • no specific site ref OR it's a region reference (e.g. "Al Ain sites") OR it's a ranking query
            bool noSpecificSite = !msgHasSpecificDeviceCode && (!hasAtSiteRef || detectedRegion3g != null || isPollutantRankingQ);
            if (hasPollutantToken && hasPollutantReadingWord && noSpecificSite
                && routeResult.Template?.Id is not "all_sites_latest_pollutant")
            {
                var allPollTemplate = _queryRouter.GetPermittedTemplates(user)
                    .FirstOrDefault(t => t.Id == "all_sites_latest_pollutant");
                if (allPollTemplate is not null)
                {
                    var matchedToken = pollutantTokens3g.First(p => lowerMsg.Contains(p));
                    var canonicalParam3g = pollutantAliasMap3g.TryGetValue(matchedToken, out var cp3g) ? cp3g : matchedToken.ToUpperInvariant();
                    _log.Information("ChatController: overriding '{From}' → all_sites_latest_pollutant (pollutant={Param}, region={Region})",
                        routeResult.Template?.Id ?? "candidates", canonicalParam3g, detectedRegion3g ?? "all");
                    routeResult = new QueryRouterResult
                    {
                        Template = allPollTemplate,
                        Score = 1.0,
                        Candidates = [],
                        PreviousTemplateId = routeResult.PreviousTemplateId
                    };
                    // Pre-fill params — parameterName required, regionName optional
                    var allPollMsgId = Guid.NewGuid();
                    var allPollParams = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                    {
                        ["parameterName"] = canonicalParam3g
                    };
                    if (detectedRegion3g != null)
                        allPollParams["regionName"] = detectedRegion3g;
                    var (allPollSql, allPollSqlParams) = _rbac.BuildSafeQuery(allPollTemplate, allPollParams, user, scope);
                    var (allPollValid, _) = SqlValidator.Validate(allPollSql);
                    if (allPollValid)
                    {
                        var allPollResult = await _sqlExecutor.ExecuteAsync(allPollSql, allPollSqlParams, ct);
                        if (allPollResult.Success && allPollResult.Rows.Count > 0)
                        {
                            var allPollSummary = await _azureAI.SummarizeResultsAsync(request.Message, allPollTemplate, allPollResult.Rows, scope, ct);
                            await _sessionService.SaveBothMessagesAsync(
                                sessionId: session.SessionId,
                                userContent: request.Message,
                                assistantMessageId: allPollMsgId,
                                assistantContent: allPollSummary,
                                sql: allPollSql,
                                responseType: allPollTemplate.ResponseType,
                                chartType: null,
                                chartDataJson: SerializeRowsForHistory(allPollTemplate.ResponseType, allPollResult.Rows),
                                dataSource: allPollTemplate.TableUsed,
                                dateRange: null,
                                modelUsed: null,
                                executionTimeMs: (int)allPollResult.ExecutionTimeMs,
                                tokenCount: 0,
                                ct: ct);
                            if (session.MessageCount <= 1)
                                _ = _sessionService.UpdateSessionTitleAsync(session.SessionId, request.Message, CancellationToken.None);
                            return Ok(_formatter.Format(
                                sessionId: session.SessionId,
                                messageId: allPollMsgId,
                                template: allPollTemplate,
                                llmSummary: allPollSummary,
                                rows: allPollResult.Rows,
                                rowCount: allPollResult.Rows.Count,
                                executionTimeMs: (int)allPollResult.ExecutionTimeMs,
                                scope: scope,
                                modelUsed: null,
                                tokenCount: 0));
                        }
                    }
                }
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
                    "I can only answer questions about HAWAQM air-quality data. Try asking about a site, device, or pollutant — for example: 'Which devices are offline right now?' or 'What is the current AQI at my school?'",
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

        // ── Override: inactive/offline site queries → force liveStatus=Inactive ──
        // The LLM sometimes omits liveStatus when user asks "total inactive sites" or "how many sites are inactive".
        // Without this, BuildAqiCategoryRows defaults to "All" and drops null-AQI inactive sites.
        var inactiveKeywords = new[] { "inactive sites", "inactive schools", "offline sites", "offline schools",
            "total inactive", "how many inactive", "how many are inactive", "how many sites are inactive",
            "how many schools are inactive", "inactive site count", "sites that are inactive",
            "sites not active", "sites that are offline", "sites currently inactive", "sites currently offline",
            "what are the total inactive", "what are the inactive" };
        // Count/number intent must NOT be forced to sites_filtered (list) — it should use site_count (number)
        bool msgIsSiteCountIntent = (lowerMsg.Contains("count") || lowerMsg.Contains("how many")
            || lowerMsg.Contains("total") || lowerMsg.Contains("number of"))
            && (lowerMsg.Contains("site") || lowerMsg.Contains("station") || lowerMsg.Contains("school"));
        if (inactiveKeywords.Any(k => lowerMsg.Contains(k)) && !msgIsSiteCountIntent)
        {
            // Force sites_filtered — it uses site_list_with_status responseShape which sources from
            // DMN_Stations directly, so all inactive sites appear (including those with no devices
            // that are missing from GetAllSiteData). This matches the Site Overview UI count.
            var inactiveTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct))
                .FirstOrDefault(t => t.Id == "sites_filtered");
            if (inactiveTemplate is not null)
            {
                template = inactiveTemplate;
                var inactiveFill = await _azureAI.FillParametersAsync(enrichedMessage, inactiveTemplate, history, scope, ct);
                llmParams = inactiveFill.Parameters;
                llmParams["liveStatus"] = "Inactive";
                llmParams.Remove("aqiCategory");
                _log.Debug("ChatController: inactive-sites override → sites_filtered liveStatus=Inactive");
            }
        }

        // ── Override: DSR + AQI exceedance → force sites_repeated_aqi_exceedance_with_dsr ──
        // The LLM sometimes routes "schools that repeatedly exceeded AQI 100 with DSR ≥ 75%"
        // to sites_by_filter or schools_latest_pollutant. Correct it here after template selection.
        if (hasDsrExceedanceSignal && template.Id != "sites_repeated_aqi_exceedance_with_dsr")
        {
            var dsrExcTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct))
                .FirstOrDefault(t => t.Id == "sites_repeated_aqi_exceedance_with_dsr");
            if (dsrExcTemplate is not null)
            {
                _log.Debug("ChatController: DSR-exceedance override applied — switching from {From} to sites_repeated_aqi_exceedance_with_dsr", template.Id);
                template = dsrExcTemplate;
                var dsrReFill = await _azureAI.FillParametersAsync(enrichedMessage, dsrExcTemplate, history, scope, ct);
                llmParams = dsrReFill.Parameters;
                // Ensure defaults when LLM doesn't extract them
                if (!llmParams.ContainsKey("aqiThreshold")) llmParams["aqiThreshold"] = "100";
                if (!llmParams.ContainsKey("aqiOperator"))  llmParams["aqiOperator"]  = ">";
                if (!llmParams.ContainsKey("dsrThreshold")) llmParams["dsrThreshold"] = "75";
                // Sector: if question mentions "school" or "schools", scope to school sector
                if (!llmParams.ContainsKey("sectorName")
                    && (lowerMsg.Contains("school") || lowerMsg.Contains("schools")))
                    llmParams["sectorName"] = "Public & Govt-School";
            }
        }

        // ── Override: DSR site-list question → force sites_dsr_below_threshold or sites_dsr_above_threshold ──
        // "show me all school sites with a Data Success Rate below 75%" must NOT route to site_count.
        var dsrSiteListTemplates = new[] { "sites_dsr_below_threshold", "sites_dsr_above_threshold", "sites_repeated_aqi_exceedance_with_dsr" };
        if (hasDsrSiteListSignal && !dsrSiteListTemplates.Contains(template.Id))
        {
            // Pick below vs above based on question phrasing
            bool isAboveDsr = lowerMsg.Contains("above") || lowerMsg.Contains("at least") || lowerMsg.Contains("more than")
                || lowerMsg.Contains("greater than") || lowerMsg.Contains("higher than") || lowerMsg.Contains("over")
                || lowerMsg.Contains("good dsr") || lowerMsg.Contains("high dsr");
            var dsrSiteTemplateId = isAboveDsr ? "sites_dsr_above_threshold" : "sites_dsr_below_threshold";
            var dsrSiteTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct))
                .FirstOrDefault(t => t.Id == dsrSiteTemplateId)
                // fall back to below-threshold if above doesn't exist
                ?? (await _queryRouter.GetPermittedTemplatesAsync(user, ct))
                    .FirstOrDefault(t => t.Id == "sites_dsr_below_threshold");
            if (dsrSiteTemplate is not null)
            {
                _log.Debug("ChatController: DSR site-list override — switching from {From} to {To}", template.Id, dsrSiteTemplate.Id);
                template = dsrSiteTemplate;
                var dsrSiteFill = await _azureAI.FillParametersAsync(enrichedMessage, dsrSiteTemplate, history, scope, ct);
                llmParams = dsrSiteFill.Parameters;
                // Default dsrThreshold to 75 when LLM didn't extract it
                if (!llmParams.ContainsKey("dsrThreshold")) llmParams["dsrThreshold"] = "75";
                // Sector: scope to school sector when question mentions school
                if (!llmParams.ContainsKey("sectorName")
                    && (lowerMsg.Contains("school") || lowerMsg.Contains("schools")))
                    llmParams["sectorName"] = "Public & Govt-School";
            }
        }

        // ── Override: ambient station comparison → force indoor_ambient_compare ──
        // "Compare the current indoor AQI at Al Reyada School with the nearest ambient station"
        // often routes to site_aqi_single — detect and correct here.
        bool hasAmbientCompareSignal =
            (lowerMsg.Contains("ambient station") || lowerMsg.Contains("outdoor station") ||
             lowerMsg.Contains("nearest ambient") || lowerMsg.Contains("nearest outdoor") ||
             lowerMsg.Contains("ambient aqi") || lowerMsg.Contains("outdoor aqi") ||
             // plain "ambient" with a site ref and compare word is enough
             (lowerMsg.Contains("ambient") && hasAtSiteRef))
            && (lowerMsg.Contains("compare") || lowerMsg.Contains("vs") || lowerMsg.Contains("versus") ||
                lowerMsg.Contains("with the nearest") || lowerMsg.Contains("and the") ||
                lowerMsg.Contains("indoor") || lowerMsg.Contains("indoor aqi") ||
                // "what is the nearest ambient station at X" / "which ambient station is nearest to X"
                lowerMsg.Contains("nearest") || lowerMsg.Contains("what is the ambient") ||
                lowerMsg.Contains("which ambient") || lowerMsg.Contains("what ambient"));

        if (hasAmbientCompareSignal && template.Id != "indoor_ambient_compare")
        {
            var ambientTemplate = (await _queryRouter.GetPermittedTemplatesAsync(user, ct))
                .FirstOrDefault(t => t.Id == "indoor_ambient_compare");
            if (ambientTemplate is not null)
            {
                _log.Debug("ChatController: ambient-compare override — switching from {From} to indoor_ambient_compare", template.Id);
                template = ambientTemplate;
                var ambientFill = await _azureAI.FillParametersAsync(enrichedMessage, ambientTemplate, history, scope, ct);
                llmParams = ambientFill.Parameters;
            }
        }

        _log.Debug("ChatController: template={Template}, llmParams={Params}",
            template.Id, string.Join(", ", llmParams.Select(kv => $"{kv.Key}={kv.Value}")));

        // Hard-override sectorName for devices_by_filter when LLM missed it from sector keywords
        if (template.Id is "devices_by_filter" or "count_devices_by_filter"
            && string.IsNullOrWhiteSpace(llmParams.GetValueOrDefault("sectorName"))
            && string.IsNullOrWhiteSpace(llmParams.GetValueOrDefault("stationName")))
        {
            if (lowerMsg.Contains("school site") || lowerMsg.Contains("school sites")
                || (lowerMsg.Contains("school") && !lowerMsg.Contains(" at ") && !lowerMsg.Contains(" for ")))
                llmParams["sectorName"] = "Public & Govt-School";
            else if (lowerMsg.Contains("commercial"))
                llmParams["sectorName"] = "Commercial";
            else if (lowerMsg.Contains("residential"))
                llmParams["sectorName"] = "Residential";
        }

        // Inject pre-extracted year from 3c-year override (year was detected before llmParams existed)
        if (preExtractedYear is not null && template.Id == "site_yearly_averages" && !llmParams.ContainsKey("year"))
            llmParams["year"] = preExtractedYear;

        // Inject pre-extracted month/year from 3c-month override
        if (template.Id == "site_monthly_averages")
        {
            if (preExtractedMonth is not null && !llmParams.ContainsKey("month"))
                llmParams["month"] = preExtractedMonth;
            if (preExtractedMonthYear is not null && !llmParams.ContainsKey("year"))
                llmParams["year"] = preExtractedMonthYear;
        }

        // ── 4b-pre0z. Fallback stationName extraction for site_yearly_averages ──
        // Handles short-form patterns like "Al Falahiyya School AQI — 2019" where the LLM
        // may fail to extract stationName because there is no "at/for" preposition.
        if (template.Id is "site_yearly_averages" or "site_yearly_averages_pay"
            && !llmParams.ContainsKey("stationName"))
        {
            // Strip trailing year, AQI, dash separators and extract the site name
            var siteNameFallback = System.Text.RegularExpressions.Regex.Replace(
                request.Message,
                @"[\s\-–—]*(aqi|air quality|iaq|iaqi)?[\s\-–—]*\b(19|20)\d{2}\b.*$",
                "", System.Text.RegularExpressions.RegexOptions.IgnoreCase).Trim();
            // Also strip leading/trailing punctuation and filler words
            siteNameFallback = System.Text.RegularExpressions.Regex.Replace(
                siteNameFallback, @"^(show me|what was|what is|get|give me)\s+", "",
                System.Text.RegularExpressions.RegexOptions.IgnoreCase).Trim().Trim('-', '–', '—', ' ');
            if (!string.IsNullOrWhiteSpace(siteNameFallback))
            {
                llmParams["stationName"] = siteNameFallback;
                _log.Information("ChatController: 4b-pre0z fallback stationName='{Name}' for {Template}",
                    siteNameFallback, template.Id);
            }
        }

        // ── 4b-pre0. Fallback stationName extraction for ambient/device queries ──
        // When LLM fails to extract stationName for ambient_station_readings, try regex fallback.
        if (template.Id is "ambient_station_readings" or "indoor_source_analysis"
            && !llmParams.ContainsKey("stationName"))
        {
            var atMatch = System.Text.RegularExpressions.Regex.Match(
                enrichedMessage, @"\bat\s+([A-Za-z][A-Za-z0-9\s]{2,40}?)(?:\s+on|\s+for|\s+in|\s*\?|$)",
                System.Text.RegularExpressions.RegexOptions.IgnoreCase);
            if (atMatch.Success)
            {
                llmParams["stationName"] = atMatch.Groups[1].Value.Trim();
                _log.Debug("ChatController: 4b-pre0 fallback stationName='{Name}' for {Template}",
                    llmParams["stationName"], template.Id);
            }
        }

        // ── 4b-pre0a. dust_event_response / indoor_source_analysis stationName resolution ──
        // The LLM may extract a slightly different name than what is stored in DMN_Stations
        // (e.g. "Al Saad Indian School" vs "Al Saad Indian"). Resolve to the actual DB name
        // using space-stripped LIKE so the SQL LIKE filter in the template SQL always matches.
        if (template.Id is "dust_event_response" or "indoor_source_analysis"
            && llmParams.TryGetValue("stationName", out var dustStationRaw)
            && !string.IsNullOrWhiteSpace(dustStationRaw))
        {
            var resolvedNameResult = await _sqlExecutor.ExecuteAsync(
                "SELECT TOP 1 StationName FROM DMN_Stations WHERE (REPLACE(LOWER(StationName),' ','') LIKE '%' + REPLACE(LOWER(@name),' ','') + '%' OR REPLACE(LOWER(@name),' ','') LIKE '%' + REPLACE(LOWER(StationName),' ','') + '%') AND ISNULL(IsDeleted,0) = 0 ORDER BY LEN(StationName) ASC",
                new Dictionary<string, object> { ["name"] = dustStationRaw.Trim() }, ct);
            if (resolvedNameResult.Success && resolvedNameResult.Rows.Count > 0)
            {
                var resolvedName = resolvedNameResult.Rows[0]["StationName"]?.ToString();
                if (!string.IsNullOrWhiteSpace(resolvedName) &&
                    !resolvedName.Equals(dustStationRaw, StringComparison.OrdinalIgnoreCase))
                {
                    llmParams["stationName"] = resolvedName;
                    _log.Information("ChatController: 4b-pre0a resolved stationName '{Raw}' → '{Resolved}' for {Template}",
                        dustStationRaw, resolvedName, template.Id);
                }
            }
            else
            {
                _log.Warning("ChatController: 4b-pre0a could not resolve stationName '{Raw}' for {Template} — proceeding with LLM-extracted name",
                    dustStationRaw, template.Id);
            }
        }

        // ── 4b-pre0b. indoor_source_analysis date defaulting ──
        // If no dates provided, try to extract a date directly from the raw message before
        // falling back to the last-30-days default.
        // If single date given (either from LLM or message regex), set both to same date.
        if (template.Id == "indoor_source_analysis")
        {
            bool hasStart = llmParams.ContainsKey("startDate") && !string.IsNullOrWhiteSpace(llmParams["startDate"]);
            bool hasEnd   = llmParams.ContainsKey("endDate")   && !string.IsNullOrWhiteSpace(llmParams["endDate"]);

            // If LLM missed the date, try to pull it directly from the raw message via regex.
            // Matches patterns like: "on 2026-03-20", "on March 20 2026", "on 20-03-2026", etc.
            if (!hasStart && !hasEnd)
            {
                var datePatterns = new[]
                {
                    @"\b(\d{4}-\d{2}-\d{2})\b",                              // 2026-03-20
                    @"\b(\d{2}-\d{2}-\d{4})\b",                              // 20-03-2026
                    @"\b(\d{1,2}/\d{1,2}/\d{4})\b",                          // 3/20/2026
                    @"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",  // March 20, 2026
                    @"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b",    // 20 March 2026
                };
                string? extractedDate = null;
                foreach (var pat in datePatterns)
                {
                    var m = System.Text.RegularExpressions.Regex.Match(
                        enrichedMessage, pat,
                        System.Text.RegularExpressions.RegexOptions.IgnoreCase);
                    if (m.Success && DateTime.TryParse(m.Groups[1].Value, out var parsedDt))
                    {
                        extractedDate = parsedDt.ToString("yyyy-MM-dd");
                        break;
                    }
                }

                if (extractedDate != null)
                {
                    llmParams["startDate"] = extractedDate;
                    llmParams["endDate"]   = extractedDate;
                    _log.Debug("ChatController: 4b-pre0b regex-extracted date '{Date}' for indoor_source_analysis", extractedDate);
                }
                else
                {
                    llmParams["startDate"] = DateTime.Today.AddDays(-30).ToString("yyyy-MM-dd");
                    llmParams["endDate"]   = DateTime.Today.AddDays(-1).ToString("yyyy-MM-dd");
                    _log.Debug("ChatController: 4b-pre0b defaulted startDate/endDate for indoor_source_analysis (no date in question)");
                }
            }
            else if (hasStart && !hasEnd)
            {
                llmParams["endDate"] = llmParams["startDate"];
                _log.Debug("ChatController: 4b-pre0b single date → endDate = startDate for indoor_source_analysis");
            }
            else if (!hasStart && hasEnd)
            {
                llmParams["startDate"] = llmParams["endDate"];
                _log.Debug("ChatController: 4b-pre0b single date (endDate only) → startDate = endDate for indoor_source_analysis");
            }
        }

        // ── 4b-pre. Fix: "available at X site" is a location query, not a status filter.
        // Remove statusFilter=Active when user says "available at/in/under" with a stationName —
        // "available" means "which devices exist here", not "which devices are Active".
        if (template.Id == "devices_by_filter"
            && llmParams.TryGetValue("statusFilter", out var sf) && sf == "Active"
            && llmParams.ContainsKey("stationName"))
        {
            var lowerMsgAvail = enrichedMessage.ToLowerInvariant();
            if (lowerMsgAvail.Contains("available at") || lowerMsgAvail.Contains("available in")
                || lowerMsgAvail.Contains("available under") || lowerMsgAvail.Contains("installed at")
                || lowerMsgAvail.Contains("installed in") || lowerMsgAvail.Contains("installed under"))
            {
                llmParams.Remove("statusFilter");
                _log.Debug("ChatController: cleared statusFilter=Active — 'available/installed at site' is location intent, not status filter");
            }
        }

        // ── 4b. Override chartType when user explicitly asked for pie/bar ──
        // Handles follow-up questions like "plot a pie chart for the above" where the
        // previous template's default chartType may be "bar" or "line".
        if (template.ResponseType == "chart")
        {
            if (userWantsPie && !userWantsBar)
                template.ChartType = "pie";
            else if (userWantsBar && !userWantsPie)
                template.ChartType = "bar";
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
            var faqAnswer = SubstituteFaqPlaceholders(
                await _azureAI.AnswerFaqAsync(request.Message, faqContext, ct),
                request.Message);

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
            {
                // Render as a friendly white assistant bubble (Success=true, text type)
                // so the user sees a calm message, not a red technical error.
                var errMsgId = Guid.NewGuid();

                // Use the specific error text when the site wasn't found or access was denied —
                // these are permanent user errors, not transient API failures.
                string friendlyMsg;
                if (!string.IsNullOrWhiteSpace(apiResult.Error)
                    && apiResult.Error.Contains("do not have access", StringComparison.OrdinalIgnoreCase))
                {
                    // Site exists but is outside the user's permitted scope — show access-denied message
                    var askedSite = llmParams.GetValueOrDefault("stationName")
                                ?? llmParams.GetValueOrDefault("siteName")
                                ?? llmParams.GetValueOrDefault("station")
                                ?? "That site";
                    friendlyMsg = BuildAccessDeniedMessage(askedSite, user, scope);
                }
                else if (!string.IsNullOrWhiteSpace(apiResult.Error)
                    && apiResult.Error.Contains("doesn't match any known site", StringComparison.OrdinalIgnoreCase))
                {
                    // Site genuinely not found in the system
                    var askedSite = llmParams.GetValueOrDefault("stationName")
                                ?? llmParams.GetValueOrDefault("siteName")
                                ?? llmParams.GetValueOrDefault("station");
                    friendlyMsg = string.IsNullOrWhiteSpace(askedSite)
                        ? apiResult.Error
                        : $"There is no site called \"{askedSite}\" in the system. Please check the name and try again. You can ask \"Show me all sites\" to see the full list.";
                }
                else
                {
                    friendlyMsg = "Sorry, I'm unable to retrieve that information right now. Please try again in a moment.";
                }
                await _sessionService.SaveBothMessagesAsync(
                    sessionId: session.SessionId,
                    userContent: request.Message,
                    assistantMessageId: errMsgId,
                    assistantContent: friendlyMsg,
                    sql: null,
                    responseType: "text",
                    chartType: null,
                    chartDataJson: null,
                    dataSource: null,
                    dateRange: null,
                    modelUsed: null,
                    executionTimeMs: (int)apiResult.ExecutionTimeMs,
                    tokenCount: 0,
                    ct: ct);
                var savedRtErr = template.ResponseType;
                template.ResponseType = "text";
                var errResponse = _formatter.Format(
                    sessionId: session.SessionId,
                    messageId: errMsgId,
                    template: template,
                    llmSummary: friendlyMsg,
                    rows: [],
                    rowCount: 0,
                    executionTimeMs: (int)(apiResult.ExecutionTimeMs),
                    scope: scope,
                    modelUsed: null,
                    tokenCount: 0);
                template.ResponseType = savedRtErr;
                return Ok(errResponse);
            }

            resultRows = apiResult.Rows;
            executionTimeMs = apiResult.ExecutionTimeMs;
            dataSourceLabel = template.TableUsed; // descriptive label e.g. "AirQuality API"

            // Apply topN: if user asked "top N" sort by Value descending and take N rows.
            // schools_pollutant and sector_pollutant rows use "Value" column for the parameter reading.
            if (llmParams.TryGetValue("topN", out var topNStr)
                && int.TryParse(topNStr, out var topNVal) && topNVal > 0)
            {
                resultRows = resultRows
                    .OrderByDescending(r =>
                    {
                        // "Value" is the column name in schools_pollutant / sector_pollutant rows
                        var raw = r.GetValueOrDefault("Value") ?? r.GetValueOrDefault("AQI") ?? r.GetValueOrDefault("ParameterValue");
                        return raw is double d ? d : double.TryParse(raw?.ToString(), out var pd) ? pd : 0.0;
                    })
                    .Take(topNVal)
                    .ToList();
                _log.Information("ChatController: applied topN={N} filter on API results, {Count} rows returned", topNVal, resultRows.Count);
            }
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

            // ── Multi-site handling for devices_by_filter with stationName ────────
            // When stationName contains multiple site names separated by "and"/commas,
            // run one query per site and merge the results.
            if (template.Id == "devices_by_filter"
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
            // For sites_by_filter: default regionName to "Abu Dhabi" when the user didn't specify one.
            // This matches the Executive Dashboard default view which shows Abu Dhabi sector breakdown.
            if (template.Id == "sites_by_filter"
                && (!llmParams.TryGetValue("regionName", out var sectorRegion)
                    || string.IsNullOrWhiteSpace(sectorRegion)))
            {
                llmParams["regionName"] = "Abu Dhabi";
                _log.Information("sites_by_filter: no region specified — defaulting to Abu Dhabi");
            }

            // run one query per region and merge the results.
            var multiRegionTemplates = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "sites_by_filter", "count_sites_in_region",
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

                // Keyword fallback: inject regionName/sectorName if LLM missed them
                if (!llmParams.ContainsKey("regionName") || string.IsNullOrWhiteSpace(llmParams.GetValueOrDefault("regionName")))
                {
                    if (lowerMsg.Contains("abu dhabi") || lowerMsg.Contains("abudhabi") || lowerMsg.Contains("abu-dhabi"))
                        llmParams["regionName"] = "Abu Dhabi";
                    else if (lowerMsg.Contains("al ain") || lowerMsg.Contains("alain") || lowerMsg.Contains("al-ain"))
                        llmParams["regionName"] = "Al Ain";
                    else if (lowerMsg.Contains("al dhafra") || lowerMsg.Contains("aldhafra") || lowerMsg.Contains("dhafra"))
                        llmParams["regionName"] = "Al Dhafra";
                }
                if (!llmParams.ContainsKey("sectorName") || string.IsNullOrWhiteSpace(llmParams.GetValueOrDefault("sectorName")))
                {
                    if (lowerMsg.Contains("school") || lowerMsg.Contains("govt") || lowerMsg.Contains("government"))
                        llmParams["sectorName"] = "Public & Govt-School";
                    else if (lowerMsg.Contains("commercial"))
                        llmParams["sectorName"] = "Commercial";
                    else if (lowerMsg.Contains("residential") || lowerMsg.Contains("residence"))
                        llmParams["sectorName"] = "Residential";
                }
            }

            // ── device_live_reading: resolve deviceName → deviceId, call GetDeviceLatestData API ──
            if (template.Id == "device_live_reading")
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
                    WHERE ISNULL(d.IsDeleted, 0) = 0
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

                _log.Information("ChatController: device_live_reading deviceId={Id} deviceName='{Name}' param={Param}",
                    latestDeviceId, latestDeviceName, latestParamName ?? "all");

                var latestApiResult = await _externalApi.GetDeviceLatestDataAsync(
                    latestDeviceId, latestDeviceName, latestParamName, bearerToken, ct);

                if (!latestApiResult.Success)
                {
                    _log.Warning("device_live_reading API_ERROR: {Error}", latestApiResult.Error);
                    return Ok(_formatter.FormatError(session.SessionId,
                        "Could not retrieve the latest data for that device. Please try again.", "API_ERROR"));
                }

                resultRows      = latestApiResult.Rows;
                executionTimeMs = latestApiResult.ExecutionTimeMs;
                dataSourceLabel = "GetDeviceLatestData";
                goto afterSqlExecution;
            }

            // ── device_status / device_info / device_pollutants: resolve partial device name → canonical ──
            // Users sometimes type "B 0001" instead of "BA 0001". Resolve the canonical name via a
            // fuzzy DB lookup so the template SQL gets the exact DeviceName from DMN_Devices.
            if (template.Id is "device_status" or "device_info" or "device_pollutants")
            {
                var devNameRaw = llmParams.GetValueOrDefault("deviceName") ?? "";
                if (!string.IsNullOrWhiteSpace(devNameRaw))
                {
                    var normDevRaw = devNameRaw.Replace(" ", "").ToUpperInvariant();
                    var paddedDevRaw = normDevRaw;
                    var prefixMatchRaw = System.Text.RegularExpressions.Regex.Match(normDevRaw, @"^([A-Z]+(?:\d+[A-Z]+)*)(\d+)$");
                    string canonicalSpacedRaw = devNameRaw;
                    if (prefixMatchRaw.Success)
                    {
                        var pfx = prefixMatchRaw.Groups[1].Value;
                        var digs = prefixMatchRaw.Groups[2].Value.PadLeft(4, '0');
                        paddedDevRaw = pfx + digs;
                        canonicalSpacedRaw = pfx + " " + digs;
                    }

                    // Extract leading letter(s) and digit suffix for LIKE fuzzy match
                    // e.g. "B0001" → firstChar="B", digitSuffix="0001"
                    var firstCharRaw = normDevRaw.Length > 0 ? normDevRaw[0].ToString() : "";
                    var digitSuffixMatch = System.Text.RegularExpressions.Regex.Match(normDevRaw, @"(\d{1,4})$");
                    var digitSuffix = digitSuffixMatch.Success ? digitSuffixMatch.Groups[1].Value.PadLeft(4, '0') : "";

                    var devLookupRaw = await _sqlExecutor.ExecuteAsync(
                        """
                        SELECT TOP 1 d.DeviceId, d.DeviceName, d.StationID
                        FROM DMN_Devices d
                        WHERE ISNULL(d.IsDeleted, 0) = 0
                          AND (d.DeviceName = @deviceName
                            OR d.DeviceName = @canonicalSpaced
                            OR REPLACE(UPPER(d.DeviceName),' ','') = @normalised
                            OR REPLACE(UPPER(d.DeviceName),' ','') = @paddedCanonical
                            OR (LEFT(REPLACE(UPPER(d.DeviceName),' ',''),1) = @firstChar
                                AND REPLACE(UPPER(d.DeviceName),' ','') LIKE '%' + @digitSuffix))
                        ORDER BY CASE WHEN d.DeviceName = @deviceName THEN 0
                                      WHEN d.DeviceName = @canonicalSpaced THEN 1
                                      WHEN REPLACE(UPPER(d.DeviceName),' ','') = @normalised THEN 2
                                      WHEN REPLACE(UPPER(d.DeviceName),' ','') = @paddedCanonical THEN 3
                                      ELSE 4 END, d.DeviceName
                        """,
                        new Dictionary<string, object>
                        {
                            ["deviceName"]      = devNameRaw,
                            ["canonicalSpaced"] = canonicalSpacedRaw,
                            ["normalised"]      = normDevRaw,
                            ["paddedCanonical"] = paddedDevRaw,
                            ["firstChar"]       = firstCharRaw,
                            ["digitSuffix"]     = digitSuffix
                        }, ct);

                    if (devLookupRaw.Success && devLookupRaw.Rows.Count > 0)
                    {
                        // Override deviceName with the exact canonical name from DB
                        var resolvedName = devLookupRaw.Rows[0]["DeviceName"]?.ToString() ?? devNameRaw;
                        llmParams["deviceName"] = resolvedName;
                        _log.Information("Device name resolved: '{Input}' → '{Resolved}' for template {Id}",
                            devNameRaw, resolvedName, template.Id);
                    }
                    else
                    {
                        _log.Warning("No device found matching '{Input}' for template {Id}", devNameRaw, template.Id);
                        // Let the template SQL run — it will return 0 rows and the LLM prompt handles "no device found"
                    }
                }
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
                    SELECT TOP 1 d.DeviceId, d.ID AS DevicePk, d.DeviceName, d.StationID
                    FROM DMN_Devices d
                    WHERE ISNULL(d.IsDeleted, 0) = 0
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

                var resolvedDeviceId   = devLookup.Rows[0]["DeviceId"]?.ToString() ?? "";  // DMN_Devices.DeviceId — FK used by DMN_Parameters.DeviceID
                var resolvedDevicePk   = devLookup.Rows[0]["DevicePk"]?.ToString() ?? resolvedDeviceId; // DMN_Devices.ID — used by API calls
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

                // Also scan the raw message for explicit interval words the LLM may have missed.
                // This catches "1-hour average", "8 hour", "24-hour" etc. in the user's phrasing.
                static string DetectIntervalInMessage(string msg)
                {
                    var m = msg.ToLowerInvariant();
                    if (System.Text.RegularExpressions.Regex.IsMatch(m, @"\b8.?hour\b|\b8h\b")) return "8hour";
                    if (System.Text.RegularExpressions.Regex.IsMatch(m, @"\b24.?hour\b|\bdaily\b|\b24h\b")) return "24hour";
                    if (System.Text.RegularExpressions.Regex.IsMatch(m, @"\b1.?hour\b|\bhourly\b|\b1h\b")) return "1hour";
                    if (System.Text.RegularExpressions.Regex.IsMatch(m, @"\bmonthly\b|\bmonth\b")) return "monthly";
                    if (System.Text.RegularExpressions.Regex.IsMatch(m, @"\byearly\b|\bannual\b")) return "yearly";
                    if (System.Text.RegularExpressions.Regex.IsMatch(m, @"\b5.?min\b|\b5m\b")) return "5min";
                    return "";
                }

                var normalisedInterval = string.IsNullOrWhiteSpace(rawInterval) ? "" : NormaliseInterval(rawInterval);

                // Priority: LLM-extracted interval → raw message keyword → span-based inference
                if (!string.IsNullOrWhiteSpace(normalisedInterval))
                {
                    interval = normalisedInterval;
                }
                else
                {
                    // Check raw message before falling back to span inference
                    var msgInterval = DetectIntervalInMessage(request.Message);

                    // Parse the date range the LLM extracted to determine span
                    DateTime.TryParse(llmParams.GetValueOrDefault("startDate"), out iStart);
                    DateTime.TryParse(llmParams.GetValueOrDefault("endDate"),   out var iEnd);

                    var spanEnd   = iEnd   == default ? DateTime.UtcNow : iEnd;
                    var spanStart = iStart == default ? spanEnd.AddDays(-1) : iStart;
                    var spanHours = (spanEnd - spanStart).TotalHours;

                    isSpecificPastDay = iStart != default
                        && (DateTime.UtcNow - iStart).TotalHours > 24
                        && spanHours <= 25;

                    if (!string.IsNullOrWhiteSpace(msgInterval))
                    {
                        // User explicitly named the interval in the message — always respect it.
                        // Clear isSpecificPastDay so the full-day date override doesn't run.
                        interval = msgInterval;
                        isSpecificPastDay = false;
                        _log.Information("device_reading_history: interval detected from message text → '{Interval}'", interval);
                    }
                    else
                    {
                        interval = isSpecificPastDay ? "24hour" : spanHours switch
                        {
                            < 2    => "5min",
                            < 48   => "1hour",
                            < 1440 => "24hour",
                            < 8760 => "monthly",
                            _      => "yearly"
                        };
                        _log.Information(
                            "device_reading_history: no interval extracted, span={Hours:F1}h pastDay={PastDay} → defaulted to '{Interval}'",
                            spanHours, isSpecificPastDay, interval);
                    }
                }

                var paramName = llmParams.GetValueOrDefault("parameterName");
                var paramFilter = string.IsNullOrWhiteSpace(paramName)
                    ? "" : " AND p.ParameterName = @parameterName";

                bool intervalSnappedToPoint = false; // set true when 1H/8H is snapped to exact timestamp

                // ── 1H / 8H: detect time from message and snap to exact interval boundary ──
                // The LLM often puts time directly into startDate (e.g. "2026-09-21 02:05:00")
                // rather than targetTime, so we can't rely on llmParams["targetTime"].
                // For 1H/8H we always want a single reading, not a range:
                //   1. Try to extract a clock time from the raw message using regex.
                //   2. If found: snap startDate/endDate to the exact interval timestamp.
                //   3. If not found: ask the user for a specific time.
                if (interval is "1hour" or "8hour")
                {
                    // Extract clock time from raw message. Must be one of:
                    //   "2:05 AM", "14:30" (colon form), or "2AM"/"9PM" (digit+am/pm, no space ambiguity)
                    // Plain bare numbers like "21" (from "Sep 21st") are NOT matched.
                    var timeMatch = System.Text.RegularExpressions.Regex.Match(
                        request.Message,
                        @"\b(\d{1,2}):(\d{2})\s*(am|pm)?\b|\b(\d{1,2})\s*(am|pm)\b",
                        System.Text.RegularExpressions.RegexOptions.IgnoreCase);

                    // Also check if LLM put a non-midnight time in startDate
                    DateTime.TryParse(llmParams.GetValueOrDefault("startDate"), out var llmStart);
                    bool llmHasTime = llmStart != default && (llmStart.Hour != 0 || llmStart.Minute != 0);

                    if (!timeMatch.Success && !llmHasTime)
                    {
                        // No time found anywhere — ask for clarification
                        var intervalLabel = interval == "1hour" ? "1-hour" : "8-hour";
                        var example       = interval == "1hour" ? "2 AM or 14:00" : "8 AM or midnight";
                        await _sessionService.SaveUserMessageAsync(session.SessionId, request.Message, ct);
                        return Ok(_formatter.FormatError(
                            session.SessionId,
                            $"To show you the {intervalLabel} average for that device, I need to know which specific time you want. " +
                            $"Please tell me the time (e.g. '{example}') and I'll give you the reading for that exact hour.",
                            "CLARIFY"));
                    }

                    // Determine the hour to snap to
                    int snapHour;
                    int snapMinute = 0;
                    if (timeMatch.Success)
                    {
                        // Group 1+2+3 = colon form "2:05 AM"; Group 4+5 = ampm-only "2AM"
                        bool colonForm = timeMatch.Groups[1].Success;
                        snapHour   = int.Parse(colonForm ? timeMatch.Groups[1].Value : timeMatch.Groups[4].Value);
                        snapMinute = colonForm && timeMatch.Groups[2].Success ? int.Parse(timeMatch.Groups[2].Value) : 0;
                        var ampm   = (colonForm ? timeMatch.Groups[3].Value : timeMatch.Groups[5].Value).ToLowerInvariant();
                        if (ampm == "pm" && snapHour < 12) snapHour += 12;
                        if (ampm == "am" && snapHour == 12) snapHour = 0;
                    }
                    else
                    {
                        // Use the time from LLM's startDate
                        snapHour   = llmStart.Hour;
                        snapMinute = llmStart.Minute;
                    }

                    // Anchor date: from LLM startDate if valid, else yesterday
                    var anchorDay = llmStart != default ? llmStart.Date : DateTime.UtcNow.Date.AddDays(-1);

                    DateTime snappedPt;
                    if (interval == "8hour")
                    {
                        // Backward 8H average stored at end of block (0, 8, 16)
                        // User says "9 AM" → block ending at 8 AM → snap to 8:00
                        var blockEnd = ((snapHour + (snapMinute > 0 ? 1 : 0)) / 8) * 8;
                        if (blockEnd == 0 && snapHour >= 16) blockEnd = 24; // edge: after 16:00 → next block end
                        snappedPt = anchorDay.AddHours(blockEnd == 0 ? 8 : blockEnd); // avoid midnight ambiguity
                    }
                    else
                    {
                        // Backward 1H average stored at end of hour: user says "2:05" → record at 3:00
                        // Floor to hour, if minutes > 0 the record is at next whole hour boundary
                        var snappedHour = snapMinute > 0 ? snapHour + 1 : snapHour;
                        snappedPt = anchorDay.AddHours(snappedHour);
                    }

                    llmParams["startDate"]     = snappedPt.ToString("yyyy-MM-dd HH:mm:ss");
                    llmParams["endDate"]       = snappedPt.ToString("yyyy-MM-dd HH:mm:ss");
                    intervalSnappedToPoint     = true;  // skip isSpecificPastDay override below

                    _log.Information(
                        "device_reading_history: {Interval} snapped from raw='{Raw}' llmStart={LlmStart} → exact point {Point}",
                        interval, request.Message, llmStart, snappedPt);
                }

                // ── targetTime snapping ──────────────────────────────────────────────────
                // If the user asked for a specific clock time (e.g. "at 4AM", "at 3:03AM",
                // "at 4:30PM"), snap it to the correct boundary for the chosen interval:
                //   5min table  → floor to nearest 5-minute mark  (3:03 → 3:00, 2:59 → 2:55)
                //   1H/8H/24H  → floor to nearest whole hour       (3:05PM → 3:00PM, 4:30PM → 4:00PM)
                // The anchor date (day) comes from startDate if already set, else "yesterday"
                // (since users asking "yesterday at 4AM" trigger this path most often).
                var rawTargetTime = llmParams.GetValueOrDefault("targetTime");
                if (!string.IsNullOrWhiteSpace(rawTargetTime) && !intervalSnappedToPoint)
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
                        if (interval == "5min")
                        {
                            // Floor to nearest 5-minute boundary (e.g. 2:37 → 2:35)
                            var snappedMinute = (m / 5) * 5;
                            snappedStart = anchorDate.AddHours(h).AddMinutes(snappedMinute);
                            snappedEnd   = snappedStart.AddMinutes(5);
                        }
                        else if (interval == "8hour")
                        {
                            // Floor to nearest 8-hour block: 0, 8, 16
                            // Averages are stored at the END of the window (backward average):
                            // e.g. user says "8 AM" → 8-hour average ending at 8:00 → Interval = 08:00:00
                            // user says "9 AM" → still the 8 AM block → Interval = 08:00:00
                            var blockStart = (h / 8) * 8;
                            snappedStart = anchorDate.AddHours(blockStart);
                            snappedEnd   = snappedStart; // exact point match — same timestamp
                        }
                        else
                        {
                            // 1hour: average stored at the END of the window (backward average).
                            // e.g. user says "2:30 AM" → floor to 2 AM → Interval = 02:00:00
                            // Query exact timestamp: startDate = endDate = 02:00:00
                            snappedStart = anchorDate.AddHours(h);
                            snappedEnd   = snappedStart; // exact point match — one record only
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
                // Skip this when 1H/8H has already snapped to an exact timestamp.
                else if (isSpecificPastDay && iStart != default && !intervalSnappedToPoint)
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
                //
                // ParameterAverages.DeviceID is nvarchar and stores DMN_Devices.DeviceId as a string.
                // Filter by pa.DeviceID = @deviceId (nvarchar=nvarchar) — exactly like the web API does:
                //   a.DeviceID=" + objfilter.DeviceID
                // This avoids the int-conversion error caused by dirty nvarchar values in that column.

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

                // ParameterAverages columns (confirmed from DB):
                //   DeviceID    = int    → pass @deviceId as int
                //   ParameterID = int    → join pa.ParameterID = p.ID (both int, no conversion)
                //   Type        = nvarchar ('1H', '8H', '24H') → filter pa.Type = '1H' etc.
                //   TypeID      = int (60) → unrelated, do NOT use for interval filtering
                int.TryParse(resolvedDeviceId, out var deviceIdInt);

                var paramNameFilter2 = string.IsNullOrWhiteSpace(paramName)
                    ? "" : " AND p.ParameterName = @parameterName";

                string historySql;
                var historyParams = new Dictionary<string, object>
                {
                    ["deviceId"]           = deviceIdInt,
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
                            WHERE pa.DeviceID = @deviceId AND pa.TypeID = 60{paramNameFilter2}
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
                            WHERE pa.DeviceID = @deviceId AND pa.TypeID = 480{paramNameFilter2}
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
                            WHERE pa.DeviceID = @deviceId AND pa.TypeID = 1440{paramNameFilter2}
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
                            WHERE pam.DeviceID = @deviceId{paramNameFilter2}
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
                            WHERE pay.DeviceID = @deviceId{paramNameFilter2}
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
                                WHERE pr.DeviceID = @deviceId{paramNameFilter2}
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
                                WHERE pr.DeviceID = @deviceId{paramNameFilter2}
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
                var sqlInvalidMsg = template.Id switch
                {
                    "dust_event_response" =>
                        "I wasn't able to retrieve the dust event data for that site and date. Please make sure you include both a site name and a specific date, for example: \"How did indoor air quality at Al Saad Indian School respond during the dust event on 2026-03-15?\"",
                    _ =>
                        "Something went wrong while fetching your results. Please try again shortly."
                };
                return Ok(_formatter.FormatError(session.SessionId, sqlInvalidMsg, "SQL_INVALID"));
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
                var userFriendlyError = template.Id switch
                {
                    "dust_event_response" =>
                        "I wasn't able to retrieve dust event data for that site and date. Please check that the site name is correct and that a date is included, for example: \"How did indoor air quality at Al Saad Indian School respond during the dust event on 2026-03-15?\"",
                    _ => sqlResult.Error!.Contains("scalar variable", StringComparison.OrdinalIgnoreCase)
                        ? "I couldn't gather enough information from your question to run that query. Could you be more specific? For example: 'Show me commercial sites in Abu Dhabi' or 'What is the CO2 at Al Saad Indian School?'"
                        : "Something went wrong while fetching your results. Please try again shortly."
                };

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
        if (template.Id is "device_live_reading" or "device_reading_history")
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
        // Also catches text-type templates like dust_event_response when the station
        // has no coordinates and the SQL CTE returns 0 rows entirely.
        bool isTextTemplateWithNoData = resultRows.Count == 0
            && template.ResponseType == "text"
            && template.Id is "dust_event_response" or "indoor_source_analysis";
        if ((resultRows.Count == 0 && template.ResponseType != "text") || isTextTemplateWithNoData)
        {
            var noDataMessage = BuildNoDataMessage(template.Id, scope, llmParams.GetValueOrDefault("aqiCategory"), llmParams.GetValueOrDefault("sectorName"), llmParams.GetValueOrDefault("liveStatus"), llmParams.GetValueOrDefault("stationName"), llmParams.GetValueOrDefault("parameterName"));

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

            // Use Format (not FormatError) so the no-data message renders as a normal
            // white assistant bubble, not a red error. Success=true, rows=[], type=text.
            // Temporarily override ResponseType to "text" so the formatter skips table rendering.
            var savedResponseType = template.ResponseType;
            template.ResponseType = "text";
            var noDataResponse = _formatter.Format(
                sessionId: session.SessionId,
                messageId: noDataMsgId,
                template: template,
                llmSummary: noDataMessage,
                rows: [],
                rowCount: 0,
                executionTimeMs: (int)Math.Min(executionTimeMs, int.MaxValue),
                scope: scope,
                modelUsed: modelUsed,
                tokenCount: tokensUsed);
            template.ResponseType = savedResponseType;
            return Ok(noDataResponse);
        }

        // For table/chart responses the data is rendered visually — skip LLM summarization.
        // Generate a zero-cost summary string instead (~0ms vs ~10s).
        // Only call SummarizeResultsAsync for text responses where the answer IS the summary.
        string summary;
        if (template.ResponseType is "table" or "chart")
        {
            // site_daily_averages: build a meaningful intro sentence and strip
            // StationName, Date, DeviceCount from the table rows — user sees
            // only Parameter and Daily Average columns.
            if (template.Id == "site_daily_averages" && resultRows.Count > 0)
            {
                var siteName  = resultRows[0].GetValueOrDefault("StationName")?.ToString() ?? "";
                var dateVal   = resultRows[0].GetValueOrDefault("Date")?.ToString() ?? "";
                // Trim the timestamp if present (e.g. "2026-03-20T00:00:00" → "2026-03-20")
                if (dateVal.Contains('T')) dateVal = dateVal[..dateVal.IndexOf('T')];
                summary = $"Here are the 24-hour average readings at {siteName} on {dateVal}. The table below shows all available parameters.";
                // Strip columns that must not appear in the table
                foreach (var row in resultRows)
                {
                    row.Remove("StationName");
                    row.Remove("Date");
                    row.Remove("DeviceCount");
                }
            }
            else if (template.Id == "site_monthly_averages" && resultRows.Count > 0)
            {
                var siteName  = resultRows[0].GetValueOrDefault("StationName")?.ToString() ?? "";
                var monthVal  = resultRows[0].GetValueOrDefault("Month")?.ToString() ?? "";
                summary = $"Here are the monthly average readings at {siteName} for {monthVal}. The table below shows all available parameters.";
                foreach (var row in resultRows)
                {
                    row.Remove("StationName");
                    row.Remove("DeviceCount");
                }
            }
            else if (template.Id == "dust_event_response" && resultRows.Count > 0)
            {
                var r          = resultRows[0];
                var indoorSite = r.GetValueOrDefault("IndoorSite")?.ToString() ?? "";
                var ambStation = r.GetValueOrDefault("NearestAmbientStation")?.ToString() ?? "";
                var distKm     = r.GetValueOrDefault("Distance_km")?.ToString() ?? "";
                var eventDate  = r.GetValueOrDefault("EventDate")?.ToString() ?? "";
                if (eventDate.Contains('T')) eventDate = eventDate[..eventDate.IndexOf('T')];
                var parameter  = r.GetValueOrDefault("Parameter")?.ToString() ?? "PM10";
                var indoorAvg  = r.GetValueOrDefault("IndoorAvg");
                var outdoorAvg = r.GetValueOrDefault("OutdoorDayAvg");
                var baseline30 = r.GetValueOrDefault("OutdoorBaseline30");
                var ioRatio    = r.GetValueOrDefault("IORatio");
                var indoorVal  = indoorAvg  != null ? $"{indoorAvg} µg/m³"  : "no data";
                var outdoorVal = outdoorAvg != null ? $"{outdoorAvg} µg/m³" : "no data";
                var ratioStr   = ioRatio    != null ? $"{ioRatio}"           : "N/A";

                // Build contextual dust-event assessment
                bool isDustEvent = false;
                string contextLine = "";
                if (outdoorAvg != null && double.TryParse(outdoorAvg.ToString(), out var outVal))
                {
                    isDustEvent = outVal >= 150;
                    if (baseline30 != null && double.TryParse(baseline30.ToString(), out var baseVal) && baseVal > 0)
                    {
                        var twiceBaseline = Math.Round(baseVal * 2, 1);
                        if (isDustEvent)
                            contextLine = $"\nOutdoor {parameter} of {outdoorVal} is above the dust event threshold (~150 µg/m³) and {(outVal >= twiceBaseline ? "above" : "below")} twice the station's 30-day average of {baseline30} µg/m³ — conditions consistent with an elevated dust episode.";
                        else
                            contextLine = $"\nFor context, outdoor {parameter} of {outdoorVal} sits below the level typically associated with a dust event (around 150 µg/m³) and {(outVal >= twiceBaseline ? "above" : "below")} twice the station's 30-day average of {baseline30} µg/m³ — so the recorded conditions for this date don't point to an elevated dust episode.";
                    }
                }

                summary = $"On {eventDate}, the nearest ambient station, {ambStation}, recorded a 24-hour average {parameter} of {outdoorVal}. " +
                          $"Indoor {parameter} at {indoorSite} averaged {indoorVal} on the same day" +
                          (ioRatio != null ? $", giving an indoor/outdoor ratio of {ratioStr}." : ".") +
                          contextLine +
                          $"\nHAWAQM gives you the measured indoor and outdoor readings and this comparison, so you have the full recorded picture for the day. " +
                          $"As the {ambStation} station is about {distKm} km away, the outdoor conditions right at {indoorSite} may differ a little from the station's readings." +
                          $"\nIf it's helpful, I can show the other pollutant readings at {indoorSite} for that day, or compare this reading with nearby sites within your access.";
                tokensUsed = 0;
            }
            else if (template.Id == "site_yearly_averages" && resultRows.Count > 0)
            {
                var r = resultRows[0];
                var siteName = r.GetValueOrDefault("StationName")?.ToString() ?? "";
                var year     = r.GetValueOrDefault("Year")?.ToString() ?? r.GetValueOrDefault("[Year]")?.ToString() ?? "";
                summary = $"Here are the yearly averages for {siteName} in {year}. The table below shows all available parameters.";
                foreach (var row in resultRows)
                    row.Remove("StationName");
            }
            else if (template.Id == "site_aqi_by_category" && resultRows.Count > 0 && isAqiCategoryGroupByQuestion)
            {
                // User asked for a category count breakdown — group raw site rows into a summary table
                bool wantsByRegion2 = lowerMsg.Contains("by region") || lowerMsg.Contains("per region")
                    || lowerMsg.Contains("each region") || lowerMsg.Contains("region");
                resultRows = GroupAqiCategoryRows(resultRows, wantsByRegion2);
                summary = BuildAqiCategorySummary(resultRows, request.Message, lowerMsg, true);
            }
            else if (template.Id is "schools_latest_pollutant" or "sector_latest_pollutant")
            {
                summary = BuildSchoolPollutantSummary(resultRows, llmParams);
            }
            else if (template.Id == "sites_by_filter")
            {
                var sbfSector = llmParams.GetValueOrDefault("sectorName") ?? "";
                var sbfRegion = llmParams.GetValueOrDefault("regionName") ?? "";
                var sbfCount  = resultRows.Count;
                var sbfDesc   = (!string.IsNullOrWhiteSpace(sbfSector) && !string.IsNullOrWhiteSpace(sbfRegion))
                    ? $"{sbfSector} sites in {sbfRegion}"
                    : !string.IsNullOrWhiteSpace(sbfSector)
                        ? $"{sbfSector} sites"
                        : !string.IsNullOrWhiteSpace(sbfRegion)
                            ? $"sites in {sbfRegion}"
                            : "sites";
                summary = $"Found {sbfCount} {sbfDesc}.";
            }
            else
            {
                summary = resultRows.Count == 1
                    ? "1 result found."
                    : $"{resultRows.Count} results found.";
            }
            tokensUsed = 0;
        }
        else if (template.Id == "site_count" && resultRows.Count > 0)
        {
            // Detect region-wise breakdown (multiple regions, no single-region filter)
            var regionParam = llmParams.GetValueOrDefault("regionName") ?? "";
            var sectorParam = llmParams.GetValueOrDefault("sectorName") ?? "";
            var groupByParam = llmParams.GetValueOrDefault("groupBy") ?? "region";
            bool isRegionBreakdown = string.IsNullOrWhiteSpace(regionParam)
                && string.IsNullOrWhiteSpace(sectorParam)
                && groupByParam != "sector"
                && resultRows.Count > 1
                && resultRows.All(r => r.ContainsKey("RegionName"));

            // Detect when the user wants a plain total count with no breakdown
            bool isTotalOnlyQuestion = string.IsNullOrWhiteSpace(regionParam)
                && string.IsNullOrWhiteSpace(sectorParam)
                && (lowerMsg.Contains("total number") || lowerMsg.Contains("how many sites") || lowerMsg.Contains("total sites")
                    || lowerMsg.Contains("number of sites") || lowerMsg.Contains("sites registered") || lowerMsg.Contains("registered sites")
                    || lowerMsg.Contains("sites so far") || lowerMsg.Contains("total") && lowerMsg.Contains("sites"))
                && !lowerMsg.Contains("region") && !lowerMsg.Contains("sector") && !lowerMsg.Contains("breakdown") && !lowerMsg.Contains("each");

            // Detect active/inactive/offline intent in the user's question
            bool scWantsActive   = lowerMsg.Contains("active")   && !lowerMsg.Contains("inactive");
            bool scWantsInactive = lowerMsg.Contains("inactive") || lowerMsg.Contains("offline");

            bool siteCountUsedLlm = false;
            if (isTotalOnlyQuestion && isRegionBreakdown)
            {
                if (scWantsActive)
                {
                    int scActive = resultRows.Sum(r => int.TryParse(r.GetValueOrDefault("ActiveSites")?.ToString(), out var n) ? n : 0);
                    summary = $"There are currently {scActive} active sites.";
                }
                else if (scWantsInactive)
                {
                    int scInactive = resultRows.Sum(r => int.TryParse(r.GetValueOrDefault("InactiveSites")?.ToString(), out var n) ? n : 0);
                    summary = $"There are currently {scInactive} inactive sites.";
                }
                else
                {
                    int scTotal = resultRows.Sum(r => int.TryParse(r.GetValueOrDefault("TotalSites")?.ToString(), out var n) ? n : 0);
                    summary = $"HAWAQM currently has {scTotal} registered sites.";
                }
                tokensUsed = 0;
            }
            else if (isRegionBreakdown)
            {
                // Format: "HAWAQM has 3 regions — Abu Dhabi (133 sites, 87 active), Al Ain (54 sites, 37 active), and Al Dhafra (13 sites, 7 active)."
                int scTotal = resultRows.Sum(r => int.TryParse(r.GetValueOrDefault("TotalSites")?.ToString(), out var n) ? n : 0);
                var scParts = resultRows.Select(r =>
                {
                    var region = r.GetValueOrDefault("RegionName")?.ToString() ?? "";
                    int sites  = int.TryParse(r.GetValueOrDefault("TotalSites")?.ToString(),  out var ts) ? ts : 0;
                    int active = int.TryParse(r.GetValueOrDefault("ActiveSites")?.ToString(), out var ac) ? ac : 0;
                    return (region, sites, active);
                }).ToList();

                var scCount = scParts.Count;
                var scLabels = scParts.Select(p => $"{p.region} ({p.sites} sites, {p.active} active)").ToList();
                string regionList = scLabels.Count == 1 ? scLabels[0]
                    : scLabels.Count == 2 ? $"{scLabels[0]} and {scLabels[1]}"
                    : $"{string.Join(", ", scLabels.Take(scLabels.Count - 1))}, and {scLabels.Last()}";

                summary = $"HAWAQM has {scCount} regions — {regionList}. There are {scTotal} sites in total.";
            }
            else if (!string.IsNullOrWhiteSpace(regionParam) && resultRows.Count == 1)
            {
                // Single-region query
                var r = resultRows[0];
                var scRegion   = r.GetValueOrDefault("RegionName")?.ToString() ?? regionParam;
                int scTotal2   = int.TryParse(r.GetValueOrDefault("TotalSites")?.ToString(),    out var ts2) ? ts2 : 0;
                int scActive2  = int.TryParse(r.GetValueOrDefault("ActiveSites")?.ToString(),   out var ac2) ? ac2 : 0;
                int scInactive2 = int.TryParse(r.GetValueOrDefault("InactiveSites")?.ToString(), out var ic2) ? ic2 : 0;
                if (scWantsActive)
                    summary = $"There are currently {scActive2} active sites in {scRegion}.";
                else if (scWantsInactive)
                    summary = $"There are currently {scInactive2} inactive sites in {scRegion}.";
                else
                    summary = $"The {scRegion} region has {scTotal2} sites in total — {scActive2} active and {scInactive2} inactive.";
            }
            else
            {
                // Total or sector-grouped — let LLM summarize
                summary = await _azureAI.SummarizeResultsAsync(enrichedMessage, template, resultRows, scope, ct);
                siteCountUsedLlm = true;
            }
            if (!siteCountUsedLlm) tokensUsed = 0;
        }
        else if (template.Id == "indoor_source_analysis" && resultRows.Count > 0)
        {
            var r = resultRows[0];
            var siteName    = r.GetValueOrDefault("ResidenceName")?.ToString() ?? "";
            var ambStation  = r.GetValueOrDefault("NearestAmbientStation")?.ToString() ?? "";
            var distKm      = r.GetValueOrDefault("Distance_km")?.ToString() ?? "";
            var parameter   = r.GetValueOrDefault("Parameter")?.ToString() ?? "PM2.5";
            var periodStart = r.GetValueOrDefault("PeriodStart")?.ToString() ?? "";
            var periodEnd   = r.GetValueOrDefault("PeriodEnd")?.ToString() ?? "";
            if (periodStart.Contains('T')) periodStart = periodStart[..periodStart.IndexOf('T')];
            if (periodEnd.Contains('T'))   periodEnd   = periodEnd[..periodEnd.IndexOf('T')];
            var dateLabel   = periodStart == periodEnd || string.IsNullOrEmpty(periodEnd) ? periodStart : $"{periodStart} to {periodEnd}";
            var indoorVal   = r.GetValueOrDefault("IndoorVal");
            var outdoorVal  = r.GetValueOrDefault("OutdoorVal");
            var ioRatio     = r.GetValueOrDefault("IORatio");
            var indoorStr   = indoorVal  != null ? $"{indoorVal} µg/m³"  : "no indoor data available";
            var outdoorStr  = outdoorVal != null ? $"{outdoorVal} µg/m³" : "no outdoor data available";
            var ratioStr    = ioRatio    != null ? $"{ioRatio}"           : "N/A";

            var ratioLine = ioRatio != null
                ? $", giving an indoor/outdoor ratio of {ratioStr}."
                : ".";
            summary = $"On {dateLabel}, the indoor {parameter} at {siteName} was {indoorStr}, based on the 24-hour average for that day. " +
                      $"The nearest ambient station, {ambStation}, recorded {outdoorStr} on the same day{ratioLine}\n\n" +
                      $"HAWAQM gives you the measured readings and this ratio to review, so you have the full recorded picture for that day. " +
                      $"Pinpointing the exact source of the {parameter} goes beyond what the monitoring data covers, and is best looked at together with what was happening at the site.\n\n" +
                      $"If it's helpful, I can show the other pollutant readings at {siteName} for that day, or compare this reading with nearby sites within your access.";
            tokensUsed = 0;
        }
        else
        {
            summary = await _azureAI.SummarizeResultsAsync(
                enrichedMessage, template, resultRows, scope, ct);
        }

        // ── 9. Persist conversation (single DB round-trip) ──────────────────
        var messageId = Guid.NewGuid();
        // Persist row data for both table and chart so history can reconstruct visuals.
        var chartDataJson = resultRows.Count > 0 && template.ResponseType is "table" or "chart"
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
        catch (TimeoutException tex)
        {
            _log.Error(tex, "LLM request timed out for user {UserId}", user.UserId);
            var timeoutSession = await _sessionService.GetOrCreateSessionAsync(request.SessionId, user.UserId, CancellationToken.None);
            return Ok(_formatter.FormatError(timeoutSession.SessionId,
                "The AI service took too long to respond. Please try again in a moment.",
                "LLM_TIMEOUT"));
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            // Client disconnected — nothing to return
            _log.Information("Chat request cancelled by client for user {UserId}", user.UserId);
            return StatusCode(499);
        }
        catch (Exception ex)
        {
            _log.Error(ex, "Unhandled error in Chat for user {UserId}", user.UserId);
            var errorSession = await _sessionService.GetOrCreateSessionAsync(request.SessionId, user.UserId, CancellationToken.None);

            var friendlyError = ex switch
            {
                HttpRequestException =>
                    "I wasn't able to reach the data service right now. Please try again in a moment.",
                TaskCanceledException =>
                    "Something went wrong while fetching your results. Please try again shortly.",
                InvalidOperationException when ex.Message.Contains("variable", StringComparison.OrdinalIgnoreCase) =>
                    "I couldn't find any results for that question. Please make sure you include a site name and date, then try again.",
                _ when ex.Message.Contains("scalar variable", StringComparison.OrdinalIgnoreCase) =>
                    "I couldn't find any results — please be more specific, for example include a site name and date.",
                _ when ex.Message.Contains("timeout", StringComparison.OrdinalIgnoreCase) =>
                    "Something went wrong while fetching your results. Please try again shortly.",
                _ =>
                    "Something went wrong while fetching your results. Please try again shortly."
            };

            return Ok(_formatter.FormatError(errorSession.SessionId, friendlyError, "INTERNAL_ERROR"));
        }
    }

    // ── Debug: route-test endpoint ───────────────────────────────────────────

    /// <summary>
    /// Debug endpoint — returns the routing decision for a question without executing SQL or LLM summarization.
    /// Shows which template was chosen, its score, top-5 candidates, and which pre-check (if any) fired.
    /// Only available in Development environment.
    /// POST /api/chat/route-test  { "message": "your question here" }
    /// </summary>
    [HttpPost("route-test")]
    [AllowAnonymous]
    public async Task<IActionResult> RouteTest([FromBody] RouteTestRequest request, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(request.Message))
            return BadRequest(new { error = "message is required" });

        var lowerMsg = request.Message.ToLowerInvariant();
        lowerMsg = System.Text.RegularExpressions.Regex.Replace(
            lowerMsg, @"\bresidence\b", "residential", System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        var result = new RouteTestResult { Question = request.Message };

        // ── Fake user context for routing only (no site filtering — route-test checks templates only) ──
        var fakeUser = new UserContext
        {
            UserId = "debug",
            Role = "admin",
            PermittedCategories = ["air_quality", "device_health", "admin"],
            PermittedSiteIds = [-1]  // deny-all sentinel — route-test never executes SQL
        };

        var templates = await _queryRouter.GetPermittedTemplatesAsync(fakeUser, ct);

        // ── Check AQI definition pre-check ───────────────────────────
        var aqiDefinitionKeywords = new[] {
            "what aqi range", "aqi range is considered", "aqi categories", "aqi levels",
            "what does aqi mean", "how is aqi calculated", "aqi scale", "aqi meaning",
            "considered good", "considered moderate", "considered unhealthy", "considered hazardous",
            "what is good aqi", "what is moderate aqi", "recording frequency", "weather forecast",
            "can i export", "how often do hawaqm devices", "when is a device considered offline"
        };
        var safetyWords = new[] { "safe", "safety", "attend", "suitable", "is it good", "is it bad",
            "how is the air", "good or bad", "is it healthy", "is it unhealthy", "can children", "should children" };
        var siteTypeWords = new[] { "school", "residential", "commercial", "site", "station", "reyada", "saad", "naeem" };
        bool hasAtSitePattern = System.Text.RegularExpressions.Regex.IsMatch(
            request.Message,
            @"\b(?:at|in|for|of|about|near)\s+(?:the\s+)?(?:site\s+|school\s+|station\s+)?[A-Z][a-zA-Z]{1,}(?:\s+[A-Za-z]{2,}){0,5}");
        bool isSiteSafetyQ = safetyWords.Any(s => lowerMsg.Contains(s))
            && (siteTypeWords.Any(s => lowerMsg.Contains(s)) || hasAtSitePattern);

        if (!isSiteSafetyQ && aqiDefinitionKeywords.Any(k => lowerMsg.Contains(k)))
        {
            result.PreCheckFired = "AQI_DEFINITION → faq_answer";
            result.SelectedTemplate = "faq_answer";
            result.Note = "Intercepted before routing. Answered from faq.json.";
            return Ok(result);
        }

        // ── Check site safety pre-check ──────────────────────────────
        if (isSiteSafetyQ)
        {
            result.PreCheckFired = "SITE_SAFETY → site_aqi_single";
            result.SelectedTemplate = "site_aqi_single";
            result.Note = "Intercepted before routing. Live AQI data will be fetched.";
            return Ok(result);
        }

        // ── Check generic AQI pre-check ──────────────────────────────
        var genericAqiTriggers = new[] { "show me the aqi", "what is the aqi", "display aqi", "current aqi", "aqi now" };
        var hasRegionWord = new[] { "abu dhabi", "al ain", "al dhafra" }.Any(r => lowerMsg.Contains(r));
        var hasSiteWord = new[] { "school", "site", "station" }.Any(s => lowerMsg.Contains(s));
        bool hasDeviceName = System.Text.RegularExpressions.Regex.IsMatch(lowerMsg, @"\b(ba\s*\d{1,4}|sei\s*100\s*m\s*\d{1,4})\b");
        if (genericAqiTriggers.Any(t => lowerMsg.Contains(t)) && !hasRegionWord && !hasSiteWord && !hasDeviceName && !hasAtSitePattern)
        {
            result.PreCheckFired = "GENERIC_AQI → clarification prompt";
            result.SelectedTemplate = null;
            result.Note = "No site/region/device specified. User will be asked to clarify.";
            return Ok(result);
        }

        // ── Normal routing ────────────────────────────────────────────
        var routerResult = await _queryRouter.RouteAsync(request.Message, fakeUser, null, ct);

        // Top-5 scores by re-scoring all templates
        var questionTokens = request.Message.ToLowerInvariant()
            .Split([' ', ',', '?', '!', '.'], StringSplitOptions.RemoveEmptyEntries);
        result.Top5Candidates = templates
            .Select(t => new CandidateScore
            {
                TemplateId = t.Id,
                Description = t.Description ?? string.Empty,
                PatternMatchCount = t.Patterns.Count(p => questionTokens.Any(tok => p.ToLowerInvariant().Contains(tok)))
            })
            .OrderByDescending(x => x.PatternMatchCount)
            .Take(5)
            .ToList();

        if (routerResult.Template is not null)
        {
            result.SelectedTemplate = routerResult.Template.Id;
            result.Score = routerResult.Score;
            result.Confidence = routerResult.Score >= 0.85 ? "HIGH" : routerResult.Score >= 0.50 ? "MEDIUM" : "LOW";
            result.Note = "Template selected directly by keyword scoring.";
        }
        else if (routerResult.Candidates.Count > 0)
        {
            result.SelectedTemplate = null;
            result.Score = routerResult.Score;
            result.Confidence = "MEDIUM";
            result.LlmCandidates = routerResult.Candidates.Select(c => c.Id).ToList();
            result.Note = "Score in medium range — LLM would pick between these candidates.";
        }
        else if (!string.IsNullOrEmpty(routerResult.ClarificationPrompt))
        {
            result.SelectedTemplate = null;
            result.Confidence = "LOW";
            result.Note = $"Low confidence. Clarification prompt: {routerResult.ClarificationPrompt}";
        }

        return Ok(result);
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    private UserContext? GetUserContext()
        => HttpContext.Items.TryGetValue("UserContext", out var ctx) ? ctx as UserContext : null;

    /// <summary>
    /// Serializes data rows to JSON for persistence when responseType is "table" or "chart".
    /// Stored in the ChartData column so history can reconstruct table and chart visuals.
    /// Returns null for text responses or when there are no rows.
    /// </summary>
    private static string? SerializeRowsForHistory(string responseType, List<Dictionary<string, object?>> rows)
    {
        if (rows.Count == 0) return null;
        if (responseType is "table" or "chart")
            return System.Text.Json.JsonSerializer.Serialize(rows);
        return null;
    }

    /// <summary>
    /// Groups raw site_aqi_by_category rows (one row per site) into a summary table
    /// showing the count of sites in each AQI category. When wantsByRegion=true the
    /// table is further broken down by region: columns Region, AQI Category, Count.
    /// When false: columns AQI Category, Count.
    /// Rows are ordered by the standard AQI category severity sequence.
    /// </summary>
    private static List<Dictionary<string, object?>> GroupAqiCategoryRows(
        List<Dictionary<string, object?>> rawRows, bool wantsByRegion)
    {
        var categoryOrder = new[]
        {
            "Good", "Moderate", "Unhealthy for Sensitive Groups",
            "Unhealthy", "Very Unhealthy", "Hazardous"
        };

        if (wantsByRegion)
        {
            // Group by (Region, AQI Category) → {Region, AQI Category, Count}
            var grouped = rawRows
                .GroupBy(r => (
                    Region:   (r.GetValueOrDefault("Region") ?? r.GetValueOrDefault("RegionName"))?.ToString() ?? "Unknown",
                    Category: (r.GetValueOrDefault("AQI Category") ?? r.GetValueOrDefault("AQICategory"))?.ToString() ?? "Unknown"))
                .Select(g => new Dictionary<string, object?>
                {
                    ["Region"]       = g.Key.Region,
                    ["AQI Category"] = g.Key.Category,
                    ["Count"]        = g.Count()
                })
                .OrderBy(r => r["Region"]?.ToString())
                .ThenBy(r =>
                {
                    var idx = Array.IndexOf(categoryOrder, r["AQI Category"]?.ToString() ?? "");
                    return idx < 0 ? 99 : idx;
                })
                .ToList();
            return grouped;
        }
        else
        {
            // Group by AQI Category only → {AQI Category, Count}
            var grouped = rawRows
                .GroupBy(r => (r.GetValueOrDefault("AQI Category") ?? r.GetValueOrDefault("AQICategory"))?.ToString() ?? "Unknown")
                .Select(g => new Dictionary<string, object?>
                {
                    ["AQI Category"] = g.Key,
                    ["Count"]        = g.Count()
                })
                .OrderBy(r =>
                {
                    var idx = Array.IndexOf(categoryOrder, r["AQI Category"]?.ToString() ?? "");
                    return idx < 0 ? 99 : idx;
                })
                .ToList();
            return grouped;
        }
    }

    /// <summary>
    /// Builds a one-sentence summary for AQI category breakdown responses.
    /// </summary>
    private static string BuildAqiCategorySummary(
        List<Dictionary<string, object?>> groupedRows, string originalMessage,
        string lowerMsg, bool isGrouped)
    {
        if (!isGrouped || groupedRows.Count == 0)
            return $"{groupedRows.Count} result(s) found.";

        // Infer site label from question
        string siteLabel = lowerMsg.Contains("school") || lowerMsg.Contains("moe") || lowerMsg.Contains("gov")
            ? "school site(s)"
            : lowerMsg.Contains("commercial") ? "commercial site(s)"
            : lowerMsg.Contains("residential") ? "residential site(s)"
            : "site(s)";

        // Infer region context
        string regionSuffix = lowerMsg.Contains("abu dhabi") ? " in Abu Dhabi"
            : lowerMsg.Contains("al ain") ? " in Al Ain"
            : lowerMsg.Contains("al dhafra") || lowerMsg.Contains("dhafra") ? " in Al Dhafra"
            : "";

        bool byRegion = groupedRows.Any(r => r.ContainsKey("Region"));
        int totalSites = groupedRows.Sum(r => r.TryGetValue("Count", out var cv) ? Convert.ToInt32(cv ?? 0) : 0);

        if (byRegion)
            return $"Here is the AQI category breakdown for {totalSites} {siteLabel}{regionSuffix}, grouped by region. The table below shows the count per category per region.";
        else
            return $"Here is the AQI category breakdown for {totalSites} {siteLabel}{regionSuffix}. The table below shows the count per category.";
    }

    private static string BuildHistoricalNoDataMessage(string? parameterName, string? siteName, string date)
    {
        var topic = !string.IsNullOrWhiteSpace(parameterName) ? parameterName : "data";
        var site  = !string.IsNullOrWhiteSpace(siteName)      ? siteName      : "that site";
        var period = !string.IsNullOrWhiteSpace(date)          ? date          : "the requested period";
        return $"I couldn't find any results for {topic} at {site} for {period}. Try widening the date range or choosing a different site.";
    }

    private static string BuildNoDataMessage(string templateId, ResolvedScope scope, string? aqiCategory = null, string? sectorName = null, string? liveStatus = null, string? siteName = null, string? parameterName = null)
    {
        var region = string.IsNullOrWhiteSpace(scope.Region) ? null : scope.Region;
        var sector = sectorName ?? (string.IsNullOrWhiteSpace(scope.Sector) ? null : scope.Sector);
        var date = scope.DateRangeLabel ?? "the selected date";

        return templateId switch
        {
            "current_all_stations" or "site_aqi_all" =>
                $"Readings{(region != null ? $" for {region}" : "")} are still being updated. Please check back in a few minutes.",

            "sites_by_filter" or "count_sites_in_region" =>
                liveStatus?.ToLowerInvariant() switch
                {
                    "inactive" => sector != null && region != null
                        ? $"Good news — all {sector} sites in {region} are currently active. There are no inactive sites right now."
                        : sector != null
                            ? $"Good news — all {sector} sites are currently active. There are no inactive sites right now."
                            : region != null
                                ? $"Good news — all monitored sites in {region} are currently active. There are no inactive sites right now."
                                : "Good news — all monitored sites are currently active. There are no inactive sites right now.",
                    "active" => sector != null && region != null
                        ? $"No active {sector} sites were found in {region}. All sites in that filter may be inactive or have no devices attached."
                        : sector != null
                            ? $"No active {sector} sites were found. All sites in that sector may be inactive or have no devices."
                            : region != null
                                ? $"No active sites were found in {region}. All sites may be inactive or have no devices attached."
                                : "No active sites were found. All sites may currently be inactive or have no devices attached.",
                    "nodevice" => sector != null && region != null
                        ? $"All {sector} sites in {region} currently have at least one device. There are no sites without devices."
                        : sector != null
                            ? $"All {sector} sites currently have at least one device. There are no sites without devices."
                            : region != null
                                ? $"All sites in {region} currently have at least one device attached. No sites without devices were found."
                                : "All sites currently have at least one device attached. There are no sites without devices right now.",
                    _ => sector != null && region != null
                        ? $"No {sector} sites were found in {region}."
                        : sector != null
                            ? $"No {sector} sites were found."
                            : region != null
                                ? $"No sites were found in {region}."
                                : "No sites were found matching the selected filter."
                },

            "sites_outside_safe_range" or "site_aqi_by_category" =>
                !string.IsNullOrWhiteSpace(aqiCategory) && !string.Equals(aqiCategory, "critical", StringComparison.OrdinalIgnoreCase)
                    ? (sector != null && region != null
                        ? $"Good news — no {sector} sites in {region} are currently in the '{aqiCategory}' AQI category. All sites within your access appear to be within acceptable levels."
                        : sector != null
                            ? $"Good news — no {sector} sites are currently in the '{aqiCategory}' AQI category. All sites within your access appear to be within acceptable levels."
                            : region != null
                                ? $"Good news — no sites in {region} are currently in the '{aqiCategory}' AQI category. All monitored sites appear to be within acceptable levels."
                                : $"Good news — no sites are currently in the '{aqiCategory}' AQI category. All monitored sites appear to be within acceptable levels.")
                    : sector != null && region != null
                        ? $"No {sector} sites in {region} were found with an AQI in that range at this time."
                        : sector != null
                            ? $"No {sector} sites were found with an AQI in that range at this time."
                            : region != null
                                ? $"No sites in {region} were found with an AQI in that range at this time."
                                : "No sites were found with an AQI in that range at this time.",

            "critical_sites" or "critical_sites_today" =>
                $"No critical sites were found in {region} for {date}. All monitored sites appear to be within acceptable limits.",

            "offline_devices" or "devices_offline" =>
                $"All devices in {region} appear to be online. No offline devices were detected at this time.",

            "top_polluted_sites" =>
                $"No pollution data is available for {region} at the moment. Devices may still be syncing. Please try again shortly.",

            "device_reading_history" =>
                $"I couldn't find any results for that device for the requested period. Try widening the date range, choosing a different interval, or check that the device was active at that time.",

            "device_live_reading" =>
                "Readings for this device are still being updated. Please check back in a few minutes.",

            "site_daily_averages" or "site_single_parameter_reading" =>
                BuildHistoricalNoDataMessage(parameterName, siteName, date),

            "site_monthly_averages" =>
                BuildHistoricalNoDataMessage(parameterName, siteName, date),

            "site_yearly_averages" or "site_yearly_averages_pay" =>
                BuildHistoricalNoDataMessage(parameterName, siteName, date),

            "dust_event_response" or "indoor_source_analysis" =>
                "I couldn't find any results for that site and date. This can happen if: (1) the site name doesn't exactly match a registered site, (2) the site has no GPS coordinates configured (required for nearest ambient station lookup), or (3) no daily average data exists for that date. Please verify the site name and date, then try again. Example: \"How did indoor air quality at Al Reyada School respond during the dust event on 2026-03-15?\"",

            "sites_dsr_above_threshold" =>
                "No sites were found with a Data Success Rate above that threshold. This means all monitored sites currently have a DSR below the specified percentage. Try lowering the threshold (e.g. above 50%) to see more results.",

            "sites_dsr_below_threshold" =>
                "No sites were found with a Data Success Rate below that threshold. This is a positive sign — all monitored sites are currently meeting or exceeding the specified data availability percentage.",

            "schools_aqi_pie" or "sites_aqi_level_chart" =>
                sector != null && region != null
                    ? $"No {sector} sites with active devices were found in {region}. The pie chart cannot be generated without data — please check that devices are online in that region."
                    : sector != null
                        ? $"No {sector} sites with active devices were found. The pie chart cannot be generated without data."
                        : region != null
                            ? $"No school or site data was found for {region}. The pie chart cannot be generated — devices in that region may be offline or not yet reporting."
                            : "No site data is available to generate the pie chart. Devices may be offline or still syncing. Please try again shortly.",

            _ =>
                "I couldn't find any results for that query. Try widening the date range, choosing a different site, or rephrasing your question."
        };
    }

    /// <summary>
    /// Returns the out-of-scope message when the user asks about a site they cannot access.
    /// </summary>
    private static string BuildAccessDeniedMessage(string askedSiteName, UserContext? user, ResolvedScope scope)
    {
        return "I can only answer questions about HAWAQM air-quality data. Try asking about a site, device or pollutant.";
    }

    /// <summary>
    /// Replaces FAQ answer placeholders with actual values derived from the user's question and current time.
    /// </summary>
    private static string SubstituteFaqPlaceholders(string answer, string userMessage)
    {
        // {DEVICE_NAME} — extract device code from question (letter-prefix + digit-suffix pattern)
        var deviceMatch = System.Text.RegularExpressions.Regex.Match(
            userMessage,
            @"\b([A-Za-z]+[0-9A-Za-z]*\s+\d{4}|\b[A-Za-z]+\d{4,}\b)",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        var deviceName = deviceMatch.Success ? deviceMatch.Value.Trim() : "the device";
        answer = answer.Replace("{DEVICE_NAME}", deviceName);

        // {FROM_TIME} / {TO_TIME} — last 24 hours window
        var toTime   = DateTime.Now;
        var fromTime = toTime.AddHours(-24);
        answer = answer.Replace("{FROM_TIME}", fromTime.ToString("dd MMM yyyy HH:mm"));
        answer = answer.Replace("{TO_TIME}",   toTime.ToString("dd MMM yyyy HH:mm"));

        return answer;
    }

    /// <summary>
    /// Deterministically builds the summary text for schools_latest_pollutant / sector_latest_pollutant
    /// results directly from the actual row data — no LLM involved, so values always match the table.
    /// </summary>
    private static string BuildSchoolPollutantSummary(
        List<Dictionary<string, object?>> rows,
        Dictionary<string, string> llmParams)
    {
        if (rows.Count == 0)
        {
            var param0 = llmParams.GetValueOrDefault("parameterName") ?? "AQI Index";
            return $"No school sites with {param0} data were found for the selected filters.";
        }

        var parameterName = llmParams.GetValueOrDefault("parameterName") ?? "AQI Index";
        var sortOrder     = llmParams.GetValueOrDefault("sortOrder") ?? "";
        int topN = 0;
        bool hasTopN = llmParams.TryGetValue("topN", out var topNStr) && int.TryParse(topNStr, out topN) && topN > 0;
        bool isDesc       = !string.Equals(sortOrder, "asc", StringComparison.OrdinalIgnoreCase);

        var sb = new System.Text.StringBuilder();

        if (hasTopN)
        {
            var direction = isDesc ? "highest" : "lowest";
            sb.AppendLine(isDesc
                ? $"Here are the top {topN} schools with the highest {parameterName} right now:"
                : $"Here are the {topN} schools with the lowest {parameterName} right now:");
        }
        else
        {
            sb.AppendLine($"There are {rows.Count} school site(s) with {parameterName} data.");
        }

        int rank = 1;
        foreach (var row in rows)
        {
            var site   = row.GetValueOrDefault("Site Name")?.ToString() ?? "";
            var region = row.GetValueOrDefault("Region")?.ToString() ?? "";
            var value  = row.GetValueOrDefault("Value");
            var unit   = row.GetValueOrDefault("Unit")?.ToString() ?? "";
            var ts     = row.GetValueOrDefault("Last Updated")?.ToString() ?? "";

            var unitStr = string.IsNullOrWhiteSpace(unit) || parameterName.Equals("AQI Index", StringComparison.OrdinalIgnoreCase)
                ? "" : $" {unit}";

            if (hasTopN)
                sb.AppendLine($"{rank}. {site} ({region}): {value}{unitStr}");
            else
                sb.AppendLine($"{site} in {region}: {parameterName} = {value}{unitStr}{(string.IsNullOrWhiteSpace(ts) ? "" : $" (last updated: {ts})")}.");

            rank++;
        }

        return sb.ToString().TrimEnd();
    }
}
