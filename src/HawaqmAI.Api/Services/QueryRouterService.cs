using System.Text.Json;
using HawaqmAI.Api.Configuration;
using HawaqmAI.Api.Models;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Options;
using Serilog;

namespace HawaqmAI.Api.Services;

/// <summary>
/// Keyword-based query router (v1 — no embeddings required).
/// Loads approved-queries.json once on startup and caches it.
/// Scores each template's patterns against the user's question using
/// token overlap to produce a confidence score.
/// </summary>
public sealed class QueryRouterService : IQueryRouterService
{
    private static readonly Serilog.ILogger _log = Log.ForContext<QueryRouterService>();
    private readonly IMemoryCache _cache;
    private readonly ChatOptions _chatOptions;
    private const string CacheKey = "approved_queries";

    public QueryRouterService(IMemoryCache cache, IOptions<ChatOptions> chatOptions)
    {
        _cache = cache;
        _chatOptions = chatOptions.Value;
    }

    // Keywords that indicate the user is referring to a previous result rather than asking a new question
    private static readonly string[] FollowUpPhrases =
    [
        "above", "previous", "that list", "the list", "same", "again", "sort", "order",
        "ascending", "descending", "asc", "desc", "filter", "group", "make it",
        "change", "modify", "show only", "now show", "instead", "but only",
        "also show", "add to", "remove from", "without", "exclude", "include only"
    ];

    // Keywords that clearly signal a site query (not a device or region)
    private static readonly string[] SiteKeywords =
        ["site", "station", "school", "building", "facility", "location", "readings at", "aqi at", "data for", "reading at"];

    // Keywords that clearly signal a device query
    private static readonly string[] DeviceKeywords =
        ["device", "sensor", "reading for", "data from", "ba 0", "sei100m 0", "sei100m"];

    // Keywords that clearly signal a region query
    private static readonly string[] RegionKeywords =
        ["abu dhabi", "abudhabi", "al ain", "alain", "al dhafra", "aldhafra", "dhafra", "region"];

    /// <inheritdoc/>
    public async Task<QueryRouterResult> RouteAsync(
        string question,
        UserContext user,
        IReadOnlyList<ConversationTurn>? history = null,
        CancellationToken ct = default)
    {
        var templates = await LoadTemplatesAsync(ct);
        var permitted = templates.Where(t => user.PermittedCategories.Contains(t.Category)).ToList();

        // ── Follow-up detection ───────────────────────────────────────────────
        // If the question contains follow-up phrases AND there is prior conversation,
        // send all permitted templates to the LLM to pick the right one in context.
        // EXCEPT: never treat FAQ/export/policy questions as follow-ups — they have
        // strong independent signals and must go through normal scoring.
        var lowerQ = question.ToLowerInvariant();
        var exportGuardPhrases = new[] {
            "raw-data export", "raw data export", "one-year export", "one year export",
            "prepare a one-year", "prepare an export", "statistical report", "exceedance report",
            "averaging intervals", "90-day", "90 day", "data success rate", "dsr"
        };
        bool isStrongFaqSignal = exportGuardPhrases.Any(p => lowerQ.Contains(p));
        if (history is { Count: > 0 } && IsFollowUpQuestion(question) && !isStrongFaqSignal)
        {
            // Find the last template used from history (assistant messages have DataSource)
            var lastTemplateId = history
                .LastOrDefault(h => h.Role == "assistant" && !string.IsNullOrEmpty(h.TemplateId))
                ?.TemplateId;

            _log.Debug("Follow-up detected for question='{Q}', last template='{T}'",
                question[..Math.Min(80, question.Length)], lastTemplateId);

            return new QueryRouterResult
            {
                Template = null,
                Score = 0.75,
                IsFollowUp = true,
                PreviousTemplateId = lastTemplateId,
                Candidates = permitted
            };
        }

        // ── Standard pattern scoring ──────────────────────────────────────────
        var questionTokens = Tokenize(question);
        var scored = permitted
            .Select(t => (Template: t, Score: ScoreTemplate(t, questionTokens, question)))
            .OrderByDescending(x => x.Score)
            .ToList();

        var best = scored.FirstOrDefault();
        var high = _chatOptions.HighConfidenceThreshold;
        var low = _chatOptions.LowConfidenceThreshold;

        _log.Debug("Query routing: question='{Q}', best='{Id}' score={Score:F2}",
            question[..Math.Min(80, question.Length)], best.Template?.Id, best.Score);

        if (best.Score >= high)
        {
            return new QueryRouterResult
            {
                Template = best.Template,
                Score = best.Score,
                Candidates = []
            };
        }

        if (best.Score >= low)
        {
            var candidates = scored.Take(3).Select(x => x.Template).ToList();
            return new QueryRouterResult
            {
                Template = null,
                Score = best.Score,
                Candidates = candidates!
            };
        }

        // ── Disambiguation: ask user to clarify site vs device vs region ────────
        // When confidence is too low AND the question lacks clear context keywords,
        // return a targeted clarification prompt instead of a generic error.
        return new QueryRouterResult
        {
            Template = null,
            Score = best.Score,
            Candidates = [],
            ClarificationPrompt = BuildDisambiguationPrompt(question)
        };
    }

    private static string BuildDisambiguationPrompt(string question)
    {
        var lower = question.ToLowerInvariant();

        // Already has a clear context — generic fallback
        bool hasSite   = SiteKeywords.Any(k => lower.Contains(k));
        bool hasDevice = DeviceKeywords.Any(k => lower.Contains(k));
        bool hasRegion = RegionKeywords.Any(k => lower.Contains(k));

        if (hasSite || hasDevice || hasRegion)
            return "Try asking about: AQI at a site, readings for a device (e.g. BA 0001, SEI100M 0008), or AQI for a region (Abu Dhabi, Al Ain, Al Dhafra).";

        // Ambiguous — could be site, device, or region name
        return "I'm not sure what you're referring to. Are you asking about:\n• A site (e.g. \"Al Saad Indian School\")?\n• A device (e.g. \"BA 0001\" or \"SEI100M 0008\")?\n• A region (Abudhabi, AL Ain, or AlDhafra)?\nPlease clarify and try again.";
    }

    private static bool IsFollowUpQuestion(string question)
    {
        var lower = question.ToLowerInvariant();
        return FollowUpPhrases.Any(phrase => lower.Contains(phrase));
    }

    // Year tokens that indicate a historical question (not current data)
    private static readonly System.Text.RegularExpressions.Regex _yearPattern =
        new(@"\b(19|20)\d{2}\b", System.Text.RegularExpressions.RegexOptions.Compiled);

    // Causation phrases — questions asking to PROVE or EXPLAIN a cause are out of scope for SQL
    private static readonly string[] CausationPhrases =
        ["can you prove", "prove that", "prove the", "prove what caused",
         "what caused the spike", "what caused the pollution", "caused the spike",
         "caused yesterday", "caused the reading", "caused this reading",
         "ventilation caused", "ventilation system caused"];

    private static bool IsCausationQuestion(string lowerQuestion)
        => CausationPhrases.Any(p => lowerQuestion.Contains(p));

    // Past-tense phrases that signal historical site-level queries
    private static readonly string[] HistoricalPhrases =
        ["what was", "was the aqi", "was the air quality", "was the reading",
         "was the pm", "was the co2", "was the no2", "was the average",
         "was the main pollutant", "was the worst", "last year", "previous year",
         "historical", "in the past", "years ago"];

    /// <summary>
    /// Returns true when the question is asking for site-level historical data
    /// that HAWAQM does not store — site AQI is calculated on the fly from live devices.
    /// </summary>
    private static bool IsHistoricalSiteLevelQuestion(string lowerQuestion)
    {
        // Must contain a past-year reference OR a past-tense phrase
        var hasYear = _yearPattern.IsMatch(lowerQuestion);
        var hasPastPhrase = HistoricalPhrases.Any(p => lowerQuestion.Contains(p));

        if (!hasYear && !hasPastPhrase) return false;

        // Must also reference a site-level entity (school, site, station, location)
        // to avoid penalising region/network-level historical questions unnecessarily
        var hasSiteRef = SiteKeywords.Any(k => lowerQuestion.Contains(k));
        return hasSiteRef;
    }

    /// <inheritdoc/>
    public IReadOnlyList<ApprovedQuery> GetPermittedTemplates(UserContext user)
    {
        var templates = _cache.Get<List<ApprovedQuery>>(CacheKey) ?? [];
        return templates.Where(t => user.PermittedCategories.Contains(t.Category)).ToList();
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    private async Task<List<ApprovedQuery>> LoadTemplatesAsync(CancellationToken ct)
    {
        if (_cache.TryGetValue(CacheKey, out List<ApprovedQuery>? cached) && cached != null)
            return cached;

        var path = Path.Combine(AppContext.BaseDirectory, "Knowledge", "approved-queries.json");
        if (!File.Exists(path))
        {
            _log.Error("approved-queries.json not found at {Path}", path);
            return [];
        }

        await using var fs = File.OpenRead(path);
        var file = await JsonSerializer.DeserializeAsync<ApprovedQueriesFile>(fs,
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true }, ct);

        var queries = file?.Queries ?? [];
        _cache.Set(CacheKey, queries, TimeSpan.FromHours(1));
        _log.Information("Loaded {Count} approved query templates", queries.Count);
        return queries;
    }

    // Parameter names that indicate a pollutant-reading request
    private static readonly HashSet<string> PollutantTokens = new(StringComparer.OrdinalIgnoreCase)
        { "co", "no2", "so2", "co2", "pm2.5", "pm10", "o3", "tvoc", "ch2o", "aqi", "temperature", "humidity", "voc", "noise" };

    // Regex-style check: question contains a device name pattern.
    // Strategy: strip all spaces from the question, then match the normalised device-name
    // pattern used by ChatController: letters+digits+letters prefix followed by 1-4 digits.
    // Handles all variants: "BA 0001", "BA0010", "SEI100M 0014", "SEI 100M 0008",
    //                        "sei100m0085", "sei 100 m 8", "BA1", "SEI100M0008" etc.
    private static bool QuestionHasDeviceName(string lowerQuestion)
    {
        // Also check the original (for word-boundary accuracy on short names like "ba 1")
        if (System.Text.RegularExpressions.Regex.IsMatch(lowerQuestion,
            @"\b(ba\s*\d{1,4}|sei\s*100\s*m\s*\d{1,4})\b"))
            return true;

        // Collapse spaces and re-check — catches "SEI 100M 0008", "sei 100 m 0014" etc.
        var collapsed = lowerQuestion.Replace(" ", "");
        return System.Text.RegularExpressions.Regex.IsMatch(collapsed,
            @"(ba\d{1,4}|sei100m\d{1,4})");
    }

    // Site-name indicators — words that appear in station names but NOT in region-only queries.
    // If any of these are present alongside a region word, the question is a site query, not a region query.
    // NOTE: "school" is intentionally NOT in this list — it is a generic noun used in "all schools",
    // "schools in Abu Dhabi" etc. and must NOT be treated as a site name indicator.
    private static readonly string[] SiteNameIndicators =
    [
        "residential", "commercial", "institutional", "institution",
        "building", "facility", "centre", "center", "hospital", "clinic",
        "office", "park", "mall", "tower", "villa", "compound", "camp",
        "indian", "british", "american", "international", "national",
        "bateen", "saad", "naeem", "naim", "khalifa", "zayed", "hamdan",
        "mushrif", "mussafah", "baniyas", "karama", "shakhbout", "shahama",
        "madinat", "khalidiyah", "corniche", "mangrove", "reem", "yas",
        "wahda", "rowdah", "muroor", "electra", "hameem", "liwa", "gayathi",
        "reyada", "reayada", "kaltham", "obeidli", "gems", "british"
    ];

    // Specific named-school indicators — unique proper nouns that only appear in a named school, not in
    // generic "schools in X" queries. Used to distinguish site_aqi_single from schools_latest_pollutant.
    private static readonly string[] NamedSchoolIndicators =
    [
        "reyada", "reayada", "saad", "naeem", "naim", "bateen", "gems", "kaltham",
        "indian school", "british school", "american school", "international school",
        "al ain school", "abu dhabi school", "al dhafra school"
    ];

    // Generic school question patterns — "Abu Dhabi schools", "schools in Al Ain", "all schools" etc.
    // These refer to the school SECTOR, not a specific named school.
    private static readonly string[] GenericSchoolPhrases =
    [
        "all schools", "schools in", "schools with", "schools co2", "schools aqi", "schools pm",
        "abu dhabi schools", "al ain schools", "al dhafra schools", "abudhabi schools",
        "alain schools", "aldhafra schools", "show me schools", "table of schools",
        "list schools", "schools air quality", "school sites", "school co2", "school aqi",
        "schools latest", "latest co2 reading", "latest aqi", "latest pm"
    ];

    /// <summary>
    /// Returns true when the question contains site-name indicators beyond just a region keyword.
    /// Used to prevent region_aqi_geographical from winning when the user names a specific site.
    /// </summary>
    private static bool QuestionHasSiteName(string lowerQuestion) =>
        SiteNameIndicators.Any(indicator => lowerQuestion.Contains(indicator));

    // Words that indicate a device status check (not an FAQ about rules)
    private static readonly string[] DeviceStatusKeywords = ["active", "inactive", "online", "offline", "working", "sending"];

    private static double ScoreTemplate(ApprovedQuery template, HashSet<string> questionTokens, string originalQuestion)
    {
        if (template.Patterns.Count == 0) return 0;

        double maxScore = 0;
        var lowerQuestion = originalQuestion.ToLowerInvariant();

        // Detect if question contains a pollutant parameter token
        bool questionHasPollutant = questionTokens.Any(t => PollutantTokens.Contains(t));

        foreach (var pattern in template.Patterns)
        {
            // Exact substring match gets highest score
            var cleanPattern = pattern.Replace("{pollutant}", "").Replace("{pollutantCol}", "").Trim().ToLowerInvariant();
            if (!string.IsNullOrWhiteSpace(cleanPattern) && lowerQuestion.Contains(cleanPattern))
            {
                maxScore = Math.Max(maxScore, 0.95);
                continue;
            }

            // Token overlap scoring: score = overlap / patternTokens (how much of pattern is covered)
            var patternTokens = Tokenize(pattern);
            if (patternTokens.Count == 0) continue;

            var overlap = patternTokens.Count(t => questionTokens.Contains(t));
            var score = (double)overlap / patternTokens.Count;

            // If the question asks for a specific pollutant reading but this template has no
            // pollutant params, penalise it — prevents "commercial sites" from winning over
            // "sector_latest_pollutant" when the user asks "CO reading for commercial sites"
            if (questionHasPollutant)
            {
                bool templateHasPollutantParam = template.Params.Any(p =>
                    p.Name.Equals("parameterName", StringComparison.OrdinalIgnoreCase));
                if (!templateHasPollutantParam)
                    score *= 0.5; // halve score for non-pollutant templates
            }

            maxScore = Math.Max(maxScore, score);
        }

        // Penalise faq_answer when the user is asking for the CURRENT/ACTUAL Data Success Rate
        // value — those must route to data_success_rate which calls the live Compliancestatas API.
        // FAQ only knows the definition/rules, not the live percentage.
        var dsrValueKeywords = new[] {
            "what is the current data success rate", "current data success rate",
            "current dsr", "what is the dsr", "dsr value", "dsr percentage", "dsr %",
            "overall data availability", "data availability rate", "data availability percentage",
            "how many devices are active", "active device count", "how many active devices",
            "how many critical sites", "critical site count", "priority hotspot count",
            "how many devices are currently active"
        };
        if (template.Id == "faq_answer"
            && dsrValueKeywords.Any(k => lowerQuestion.Contains(k)))
        {
            maxScore *= 0.1;
        }

        // Penalise faq_answer when question contains a specific device name + status word —
        // those questions should route to device_status, not the FAQ knowledge base.
        if (template.Id == "faq_answer"
            && QuestionHasDeviceName(lowerQuestion)
            && DeviceStatusKeywords.Any(k => lowerQuestion.Contains(k)))
        {
            maxScore *= 0.2;
        }

        // Penalise device_last_reading when question asks WHEN the device last sent data —
        // those should route to device_status which queries ParameterReadingUpdateTime directly.
        if (template.Id == "device_last_reading"
            && QuestionHasDeviceName(lowerQuestion)
            && (lowerQuestion.Contains("when did") || lowerQuestion.Contains("last send") || lowerQuestion.Contains("last sent")))
        {
            maxScore *= 0.1;
        }

        // Boost all device templates when a device name pattern is present —
        // prevents site templates from stealing device queries (e.g. "PM2.5 for BA0010").
        if ((template.Id == "device_last_reading" || template.Id == "device_reading_history"
             || template.Id == "device_info" || template.Id == "device_status"
             || template.Id == "device_pollutants")
            && QuestionHasDeviceName(lowerQuestion))
        {
            maxScore = Math.Max(maxScore, 0.92);
        }

        // Route device_info when question has device name + metadata keywords (site/region/sector/deployed etc.)
        var deviceMetaKeywords = new[] {
            "which site", "what site", "site name", "which station", "what station",
            "which region", "what region", "which sector", "what sector",
            "deployed", "created", "installed", "when was", "registration date",
            "device model", "device type", "details", "info about", "tell me about",
            "last measured time", "last data time", "last measurement time",
            "where is", "belong to"
        };
        if (template.Id == "device_info"
            && QuestionHasDeviceName(lowerQuestion)
            && deviceMetaKeywords.Any(k => lowerQuestion.Contains(k)))
        {
            maxScore = Math.Max(maxScore, 0.96);
        }

        // Route device_pollutants when question asks what a device monitors/measures
        var pollutantQueryWords = new[] { "pollutants", "parameters", "what does", "what it measures", "monitors", "what can" };
        if (template.Id == "device_pollutants"
            && QuestionHasDeviceName(lowerQuestion)
            && pollutantQueryWords.Any(k => lowerQuestion.Contains(k)))
        {
            maxScore = Math.Max(maxScore, 0.96);
        }

        // Penalise device_last_reading when question is asking for metadata (not readings)
        if (template.Id == "device_last_reading"
            && QuestionHasDeviceName(lowerQuestion)
            && deviceMetaKeywords.Any(k => lowerQuestion.Contains(k))
            && !lowerQuestion.Contains("reading") && !lowerQuestion.Contains("value")
            && !lowerQuestion.Contains("pm2.5") && !lowerQuestion.Contains("co2")
            && !lowerQuestion.Contains("aqi") && !lowerQuestion.Contains("temperature"))
        {
            maxScore *= 0.1;
        }

        // Between device_last_reading and device_reading_history:
        // Pick history ONLY when an explicit time interval OR specific date/time is present.
        // "reading" and "value" alone do NOT imply history — they appear in current-reading questions too.
        var historyIntervalWords = new[] {
            "5 min", "5min", "5-min", "5-minute",
            "1 hour", "1h", "1hour", "1-hour", "1h average", "1hour average", "1h averages",
            "8 hour", "8h", "8hour", "8-hour", "8h average",
            "24 hour", "24h", "24hour", "24-hour", "daily", "daily average",
            "monthly", "month", "monthly average",
            "yearly", "year", "annual", "yearly average",
            "history", "historical", "past", "average", "avg", "mean",
            "interval", "trend",
            "last hour", "last 24", "last month", "last year",
            "yesterday", "last week"
        };
        // "current", "now", "latest", "live", "right now", "at this moment" → force device_last_reading
        var currentReadingWords = new[] { "current", "now", "latest", "live", "right now", "at this moment", "today's reading", "present" };
        bool isCurrentReading = currentReadingWords.Any(w => lowerQuestion.Contains(w));

        // Detect specific date mentions (e.g. "11-aug-2026", "2026-08-11", "yesterday", "on monday")
        var hasSpecificDate = System.Text.RegularExpressions.Regex.IsMatch(lowerQuestion,
            @"\b(yesterday|last\s+night|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b");
        // Detect specific time mentions (e.g. "2:02am", "4am", "13:00")
        var hasSpecificTime = System.Text.RegularExpressions.Regex.IsMatch(lowerQuestion,
            @"\b(\d{1,2}:\d{2}\s*(am|pm)?|\d{1,2}\s*(am|pm))\b");

        // History signal: must have an interval word OR a specific date/time — NOT just "reading"
        bool isHistorySignal = !isCurrentReading &&
                               (historyIntervalWords.Any(w => lowerQuestion.Contains(w))
                                || hasSpecificDate || hasSpecificTime);

        // Force device_last_reading when "current/now/latest" is present with a device name
        if (template.Id == "device_last_reading"
            && QuestionHasDeviceName(lowerQuestion)
            && isCurrentReading)
        {
            maxScore = Math.Max(maxScore, 0.97);
        }

        if (template.Id == "device_last_reading"
            && QuestionHasDeviceName(lowerQuestion)
            && isHistorySignal)
        {
            maxScore *= 0.05; // let device_reading_history win decisively
        }
        if (template.Id == "device_reading_history"
            && QuestionHasDeviceName(lowerQuestion)
            && isHistorySignal)
        {
            maxScore = Math.Max(maxScore, 0.97); // beats all non-device templates (0.95) and device_last_reading
        }

        // Penalise ALL non-device templates when a device name is present —
        // device name is unambiguous and must never route to site/station SQL templates.
        var nonDeviceTemplates = new[] {
            "site_aqi_single", "site_readings_all", "site_aqi_all",
            "site_aqi_by_category", "schools_latest_pollutant", "sector_latest_pollutant",
            "average_at_station", "pollutant_trend_hourly", "exceedance_count",
            "exceedance_all_stations", "current_all_stations", "rank_stations_by_pollutant",
            "daily_aqi_extremes", "main_pollutant_at_station", "worst_month_pollutants",
            "site_aqi_trend_yearly", "compare_two_stations"
        };
        if (nonDeviceTemplates.Contains(template.Id) && QuestionHasDeviceName(lowerQuestion))
        {
            maxScore *= 0.02;
        }

        // Penalise ALL device templates when there is NO device name in the question —
        // without a BA/SEI100M code the device templates cannot resolve and will error.
        var deviceTemplates = new[] {
            "device_last_reading", "device_status", "device_info",
            "device_pollutants", "device_reading_history"
        };
        if (deviceTemplates.Contains(template.Id) && !QuestionHasDeviceName(lowerQuestion))
        {
            maxScore *= 0.02;
        }

        // Detect generic school questions: "Abu Dhabi schools", "schools in Al Ain", "all schools" etc.
        // These refer to the school SECTOR and must route to schools_latest_pollutant, NOT site_aqi_single.
        bool isGenericSchoolQuestion = GenericSchoolPhrases.Any(p => lowerQuestion.Contains(p))
            || (lowerQuestion.Contains("schools") && !NamedSchoolIndicators.Any(k => lowerQuestion.Contains(k)));

        // Penalise site_aqi_all when the question contains a specific site name —
        // a named-site question must use site_aqi_single, not the all-sites table.
        bool hasSiteName = !isGenericSchoolQuestion
            && (QuestionHasSiteName(lowerQuestion)
                || NamedSchoolIndicators.Any(k => lowerQuestion.Contains(k)));
        if (template.Id == "site_aqi_all" && hasSiteName)
            maxScore *= 0.02;

        // Boost site_aqi_single when the question contains a specific site name and asks for AQI or readings.
        // Generic school questions (e.g. "Abu Dhabi schools CO2") must NOT trigger this boost —
        // they route to schools_latest_pollutant instead.
        bool isAqiOrReadingQuestion = lowerQuestion.Contains("aqi") || lowerQuestion.Contains("air quality")
            || lowerQuestion.Contains("reading") || lowerQuestion.Contains("co2")
            || lowerQuestion.Contains("pm2.5") || lowerQuestion.Contains("pm10")
            || lowerQuestion.Contains("temperature") || lowerQuestion.Contains("humidity");
        if (template.Id == "site_aqi_single" && hasSiteName && isAqiOrReadingQuestion && !isGenericSchoolQuestion)
            maxScore = Math.Max(maxScore, 0.96);

        // Boost schools_latest_pollutant for generic school questions with a pollutant signal.
        // "Abu Dhabi schools CO2", "schools in Al Ain with AQI", "all schools PM2.5" etc.
        if (template.Id == "schools_latest_pollutant" && isGenericSchoolQuestion)
            maxScore = Math.Max(maxScore, 0.96);
        // Penalise site_aqi_single for generic school questions — must not ask for a site name
        if (template.Id == "site_aqi_single" && isGenericSchoolQuestion)
            maxScore *= 0.1;

        // Boost region_aqi_geographical when question has a region keyword + AQI/air quality signal
        // but NO specific site name — e.g. "what is the AQI in AL Ain", "AQI in Abudhabi"
        bool hasRegionKeyword = RegionKeywords.Any(k => lowerQuestion.Contains(k));
        bool isAqiQuestion = lowerQuestion.Contains("aqi") || lowerQuestion.Contains("air quality");
        if (template.Id == "region_aqi_geographical" && hasRegionKeyword && isAqiQuestion && !hasSiteName)
            maxScore = Math.Max(maxScore, 0.96);

        // Penalise site_aqi_all when a region keyword is present — region queries must go to region_aqi_geographical
        if (template.Id == "site_aqi_all" && hasRegionKeyword && isAqiQuestion && !hasSiteName)
            maxScore *= 0.1;

        // Boost list_stations / sites_filtered when question asks for a SITE LIST (not readings).
        // "what are the sites under X", "sites in X and Y", "sites under X region" etc.
        // These must NOT route to current_all_stations (which returns readings).
        var siteListPhrases = new[] {
            "what are the sites", "what sites are", "which sites are", "list the sites",
            "sites under", "sites in", "sites within", "show sites", "give sites",
            "sites available", "all sites", "list sites", "show all sites"
        };
        bool isSiteListQuestion = siteListPhrases.Any(p => lowerQuestion.Contains(p));
        bool hasRegionWord = RegionKeywords.Any(k => lowerQuestion.Contains(k));

        if (isSiteListQuestion && (template.Id == "list_stations" || template.Id == "sites_filtered"))
            maxScore = Math.Max(maxScore, 0.93);

        // Penalise current_all_stations for site-list questions — it returns readings, not a list
        if (template.Id == "current_all_stations" && isSiteListQuestion && !isAqiOrReadingQuestion)
            maxScore *= 0.1;

        // Boost worst_month_pollutants for "what drove the worst month" questions
        var worstMonthPhrases = new[] {
            "worst air quality month", "worst aqi month", "what drove the worst month",
            "what caused the worst month", "pollutants behind the worst month",
            "worst month air quality", "which month was worst", "worst monthly aqi",
            "what made the worst month"
        };
        bool isWorstMonth = worstMonthPhrases.Any(p => lowerQuestion.Contains(p));
        if (template.Id == "worst_month_pollutants" && isWorstMonth)
            maxScore = Math.Max(maxScore, 0.96);
        if (template.Id != "worst_month_pollutants" && isWorstMonth)
            maxScore *= 0.05;

        // Boost region_aqi_trend_yearly for multi-year regional trend questions
        var regionTrendPhrases = new[] {
            "poorest air quality over the last", "worst air quality over the last",
            "which region had the poorest", "which region had the worst",
            "region aqi trend", "regional air quality trend", "regional aqi trend",
            "compare regions over the last", "region comparison over years",
            "region aqi over last", "worst region for air quality over"
        };
        bool isRegionTrend = regionTrendPhrases.Any(p => lowerQuestion.Contains(p));
        if (template.Id == "region_aqi_trend_yearly" && isRegionTrend)
            maxScore = Math.Max(maxScore, 0.96);
        if (template.Id != "region_aqi_trend_yearly" && isRegionTrend)
            maxScore *= 0.05;

        // Boost region_aqi_single_year for single-year regional best/worst questions
        var regionSingleYearPhrases = new[] {
            "which region performed best in", "which region was best in",
            "which region performed worst in", "which region was worst in",
            "region performance in", "best region in", "worst region in",
            "region comparison for", "region with best aqi in", "region with worst aqi in",
            "how did regions perform in"
        };
        bool isRegionSingleYear = regionSingleYearPhrases.Any(p => lowerQuestion.Contains(p))
            && !isRegionTrend;
        if (template.Id == "region_aqi_single_year" && isRegionSingleYear)
            maxScore = Math.Max(maxScore, 0.96);
        if (template.Id != "region_aqi_single_year" && isRegionSingleYear)
            maxScore *= 0.05;

        // Boost site_aqi_trend_yearly for multi-year trend questions at a specific site
        var yearlyTrendPhrases = new[] {
            "improved or worsened", "has air quality improved", "has aqi improved",
            "has air quality worsened", "has aqi worsened", "aqi trend at",
            "air quality trend at", "trend over the last", "year over year aqi",
            "year on year aqi", "yearly aqi trend", "aqi over the last",
            "how has air quality changed", "how has aqi changed",
            "aqi improvement at", "aqi worsening at", "annual aqi trend",
            "multi-year aqi", "aqi change over years", "aqi history at"
        };
        bool isYearlyTrend = yearlyTrendPhrases.Any(p => lowerQuestion.Contains(p));
        if (template.Id == "site_aqi_trend_yearly" && isYearlyTrend)
        {
            maxScore = Math.Max(maxScore, 0.96);
        }
        if (template.Id != "site_aqi_trend_yearly" && isYearlyTrend
            && (lowerQuestion.Contains("school") || lowerQuestion.Contains("site") || lowerQuestion.Contains("station")
                || QuestionHasSiteName(lowerQuestion)))
        {
            maxScore *= 0.05;
        }

        // Boost indoor_ambient_compare for questions comparing indoor AQI with ambient/outdoor station
        var indoorAmbientPhrases = new[] {
            "ambient station", "indoor vs outdoor", "indoor and outdoor", "nearest ambient",
            "nearest outdoor station", "outdoor aqi near", "ambient aqi near",
            "compare indoor aqi", "indoor aqi with ambient", "indoor aqi vs ambient",
            "compare with outdoor", "compare with ambient", "indoor outdoor aqi"
        };
        bool isIndoorAmbientCompare = indoorAmbientPhrases.Any(p => lowerQuestion.Contains(p));
        if (template.Id == "indoor_ambient_compare" && isIndoorAmbientCompare)
        {
            maxScore = Math.Max(maxScore, 0.96);
        }
        if (template.Id != "indoor_ambient_compare" && isIndoorAmbientCompare)
        {
            maxScore *= 0.05;
        }

        // Boost lowest_dsr_devices when question asks which devices had the lowest DSR last month
        var lowestDsrPhrases = new[] {
            "lowest data success rate", "lowest dsr", "worst data success rate", "worst dsr",
            "devices with lowest dsr", "bottom devices by dsr", "least data last month",
            "poorest compliance", "lowest compliance devices", "dsr by device last month",
            "data success rate by device"
        };
        bool isLowestDsr = lowestDsrPhrases.Any(p => lowerQuestion.Contains(p));
        if (template.Id == "lowest_dsr_devices" && isLowestDsr)
        {
            maxScore = Math.Max(maxScore, 0.96);
        }
        if ((template.Id == "data_success_rate" || template.Id == "device_compliance_rate")
            && isLowestDsr)
        {
            maxScore *= 0.05;
        }

        // Boost device_peak_aqi_contribution when question asks which device drove the highest site AQI
        var peakContributionPhrases = new[] {
            "contributed most to the highest", "contributed most to the peak",
            "caused the highest aqi", "drove the highest aqi",
            "responsible for the highest aqi", "device behind the highest aqi",
            "which device had the highest aqi at", "pushed the aqi highest",
            "device contributed most", "contributed to the peak aqi"
        };
        bool isPeakContribution = peakContributionPhrases.Any(p => lowerQuestion.Contains(p));
        if (template.Id == "device_peak_aqi_contribution" && isPeakContribution)
        {
            maxScore = Math.Max(maxScore, 0.96);
        }
        if (template.Id != "device_peak_aqi_contribution" && isPeakContribution)
        {
            maxScore *= 0.05;
        }

        // Boost compare_devices_monthly when question asks to compare devices at a site for a month
        var compareDevicesPhrases = new[] {
            "compare the two devices", "compare both devices", "compare devices at",
            "two devices at", "both devices at", "device comparison at",
            "devices side by side", "side by side devices", "device vs device",
            "compare the devices at"
        };
        bool isDeviceCompare = compareDevicesPhrases.Any(p => lowerQuestion.Contains(p));
        if (template.Id == "compare_devices_monthly" && isDeviceCompare)
        {
            maxScore = Math.Max(maxScore, 0.96);
        }
        if (template.Id != "compare_devices_monthly" && isDeviceCompare)
        {
            maxScore *= 0.05;
        }

        // Detect yearly AQI ranking questions: "highest/lowest average AQI last year / in 2024 / in 2025"
        var yearlyAqiRankingPhrases = new[] {
            "last year", "previous year", "in 2020", "in 2021", "in 2022", "in 2023", "in 2024", "in 2025",
            "annual aqi", "yearly aqi", "year aqi", "average aqi last year", "aqi last year",
            "aqi ranking last year", "top sites aqi", "worst sites last year", "best sites last year",
            "highest average aqi", "lowest average aqi"
        };
        bool isYearlyAqiRanking = lowerQuestion.Contains("aqi")
            && yearlyAqiRankingPhrases.Any(p => lowerQuestion.Contains(p))
            && (lowerQuestion.Contains("highest") || lowerQuestion.Contains("lowest") || lowerQuestion.Contains("top")
                || lowerQuestion.Contains("worst") || lowerQuestion.Contains("best") || lowerQuestion.Contains("ranking")
                || lowerQuestion.Contains("average"));

        bool isLowestYearlyAqi = isYearlyAqiRanking
            && (lowerQuestion.Contains("lowest") || lowerQuestion.Contains("best")
                || lowerQuestion.Contains("cleanest") || lowerQuestion.Contains("safest"));

        if (template.Id == "top_sites_aqi_yearly" && isYearlyAqiRanking && !isLowestYearlyAqi)
        {
            maxScore = Math.Max(maxScore, 0.96);
        }
        if (template.Id == "bottom_sites_aqi_yearly" && isLowestYearlyAqi)
        {
            maxScore = Math.Max(maxScore, 0.96);
        }
        // Penalise faq_answer and current-AQI templates for yearly ranking questions
        if ((template.Id == "faq_answer" || template.Id == "site_aqi_ranking"
             || template.Id == "highest_aqi_today" || template.Id == "lowest_aqi_today"
             || template.Id == "site_aqi_all")
            && isYearlyAqiRanking)
        {
            maxScore *= 0.05;
        }

        // Penalise lowest_readings / highest_readings when the question mentions "aqi" —
        // those templates query ParameterAverages and do NOT support AQI Index.
        // AQI site-ranking questions should route to lowest_aqi_today / highest_aqi_today instead.
        if ((template.Id == "lowest_readings" || template.Id == "highest_readings")
            && questionTokens.Contains("aqi"))
        {
            maxScore *= 0.1;
        }

        // Penalise current_all_stations when the question asks for "AQI" without specifying
        // a concrete pollutant (CO, NO2, PM2.5, etc.) — that template requires @parameterName
        // and will fail with "Must declare the scalar variable" if AQI is passed.
        // Generic AQI questions should route to site_aqi_all or region_aqi_geographical instead.
        if (template.Id == "current_all_stations" && questionTokens.Contains("aqi")
            && !questionTokens.Any(t => new[] { "co", "no2", "so2", "co2", "pm2.5", "pm10", "o3", "tvoc", "ch2o", "temperature", "humidity" }.Contains(t)))
        {
            maxScore *= 0.05;
        }

        // Penalise site_aqi_all when the question mentions a specific region name —
        // those should route to region_aqi_geographical which filters AQI by region.
        // site_aqi_all returns all-sites AQI with no region filter.
        if (template.Id == "site_aqi_all"
            && RegionKeywords.Any(k => lowerQuestion.Contains(k)))
        {
            maxScore *= 0.1;
        }

        // Penalise region_aqi_geographical when the question contains a SITE NAME beyond
        // just the region word — site-name questions must route to site_aqi_single / site_readings_all.
        // A "site name" is detected when the question has site-context words (school, residential,
        // commercial, institutional, building, facility, at/for/in + non-region proper noun).
        // Priority: specific site name always beats region-only routing.
        var hasSiteNameBeyondRegion = QuestionHasSiteName(lowerQuestion);
        if (template.Id == "region_aqi_geographical" && hasSiteNameBeyondRegion)
        {
            maxScore *= 0.05;
        }

        // Penalise sites_by_sector / list_stations / sites_by_region when the question explicitly
        // mentions "device" or "sensor" — those templates return sites, not devices.
        // Device listing questions must route to devices_with_readings.
        if ((template.Id == "sites_by_sector" || template.Id == "list_stations" ||
             template.Id == "sites_by_region" || template.Id == "count_sites_in_region")
            && (lowerQuestion.Contains("device") || lowerQuestion.Contains("sensor")))
        {
            maxScore *= 0.05;
        }

        // Penalise sites_count_by_sector and sites_by_sector when user is asking for sector/subsector LIST or COUNT
        // (not sites-per-sector breakdown) — route to list_sectors or list_subsectors instead.
        var isSectorCountQuestion =
            (lowerQuestion.Contains("how many sector") || lowerQuestion.Contains("number of sector") ||
             lowerQuestion.Contains("what are the sector") || lowerQuestion.Contains("list sector") ||
             lowerQuestion.Contains("sectors in") || lowerQuestion.Contains("sectors are there") ||
             lowerQuestion.Contains("total sector")) &&
            !lowerQuestion.Contains("site") && !lowerQuestion.Contains("station");

        var isSubSectorListQuestion =
            lowerQuestion.Contains("subsector") || lowerQuestion.Contains("sub sector") ||
            lowerQuestion.Contains("sub-sector");

        if ((template.Id == "sites_count_by_sector" || template.Id == "sites_by_sector")
            && (isSectorCountQuestion || isSubSectorListQuestion))
        {
            maxScore *= 0.05;
        }

        // Penalise site_aqi_by_category when user is asking about the DEFINITION or RANGE of an AQI category —
        // e.g. "What AQI range is considered Good?" is an FAQ about what the range means, NOT a request for sites.
        // Signals: "what aqi range", "what aqi is", "aqi range is", "range is considered", "what is good aqi",
        // "what does good mean", "what does moderate mean", "what is the aqi for good", "aqi for good",
        // "range for good", "range for moderate", "range for unhealthy", "what range is", "considered good".
        var aqiRangeDefinitionPhrases = new[] {
            "what aqi range", "aqi range is", "range is considered", "range considered",
            "what range is", "what is good aqi", "what is moderate aqi", "what is unhealthy aqi",
            "what is hazardous aqi", "what is very unhealthy aqi",
            "aqi range for good", "aqi range for moderate", "aqi range for unhealthy",
            "aqi range for hazardous", "aqi range for very unhealthy",
            "what aqi is good", "what aqi is moderate", "what aqi is unhealthy", "what aqi is hazardous",
            "aqi good range", "aqi moderate range", "aqi unhealthy range", "aqi hazardous range",
            "aqi level for good", "aqi level for moderate", "what level is good", "what level is moderate"
        };
        if (template.Id == "site_aqi_by_category"
            && aqiRangeDefinitionPhrases.Any(k => lowerQuestion.Contains(k)))
        {
            maxScore *= 0.02;
        }

        // Penalise site_aqi_by_category and site_aqi_all when user explicitly asks for a pie/donut chart —
        // those should route to schools_aqi_pie which returns aggregated category counts for pie rendering.
        var isPieChartQuestion =
            lowerQuestion.Contains("pie chart") || lowerQuestion.Contains("donut chart") ||
            lowerQuestion.Contains("pie graph") || lowerQuestion.Contains("distribution chart") ||
            (lowerQuestion.Contains("distribution") && lowerQuestion.Contains("aqi") && lowerQuestion.Contains("categor"));

        if ((template.Id == "site_aqi_by_category" || template.Id == "site_aqi_all")
            && isPieChartQuestion)
        {
            maxScore *= 0.05;
        }

        // Penalise site_aqi_all when the question mentions an AQI category / quality level —
        // those should route to site_aqi_by_category which filters/groups by category.
        var aqiCategoryKeywords = new[] {
            "hotspot", "critical", "unhealthy", "hazardous", "very unhealthy",
            "good air", "healthy", "moderate air", "poor air", "bad air",
            "safe sites", "unsafe sites", "sites with good", "sites with bad",
            "sites with poor", "good quality", "poor quality", "aqi category",
            "aqi level", "aqi classification", "by category", "by level"
        };
        if (template.Id == "site_aqi_all"
            && aqiCategoryKeywords.Any(k => lowerQuestion.Contains(k)))
        {
            maxScore *= 0.05;
        }

        // Penalise count_sites_in_region when multiple regions are mentioned in one question —
        // that template takes a single @regionName and fails when the user asks about all regions.
        // Multi-region questions must route to sites_count_by_region which groups by all regions.
        var mentionedRegionCount =
            (lowerQuestion.Contains("abu dhabi") || lowerQuestion.Contains("abudhabi") ? 1 : 0) +
            (lowerQuestion.Contains("al ain") || lowerQuestion.Contains("alain") ? 1 : 0) +
            (lowerQuestion.Contains("al dhafra") || lowerQuestion.Contains("aldhafra") || lowerQuestion.Contains("dhafra") ? 1 : 0);

        if (template.Id == "count_sites_in_region" && mentionedRegionCount > 1)
        {
            maxScore *= 0.05;
        }

        // Boost sites_count_by_region when all 3 regions are mentioned together —
        // user wants a cross-region count, not per-region listing.
        var isAllThreeRegionsCountQuestion =
            mentionedRegionCount == 3 &&
            (lowerQuestion.Contains("how many") || lowerQuestion.Contains("total") ||
             lowerQuestion.Contains("count") || lowerQuestion.Contains("number of"));

        if (template.Id == "sites_count_by_region" && isAllThreeRegionsCountQuestion)
        {
            maxScore = Math.Max(maxScore, 0.97);
        }

        // Penalise sites_by_region and count_sites_in_region when all 3 regions are mentioned
        // in a count question — these templates take a single @regionName and will fail or
        // ask for clarification. Route to sites_count_by_region instead.
        if ((template.Id == "sites_by_region" || template.Id == "count_sites_in_region")
            && isAllThreeRegionsCountQuestion)
        {
            maxScore *= 0.05;
        }

        // Penalise list_stations when the question mentions a specific region name —
        // those should route to sites_by_region which filters by region, not list_stations
        // which returns all accessible sites regardless of region.
        if (template.Id == "list_stations"
            && RegionKeywords.Any(k => lowerQuestion.Contains(k)))
        {
            maxScore *= 0.05;
        }

        // Detect if the question is about listing sites by sector
        // (commercial, residential, public/govt school) — used by two penalty blocks below.
        var isSectorListQuestion =
            lowerQuestion.Contains("commercial") ||
            lowerQuestion.Contains("residential") ||
            lowerQuestion.Contains("govt-school") ||
            lowerQuestion.Contains("govt school") ||
            lowerQuestion.Contains("gov school") ||
            lowerQuestion.Contains("gov-school") ||
            lowerQuestion.Contains("public & govt") ||
            lowerQuestion.Contains("public & gov") ||
            lowerQuestion.Contains("public and gov") ||
            lowerQuestion.Contains("public and govt") ||
            lowerQuestion.Contains("government school") ||
            lowerQuestion.Contains("government sites");

        // Penalise device_status when question is asking device METADATA (site, region, sector, deployed, pollutants)
        // rather than active/inactive status — those must route to device_info.
        var pureMetaKeywords = new[] { "region", "sector", "deployed", "created", "installed", "pollutants", "monitors", "model", "info", "details", "where is" };
        if (template.Id == "device_status"
            && QuestionHasDeviceName(lowerQuestion)
            && pureMetaKeywords.Any(k => lowerQuestion.Contains(k)))
        {
            maxScore *= 0.05;
        }

        // Penalise device_status when no specific device name is present but region/sector/site is —
        // those must route to devices_by_filter which supports those filters.
        if (template.Id == "device_status"
            && !QuestionHasDeviceName(lowerQuestion)
            && (RegionKeywords.Any(k => lowerQuestion.Contains(k))
                || isSectorListQuestion
                || lowerQuestion.Contains("site") || lowerQuestion.Contains("station")))
        {
            maxScore *= 0.05;
        }

        // Penalise active_device_count / online_devices / offline_devices when a region or sector
        // is mentioned — those templates have no filter and return all-network results.
        if ((template.Id == "active_device_count" || template.Id == "online_devices" || template.Id == "offline_devices")
            && (RegionKeywords.Any(k => lowerQuestion.Contains(k)) || isSectorListQuestion))
        {
            maxScore *= 0.1;
        }

        // Penalise sites_by_region when a sector word is also present —
        // sector+region questions must go to sites_by_sector which applies BOTH filters.
        // sites_by_region only filters by region and would return all sectors.
        if (template.Id == "sites_by_region" && isSectorListQuestion)
        {
            maxScore *= 0.05;
        }

        // Penalise schools_latest_pollutant when the question names a SPECIFIC school/site —
        // questions with a unique school proper noun must route to site_aqi_single, not the generic school list.
        if (template.Id == "schools_latest_pollutant"
            && NamedSchoolIndicators.Any(k => lowerQuestion.Contains(k)))
        {
            maxScore *= 0.02;
        }

        // Penalise schools_latest_pollutant when the question contains a NUMERIC AQI threshold —
        // "how many schools have an AQI above 30" is a filter question → site_aqi_by_category.
        // schools_latest_pollutant returns all schools' raw readings, not filtered by threshold.
        var hasNumericThreshold = System.Text.RegularExpressions.Regex.IsMatch(lowerQuestion, @"\b\d+\b")
            && (lowerQuestion.Contains("above") || lowerQuestion.Contains("below") || lowerQuestion.Contains("more than")
                || lowerQuestion.Contains("less than") || lowerQuestion.Contains("greater than")
                || lowerQuestion.Contains("over") || lowerQuestion.Contains("under") || lowerQuestion.Contains("exceeds")
                || lowerQuestion.Contains("between") || lowerQuestion.Contains("at least") || lowerQuestion.Contains("at most"));
        if (template.Id == "schools_latest_pollutant" && hasNumericThreshold)
        {
            maxScore *= 0.02;
        }

        // Boost site_aqi_by_category when question has a numeric AQI threshold —
        // these are always filter/count questions that the aqi_category_filter API handles.
        if (template.Id == "site_aqi_by_category" && hasNumericThreshold
            && (lowerQuestion.Contains("aqi") || lowerQuestion.Contains("air quality")))
        {
            maxScore = Math.Max(maxScore, 0.97);
        }

        // Penalise schools_latest_pollutant when the question mentions an AQI quality level —
        // category questions (good/moderate/unhealthy/hazardous) must route to site_aqi_by_category.
        // schools_latest_pollutant only fetches AQI values, not filters by category.
        var aqiQualityWords = new[] {
            "good", "moderate", "unhealthy", "hazardous", "very unhealthy", "critical",
            "hotspot", "safe", "unsafe", "bad air", "poor air", "healthy air",
            "sensitive groups", "vulnerable", "harmful", "dangerous"
        };
        if (template.Id == "schools_latest_pollutant"
            && aqiQualityWords.Any(k => lowerQuestion.Contains(k)))
        {
            maxScore *= 0.05;
        }

        // Detect AQI quality level question: any mention of category words + AQI or "air quality"
        bool hasAqiQualityWord = aqiQualityWords.Any(k => lowerQuestion.Contains(k));
        bool isAqiCategoryQuestion = hasAqiQualityWord
            && (lowerQuestion.Contains("aqi") || lowerQuestion.Contains("air quality")
                || lowerQuestion.Contains("where aqi") || lowerQuestion.Contains("aqi is"));

        // Boost schools_aqi_pie when the question explicitly asks for a pie/donut/distribution chart —
        // takes priority over site_aqi_by_category even when AQI category words are present.
        if (template.Id == "schools_aqi_pie" && isPieChartQuestion)
        {
            maxScore = Math.Max(maxScore, 0.97);
        }

        // Penalise site_aqi_by_category when the user explicitly asks for a pie/donut chart —
        // pie chart questions must route to schools_aqi_pie which returns aggregated category counts.
        if (template.Id == "site_aqi_by_category" && isPieChartQuestion)
        {
            maxScore *= 0.05;
        }

        // Boost site_aqi_by_category for ANY question that mentions an AQI category/level
        // regardless of whether a region, sector, or school is specified.
        if (template.Id == "site_aqi_by_category" && isAqiCategoryQuestion && !isPieChartQuestion)
        {
            maxScore = Math.Max(maxScore, 0.96);
        }

        // Penalise sites_by_region / list_stations / sites_by_sector when the question
        // asks for an AQI category — those templates return site lists without AQI data.
        if ((template.Id == "sites_by_region" || template.Id == "list_stations" || template.Id == "sites_by_sector")
            && isAqiCategoryQuestion)
        {
            maxScore *= 0.05;
        }

        // Boost site_aqi_by_category for AQI numeric range queries (more than X / less than X)
        // possibly filtered by region — "AQI more than 30 in Al Ain", "sites with AQI below 100 in Abu Dhabi"
        var aqiRangeWords = new[] { "more than", "greater than", "above", "over", "less than", "below", "under", "between" };
        var hasAqiRange = lowerQuestion.Contains("aqi") && aqiRangeWords.Any(k => lowerQuestion.Contains(k));
        if (template.Id == "site_aqi_by_category" && hasAqiRange)
        {
            maxScore = Math.Max(maxScore, 0.96);
        }
        // Penalise region_aqi_geographical for AQI range queries — that template gives regional averages,
        // not a filtered list of sites. site_aqi_by_category returns per-site results with range filter.
        if (template.Id == "region_aqi_geographical" && hasAqiRange)
        {
            maxScore *= 0.05;
        }

        // Boost sites_all_regions when user asks to list/show sites across all regions.
        var isAllRegionsSiteListQuestion =
            (lowerQuestion.Contains("all region") || lowerQuestion.Contains("each region") ||
             lowerQuestion.Contains("across region") || lowerQuestion.Contains("region wise") ||
             lowerQuestion.Contains("region-wise") || lowerQuestion.Contains("by region"))
            && (lowerQuestion.Contains("site") || lowerQuestion.Contains("station"))
            && !lowerQuestion.Contains("aqi") && !lowerQuestion.Contains("pollutant")
            && !lowerQuestion.Contains("co2") && !lowerQuestion.Contains("pm");

        if (template.Id == "sites_all_regions" && isAllRegionsSiteListQuestion)
        {
            maxScore = Math.Max(maxScore, 0.97);
        }

        // Penalise pollutant/school templates when user is asking for site listing across all regions.
        if ((template.Id == "schools_latest_pollutant" || template.Id == "sector_latest_pollutant"
             || template.Id == "sites_count_by_region")
            && isAllRegionsSiteListQuestion)
        {
            maxScore *= 0.05;
        }

        // Penalise schools_latest_pollutant / sector_latest_pollutant when the question
        // is asking for AQI region-wise — those return per-site lists, not regional summaries.
        // "region wise" or "by region" + "aqi" without school/sector context must go to region_aqi_geographical.
        var isRegionWiseAqiQuestion =
            (lowerQuestion.Contains("region wise") || lowerQuestion.Contains("regionwise") ||
             lowerQuestion.Contains("region-wise") || lowerQuestion.Contains("by region") ||
             lowerQuestion.Contains("per region") || lowerQuestion.Contains("all region") ||
             lowerQuestion.Contains("each region") || lowerQuestion.Contains("across region"))
            && (lowerQuestion.Contains("aqi") || lowerQuestion.Contains("air quality"));

        if ((template.Id == "schools_latest_pollutant" || template.Id == "sector_latest_pollutant"
             || template.Id == "site_aqi_all" || template.Id == "site_aqi_ranking")
            && isRegionWiseAqiQuestion)
        {
            maxScore *= 0.05;
        }

        // Penalise current_all_stations when the question is about listing sites by sector —
        // those should route to sites_by_sector, not current_all_stations which requires
        // a @parameterName and will throw a SQL error.
        if (template.Id == "current_all_stations"
            && isSectorListQuestion
            && (lowerQuestion.Contains("site") || lowerQuestion.Contains("station") || lowerQuestion.Contains("show") || lowerQuestion.Contains("list") || lowerQuestion.Contains("all")))
        {
            maxScore *= 0.05;
        }

        // Boost faq_answer for export/report-limit questions — these are how-to/policy questions
        // about HAWAQM features, not data queries. A school name in the question must not steal
        // routing to schools_latest_pollutant or site_readings_all.
        var exportReportPhrases = new[] {
            "raw-data export", "raw data export", "one-year export", "one year export",
            "prepare a one-year", "prepare an export", "export for a school", "export for a site",
            "year of data export", "year of raw data", "background export", "download link emailed",
            "statistical report", "exceedance report", "six months of exceedance",
            "split into two", "90-day", "90 day", "how should i split",
            "averaging intervals for",
            "site average hiding", "hiding a problem on", "device offset",
            "site aqi look acceptable while one device"
        };
        bool isExportOrReportQuestion = exportReportPhrases.Any(p => lowerQuestion.Contains(p));
        if (template.Id == "faq_answer" && isExportOrReportQuestion)
        {
            maxScore = Math.Max(maxScore, 0.96);
        }
        if (template.Id != "faq_answer" && isExportOrReportQuestion)
        {
            maxScore *= 0.05;
        }

        // Boost priority_hotspots when user asks for the hotspot site LIST (not just count).
        var hotspotListPhrases = new[] {
            "priority hotspot", "priority hotspots", "what are the hotspots", "show hotspots",
            "list hotspots", "hotspot sites", "which sites are hotspot", "which sites are critical",
            "show critical sites", "list critical sites", "sites flagged as critical",
            "dashboard hotspot", "executive dashboard hotspot", "sites with critical aqi"
        };
        bool isHotspotList = hotspotListPhrases.Any(p => lowerQuestion.Contains(p));
        if (template.Id == "priority_hotspots" && isHotspotList)
            maxScore = Math.Max(maxScore, 0.97);
        // Boost data_success_rate for overall AQI / network AQI questions from dashboard
        var overallAqiPhrases = new[] {
            "overall aqi", "network aqi", "aqi for the application", "aqi from dashboard",
            "executive dashboard aqi", "current aqi overview", "current overall aqi"
        };
        bool isOverallAqi = overallAqiPhrases.Any(p => lowerQuestion.Contains(p));
        if (template.Id == "data_success_rate" && isOverallAqi)
            maxScore = Math.Max(maxScore, 0.97);
        // When user asks for hotspot LIST, penalise data_success_rate (gives only count) and site_aqi_by_category
        if (isHotspotList && template.Id is "data_success_rate" or "site_aqi_by_category" or "faq_answer")
            maxScore *= 0.05;

        // Boost mold_reports for any question about mold tests, biological reports, or mold status.
        var moldPhrases = new[] {
            "mold report", "mold test", "mold status", "biological report", "biological test",
            "mold result", "mold reports", "mold testing", "show mold", "site mold"
        };
        bool isMoldQuestion = moldPhrases.Any(p => lowerQuestion.Contains(p));
        if (template.Id == "mold_reports" && isMoldQuestion)
            maxScore = Math.Max(maxScore, 0.97);
        if (template.Id != "mold_reports" && isMoldQuestion)
            maxScore *= 0.05;

        // Penalise ALL SQL templates (not faq_answer) when the question asks about
        // historical site-level data — HAWAQM does not store historical site-level records.
        // Historical questions must route to faq_answer which returns a "not available" response.
        if (template.Id != "faq_answer" && IsHistoricalSiteLevelQuestion(lowerQuestion))
        {
            maxScore *= 0.05;
        }

        // Penalise ALL SQL templates (not faq_answer) when the question asks to PROVE or EXPLAIN
        // causation — HAWAQM can show correlating data but cannot prove a cause.
        // These questions must route to faq_answer which returns the appropriate scoped answer.
        if (template.Id != "faq_answer" && IsCausationQuestion(lowerQuestion))
        {
            maxScore *= 0.05;
        }

        // Boost site_device_summary for device-per-site questions:
        // "sites without devices", "devices per site", "active/inactive devices per site",
        // "last reading per site", "best AQI device per site"
        var isDevicePerSiteQuestion =
            lowerQuestion.Contains("without device") ||
            lowerQuestion.Contains("no device") ||
            (lowerQuestion.Contains("don't have") && lowerQuestion.Contains("device")) ||
            lowerQuestion.Contains("devices per site") ||
            lowerQuestion.Contains("device per site") ||
            lowerQuestion.Contains("devices at each site") ||
            lowerQuestion.Contains("devices for each site") ||
            lowerQuestion.Contains("device count per site") ||
            lowerQuestion.Contains("devices by site") ||
            lowerQuestion.Contains("device summary") ||
            lowerQuestion.Contains("device status per site") ||
            (lowerQuestion.Contains("active") && lowerQuestion.Contains("inactive") && lowerQuestion.Contains("device") && lowerQuestion.Contains("site")) ||
            (lowerQuestion.Contains("last reading") && lowerQuestion.Contains("site") && lowerQuestion.Contains("device")) ||
            (lowerQuestion.Contains("best aqi") && lowerQuestion.Contains("device")) ||
            (lowerQuestion.Contains("better in aqi") && lowerQuestion.Contains("device"));

        if (template.Id == "site_device_summary" && isDevicePerSiteQuestion)
        {
            maxScore = Math.Max(maxScore, 0.97);
        }

        // Penalise templates that would wrongly capture device-per-site questions
        if ((template.Id == "online_devices" || template.Id == "offline_devices" ||
             template.Id == "active_device_count" || template.Id == "devices_by_filter")
            && isDevicePerSiteQuestion)
        {
            maxScore *= 0.05;
        }

        return maxScore;
    }

    private static HashSet<string> Tokenize(string text)
    {
        return new HashSet<string>(
            text.ToLowerInvariant()
                .Split([' ', ',', '.', '?', '!', '"', '\'', '(', ')', '{', '}'], StringSplitOptions.RemoveEmptyEntries)
                .Where(t => t.Length > 1),  // keep 2+ char tokens so "al", "in" survive for region names
            StringComparer.Ordinal);
    }
}
