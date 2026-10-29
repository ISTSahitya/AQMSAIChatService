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

    // Keywords that clearly signal a device query.
    // Intentionally excludes hardcoded device model prefixes (BA, SEI100M) — QuestionHasDeviceName()
    // handles model detection generically so new device models need no code change here.
    private static readonly string[] DeviceKeywords =
        ["device", "sensor", "reading for", "data from"];

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

        // Debug: log top-5 candidates to diagnose routing decisions
        var top5 = scored.Take(5).Select(x => $"{x.Template?.Id}={x.Score:F4}");
        _log.Debug("Top-5 candidates: {Candidates}", string.Join(", ", top5));

        if (best.Score >= high)
        {
            // High confidence: bypass LLM and use template directly.
            // With HighConfidenceThreshold = 1.10 this is only reached for
            // score = 1.0 (indoor_source_analysis / dust_event_response hard overrides).
            return new QueryRouterResult
            {
                Template = best.Template,
                Score = best.Score,
                Candidates = []
            };
        }

        if (best.Score >= low)
        {
            // Always pass ALL permitted templates to the LLM — not just top 3.
            // The LLM routing prompt is more reliable than the keyword scorer for ambiguous questions.
            return new QueryRouterResult
            {
                Template = null,
                Score = best.Score,
                Candidates = permitted
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
         "ventilation caused", "ventilation system caused",
         "ventilation system cause", "did the ventilation",
         "cause the aqi spike", "cause yesterday's aqi",
         "cause the spike", "caused the aqi spike", "caused yesterday's aqi",
         "aqi spike", "yesterday's aqi spike", "school ventilation",
         "spike — ", "spike -", "spike at school", "spike at site",
         "ventilation spike", "aqi spike —", "aqi spike -"];

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

    /// <inheritdoc/>
    public async Task<IReadOnlyList<ApprovedQuery>> GetPermittedTemplatesAsync(UserContext user, CancellationToken ct = default)
    {
        var templates = await LoadTemplatesAsync(ct);
        return templates.Where(t => user.PermittedCategories.Contains(t.Category)).ToList();
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    private async Task<List<ApprovedQuery>> LoadTemplatesAsync(CancellationToken ct)
    {
        if (_cache.TryGetValue(CacheKey, out List<ApprovedQuery>? cached) && cached != null)
            return cached;

        var knowledgeDir = Path.Combine(AppContext.BaseDirectory, "Knowledge");
        var files = Directory.GetFiles(knowledgeDir, "*-queries.json");

        if (files.Length == 0)
        {
            _log.Error("No *-queries.json files found in {Dir}", knowledgeDir);
            return [];
        }

        var opts = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        var all = new List<ApprovedQuery>();

        foreach (var filePath in files)
        {
            await using var fs = File.OpenRead(filePath);
            var file = await JsonSerializer.DeserializeAsync<ApprovedQueriesFile>(fs, opts, ct);
            if (file?.Queries is { Count: > 0 } q)
                all.AddRange(q);
            else
                _log.Warning("No queries loaded from {File}", Path.GetFileName(filePath));
        }

        _cache.Set(CacheKey, all, TimeSpan.FromHours(1));
        _log.Information("Loaded {Count} approved query templates from {FileCount} files", all.Count, files.Length);
        return all;
    }

    // Parameter names that indicate a pollutant-reading request
    private static readonly HashSet<string> PollutantTokens = new(StringComparer.OrdinalIgnoreCase)
        { "co", "no2", "so2", "co2", "pm2.5", "pm10", "o3", "tvoc", "ch2o", "aqi", "temperature", "humidity", "voc", "noise" };

    // Regex-style check: question contains a device name pattern.
    //
    // DESIGN: intentionally device-prefix-agnostic so it works even if device naming
    // conventions change (new models, new prefixes). The DB fuzzy lookup in ChatController
    // resolves the exact name; this method only needs to detect "something that looks like
    // a device code" vs. a site/person name.
    //
    // Rules:
    //   1. Generic: word boundary + 1-8 uppercase letters optionally mixed with digits
    //      (the alphanumeric prefix of a device model name) immediately followed by
    //      1-4 digits, with optional internal spaces — e.g. "BA 0001", "SEI100M 0014",
    //      "XYZ 0042", collapsed "sei100m0009".
    //      Guard: must NOT be a pure-word token (no digits at all), which would match
    //      ordinary words. Require at least one digit in the suffix portion.
    //   2. Context guard: when the word "device" or "sensor" appears in the question,
    //      a short letter+digits token nearby is also treated as a device code —
    //      covers partial/typo inputs like "device B 0001".
    //
    // Current known prefixes for reference: BA, SEI100M
    // To add a new device model, NO code change is needed — the generic rule picks it up.
    private static bool QuestionHasDeviceName(string lowerQuestion)
    {
        // Rule 1a: generic device code pattern in original text
        // Letters (1-8) optionally interspersed with digit groups, then \s* then digits (1-4)
        // \b guards prevent matching ordinary words like "abu", "al", "o3" etc.
        // The negative lookahead (?!\s*[a-z]) prevents matching "pm 2" or "co 2" type tokens.
        if (System.Text.RegularExpressions.Regex.IsMatch(lowerQuestion,
            @"\b[a-z]{1,8}(?:\d+[a-z]+)*\s*\d{1,4}\b",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase))
        {
            // Filter out common false positives: pollutant names, region words, number phrases
            // that also match the generic pattern (e.g. "pm10", "no2", "co2", "o3", "pm2", "al1")
            var fp = System.Text.RegularExpressions.Regex.Matches(lowerQuestion,
                @"\b([a-z]{1,8}(?:\d+[a-z]+)*)\s*(\d{1,4})\b",
                System.Text.RegularExpressions.RegexOptions.IgnoreCase);

            foreach (System.Text.RegularExpressions.Match m in fp)
            {
                var prefix = m.Groups[1].Value.ToUpperInvariant();
                // Skip known non-device tokens
                if (prefix is "PM" or "NO" or "SO" or "CO" or "O" or "CH" or "VOC"
                         or "AL" or "PM2" or "NO2" or "SO2" or "CO2" or "O3" or "CH2"
                         or "AQI" or "TOP" or "P" or "S" or "M" or "N" or "H" or "C"
                         or "T" or "F" or "D" or "E" or "G" or "I" or "J" or "K"
                         or "L" or "Q" or "R" or "U" or "V" or "W" or "X" or "Y" or "Z"
                         // Site-name ordinals/sequences common in UAE school/building names
                         or "CYCLE" or "PHASE" or "STAGE" or "BLOCK" or "ZONE"
                         or "WING" or "GRADE" or "PART" or "UNIT" or "BUILDING"
                         or "FLOOR" or "SECTION" or "TYPE" or "CLASS" or "LEVEL")
                    continue;

                // Prefix must be at least 2 chars OR be "B" followed by context word "device"
                if (prefix.Length >= 2)
                    return true;
            }
        }

        // Rule 1b: collapse spaces and re-check — catches "SEI 100M 0008", "sei 100 m 0014"
        var collapsed = lowerQuestion.Replace(" ", "");
        if (System.Text.RegularExpressions.Regex.IsMatch(collapsed,
            @"\b[a-z]{2,}(?:\d+[a-z]+)*\d{1,4}\b",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase))
        {
            var fp2 = System.Text.RegularExpressions.Regex.Matches(collapsed,
                @"\b([a-z]{2,}(?:\d+[a-z]+)*)\d{1,4}\b",
                System.Text.RegularExpressions.RegexOptions.IgnoreCase);
            foreach (System.Text.RegularExpressions.Match m in fp2)
            {
                var prefix = m.Groups[1].Value.ToUpperInvariant();
                if (prefix is "PM" or "NO" or "SO" or "CO" or "O" or "CH" or "VOC"
                         or "PM2" or "NO2" or "SO2" or "CO2" or "CH2" or "AQI" or "TOP"
                         // Site-name ordinals/sequences common in UAE school/building names
                         or "CYCLE" or "PHASE" or "STAGE" or "BLOCK" or "ZONE"
                         or "WING" or "GRADE" or "PART" or "UNIT" or "BUILDING"
                         or "FLOOR" or "SECTION" or "TYPE" or "CLASS" or "LEVEL")
                    continue;
                return true;
            }
        }

        // Rule 2: "device/sensor" + any short code nearby (covers partial inputs like "device B 0001")
        if (System.Text.RegularExpressions.Regex.IsMatch(lowerQuestion,
            @"\b(device|sensor)\b.{0,40}\b[a-z]{1,8}(?:\d+[a-z]+)*\s*\d{1,4}\b",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase))
            return true;

        return false;
    }

    // Site-name indicators — words that appear in station names but NOT in region-only queries.
    // If any of these are present alongside a region word, the question is a site query, not a region query.
    // NOTE: "school" is intentionally NOT in this list — it is a generic noun used in "all schools",
    // "schools in Abu Dhabi" etc. and must NOT be treated as a site name indicator.
    // NOTE: "residential", "commercial", "institutional", "institution" are sector names — they are
    // checked separately via _sectorNameWords and excluded from site-name detection when paired with
    // "sector", "sites", "all", "show", "list" context (i.e. the user is filtering by sector, not naming a site).
    private static readonly string[] SiteNameIndicators =
    [
        "building", "facility", "centre", "center", "hospital", "clinic",
        "office", "park", "mall", "tower", "villa", "compound", "camp",
        "indian", "british", "american", "international", "national",
        "bateen", "saad", "naeem", "naim", "khalifa", "zayed", "hamdan",
        "mushrif", "mussafah", "baniyas", "karama", "shakhbout", "shahama",
        "madinat", "khalidiyah", "corniche", "mangrove", "reem", "yas",
        "wahda", "rowdah", "muroor", "electra", "hameem", "liwa", "gayathi",
        "reyada", "reayada", "kaltham", "obeidli", "gems", "british",
        "shawamekh", "al shawamekh"
    ];

    // Sector name words — "residential", "commercial", "institutional" appear in both sector filters
    // ("all residential sites") and actual site names ("Abudhabi Commercial Institutional Site1").
    // QuestionHasSiteName uses context to decide which interpretation applies.
    private static readonly string[] SectorNameWords =
        ["residential", "commercial", "institutional", "institution"];

    // Generic site-listing context words — when sector words appear with these, it's a sector filter,
    // not a specific site name (e.g. "all Abu Dhabi sites under Residential sector").
    private static readonly string[] SectorFilterContext =
        ["sector", "sites under", "all sites", "show me all", "show all", "list all",
         "sites in", "all abu dhabi sites", "all al ain sites", "all al dhafra sites",
         "all sites under", "sites under residential", "sites under commercial",
         "sites under institutional", "residential sector", "commercial sector", "institutional sector",
         "residential sites", "commercial sites", "institutional sites"];

    // Specific named-school indicators — unique proper nouns that only appear in a named school, not in
    // generic "schools in X" queries. Used to distinguish site_aqi_single from schools_latest_pollutant.
    private static readonly string[] NamedSchoolIndicators =
    [
        "reyada", "reayada", "saad", "naeem", "naim", "bateen", "gems", "kaltham",
        "shawamekh", "shahama", "baniyas", "mushrif", "khalifa",
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
        "schools latest", "schools with their", "all abu dhabi schools",
        "all al ain schools", "all al dhafra schools", "table of all schools"
    ];

    /// <summary>
    /// Returns true when the question contains site-name indicators beyond just a region keyword.
    /// Sector words (residential, commercial, institutional) are only treated as site-name indicators
    /// when NOT used in a sector-filter context (e.g. "all residential sites" is NOT a site name).
    /// </summary>
    private static bool QuestionHasSiteName(string lowerQuestion)
    {
        // Non-sector site name indicators — always count
        if (SiteNameIndicators.Any(indicator => lowerQuestion.Contains(indicator)))
            return true;

        // Sector words only count as site name indicators when NOT used as a sector filter
        bool hasSectorWord = SectorNameWords.Any(w => lowerQuestion.Contains(w));
        if (!hasSectorWord) return false;

        // If question is clearly filtering by sector (not naming a specific site), return false
        bool isSectorFilter = SectorFilterContext.Any(ctx => lowerQuestion.Contains(ctx));
        return !isSectorFilter;
    }

    // Words that indicate a device status check (not an FAQ about rules)
    private static readonly string[] DeviceStatusKeywords = [
        "active", "inactive", "online", "offline", "connected", "disconnected",
        "working", "sending", "send", "status",
        "last send", "last report", "last active", "last update", "last transmit",
        "last measured", "last data", "last communicated", "most recent data",
        "most recent transmission", "recent data transmission",
        "last time", "last reading", "last reported", "last communicate",
        "when was", "when did"
    ];

    // Shared compiled regex for specific-date detection (ISO, slash, month-name, year-only, relative)
    private static readonly System.Text.RegularExpressions.Regex _specificDateRegex =
        new System.Text.RegularExpressions.Regex(
            @"(yesterday|last\s+night|last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)|" +
            @"jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?|" +
            @"\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|" +
            @"\d{1,2}(st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|" +
            @"in\s+20\d{2}|on\s+20\d{2}|20\d{2})",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase | System.Text.RegularExpressions.RegexOptions.Compiled);

    private static double ScoreTemplate(ApprovedQuery template, HashSet<string> questionTokens, string originalQuestion)
    {
        if (template.Patterns.Count == 0) return 0;

        double maxScore = 0;
        var lowerQuestion = originalQuestion.ToLowerInvariant();

        // Hard override: live-data site templates must never win when a specific past date is mentioned.
        // Site historical data must always route to site_daily_averages / site_single_parameter_reading.
        var liveOnlySiteTemplates = new[] { "site_aqi_single", "site_readings_all", "main_pollutant_at_station" };
        if (liveOnlySiteTemplates.Contains(template.Id) && _specificDateRegex.IsMatch(lowerQuestion))
            return 0.0;

        // Hard override: site_aqi_single must never win when the question refers to schools generically.
        // "all schools", "abu dhabi schools", "schools with their CO2", "all abu dhabi schools with" etc.
        // are group queries → schools_latest_pollutant.
        if (template.Id == "site_aqi_single"
            && GenericSchoolPhrases.Any(p => lowerQuestion.Contains(p)))
            return 0.0;

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
            "how many devices are currently active",
            "how much data is being successfully received", "latest data success percentage",
            "how successful is the current data collection", "how much data are we getting",
            "data completeness right now", "data collection rate", "data reception rate",
            "how well are devices reporting", "how well is data being collected",
            "how complete is our data", "current data collection rate"
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

        // Penalise device_live_reading when question is about online/offline/active/inactive status —
        // those route to device_status, not device_live_reading which returns parameter readings.
        if (template.Id == "device_live_reading"
            && QuestionHasDeviceName(lowerQuestion)
            && (lowerQuestion.Contains("online") || lowerQuestion.Contains("offline")
                || lowerQuestion.Contains("active") || lowerQuestion.Contains("inactive")
                || lowerQuestion.Contains("connected") || lowerQuestion.Contains("disconnected")))
        {
            maxScore *= 0.05;
        }

        // Penalise device_live_reading when question asks WHEN the device last sent/reported/was active —
        // those route to device_status which queries ParameterReadingUpdateTime directly.
        var lastTimestampQuestionPhrases = new[] {
            "when did", "when was", "last time", "last send", "last sent", "last report",
            "last active", "last reading", "last data", "last transmit", "last measured",
            "last communicate", "most recent data transmission", "most recent transmission"
        };
        if (template.Id == "device_live_reading"
            && QuestionHasDeviceName(lowerQuestion)
            && lastTimestampQuestionPhrases.Any(p => lowerQuestion.Contains(p)))
        {
            maxScore *= 0.05;
        }

        // Boost all device templates when a device name pattern is present —
        // prevents site templates from stealing device queries (e.g. "PM2.5 for BA0010").
        if ((template.Id == "device_live_reading" || template.Id == "device_reading_history"
             || template.Id == "device_info" || template.Id == "device_status"
             || template.Id == "device_pollutants")
            && QuestionHasDeviceName(lowerQuestion))
        {
            maxScore = Math.Max(maxScore, 0.92);
        }

        // Boost device_status decisively when device name + status/timestamp question word is present.
        // Also covers "when did X last send/report/update" — timestamp questions about a specific device.
        if (template.Id == "device_status"
            && QuestionHasDeviceName(lowerQuestion)
            && DeviceStatusKeywords.Any(k => lowerQuestion.Contains(k)))
        {
            maxScore = Math.Max(maxScore, 0.97);
        }

        // Penalise devices_by_filter when a SPECIFIC device name is present —
        // single-device status queries go to device_status, not the list template.
        if (template.Id == "devices_by_filter"
            && QuestionHasDeviceName(lowerQuestion))
        {
            maxScore *= 0.05;
        }

        // Penalise device_info for "last send/report/update/active" questions with a device name —
        // those are timestamp questions that device_status answers, not metadata questions.
        var lastTimestampPhrases = new[] {
            "last send", "last report", "last active", "last update", "last transmit",
            "last measured", "last data", "last communicated", "most recent data",
            "most recent transmission", "when did", "when was"
        };
        if (template.Id == "device_info"
            && QuestionHasDeviceName(lowerQuestion)
            && lastTimestampPhrases.Any(p => lowerQuestion.Contains(p)))
        {
            maxScore *= 0.1;
        }

        // Penalise list_devices (system-wide count, admin only) for site-specific device questions
        var deviceAtSitePhrases = new[] {
            "devices at", "devices for", "devices installed at", "devices under",
            "sensors at", "sensors for", "sensors installed at", "sensors under",
            "devices available at", "available at", "list devices", "which devices are",
            "how many devices does", "how many devices at", "how many sensors does",
            "what devices are", "what sensors are", "device count at", "device count for"
        };
        bool isDeviceAtSiteQuestion = deviceAtSitePhrases.Any(p => lowerQuestion.Contains(p))
            && !QuestionHasDeviceName(lowerQuestion)
            && (QuestionHasSiteName(lowerQuestion) || lowerQuestion.Contains("school") || lowerQuestion.Contains("site") || lowerQuestion.Contains("station"));

        if (template.Id == "list_devices" && isDeviceAtSiteQuestion)
            maxScore *= 0.05;

        // Boost devices_by_filter for site-specific device listing ("list devices for <school>",
        // "devices at <site>", "what devices are installed at <station>")
        if (template.Id == "devices_by_filter" && isDeviceAtSiteQuestion)
            maxScore = Math.Max(maxScore, 0.95);

        // Route device_info when question has device name + metadata keywords (site/region/sector/deployed etc.)
        var deviceMetaKeywords = new[] {
            "which site", "what site", "site name", "which station", "what station",
            "which region", "what region", "which sector", "what sector",
            "deployed", "created", "installed", "registration date",
            "device model", "device type", "details", "info about", "tell me about",
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

        // Penalise device_live_reading when question is asking for metadata (not readings)
        if (template.Id == "device_live_reading"
            && QuestionHasDeviceName(lowerQuestion)
            && deviceMetaKeywords.Any(k => lowerQuestion.Contains(k))
            && !lowerQuestion.Contains("reading") && !lowerQuestion.Contains("value")
            && !lowerQuestion.Contains("pm2.5") && !lowerQuestion.Contains("co2")
            && !lowerQuestion.Contains("aqi") && !lowerQuestion.Contains("temperature"))
        {
            maxScore *= 0.1;
        }

        // Between device_live_reading and device_reading_history:
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
        // "current", "now", "latest", "live", "right now", "at this moment" → force device_live_reading
        var currentReadingWords = new[] { "current", "now", "latest", "live", "right now", "at this moment", "today's reading", "present" };
        bool isCurrentReading = currentReadingWords.Any(w => lowerQuestion.Contains(w));

        // Detect specific date mentions in any common format:
        // "2026-09-25", "25/09/2026", "25-09-2026", "Sep 25", "25 Sep 2026",
        // "yesterday", "last Monday", "on the 25th", year-only "in 2025"
        var hasSpecificDate = System.Text.RegularExpressions.Regex.IsMatch(lowerQuestion,
            @"\b(yesterday|last\s+night|last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)|" +
            @"jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?|" +
            @"\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|" +
            @"\d{1,2}(st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|" +
            @"in\s+20\d{2}|on\s+20\d{2}|20\d{2})\b");
        // Detect specific time mentions (e.g. "2:02am", "4am", "13:00")
        var hasSpecificTime = System.Text.RegularExpressions.Regex.IsMatch(lowerQuestion,
            @"\b(\d{1,2}:\d{2}\s*(am|pm)?|\d{1,2}\s*(am|pm))\b");

        // History signal: must have an interval word OR a specific date/time — NOT just "reading"
        bool isHistorySignal = !isCurrentReading &&
                               (historyIntervalWords.Any(w => lowerQuestion.Contains(w))
                                || hasSpecificDate || hasSpecificTime);

        // Force device_live_reading when "current/now/latest" is present with a device name
        if (template.Id == "device_live_reading"
            && QuestionHasDeviceName(lowerQuestion)
            && isCurrentReading)
        {
            maxScore = Math.Max(maxScore, 0.97);
        }

        if (template.Id == "device_live_reading"
            && QuestionHasDeviceName(lowerQuestion)
            && isHistorySignal)
        {
            maxScore *= 0.05; // let device_reading_history win decisively
        }
        // "Last 24 hours" with a device name but NO explicit interval (5min/1hour/8hour) and NO specific
        // time is a capability/FAQ question — route to faq_answer, not device_reading_history.
        var explicitIntervalWords = new[] {
            "5 min", "5min", "5-min", "5-minute",
            "1 hour", "1h", "1hour", "1-hour",
            "8 hour", "8h", "8hour", "8-hour"
        };
        bool isLast24HoursFaq = QuestionHasDeviceName(lowerQuestion)
            && (lowerQuestion.Contains("last 24") || lowerQuestion.Contains("24 hours"))
            && !explicitIntervalWords.Any(w => lowerQuestion.Contains(w))
            && !hasSpecificTime;

        if (template.Id == "device_reading_history"
            && QuestionHasDeviceName(lowerQuestion)
            && isHistorySignal
            && !isLast24HoursFaq)
        {
            maxScore = Math.Max(maxScore, 0.97); // beats all non-device templates (0.95) and device_live_reading
        }

        // Boost faq_answer when user asks about last 24 hours for a device without specifying interval —
        // this is a capability question, not a data request.
        if (template.Id == "faq_answer" && isLast24HoursFaq)
        {
            maxScore = Math.Max(maxScore, 0.97);
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
            "device_live_reading", "device_status", "device_info",
            "device_pollutants", "device_reading_history"
        };
        if (deviceTemplates.Contains(template.Id) && !QuestionHasDeviceName(lowerQuestion))
        {
            maxScore *= 0.02;
        }

        // Detect generic school questions: "Abu Dhabi schools", "schools in Al Ain", "all schools" etc.
        // These refer to the school SECTOR and must route to schools_latest_pollutant, NOT site_aqi_single.
        // Explicitly exclude sector-filter questions — "sites under Residential sector" is NOT a school question
        // even if a GenericSchoolPhrase substring happens to match.
        bool isSectorFilterQ = SectorFilterContext.Any(ctx => lowerQuestion.Contains(ctx));
        bool isGenericSchoolQuestion = !isSectorFilterQ
            && (GenericSchoolPhrases.Any(p => lowerQuestion.Contains(p))
                || (lowerQuestion.Contains("schools") && !NamedSchoolIndicators.Any(k => lowerQuestion.Contains(k))));

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
            || lowerQuestion.Contains("temperature") || lowerQuestion.Contains("humidity")
            || lowerQuestion.Contains("categor") || lowerQuestion.Contains("fall under")
            || lowerQuestion.Contains("good or bad") || lowerQuestion.Contains("bad or good")
            || lowerQuestion.Contains("how is the air") || lowerQuestion.Contains("tell me the air")
            || lowerQuestion.Contains("air quality today") || lowerQuestion.Contains("quality today")
            || lowerQuestion.Contains("current air") || lowerQuestion.Contains("air today");
        // Do NOT boost site_aqi_single when a past date is present — historical site data
        // must route to site_daily_averages / site_single_parameter_reading instead.
        if (template.Id == "site_aqi_single" && hasSiteName && isAqiOrReadingQuestion && !isGenericSchoolQuestion && !hasSpecificDate)
            maxScore = Math.Max(maxScore, 0.96);

        // Detect DSR / AQI-exceedance questions — these must route to the DSR/exceedance templates,
        // NOT to schools_latest_pollutant, even though they mention "schools" generically.
        var dsrExceedanceSignals = new[] {
            "data success rate", "dsr", "repeatedly exceeded", "repeated exceedance",
            "exceeded aqi", "exceed aqi", "exceeded threshold", "exceed threshold",
            "exceedance with", "exceedance and", "active days", "device active",
            "below 75%", "above 75%", "below 80%", "above 80%", "at least 75%",
            "at least 80%", "less than 75%", "less than 80%", "success rate below",
            "success rate above", "success rate of at least", "data availability below",
            "data availability above", "dsr below", "dsr above"
        };
        bool isDsrOrExceedanceQuestion = dsrExceedanceSignals.Any(s => lowerQuestion.Contains(s));

        // Boost schools_latest_pollutant for generic school questions with a pollutant signal.
        // "Abu Dhabi schools CO2", "schools in Al Ain with AQI", "all schools PM2.5" etc.
        // But NOT when the question is about DSR or AQI exceedance — those route to dedicated templates.
        // AQI category questions (good/moderate/unhealthy) are handled below by site_aqi_by_category boost.
        var aqiQualitySignals = new[] {
            "good", "moderate", "unhealthy", "hazardous", "very unhealthy", "critical",
            "hotspot", "safe", "unsafe", "bad air", "poor air", "healthy air",
            "sensitive groups", "vulnerable", "harmful", "dangerous"
        };
        bool hasAqiQualitySignal = aqiQualitySignals.Any(k => lowerQuestion.Contains(k));

        // Detect device status questions — user wants device list, not site/pollutant data.
        // "online devices in all schools", "offline devices in school sector", "active devices in schools" etc.
        var deviceStatusWords = new[] { "online devices", "offline devices", "active devices", "inactive devices" };
        bool isDeviceStatusQuestion = deviceStatusWords.Any(w => lowerQuestion.Contains(w));

        if (template.Id == "schools_latest_pollutant" && isGenericSchoolQuestion && !isDsrOrExceedanceQuestion && !hasAqiQualitySignal && !isDeviceStatusQuestion)
            maxScore = Math.Max(maxScore, 0.96);
        // Penalise schools_latest_pollutant for DSR/exceedance questions
        if (template.Id == "schools_latest_pollutant" && isDsrOrExceedanceQuestion)
            maxScore *= 0.05;
        // Penalise schools_latest_pollutant for device status questions — must route to devices_by_filter
        if (template.Id == "schools_latest_pollutant" && isDeviceStatusQuestion)
            maxScore *= 0.02;
        // Boost devices_by_filter for device status questions mentioning schools/sector
        if (template.Id == "devices_by_filter" && isDeviceStatusQuestion && isGenericSchoolQuestion)
            maxScore = Math.Max(maxScore, 0.97);

        // Boost DSR/exceedance templates when the question contains DSR or exceedance signals
        if ((template.Id == "sites_repeated_aqi_exceedance_with_dsr"
             || template.Id == "sites_dsr_below_threshold"
             || template.Id == "sites_dsr_above_threshold")
            && isDsrOrExceedanceQuestion)
        {
            maxScore = Math.Max(maxScore, 0.97);
        }
        // Penalise site_aqi_single for generic school questions — must not ask for a site name
        if (template.Id == "site_aqi_single" && isGenericSchoolQuestion)
            maxScore *= 0.1;
        // Penalise pollutant_trend_hourly and sector_latest_pollutant for generic school questions —
        // "show me a table of all Abu Dhabi schools with their latest CO2 reading" must not
        // match on "show me readings" or "latest co2 for"; it is a schools-sector list query.
        if (template.Id == "pollutant_trend_hourly" && isGenericSchoolQuestion)
            maxScore *= 0.05;
        if (template.Id == "sector_latest_pollutant" && isGenericSchoolQuestion)
            maxScore *= 0.05;

        // Boost region_aqi_geographical when question has a region keyword + AQI/air quality signal
        // but NO specific site name — e.g. "what is the AQI in AL Ain", "AQI in Abudhabi"
        bool hasRegionKeyword = RegionKeywords.Any(k => lowerQuestion.Contains(k));
        bool isAqiQuestion = lowerQuestion.Contains("aqi") || lowerQuestion.Contains("air quality");
        if (template.Id == "region_aqi_geographical" && hasRegionKeyword && isAqiQuestion && !hasSiteName)
            maxScore = Math.Max(maxScore, 0.96);

        // Penalise site_aqi_all when a region keyword is present — region queries must go to region_aqi_geographical
        if (template.Id == "site_aqi_all" && hasRegionKeyword && isAqiQuestion && !hasSiteName)
            maxScore *= 0.1;

        // Penalise site_aqi_single when the question mentions a region explicitly —
        // "Abu Dhabi region", "Al Ain region" etc. are region queries, not named-site queries.
        if (template.Id == "site_aqi_single" && hasRegionKeyword && !hasSiteName)
            maxScore *= 0.05;

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

        // Detect if the question is about listing sites by sector
        // (commercial, residential, public/govt school) — used by multiple scoring blocks below.
        bool isSectorListQuestion =
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

        if (isSiteListQuestion && (template.Id == "list_stations" || template.Id == "sites_filtered"))
            maxScore = Math.Max(maxScore, 0.93);

        // Strongly boost list_stations when user asks for a SECTOR-filtered site list —
        // "what are the sites under commercial sector", "sites under residential" etc.
        // must not route to chart templates.
        bool isSectorSiteListQuestion = isSiteListQuestion && isSectorListQuestion;
        if (template.Id == "list_stations" && isSectorSiteListQuestion)
            maxScore = Math.Max(maxScore, 0.97);

        // Boost sites_by_filter and penalise sites_filtered for sector-based listing questions.
        // sites_filtered has no sectorName param — it cannot filter by sector at all.
        if (isSectorListQuestion && template.Id == "sites_by_filter")
            maxScore = Math.Max(maxScore, 0.98);
        if (isSectorListQuestion && template.Id == "sites_filtered")
            maxScore *= 0.05;

        // Penalise chart templates when user asks for a plain site list (no chart keywords) —
        // "what are the sites under commercial sector" is NOT a chart request.
        var chartTemplates = new[] {
            "schools_aqi_pie", "site_aqi_by_category", "sites_by_sector_chart",
            "sites_by_region_chart", "aqi_by_region_chart", "pollutant_by_sector_chart"
        };
        bool hasChartKeyword = lowerQuestion.Contains("pie") || lowerQuestion.Contains("chart")
            || lowerQuestion.Contains("graph") || lowerQuestion.Contains("bar chart")
            || lowerQuestion.Contains("donut") || lowerQuestion.Contains("distribution");
        if (chartTemplates.Contains(template.Id) && isSiteListQuestion && !hasChartKeyword)
            maxScore *= 0.02;

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

        // Penalise sites_by_filter / list_stations when the question explicitly
        // mentions "device" or "sensor" — those templates return sites, not devices.
        // Device listing questions must route to devices_by_filter.
        if ((template.Id == "sites_by_filter" || template.Id == "list_stations")
            && (lowerQuestion.Contains("device") || lowerQuestion.Contains("sensor")))
        {
            maxScore *= 0.05;
        }

        // Penalise site_count and sites_by_filter when user is asking for sector/subsector LIST or COUNT
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

        if ((template.Id == "site_count" || template.Id == "sites_by_filter")
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
            "aqi level for good", "aqi level for moderate", "what level is good", "what level is moderate",
            "hazardous aqi value", "hazardous aqi range", "aqi value for hazardous",
            "aqi value for good", "aqi value for moderate", "aqi value for unhealthy",
            "aqi value for very unhealthy", "good aqi value", "moderate aqi value",
            "unhealthy aqi value", "very unhealthy aqi value", "very unhealthy aqi range",
            "tell me the aqi value for", "aqi value hazardous", "aqi value good",
            "aqi value moderate", "aqi value unhealthy", "which aqi values are classified",
            "what aqi values are classified", "what aqi values fall under",
            "which aqi values fall", "aqi values classified as", "aqi numbers mean",
            "what range of aqi is classified", "which range of aqi is classified",
            "range of aqi is classified", "range of aqi classified",
            "aqi range classified as", "classified as unhealthy for sensitive",
            "classified as good", "classified as moderate", "classified as hazardous",
            "classified as very unhealthy", "classified as unhealthy",
            "fall under the unhealthy", "fall under the good", "fall under the moderate",
            "fall under the hazardous", "fall within the", "aqi fall under",
            "unhealthy for sensitive groups", "aqi range is unhealthy for sensitive",
            "what aqi range is unhealthy", "what aqi range is good", "what aqi range is moderate",
            "what aqi range is hazardous", "what aqi range is very unhealthy",
            "aqi range is good", "aqi range is moderate", "aqi range is hazardous",
            "aqi range is very unhealthy", "aqi is unhealthy for sensitive",
            "aqi for sensitive groups", "sensitive groups aqi",
            "fall within the moderate", "fall within the good", "fall within the unhealthy",
            "fall within the hazardous", "fall within the very unhealthy",
            "within the moderate category", "within the good category",
            "within the unhealthy category", "within the hazardous category",
            "within the very unhealthy category", "within the sensitive",
            "under the moderate category", "under the good category",
            "under the unhealthy category", "under the hazardous category",
            "values fall within", "aqi values fall",
            "range for good air", "range for moderate air", "range for unhealthy air",
            "range for hazardous air", "range for good air quality",
            "range for moderate air quality", "what's the range for",
            "whats the range for", "range for good", "range for moderate",
            "range for unhealthy", "range for hazardous", "range for very unhealthy",
            "acceptable aqi range for", "what is the acceptable aqi",
            "aqi range for the good", "aqi range for the moderate",
            "aqi range for the unhealthy", "aqi range for the hazardous",
            "aqi range for the very unhealthy", "aqi range for the sensitive",
            "when is aqi considered good", "when is aqi considered moderate",
            "when is aqi considered unhealthy", "when is aqi considered hazardous",
            "aqi numbers mean", "what aqi numbers mean",
            "aqi values fall under the good", "aqi values fall under the moderate",
            "aqi values fall under the unhealthy", "aqi values fall under the hazardous",
            "values fall under the good", "values fall under the moderate",
            // temperature / humidity / noise FAQ phrases
            "does hawaqm measure temperature", "does hawaqm monitor temperature",
            "does hawaqm track temperature", "does hawaqm measure humidity",
            "does hawaqm track humidity", "does hawaqm monitor humidity",
            "does hawaqm measure noise", "does hawaqm monitor noise",
            "does hawaqm track noise", "does hawaqm also measure",
            "does hawaqm also monitor", "does hawaqm also track",
            "does it track noise", "are temperature, humidity and noise",
            "are temperature and humidity monitored", "besides pollutants",
            "temperature, humidity and ambient", "temperature and humidity too",
            "track noise as well", "measure noise as well",
            "does hawaqm track environmental",
            // site AQI averaging FAQ phrases
            "how does the system calculate the overall aqi for a site",
            "how is the overall aqi for a site calculated",
            "how is site aqi calculated for multiple devices",
            "how does site aqi work with more than one device",
            "site with more than one device", "site with multiple devices"
        };
        bool isAqiRangeDefinitionQuestion = aqiRangeDefinitionPhrases.Any(k => lowerQuestion.Contains(k));
        // Guard: site-list questions override the FAQ boost even when a definition phrase matched.
        var siteListAqiPhrases = new[] {
            "show me sites", "show sites", "list sites", "which sites", "find sites",
            "sites with good", "sites with moderate", "sites with unhealthy", "sites with hazardous",
            "sites where aqi", "sites in good", "sites in moderate", "sites in unhealthy",
            "sites for good", "sites for moderate", "sites for hazardous",
            "sites currently", "sites that are", "good air quality sites", "critical sites",
            "hotspot sites", "schools with good", "schools with moderate", "schools where aqi"
        };
        bool isSiteListAqiQuery = siteListAqiPhrases.Any(k => lowerQuestion.Contains(k));
        if (template.Id == "site_aqi_by_category" && isAqiRangeDefinitionQuestion && !isSiteListAqiQuery)
            maxScore *= 0.02;
        if (template.Id == "faq_answer" && isAqiRangeDefinitionQuestion && !isSiteListAqiQuery)
            maxScore = Math.Max(maxScore, 0.98);

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

        // Detect count vs list intent — used by several blocks below.
        var mentionedRegionCount =
            (lowerQuestion.Contains("abu dhabi") || lowerQuestion.Contains("abudhabi") ? 1 : 0) +
            (lowerQuestion.Contains("al ain") || lowerQuestion.Contains("alain") ? 1 : 0) +
            (lowerQuestion.Contains("al dhafra") || lowerQuestion.Contains("aldhafra") || lowerQuestion.Contains("dhafra") ? 1 : 0);

        var countWords = new[] { "how many", "total", "count", "number of" };
        bool isCountIntent = countWords.Any(w => lowerQuestion.Contains(w));
        var listIntentPhrases = new[] { "what are the sites", "which sites are", "show me sites",
            "show sites", "list sites", "give me sites", "what sites are", "sites under", "sites in",
            "show me active sites", "show me inactive sites", "what are the active sites",
            "what are the inactive sites", "which sites are active", "which sites are inactive",
            "active sites in", "inactive sites in", "show active sites", "show inactive sites" };
        bool isListIntent = listIntentPhrases.Any(p => lowerQuestion.Contains(p));

        // Boost site_count for any count question about sites/stations.
        var hasCountWord = lowerQuestion.Contains("how many") || lowerQuestion.Contains("total") ||
                           lowerQuestion.Contains("count") || lowerQuestion.Contains("number of");
        var hasSiteWord = lowerQuestion.Contains("site") || lowerQuestion.Contains("station");
        if (template.Id == "site_count" && hasCountWord && hasSiteWord)
            maxScore = Math.Max(maxScore, 0.95);

        // Penalise site_count for list-intent questions — those go to sites_by_filter.
        if (template.Id == "site_count" && isListIntent && !isCountIntent)
            maxScore *= 0.05;

        // Penalise sites_by_filter for count-intent questions — routes to site_count instead.
        if (template.Id == "sites_by_filter" && isCountIntent && hasSiteWord && !isListIntent)
            maxScore *= 0.3;

        // Penalise list_stations when the question mentions a specific region name —
        // those should route to sites_by_filter which filters by region.
        if (template.Id == "list_stations"
            && RegionKeywords.Any(k => lowerQuestion.Contains(k)))
        {
            maxScore *= 0.05;
        }

        // Boost sector_latest_pollutant ONLY when question names a sector AND an explicit pollutant/reading.
        // "Show me Abu Dhabi commercial sites with their latest CO2" → boost.
        // "What are the sites under commercial in Abu Dhabi" (no pollutant) → penalise, goes to sites_by_filter.
        var hasPollutantWord = lowerQuestion.Contains("co2") || lowerQuestion.Contains("pm2.5")
            || lowerQuestion.Contains("pm10") || lowerQuestion.Contains("co ") || lowerQuestion.Contains(" co ")
            || lowerQuestion.Contains("no2") || lowerQuestion.Contains("so2") || lowerQuestion.Contains("o3")
            || lowerQuestion.Contains("tvoc") || lowerQuestion.Contains("ch2o")
            || lowerQuestion.Contains("temperature") || lowerQuestion.Contains("humidity")
            || lowerQuestion.Contains("reading") || lowerQuestion.Contains("aqi");
        bool isSectorPollutantQuestion = isSectorListQuestion && hasPollutantWord && !isGenericSchoolQuestion;
        if (template.Id == "sector_latest_pollutant" && isSectorPollutantQuestion)
            maxScore = Math.Max(maxScore, 0.97);
        // Penalise sector_latest_pollutant when no pollutant is mentioned — it's a plain site list question
        if (template.Id == "sector_latest_pollutant" && isSectorListQuestion && !hasPollutantWord)
            maxScore *= 0.05;
        // Penalise site_aqi_single for sector-pollutant questions — must not resolve to one site
        if (template.Id == "site_aqi_single" && isSectorPollutantQuestion)
            maxScore *= 0.05;

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

        // Penalise active_device_count when a region or sector is mentioned —
        // that template has no filter; region/sector device questions go to devices_by_filter.
        if (template.Id == "active_device_count"
            && (RegionKeywords.Any(k => lowerQuestion.Contains(k)) || isSectorListQuestion))
        {
            maxScore *= 0.1;
        }

        // Penalise sites_by_filter when a sector word is also present —
        // sector+region questions must go to sites_by_filter which applies BOTH filters.
        // sites_by_filter only filters by region and would return all sectors.
        if (template.Id == "sites_by_filter" && isSectorListQuestion)
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

        // Penalise schools_latest_pollutant when the question mentions a region + AQI but
        // has NO school/sector context — it's a regional AQI question, not a school query.
        if (template.Id == "schools_latest_pollutant"
            && hasRegionKeyword && isAqiQuestion
            && !lowerQuestion.Contains("school") && !lowerQuestion.Contains("sector"))
        {
            maxScore *= 0.02;
        }

        // Detect a numeric AQI threshold phrase — "above 50", "more than 30", "higher than 100" etc.
        var hasNumericThreshold = System.Text.RegularExpressions.Regex.IsMatch(lowerQuestion, @"\b\d+\b")
            && (lowerQuestion.Contains("above") || lowerQuestion.Contains("below") || lowerQuestion.Contains("more than")
                || lowerQuestion.Contains("less than") || lowerQuestion.Contains("greater than")
                || lowerQuestion.Contains("higher than") || lowerQuestion.Contains("lower than")
                || lowerQuestion.Contains("over") || lowerQuestion.Contains("under") || lowerQuestion.Contains("exceeds")
                || lowerQuestion.Contains("between") || lowerQuestion.Contains("at least") || lowerQuestion.Contains("at most"));

        // Detect whether the question is scoped to the school sector
        bool isSchoolSectorQuestion =
            lowerQuestion.Contains("school") || lowerQuestion.Contains("schools") ||
            lowerQuestion.Contains("govt school") || lowerQuestion.Contains("government school") ||
            lowerQuestion.Contains("private school") || lowerQuestion.Contains("public school") ||
            lowerQuestion.Contains("moe school") || lowerQuestion.Contains("moe schools") ||
            lowerQuestion.Contains(" moe ") || lowerQuestion.StartsWith("moe ") || lowerQuestion.EndsWith(" moe");

        // Penalise schools_latest_pollutant when the question has a numeric threshold
        // AND there is NO school context — e.g. "how many sites have AQI above 30" → site_aqi_by_category.
        // DO NOT penalise when school context is present — schools_latest_pollutant supports minValue/maxValue.
        if (template.Id == "schools_latest_pollutant" && hasNumericThreshold && !isSchoolSectorQuestion)
        {
            maxScore *= 0.02;
        }

        // Boost schools_latest_pollutant when school context + numeric threshold present —
        // "Al Ain schools with AQI higher than 50" must go to schools_latest_pollutant, not site_aqi_by_category.
        if (template.Id == "schools_latest_pollutant" && hasNumericThreshold && isSchoolSectorQuestion
            && (lowerQuestion.Contains("aqi") || lowerQuestion.Contains("air quality")))
        {
            maxScore = Math.Max(maxScore, 0.95);
        }

        // Boost site_aqi_by_category when question has a numeric AQI threshold —
        // but NOT when the question is about schools (those stay in schools_latest_pollutant).
        if (template.Id == "site_aqi_by_category" && hasNumericThreshold
            && (lowerQuestion.Contains("aqi") || lowerQuestion.Contains("air quality"))
            && !isSchoolSectorQuestion)
        {
            maxScore = Math.Max(maxScore, 0.97);
        }

        // Penalise site_aqi_by_category when school context is present — school threshold questions
        // must stay in schools_latest_pollutant which returns the school sector data.
        if (template.Id == "site_aqi_by_category" && isSchoolSectorQuestion && hasNumericThreshold)
        {
            maxScore *= 0.05;
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
        // Use 1.15 to exceed HighConfidenceThreshold (1.10) and bypass the LLM entirely —
        // the LLM has been observed picking schools_latest_pollutant for school+category questions.
        if (template.Id == "site_aqi_by_category" && isAqiCategoryQuestion && !isPieChartQuestion)
        {
            maxScore = Math.Max(maxScore, 1.15);
        }

        // Penalise sites_by_filter / list_stations / sites_by_filter when the question
        // asks for an AQI category — those templates return site lists without AQI data.
        if ((template.Id == "sites_by_filter" || template.Id == "list_stations" || template.Id == "sites_by_filter")
            && isAqiCategoryQuestion)
        {
            maxScore *= 0.05;
        }

        // Detect comparative category phrases: "worse than Moderate", "better than Good", "beyond Unhealthy" etc.
        // These are AQI range queries (not category queries) — must route to site_aqi_by_category with minValue/maxValue.
        var comparativeCategoryPhrases = new[] {
            "worse than good", "worse than moderate", "worse than unhealthy", "worse than very unhealthy", "worse than hazardous",
            "better than moderate", "better than unhealthy", "better than very unhealthy", "better than hazardous",
            "beyond good", "beyond moderate", "beyond unhealthy",
            "above good", "above moderate", "above unhealthy",
            "more than good", "more than moderate", "more than unhealthy",
            "not good", "not moderate", "no longer good",
            "outside good", "outside moderate",
            "air quality is worse", "air quality worse than", "quality worse than"
        };
        bool hasComparativeCategoryPhrase = comparativeCategoryPhrases.Any(p => lowerQuestion.Contains(p));

        // Boost site_aqi_by_category for AQI numeric range queries (more than X / less than X)
        // possibly filtered by region — "AQI more than 30 in Al Ain", "sites with AQI below 100 in Abu Dhabi"
        var aqiRangeWords = new[] { "more than", "greater than", "above", "over", "less than", "below", "under", "between" };
        var hasAqiRange = lowerQuestion.Contains("aqi") && aqiRangeWords.Any(k => lowerQuestion.Contains(k));
        if (template.Id == "site_aqi_by_category" && (hasAqiRange || hasComparativeCategoryPhrase))
        {
            // Use 1.15 to bypass LLM for comparative phrases too — same as category questions
            maxScore = Math.Max(maxScore, hasComparativeCategoryPhrase ? 1.15 : 0.96);
        }
        // Penalise schools_latest_pollutant for comparative category phrases
        if (template.Id == "schools_latest_pollutant" && hasComparativeCategoryPhrase)
            maxScore *= 0.05;
        // Penalise region_aqi_geographical for AQI range queries — that template gives regional averages,
        // not a filtered list of sites. site_aqi_by_category returns per-site results with range filter.
        if (template.Id == "region_aqi_geographical" && hasAqiRange)
        {
            maxScore *= 0.05;
        }

        // COUNT BY GROUP questions → site_count with groupBy param (LLM sets this).
        var isCountBySectorQuestion =
            (lowerQuestion.Contains("count") || lowerQuestion.Contains("how many") ||
             lowerQuestion.Contains("total") || lowerQuestion.Contains("number of"))
            && (lowerQuestion.Contains("by sector") || lowerQuestion.Contains("per sector") ||
                lowerQuestion.Contains("each sector"))
            && (lowerQuestion.Contains("site") || lowerQuestion.Contains("station"));

        var isCountByRegionQuestion =
            (lowerQuestion.Contains("count") || lowerQuestion.Contains("how many") ||
             lowerQuestion.Contains("total") || lowerQuestion.Contains("number of"))
            && (lowerQuestion.Contains("by region") || lowerQuestion.Contains("per region") ||
                lowerQuestion.Contains("each region") || lowerQuestion.Contains("all region"))
            && (lowerQuestion.Contains("site") || lowerQuestion.Contains("station"));

        if (template.Id == "site_count" && (isCountBySectorQuestion || isCountByRegionQuestion))
            maxScore = Math.Max(maxScore, 0.97);

        // Penalise sites_by_filter for count-by-group questions (wrong SQL — returns individual rows)
        if (template.Id == "sites_by_filter" && (isCountBySectorQuestion || isCountByRegionQuestion))
            maxScore *= 0.05;

        // LIST questions → sites_by_filter (individual rows, filterable)
        var isAllRegionsSiteListQuestion =
            (lowerQuestion.Contains("all region") || lowerQuestion.Contains("each region") ||
             lowerQuestion.Contains("across region") || lowerQuestion.Contains("region wise") ||
             lowerQuestion.Contains("region-wise") || lowerQuestion.Contains("by region") ||
             lowerQuestion.Contains("per region"))
            && (lowerQuestion.Contains("site") || lowerQuestion.Contains("station"))
            && !lowerQuestion.Contains("aqi") && !lowerQuestion.Contains("pollutant")
            && !lowerQuestion.Contains("co2") && !lowerQuestion.Contains("pm")
            && !isCountByRegionQuestion;

        if (template.Id == "sites_by_filter" && isAllRegionsSiteListQuestion)
            maxScore = Math.Max(maxScore, 0.97);

        if ((template.Id == "schools_latest_pollutant" || template.Id == "sector_latest_pollutant")
            && isAllRegionsSiteListQuestion)
            maxScore *= 0.05;

        // Penalise schools_latest_pollutant when a past date is present —
        // "yesterday's AQI at X school", "last week's readings at X" → site_daily_averages, not live poll.
        if (template.Id == "schools_latest_pollutant" && hasSpecificDate)
            maxScore *= 0.02;

        // Penalise offline_devices when "school" is mentioned — offline_devices has no sector filter
        // and will return devices from ALL sectors. School device questions must use devices_by_filter
        // which supports sectorName="Public & Govt-School" filtering.
        bool isSchoolDeviceQuestion =
            lowerQuestion.Contains("school") &&
            (lowerQuestion.Contains("offline") || lowerQuestion.Contains("inactive") ||
             lowerQuestion.Contains("device") || lowerQuestion.Contains("sensor"));
        if (template.Id == "offline_devices" && isSchoolDeviceQuestion)
            maxScore *= 0.02;

        // Boost devices_by_filter when school + device status word present
        if (template.Id == "devices_by_filter" && isSchoolDeviceQuestion)
            maxScore = Math.Max(maxScore, 0.95);

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
        // those should route to sites_by_filter, not current_all_stations which requires
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

        // Boost priority_hotspots for any question about critical/hotspot sites or their count.
        // These always want the list (which also gives the count) — never just the stats number.
        var hotspotCriticalPhrases = new[] {
            "hotspot", "critical site", "critical air quality", "critical air-quality",
            "any critical", "total critical", "number of critical", "how many critical",
            "how many hotspot", "critical category", "classified as critical",
            "currently critical", "critical right now", "critical today"
        };
        bool isHotspotCriticalQuestion = hotspotCriticalPhrases.Any(p => lowerQuestion.Contains(p));

        // "Right now / currently / at the moment" critical/hotspot questions must use
        // site_aqi_by_category (live GetAllSiteData, AQI > 100) — NOT priority_hotspots which
        // uses Compliancestatas yearly averages and almost always returns 0 sites on dev/prod.
        var liveNowWords = new[] {
            "right now", "at the moment", "currently", "current number", "right now",
            "at this moment", "today", "at the moment", "live"
        };
        bool isLiveNowCritical = isHotspotCriticalQuestion
            && liveNowWords.Any(w => lowerQuestion.Contains(w));

        if (template.Id == "site_aqi_by_category" && isLiveNowCritical)
            maxScore = Math.Max(maxScore, 0.97);

        // priority_hotspots uses yearly averages — penalise it for "right now" questions
        if (template.Id == "priority_hotspots" && isLiveNowCritical)
            maxScore *= 0.15;

        // For non-"right now" hotspot/critical questions still boost priority_hotspots
        if (template.Id == "priority_hotspots" && isHotspotCriticalQuestion && !isLiveNowCritical)
            maxScore = Math.Max(maxScore, 0.97);

        // Penalise data_success_rate when the user wants critical site NAMES or a count of hotspots —
        // data_success_rate only returns a number, not the list. priority_hotspots returns both.
        if (template.Id == "data_success_rate" && isHotspotCriticalQuestion
            && !lowerQuestion.Contains("dsr") && !lowerQuestion.Contains("data success")
            && !lowerQuestion.Contains("data availability") && !lowerQuestion.Contains("active device"))
            maxScore *= 0.1;

        // Boost highest_aqi_today for "worst/highest AQI right now / currently" questions —
        // these ask for the single site with the peak AQI at this moment, not a category filter.
        var highestAqiNowPhrases = new[] {
            "worst aqi", "highest aqi", "most polluted site", "worst air quality",
            "highest air quality index", "peak aqi", "maximum aqi", "max aqi",
            "aqi at its worst", "aqi currently worst", "currently worst",
            "currently highest", "currently reporting the highest", "identify the site with the highest",
            "identify the site with the worst", "where is the aqi highest", "where is the aqi worst",
            "where is air quality worst", "aqi currently at its worst", "at its worst",
            "worst right now", "highest right now", "which site is worst", "which site has the worst",
            "site with the worst", "site with the highest", "worst site"
        };
        bool isHighestAqiNow = highestAqiNowPhrases.Any(p => lowerQuestion.Contains(p))
            && (lowerQuestion.Contains("site") || lowerQuestion.Contains("location") || lowerQuestion.Contains("where") || lowerQuestion.Contains("which") || lowerQuestion.Contains("at which"))
            && !isYearlyAqiRanking && !isWorstMonth && !isRegionTrend && !isRegionSingleYear;

        if (template.Id == "highest_aqi_today" && isHighestAqiNow)
            maxScore = Math.Max(maxScore, 0.97);
        // Penalise site_aqi_by_category for "worst/highest single site" questions —
        // the user wants the ONE peak site, not a filtered list of all sites by category.
        if (template.Id == "site_aqi_by_category" && isHighestAqiNow)
            maxScore *= 0.05;

        // Boost lowest_aqi_today for "best/lowest/cleanest AQI right now / currently" questions —
        // these ask for the single site with the lowest AQI at this moment, not a category filter.
        var lowestAqiNowPhrases = new[] {
            "lowest aqi", "best aqi", "cleanest site", "best air quality", "least polluted site",
            "lowest air quality level", "best air today", "best air right now",
            "currently lowest", "currently best", "currently reporting the lowest",
            "currently reporting the best", "identify the site with the lowest",
            "identify the site with the best", "where is the aqi lowest", "where is the aqi best",
            "where is air quality best", "recording the best air quality"
        };
        bool isLowestAqiNow = lowestAqiNowPhrases.Any(p => lowerQuestion.Contains(p))
            && (lowerQuestion.Contains("site") || lowerQuestion.Contains("location") || lowerQuestion.Contains("where") || lowerQuestion.Contains("which"))
            && !isYearlyAqiRanking && !isWorstMonth && !isRegionTrend && !isRegionSingleYear;

        if (template.Id == "lowest_aqi_today" && isLowestAqiNow)
            maxScore = Math.Max(maxScore, 0.97);

        // Penalise site_aqi_by_category for "which single site is worst/highest/best/lowest right now" —
        // these are point queries, not category-grouped lists.
        if (template.Id == "site_aqi_by_category" && (isHighestAqiNow || isLowestAqiNow)
            && !lowerQuestion.Contains("unhealthy") && !lowerQuestion.Contains("hazardous")
            && !lowerQuestion.Contains("good") && !lowerQuestion.Contains("moderate"))
            maxScore *= 0.1;

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

        // Boost active_device_count for "how many devices are online/active" questions —
        // these are count-only questions that must NOT route to data_success_rate (which returns DSR stats)
        // or online_devices (which lists devices). Respects user's site/region scope filter.
        var activeDeviceCountPhrases = new[] {
            "how many devices are online", "how many devices are active", "how many devices online",
            "how many active devices", "how many online devices", "how many devices currently",
            "how many sensors are online", "how many sensors are active", "how many sensors online",
            "devices online at the moment", "devices active at the moment",
            "tell me how many devices", "can you tell me how many devices",
            "active devices count", "active device count", "device online count", "device active count",
            "devices online count", "devices active count", "number of active devices", "number of online devices",
            "count of active devices", "count active devices", "total active devices", "total online devices"
        };
        bool isActiveDeviceCountQuestion = activeDeviceCountPhrases.Any(p => lowerQuestion.Contains(p))
            && !lowerQuestion.Contains("dsr") && !lowerQuestion.Contains("data success")
            && !lowerQuestion.Contains("which device") && !lowerQuestion.Contains("list device")
            && !lowerQuestion.Contains("show device") && !lowerQuestion.Contains("show me device");

        if (template.Id == "active_device_count" && isActiveDeviceCountQuestion)
            maxScore = Math.Max(maxScore, 0.97);

        // Penalise data_success_rate for pure active-device count questions — that template returns DSR %,
        // not a device count. active_device_count returns the count directly from the DB with scope filter.
        if (template.Id == "data_success_rate" && isActiveDeviceCountQuestion
            && !lowerQuestion.Contains("dsr") && !lowerQuestion.Contains("data success")
            && !lowerQuestion.Contains("data availability") && !lowerQuestion.Contains("compliance"))
            maxScore *= 0.1;

        // Boost devices_by_model_chart for device model distribution chart questions
        var deviceModelChartPhrases = new[] {
            "distribution of devices by model", "devices by model", "device model distribution",
            "device model breakdown", "device model pie", "device model bar",
            "chart of device models", "devices grouped by model", "devices based on the model",
            "how many devices of each model", "devices by type", "device type distribution"
        };
        bool isDeviceModelChart = (deviceModelChartPhrases.Any(p => lowerQuestion.Contains(p))
            || (lowerQuestion.Contains("device") && lowerQuestion.Contains("model")
                && (lowerQuestion.Contains("pie") || lowerQuestion.Contains("bar") || lowerQuestion.Contains("chart") || lowerQuestion.Contains("distribution"))));
        if (template.Id == "devices_by_model_chart" && isDeviceModelChart)
            maxScore = Math.Max(maxScore, 0.97);
        if (template.Id != "devices_by_model_chart" && isDeviceModelChart)
            maxScore *= 0.05;

        // Boost sites_by_subsector_chart for school subsector / type-of-school chart questions
        var subsectorChartPhrases = new[] {
            "school subsector", "subsector chart", "type of school", "private and public school",
            "private vs public school", "school type distribution", "subsectors pie", "subsectors bar",
            "pie chart based on school subsector", "pie chart based on type of school",
            "plot pie chart based on school", "plot a pie chart based on school"
        };
        bool isSubsectorChart = subsectorChartPhrases.Any(p => lowerQuestion.Contains(p))
            && (lowerQuestion.Contains("pie") || lowerQuestion.Contains("bar") || lowerQuestion.Contains("chart") || lowerQuestion.Contains("plot"));
        if (template.Id == "sites_by_subsector_chart" && isSubsectorChart)
            maxScore = Math.Max(maxScore, 0.97);
        if (template.Id != "sites_by_subsector_chart" && isSubsectorChart)
            maxScore *= 0.05;

        // Boost sites_aqi_level_chart for AQI level distribution chart questions
        var aqiLevelChartPhrases = new[] {
            "sites based on aqi level", "aqi level distribution", "sites by aqi level",
            "aqi level chart", "aqi level pie", "aqi level bar", "sites aqi level"
        };
        bool isAqiLevelChart = aqiLevelChartPhrases.Any(p => lowerQuestion.Contains(p))
            && (lowerQuestion.Contains("pie") || lowerQuestion.Contains("bar") || lowerQuestion.Contains("chart") || lowerQuestion.Contains("distribution"));
        if (template.Id == "sites_aqi_level_chart" && isAqiLevelChart)
            maxScore = Math.Max(maxScore, 0.97);
        if (template.Id != "sites_aqi_level_chart" && isAqiLevelChart)
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

        // Shared: named-site detection (used by both complex-query boosts and site_daily_averages boost below)
        bool isNamedSiteQuestion = hasSiteName || NamedSchoolIndicators.Any(k => lowerQuestion.Contains(k));

        // ── indoor_source_analysis: pollutant source (indoor vs outdoor) over a date range ──────
        // Signals: source/indoor/bakhoor/cooking phrase + any pollutant name + date range
        var indoorSourcePhrases = new[] {
            "more likely coming from", "coming from indoors", "coming from outdoor",
            "indoor source", "outdoor source", "source analysis", "pm2.5 source",
            "pm10 source", "co2 source", "co source", "no2 source", "tvoc source",
            "bakhoor", "cooking", "incense", "indoor dust", "outdoor dust",
            "indoor or outdoor", "indoors or outdoor", "from indoor", "from outdoor",
            "is the pm2.5", "is the pm10", "is the co2", "is the co ",
            "source of pm2.5", "source of pm10", "source of co2", "source of the pollutant",
            "what is causing elevated", "indoor outdoor ratio for"
        };
        var indoorSourcePollutants = new[] {
            "pm2.5", "pm 2.5", "pm10", "pm 10", "co2", "co₂", "co ", "no2", "no₂",
            "so2", "so₂", "o3", "o₃", "tvoc", "voc", "ch2o", "formaldehyde"
        };
        bool isIndoorSourceAnalysis = indoorSourcePhrases.Any(p => lowerQuestion.Contains(p))
            && indoorSourcePollutants.Any(p => lowerQuestion.Contains(p))
            && (lowerQuestion.Contains("from") || lowerQuestion.Contains("over the last") || lowerQuestion.Contains("between") || lowerQuestion.Contains("period"));

        if (template.Id == "indoor_source_analysis" && isIndoorSourceAnalysis)
            maxScore = 1.0; // always wins — unambiguous signal

        if (template.Id != "indoor_source_analysis" && isIndoorSourceAnalysis)
            maxScore *= 0.02; // crush every competitor

        // ── dust_event_response: how indoor AQ responded during a dust event on a specific date ──
        var dustEventPhrases = new[] {
            "dust event", "dust storm", "sandstorm", "sand storm",
            "respond during", "responded during", "response during",
            "during the dust", "during a dust"
        };
        bool isDustEventResponse = dustEventPhrases.Any(p => lowerQuestion.Contains(p))
            && isNamedSiteQuestion;

        if (template.Id == "dust_event_response" && isDustEventResponse)
            maxScore = 1.0; // always wins — unambiguous signal

        if (template.Id != "dust_event_response" && isDustEventResponse)
            maxScore *= 0.02; // crush every competitor

        // SiteParameterAverages templates handle all historical site-level data.
        // When user mentions a past date (specific date, year, or past phrase) + site reference,
        // force route to site average templates and block all live-data templates.
        var siteAvgTemplates = new HashSet<string>
        {
            "site_daily_averages", "site_single_parameter_reading",
            "site_compare_two_sites", "site_monthly_averages",
            "site_yearly_averages", "site_parameter_trend"
        };
        var liveDataTemplates = new HashSet<string>
        {
            "site_aqi_single", "site_readings_all", "site_aqi_all",
            "site_aqi_by_category", "main_pollutant_at_station", "faq_answer"
        };
        bool isHistoricalSite = IsHistoricalSiteLevelQuestion(lowerQuestion);

        // Hard-penalize live-data templates when a specific past date is detected
        if (hasSpecificDate && liveDataTemplates.Contains(template.Id) && template.Id != "faq_answer")
        {
            maxScore = 0.0;
        }

        // Hard-boost site_daily_averages when past date + site name both present —
        // overrides all other scoring so this always wins over site_aqi_single.
        // EXCEPTION: must not override indoor_source_analysis or dust_event_response which also
        // mention dates + site names but are entirely different query types.
        if (template.Id == "site_daily_averages" && hasSpecificDate && isNamedSiteQuestion
            && !isIndoorSourceAnalysis && !isDustEventResponse)
        {
            maxScore = 0.99;
        }

        // Strongly boost site average templates when a past date is detected
        // (but not when indoor_source_analysis or dust_event_response already won)
        if (hasSpecificDate && siteAvgTemplates.Contains(template.Id) && !isIndoorSourceAnalysis && !isDustEventResponse)
        {
            maxScore = Math.Max(maxScore, Math.Min(maxScore * 5.0, 0.98));
        }

        // Original historical penalty for non-site-avg templates (covers "last year", "what was" etc.)
        if (template.Id != "faq_answer" && !siteAvgTemplates.Contains(template.Id) && isHistoricalSite && !hasSpecificDate)
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
        if ((template.Id == "active_device_count" || template.Id == "devices_by_filter")
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
