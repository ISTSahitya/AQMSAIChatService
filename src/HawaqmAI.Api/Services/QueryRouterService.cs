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
        "ascending", "descending", "asc", "desc", "filter", "group", "make it", "can you",
        "change", "modify", "update", "show only", "now show", "instead", "but only",
        "also show", "add to", "remove from", "without", "exclude", "include only"
    ];

    // Keywords that clearly signal a site query (not a device or region)
    private static readonly string[] SiteKeywords =
        ["site", "station", "school", "building", "facility", "location", "readings at", "aqi at", "data for", "reading at"];

    // Keywords that clearly signal a device query
    private static readonly string[] DeviceKeywords =
        ["device", "sensor", "reading for", "data from", "ba ", "sei100", "seI100m"];

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
        if (history is { Count: > 0 } && IsFollowUpQuestion(question))
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

    // Regex-style check: question contains a device name pattern (BA + digits, SEI100M + digits)
    private static bool QuestionHasDeviceName(string lowerQuestion) =>
        System.Text.RegularExpressions.Regex.IsMatch(lowerQuestion, @"\b(ba|sei100m?)\s*\d{4}\b");

    // Site-name indicators — words that appear in station names but NOT in region-only queries.
    // If any of these are present alongside a region word, the question is a site query, not a region query.
    private static readonly string[] SiteNameIndicators =
    [
        "school", "residential", "commercial", "institutional", "institution",
        "building", "facility", "centre", "center", "hospital", "clinic",
        "office", "park", "mall", "tower", "villa", "compound", "camp",
        "indian", "british", "american", "international", "national",
        "bateen", "saad", "naeem", "naim", "khalifa", "zayed", "hamdan",
        "mushrif", "mussafah", "baniyas", "karama", "shakhbout", "shahama",
        "madinat", "khalidiyah", "corniche", "mangrove", "reem", "yas",
        "wahda", "rowdah", "muroor", "electra", "hameem", "liwa", "gayathi"
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
