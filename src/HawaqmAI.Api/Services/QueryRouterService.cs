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
