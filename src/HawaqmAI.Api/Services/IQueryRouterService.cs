using HawaqmAI.Api.Models;

namespace HawaqmAI.Api.Services;

/// <summary>
/// Matches a natural language question to an approved query template.
/// Returns a scored match without calling the LLM for high-confidence hits.
/// </summary>
public interface IQueryRouterService
{
    /// <summary>
    /// Returns the best matching approved query and a confidence score (0.0–1.0).
    /// Accepts conversation history so follow-up questions can reuse the previous template.
    /// Returns null if no template matches above the minimum threshold.
    /// </summary>
    Task<QueryRouterResult> RouteAsync(string question, UserContext user, IReadOnlyList<ConversationTurn>? history = null, CancellationToken ct = default);

    /// <summary>Returns all templates the given user role is permitted to see.</summary>
    IReadOnlyList<ApprovedQuery> GetPermittedTemplates(UserContext user);
}

/// <summary>Result of the query routing step.</summary>
public sealed class QueryRouterResult
{
    /// <summary>Best matching template (null if no match above minimum threshold).</summary>
    public ApprovedQuery? Template { get; init; }

    /// <summary>Confidence score 0.0–1.0.</summary>
    public double Score { get; init; }

    /// <summary>Top-N candidates sent to LLM when score is in the middle band.</summary>
    public List<ApprovedQuery> Candidates { get; init; } = [];

    /// <summary>Whether this is a follow-up question that refers to a previous result (sort, filter, reformat, etc).</summary>
    public bool IsFollowUp { get; init; }

    /// <summary>The previous template ID to reuse for follow-up questions.</summary>
    public string? PreviousTemplateId { get; init; }

    /// <summary>Whether the LLM needs to confirm the selection among candidates.</summary>
    public bool NeedsLlmConfirmation => (Score is >= 0.50 and < 0.85 && Candidates.Count > 0) || IsFollowUp;

    /// <summary>Whether confidence was too low — ask the user to clarify.</summary>
    public bool NeedsClarification => Template is null && Candidates.Count == 0 && !IsFollowUp;

    /// <summary>Targeted clarification message to show the user when NeedsClarification is true.</summary>
    public string? ClarificationPrompt { get; init; }
}
