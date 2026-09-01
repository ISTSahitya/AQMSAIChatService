using HawaqmAI.Api.Models;

namespace HawaqmAI.Api.Services;

/// <summary>Communicates with Azure AI Foundry to select query templates and fill parameters.</summary>
public interface IAzureAIService
{
    /// <summary>
    /// Asks the LLM to select the best template from candidates and fill parameters.
    /// Returns the selected template ID and extracted parameter values.
    /// </summary>
    Task<LlmSelectionResult> SelectTemplateAsync(
        string question,
        IReadOnlyList<ApprovedQuery> candidates,
        IReadOnlyList<ConversationTurn> history,
        ResolvedScope scope,
        string? previousTemplateId = null,
        CancellationToken ct = default);

    /// <summary>
    /// Asks the LLM to fill parameters for an already-selected template.
    /// Used when QueryRouter selected the template directly (high confidence).
    /// </summary>
    Task<LlmParameterResult> FillParametersAsync(
        string question,
        ApprovedQuery template,
        IReadOnlyList<ConversationTurn> history,
        ResolvedScope scope,
        CancellationToken ct = default);

    /// <summary>
    /// Generates a natural language summary of the query results.
    /// </summary>
    Task<string> SummarizeResultsAsync(
        string question,
        ApprovedQuery template,
        List<Dictionary<string, object?>> results,
        ResolvedScope scope,
        CancellationToken ct = default);

    /// <summary>
    /// Answers a general FAQ question about HAWAQM using the faq.json knowledge base.
    /// No database or API call is made.
    /// </summary>
    Task<string> AnswerFaqAsync(
        string question,
        string faqContext,
        CancellationToken ct = default);
}

/// <summary>Result of asking the LLM to choose among candidates.</summary>
public sealed class LlmSelectionResult
{
    public string SelectedQueryId { get; init; } = string.Empty;
    public Dictionary<string, string> Parameters { get; init; } = [];
    public double Confidence { get; init; }
    public string Summary { get; init; } = string.Empty;
    public int TokensUsed { get; init; }
    public string? ModelUsed { get; init; }
}

/// <summary>Result of asking the LLM to fill parameters for a known template.</summary>
public sealed class LlmParameterResult
{
    public Dictionary<string, string> Parameters { get; init; } = [];
    public double Confidence { get; init; }
    public int TokensUsed { get; init; }
    public string? ModelUsed { get; init; }
}

/// <summary>A single turn in the conversation history sent as LLM context.</summary>
public sealed class ConversationTurn
{
    public string Role { get; init; } = string.Empty;  // "user" or "assistant"
    public string Content { get; init; } = string.Empty;
    /// <summary>The approved query template ID used to produce this assistant turn (null for user turns).</summary>
    public string? TemplateId { get; init; }
}
