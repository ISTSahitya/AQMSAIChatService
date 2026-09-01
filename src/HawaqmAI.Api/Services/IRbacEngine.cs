using HawaqmAI.Api.Models;

namespace HawaqmAI.Api.Services;

/// <summary>
/// Three-level RBAC enforcement for AI-generated queries:
/// 1. Schema filter — which templates the LLM is allowed to see
/// 2. Query validation — verifies the selected template is in the user's permitted categories
/// 3. SQL rewriting — injects site_id filter + row limits + scope constraints
/// </summary>
public interface IRbacEngine
{
    /// <summary>
    /// Level 1: Filter the full template list to only those permitted for this user's role.
    /// </summary>
    IReadOnlyList<ApprovedQuery> FilterTemplates(IReadOnlyList<ApprovedQuery> allTemplates, UserContext user);

    /// <summary>
    /// Level 2: Validate that the selected template is allowed for this user.
    /// Returns (valid: true) or (valid: false, reason).
    /// </summary>
    (bool IsValid, string? Reason) ValidateSelection(ApprovedQuery template, UserContext user);

    /// <summary>
    /// Level 3: Rewrite the final SQL to inject site_id IN (...), scope constraints, and TOP limit.
    /// Also substitutes column name placeholders after validation.
    /// </summary>
    (string Sql, Dictionary<string, object> Parameters) BuildSafeQuery(
        ApprovedQuery template,
        Dictionary<string, string> llmParameters,
        UserContext user,
        ResolvedScope scope);
}
