using HawaqmAI.Api.Models;

namespace HawaqmAI.Api.Services;

/// <summary>
/// Manages chat sessions and conversation memory.
/// Dual-writes to in-memory cache (fast reads) and Azure SQL (persistence).
/// </summary>
public interface ISessionService
{
    /// <summary>Gets or creates a session. Returns the session entity.</summary>
    Task<ChatSession> GetOrCreateSessionAsync(Guid? sessionId, string userId, CancellationToken ct = default);

    /// <summary>Returns the last N messages as conversation turns for LLM context.</summary>
    Task<IReadOnlyList<ConversationTurn>> GetConversationHistoryAsync(Guid sessionId, int maxTurns = 15, CancellationToken ct = default);

    /// <summary>Persists a user message and returns its ID.</summary>
    Task<Guid> SaveUserMessageAsync(Guid sessionId, string content, CancellationToken ct = default);

    /// <summary>Persists an assistant response.</summary>
    Task SaveAssistantMessageAsync(Guid sessionId, Guid messageId, string content, string? sql,
        string responseType, string? chartType, string? chartDataJson,
        string? dataSource, string? dateRange, string? modelUsed,
        int executionTimeMs, int tokenCount, CancellationToken ct = default);

    /// <summary>Returns all sessions for a user (last 30 days).</summary>
    Task<IReadOnlyList<SessionSummaryDto>> GetUserSessionsAsync(string userId, CancellationToken ct = default);

    /// <summary>Returns all messages for a session (ownership verified).</summary>
    Task<IReadOnlyList<ChatMessage>> GetSessionMessagesAsync(Guid sessionId, string userId, CancellationToken ct = default);

    /// <summary>Deletes a session (ownership verified).</summary>
    Task<bool> DeleteSessionAsync(Guid sessionId, string userId, CancellationToken ct = default);

    /// <summary>Renames a session title (ownership verified).</summary>
    Task<bool> RenameSessionAsync(Guid sessionId, string userId, string newTitle, CancellationToken ct = default);

    /// <summary>Auto-generates a session title from the first question (called after first message).</summary>
    Task UpdateSessionTitleAsync(Guid sessionId, string firstQuestion, CancellationToken ct = default);
}
