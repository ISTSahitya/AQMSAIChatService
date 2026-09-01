using System.Text.Json;
using HawaqmAI.Api.Configuration;
using HawaqmAI.Api.Data;
using HawaqmAI.Api.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Options;
using Serilog;

namespace HawaqmAI.Api.Services;

/// <summary>
/// Session management with dual-write: in-memory cache for speed + Azure SQL for persistence.
/// Background cleanup deletes sessions older than 30 days.
/// </summary>
public sealed class SessionService : ISessionService
{
    private static readonly Serilog.ILogger _log = Log.ForContext<SessionService>();
    private readonly ChatHistoryDbContext _db;
    private readonly IMemoryCache _cache;
    private readonly ChatOptions _chatOptions;

    private static string CacheKey(Guid id) => $"session_history_{id}";

    public SessionService(
        ChatHistoryDbContext db,
        IMemoryCache cache,
        IOptions<ChatOptions> chatOptions)
    {
        _db = db;
        _cache = cache;
        _chatOptions = chatOptions.Value;
    }

    /// <inheritdoc/>
    public async Task<ChatSession> GetOrCreateSessionAsync(
        Guid? sessionId,
        string userId,
        CancellationToken ct = default)
    {
        if (sessionId.HasValue)
        {
            var existing = await _db.ChatSessions
                .FirstOrDefaultAsync(s => s.SessionId == sessionId.Value && s.UserId == userId, ct);

            if (existing is not null)
                return existing;
        }

        // Create new session
        var session = new ChatSession
        {
            UserId = userId,
            Title = "New conversation",
            IsActive = true
        };

        _db.ChatSessions.Add(session);
        await _db.SaveChangesAsync(ct);

        _log.Information("Created new session {SessionId} for user {UserId}", session.SessionId, userId);
        return session;
    }

    /// <inheritdoc/>
    public async Task<IReadOnlyList<ConversationTurn>> GetConversationHistoryAsync(
        Guid sessionId,
        int maxTurns = 15,
        CancellationToken ct = default)
    {
        // Try cache first
        if (_cache.TryGetValue(CacheKey(sessionId), out List<ConversationTurn>? cached) && cached != null)
            return cached.TakeLast(maxTurns).ToList();

        // Load from DB
        var messages = await _db.ChatMessages
            .Where(m => m.SessionId == sessionId)
            .OrderByDescending(m => m.CreatedAt)
            .ThenByDescending(m => m.MessageId)
            .Take(maxTurns)
            .Select(m => new ConversationTurn { Role = m.Role, Content = m.Content, TemplateId = m.DataSource })
            .ToListAsync(ct);

        messages.Reverse();

        _cache.Set(CacheKey(sessionId), messages, TimeSpan.FromMinutes(_chatOptions.SessionTimeoutMinutes));
        return messages;
    }

    /// <inheritdoc/>
    public async Task<Guid> SaveUserMessageAsync(
        Guid sessionId,
        string content,
        CancellationToken ct = default)
    {
        var message = new ChatMessage
        {
            SessionId = sessionId,
            Role = "user",
            Content = content
        };

        _db.ChatMessages.Add(message);
        await UpdateSessionStats(sessionId, ct);
        await _db.SaveChangesAsync(ct);

        AppendToCache(sessionId, "user", content);
        return message.MessageId;
    }

    /// <inheritdoc/>
    public async Task SaveAssistantMessageAsync(
        Guid sessionId,
        Guid messageId,
        string content,
        string? sql,
        string responseType,
        string? chartType,
        string? chartDataJson,
        string? dataSource,
        string? dateRange,
        string? modelUsed,
        int executionTimeMs,
        int tokenCount,
        CancellationToken ct = default)
    {
        var message = new ChatMessage
        {
            MessageId = messageId,
            SessionId = sessionId,
            Role = "assistant",
            Content = content,
            SqlQuery = sql,
            ResponseType = responseType,
            ChartType = chartType,
            ChartData = chartDataJson,
            DataSource = dataSource,
            DateRange = dateRange,
            ModelUsed = modelUsed,
            ExecutionTimeMs = executionTimeMs,
            TokenCount = tokenCount
        };

        _db.ChatMessages.Add(message);
        await _db.SaveChangesAsync(ct);

        AppendToCache(sessionId, "assistant", content, templateId: dataSource);
    }

    /// <inheritdoc/>
    public async Task<IReadOnlyList<SessionSummaryDto>> GetUserSessionsAsync(
        string userId,
        CancellationToken ct = default)
    {
        var cutoff = DateTime.UtcNow.AddDays(-_chatOptions.HistoryRetentionDays);

        return await _db.ChatSessions
            .Where(s => s.UserId == userId && s.IsActive && s.LastMessageAt >= cutoff)
            .OrderByDescending(s => s.LastMessageAt)
            .Select(s => new SessionSummaryDto
            {
                SessionId = s.SessionId,
                Title = s.Title,
                SiteName = s.SiteName,
                MessageCount = s.MessageCount,
                LastMessageAt = s.LastMessageAt,
                CreatedAt = s.CreatedAt
            })
            .ToListAsync(ct);
    }

    /// <inheritdoc/>
    public async Task<IReadOnlyList<ChatMessage>> GetSessionMessagesAsync(
        Guid sessionId,
        string userId,
        CancellationToken ct = default)
    {
        var session = await _db.ChatSessions
            .FirstOrDefaultAsync(s => s.SessionId == sessionId && s.UserId == userId, ct);

        if (session is null) return [];

        return await _db.ChatMessages
            .Where(m => m.SessionId == sessionId)
            .OrderBy(m => m.CreatedAt)
            .ToListAsync(ct);
    }

    /// <inheritdoc/>
    public async Task<bool> DeleteSessionAsync(
        Guid sessionId,
        string userId,
        CancellationToken ct = default)
    {
        var session = await _db.ChatSessions
            .FirstOrDefaultAsync(s => s.SessionId == sessionId && s.UserId == userId, ct);

        if (session is null) return false;

        _db.ChatSessions.Remove(session);
        await _db.SaveChangesAsync(ct);

        _cache.Remove(CacheKey(sessionId));
        _log.Information("Deleted session {SessionId} for user {UserId}", sessionId, userId);
        return true;
    }

    /// <inheritdoc/>
    public async Task<bool> RenameSessionAsync(
        Guid sessionId,
        string userId,
        string newTitle,
        CancellationToken ct = default)
    {
        var session = await _db.ChatSessions
            .FirstOrDefaultAsync(s => s.SessionId == sessionId && s.UserId == userId, ct);

        if (session is null) return false;

        session.Title = newTitle[..Math.Min(newTitle.Length, 255)];
        await _db.SaveChangesAsync(ct);
        return true;
    }

    /// <inheritdoc/>
    public async Task UpdateSessionTitleAsync(
        Guid sessionId,
        string firstQuestion,
        CancellationToken ct = default)
    {
        var session = await _db.ChatSessions.FindAsync([sessionId], ct);
        if (session is null) return;

        // Auto-generate title from first question (truncate to 60 chars)
        var title = firstQuestion.Length > 60
            ? firstQuestion[..57] + "..."
            : firstQuestion;

        session.Title = title;
        await _db.SaveChangesAsync(ct);
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    private async Task UpdateSessionStats(Guid sessionId, CancellationToken ct)
    {
        var session = await _db.ChatSessions.FindAsync([sessionId], ct);
        if (session is null) return;

        session.MessageCount++;
        session.LastMessageAt = DateTime.UtcNow;
    }

    private void AppendToCache(Guid sessionId, string role, string content, string? templateId = null)
    {
        var key = CacheKey(sessionId);
        var history = _cache.GetOrCreate(key, _ => new List<ConversationTurn>()) ?? [];

        history.Add(new ConversationTurn { Role = role, Content = content, TemplateId = templateId });

        // Keep only the last MaxConversationTurns
        if (history.Count > _chatOptions.MaxConversationTurns)
            history.RemoveRange(0, history.Count - _chatOptions.MaxConversationTurns);

        _cache.Set(key, history, TimeSpan.FromMinutes(_chatOptions.SessionTimeoutMinutes));
    }
}

/// <summary>
/// Background service that purges expired chat sessions nightly.
/// </summary>
public sealed class SessionCleanupService : BackgroundService
{
    private static readonly Serilog.ILogger _log = Log.ForContext<SessionCleanupService>();
    private readonly IServiceScopeFactory _scopeFactory;
    private readonly IOptions<ChatOptions> _chatOptions;

    public SessionCleanupService(IServiceScopeFactory scopeFactory, IOptions<ChatOptions> chatOptions)
    {
        _scopeFactory = scopeFactory;
        _chatOptions = chatOptions;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _log.Information("Session cleanup service started");

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                // Run once per day
                await Task.Delay(TimeSpan.FromHours(24), stoppingToken);
                await using var scope = _scopeFactory.CreateAsyncScope();
                var db = scope.ServiceProvider.GetRequiredService<ChatHistoryDbContext>();

                var cutoff = DateTime.UtcNow.AddDays(-_chatOptions.Value.HistoryRetentionDays);
                var expired = await db.ChatSessions
                    .Where(s => s.LastMessageAt < cutoff)
                    .ToListAsync(stoppingToken);

                if (expired.Count > 0)
                {
                    db.ChatSessions.RemoveRange(expired);
                    await db.SaveChangesAsync(stoppingToken);
                    _log.Information("Cleaned up {Count} expired chat sessions older than {Days} days",
                        expired.Count, _chatOptions.Value.HistoryRetentionDays);
                }
            }
            catch (OperationCanceledException)
            {
                // Host is shutting down — exit the loop gracefully
                break;
            }
            catch (Exception ex)
            {
                _log.Error(ex, "Session cleanup failed");
            }
        }
    }
}
