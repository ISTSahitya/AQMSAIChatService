using HawaqmAI.Api.Models;
using HawaqmAI.Api.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Serilog;

namespace HawaqmAI.Api.Controllers;

/// <summary>
/// CRUD endpoints for chat sessions and conversation history.
/// All endpoints require authentication and enforce ownership.
/// </summary>
[ApiController]
[Route("api/chat/sessions")]
[Authorize]
public sealed class SessionsController : ControllerBase
{
    private static readonly Serilog.ILogger _log = Log.ForContext<SessionsController>();
    private readonly ISessionService _sessionService;

    public SessionsController(ISessionService sessionService)
    {
        _sessionService = sessionService;
    }

    /// <summary>List all sessions for the authenticated user (last 30 days).</summary>
    [HttpGet]
    [ProducesResponseType(typeof(IReadOnlyList<SessionSummaryDto>), 200)]
    public async Task<IActionResult> GetSessions(CancellationToken ct)
    {
        var userId = GetUserId();
        if (userId is null) return Unauthorized();

        var sessions = await _sessionService.GetUserSessionsAsync(userId, ct);
        return Ok(sessions);
    }

    /// <summary>Get all messages for a specific session (ownership enforced).</summary>
    [HttpGet("{sessionId:guid}")]
    [ProducesResponseType(typeof(IReadOnlyList<ChatMessage>), 200)]
    [ProducesResponseType(404)]
    public async Task<IActionResult> GetSession(Guid sessionId, CancellationToken ct)
    {
        var userId = GetUserId();
        if (userId is null) return Unauthorized();

        var messages = await _sessionService.GetSessionMessagesAsync(sessionId, userId, ct);
        if (messages.Count == 0)
            return NotFound(new { error = "Session not found or not accessible." });

        return Ok(messages);
    }

    /// <summary>Delete a session and all its messages (ownership enforced).</summary>
    [HttpDelete("{sessionId:guid}")]
    [ProducesResponseType(204)]
    [ProducesResponseType(404)]
    public async Task<IActionResult> DeleteSession(Guid sessionId, CancellationToken ct)
    {
        var userId = GetUserId();
        if (userId is null) return Unauthorized();

        var deleted = await _sessionService.DeleteSessionAsync(sessionId, userId, ct);
        if (!deleted)
            return NotFound(new { error = "Session not found or not accessible." });

        _log.Information("User {UserId} deleted session {SessionId}", userId, sessionId);
        return NoContent();
    }

    /// <summary>Rename a session title (ownership enforced).</summary>
    [HttpPatch("{sessionId:guid}")]
    [ProducesResponseType(200)]
    [ProducesResponseType(404)]
    [ProducesResponseType(400)]
    public async Task<IActionResult> RenameSession(
        Guid sessionId,
        [FromBody] RenameSessionRequest request,
        CancellationToken ct)
    {
        if (!ModelState.IsValid) return BadRequest(ModelState);

        var userId = GetUserId();
        if (userId is null) return Unauthorized();

        var renamed = await _sessionService.RenameSessionAsync(sessionId, userId, request.Title, ct);
        if (!renamed)
            return NotFound(new { error = "Session not found or not accessible." });

        return Ok(new { message = "Session renamed successfully." });
    }

    /// <summary>Resume an archived session (re-activates it for the scope bar).</summary>
    [HttpPost("{sessionId:guid}/resume")]
    [ProducesResponseType(200)]
    [ProducesResponseType(404)]
    public async Task<IActionResult> ResumeSession(Guid sessionId, CancellationToken ct)
    {
        var userId = GetUserId();
        if (userId is null) return Unauthorized();

        // Validate ownership by attempting to get messages
        var messages = await _sessionService.GetSessionMessagesAsync(sessionId, userId, ct);
        if (messages.Count == 0)
            return NotFound(new { error = "Session not found or not accessible." });

        return Ok(new { sessionId, message = "Session resumed." });
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    private string? GetUserId()
    {
        var userCtx = HttpContext.Items.TryGetValue("UserContext", out var ctx) ? ctx as UserContext : null;
        return userCtx?.UserId
            ?? User.FindFirst("sub")?.Value
            ?? User.FindFirst("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/nameidentifier")?.Value;
    }
}
