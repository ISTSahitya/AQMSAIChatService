using HawaqmAI.Api.Data;
using HawaqmAI.Api.Models;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Serilog;

namespace HawaqmAI.Api.Controllers;

/// <summary>
/// Feedback and pin management for chat messages.
/// POST /api/chat/feedback — thumbs up/down
/// POST /api/chat/pins    — pin a message
/// DELETE /api/chat/pins/{messageId} — unpin
/// </summary>
[ApiController]
[Route("api/chat")]
[Authorize]
public sealed class FeedbackController : ControllerBase
{
    private static readonly Serilog.ILogger _log = Log.ForContext<FeedbackController>();
    private readonly ChatHistoryDbContext _db;

    public FeedbackController(ChatHistoryDbContext db)
    {
        _db = db;
    }

    /// <summary>Submit thumbs up (+1) or thumbs down (-1) feedback on a message.</summary>
    [HttpPost("feedback")]
    [ProducesResponseType(201)]
    [ProducesResponseType(400)]
    [ProducesResponseType(404)]
    public async Task<IActionResult> SubmitFeedback(
        [FromBody] FeedbackRequest request,
        CancellationToken ct)
    {
        if (!ModelState.IsValid) return BadRequest(ModelState);
        if (request.Rating != 1 && request.Rating != -1)
            return BadRequest(new { error = "Rating must be 1 (positive) or -1 (negative)." });

        var userId = GetUserId();
        if (userId is null) return Unauthorized();

        var message = await _db.ChatMessages.FindAsync([request.MessageId], ct);
        if (message is null)
            return NotFound(new { error = "Message not found." });

        // Upsert — replace existing feedback from same user on same message
        var existing = await _db.ChatFeedback
            .FirstOrDefaultAsync(f => f.MessageId == request.MessageId && f.UserId == userId, ct);

        if (existing is not null)
        {
            existing.Rating = request.Rating;
            existing.Comment = request.Comment;
            existing.CreatedAt = DateTime.UtcNow;
        }
        else
        {
            var feedback = new ChatFeedback
            {
                MessageId = request.MessageId,
                UserId = userId,
                Rating = request.Rating,
                Comment = request.Comment
            };
            _db.ChatFeedback.Add(feedback);
        }

        await _db.SaveChangesAsync(ct);

        _log.Information("Feedback {Rating} from user {UserId} on message {MessageId}",
            request.Rating, userId, request.MessageId);

        return Created(string.Empty, new { message = "Feedback recorded." });
    }

    /// <summary>Pin a message for quick reference.</summary>
    [HttpPost("pins")]
    [ProducesResponseType(201)]
    [ProducesResponseType(400)]
    [ProducesResponseType(404)]
    [ProducesResponseType(409)]
    public async Task<IActionResult> PinMessage([FromBody] PinRequest request, CancellationToken ct)
    {
        if (!ModelState.IsValid) return BadRequest(ModelState);

        var userId = GetUserId();
        if (userId is null) return Unauthorized();

        var message = await _db.ChatMessages.FindAsync([request.MessageId], ct);
        if (message is null)
            return NotFound(new { error = "Message not found." });

        var alreadyPinned = await _db.UserPins
            .AnyAsync(p => p.MessageId == request.MessageId && p.UserId == userId, ct);

        if (alreadyPinned)
            return Conflict(new { error = "Message is already pinned." });

        var pin = new UserPin { MessageId = request.MessageId, UserId = userId };
        _db.UserPins.Add(pin);
        await _db.SaveChangesAsync(ct);

        _log.Information("User {UserId} pinned message {MessageId}", userId, request.MessageId);
        return Created(string.Empty, new { pinId = pin.PinId });
    }

    /// <summary>Unpin a message.</summary>
    [HttpDelete("pins/{messageId:guid}")]
    [ProducesResponseType(204)]
    [ProducesResponseType(404)]
    public async Task<IActionResult> UnpinMessage(Guid messageId, CancellationToken ct)
    {
        var userId = GetUserId();
        if (userId is null) return Unauthorized();

        var pin = await _db.UserPins
            .FirstOrDefaultAsync(p => p.MessageId == messageId && p.UserId == userId, ct);

        if (pin is null)
            return NotFound(new { error = "Pin not found." });

        _db.UserPins.Remove(pin);
        await _db.SaveChangesAsync(ct);

        _log.Information("User {UserId} unpinned message {MessageId}", userId, messageId);
        return NoContent();
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
