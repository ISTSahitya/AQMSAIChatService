using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace HawaqmAI.Api.Models;

// ─── EF Core Entities (persisted to chat DB) ───────────────────────────────

/// <summary>A user's chat conversation session.</summary>
public sealed class ChatSession
{
    [Key]
    public Guid SessionId { get; set; } = Guid.NewGuid();

    [Required, MaxLength(100)]
    public string UserId { get; set; } = string.Empty;

    [MaxLength(255)]
    public string Title { get; set; } = "New conversation";

    /// <summary>Serialised JSON scope bar state for this session.</summary>
    public string? Scope { get; set; }

    public int MessageCount { get; set; } = 0;

    [MaxLength(100)]
    public string? SiteName { get; set; }

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime LastMessageAt { get; set; } = DateTime.UtcNow;
    public bool IsActive { get; set; } = true;

    public ICollection<ChatMessage> Messages { get; set; } = [];
}

/// <summary>A single message (user or assistant) within a chat session.</summary>
public sealed class ChatMessage
{
    [Key]
    public Guid MessageId { get; set; } = Guid.NewGuid();

    [Required]
    public Guid SessionId { get; set; }

    [ForeignKey(nameof(SessionId))]
    public ChatSession? Session { get; set; }

    /// <summary>"user" or "assistant".</summary>
    [Required, MaxLength(10)]
    public string Role { get; set; } = string.Empty;

    [Required]
    public string Content { get; set; } = string.Empty;

    public string? SqlQuery { get; set; }

    [MaxLength(20)]
    public string? ResponseType { get; set; }

    [MaxLength(20)]
    public string? ChartType { get; set; }

    /// <summary>JSON array of chart/table data points.</summary>
    public string? ChartData { get; set; }

    [MaxLength(100)]
    public string? DataSource { get; set; }

    [MaxLength(100)]
    public string? DateRange { get; set; }

    [MaxLength(50)]
    public string? ModelUsed { get; set; }

    public int? ExecutionTimeMs { get; set; }
    public int? TokenCount { get; set; }

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    public ICollection<ChatFeedback> Feedback { get; set; } = [];
    public ICollection<UserPin> Pins { get; set; } = [];
}

/// <summary>Thumbs up/down feedback on a specific message.</summary>
public sealed class ChatFeedback
{
    [Key]
    public Guid FeedbackId { get; set; } = Guid.NewGuid();

    [Required]
    public Guid MessageId { get; set; }

    [ForeignKey(nameof(MessageId))]
    public ChatMessage? Message { get; set; }

    [Required, MaxLength(100)]
    public string UserId { get; set; } = string.Empty;

    /// <summary>1 = thumbs up, -1 = thumbs down.</summary>
    [Required]
    public int Rating { get; set; }

    public string? Comment { get; set; }

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}

/// <summary>A message pinned by the user for quick reference.</summary>
public sealed class UserPin
{
    [Key]
    public Guid PinId { get; set; } = Guid.NewGuid();

    [Required]
    public Guid MessageId { get; set; }

    [ForeignKey(nameof(MessageId))]
    public ChatMessage? Message { get; set; }

    [Required, MaxLength(100)]
    public string UserId { get; set; } = string.Empty;

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}

// ─── API DTOs ──────────────────────────────────────────────────────────────

/// <summary>Session summary returned in GET /api/chat/sessions list.</summary>
public sealed class SessionSummaryDto
{
    public Guid SessionId { get; set; }
    public string Title { get; set; } = string.Empty;
    public string? SiteName { get; set; }
    public int MessageCount { get; set; }
    public DateTime LastMessageAt { get; set; }
    public DateTime CreatedAt { get; set; }
}

/// <summary>Request body for renaming a session (PATCH).</summary>
public sealed class RenameSessionRequest
{
    [Required, MaxLength(255)]
    public string Title { get; set; } = string.Empty;
}

/// <summary>Feedback submission request.</summary>
public sealed class FeedbackRequest
{
    [Required]
    public Guid MessageId { get; set; }

    /// <summary>1 = positive, -1 = negative.</summary>
    [Required]
    [Range(-1, 1)]
    public int Rating { get; set; }

    public string? Comment { get; set; }
}

/// <summary>Pin creation request.</summary>
public sealed class PinRequest
{
    [Required]
    public Guid MessageId { get; set; }
}
