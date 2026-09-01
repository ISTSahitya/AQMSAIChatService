using HawaqmAI.Api.Models;
using Microsoft.EntityFrameworkCore;

namespace HawaqmAI.Api.Data;

/// <summary>
/// EF Core DbContext for chat session persistence (read-write).
/// Uses the hawaqm_ai_chat SQL user connection.
/// Air quality tables are accessed separately via Dapper (read-only).
/// </summary>
public sealed class ChatHistoryDbContext : DbContext
{
    public ChatHistoryDbContext(DbContextOptions<ChatHistoryDbContext> options)
        : base(options)
    {
    }

    public DbSet<ChatSession> ChatSessions => Set<ChatSession>();
    public DbSet<ChatMessage> ChatMessages => Set<ChatMessage>();
    public DbSet<ChatFeedback> ChatFeedback => Set<ChatFeedback>();
    public DbSet<UserPin> UserPins => Set<UserPin>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        base.OnModelCreating(modelBuilder);

        // ── ChatSession ───────────────────────────────────────────────────────
        modelBuilder.Entity<ChatSession>(entity =>
        {
            entity.ToTable("chat_sessions");
            entity.HasKey(e => e.SessionId);
            entity.Property(e => e.SessionId).HasColumnName("session_id").HasDefaultValueSql("NEWID()");
            entity.Property(e => e.UserId).HasColumnName("user_id").IsRequired().HasMaxLength(100);
            entity.Property(e => e.Title).HasColumnName("title").HasMaxLength(255).HasDefaultValue("New conversation");
            entity.Property(e => e.Scope).HasColumnName("scope").HasColumnType("nvarchar(max)");
            entity.Property(e => e.SiteName).HasColumnName("site_name").HasMaxLength(100);
            entity.Property(e => e.CreatedAt).HasColumnName("created_at").HasDefaultValueSql("GETUTCDATE()");
            entity.Property(e => e.LastMessageAt).HasColumnName("last_message_at").HasDefaultValueSql("GETUTCDATE()");
            entity.Property(e => e.MessageCount).HasColumnName("message_count");
            entity.Property(e => e.IsActive).HasColumnName("is_active").HasDefaultValue(true);

            entity.HasIndex(e => new { e.UserId, e.LastMessageAt })
                  .HasDatabaseName("IX_chat_sessions_user")
                  .IsDescending(false, true);

            entity.HasMany(s => s.Messages)
                  .WithOne(m => m.Session)
                  .HasForeignKey(m => m.SessionId)
                  .OnDelete(DeleteBehavior.Cascade);
        });

        // ── ChatMessage ───────────────────────────────────────────────────────
        modelBuilder.Entity<ChatMessage>(entity =>
        {
            entity.ToTable("chat_messages");
            entity.HasKey(e => e.MessageId);
            entity.Property(e => e.MessageId).HasColumnName("message_id").HasDefaultValueSql("NEWID()");
            entity.Property(e => e.SessionId).HasColumnName("session_id");
            entity.Property(e => e.Role).HasColumnName("role").IsRequired().HasMaxLength(10);
            entity.Property(e => e.Content).HasColumnName("content").IsRequired().HasColumnType("nvarchar(max)");
            entity.Property(e => e.SqlQuery).HasColumnName("sql_query").HasColumnType("nvarchar(max)");
            entity.Property(e => e.ResponseType).HasColumnName("response_type").HasMaxLength(20);
            entity.Property(e => e.ChartType).HasColumnName("chart_type").HasMaxLength(20);
            entity.Property(e => e.ChartData).HasColumnName("chart_data").HasColumnType("nvarchar(max)");
            entity.Property(e => e.DataSource).HasColumnName("data_source").HasMaxLength(100);
            entity.Property(e => e.DateRange).HasColumnName("date_range").HasMaxLength(100);
            entity.Property(e => e.ModelUsed).HasColumnName("model_used").HasMaxLength(50);
            entity.Property(e => e.ExecutionTimeMs).HasColumnName("execution_time_ms");
            entity.Property(e => e.TokenCount).HasColumnName("token_count");
            entity.Property(e => e.CreatedAt).HasColumnName("created_at").HasDefaultValueSql("GETUTCDATE()");

            entity.HasIndex(e => new { e.SessionId, e.CreatedAt })
                  .HasDatabaseName("IX_chat_messages_session");

            // Role check constraint — only "user" or "assistant" allowed
            entity.ToTable(t => t.HasCheckConstraint("CK_chat_messages_role", "role IN ('user', 'assistant')"));

            entity.HasMany(m => m.Feedback)
                  .WithOne(f => f.Message)
                  .HasForeignKey(f => f.MessageId)
                  .OnDelete(DeleteBehavior.Cascade);

            entity.HasMany(m => m.Pins)
                  .WithOne(p => p.Message)
                  .HasForeignKey(p => p.MessageId)
                  .OnDelete(DeleteBehavior.Cascade);
        });

        // ── ChatFeedback ──────────────────────────────────────────────────────
        modelBuilder.Entity<ChatFeedback>(entity =>
        {
            entity.ToTable("chat_feedback");
            entity.HasKey(e => e.FeedbackId);
            entity.Property(e => e.FeedbackId).HasColumnName("feedback_id").HasDefaultValueSql("NEWID()");
            entity.Property(e => e.MessageId).HasColumnName("message_id");
            entity.Property(e => e.UserId).HasColumnName("user_id").IsRequired().HasMaxLength(100);
            entity.Property(e => e.Rating).HasColumnName("rating");
            entity.Property(e => e.Comment).HasColumnName("comment").HasColumnType("nvarchar(max)");
            entity.Property(e => e.CreatedAt).HasColumnName("created_at").HasDefaultValueSql("GETUTCDATE()");

            entity.ToTable(t => t.HasCheckConstraint("CK_chat_feedback_rating", "rating IN (-1, 1)"));
        });

        // ── UserPin ───────────────────────────────────────────────────────────
        modelBuilder.Entity<UserPin>(entity =>
        {
            entity.ToTable("user_pins");
            entity.HasKey(e => e.PinId);
            entity.Property(e => e.PinId).HasColumnName("pin_id").HasDefaultValueSql("NEWID()");
            entity.Property(e => e.MessageId).HasColumnName("message_id");
            entity.Property(e => e.UserId).HasColumnName("user_id").IsRequired().HasMaxLength(100);
            entity.Property(e => e.CreatedAt).HasColumnName("created_at").HasDefaultValueSql("GETUTCDATE()");

            // Unique: one user can only pin a message once
            entity.HasIndex(e => new { e.MessageId, e.UserId })
                  .IsUnique()
                  .HasDatabaseName("UQ_user_pins_message_user");
        });
    }
}
