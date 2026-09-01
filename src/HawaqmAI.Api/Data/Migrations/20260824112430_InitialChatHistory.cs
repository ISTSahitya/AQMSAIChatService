using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace HawaqmAI.Api.Data.Migrations
{
    /// <inheritdoc />
    public partial class InitialChatHistory : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "chat_sessions",
                columns: table => new
                {
                    session_id = table.Column<Guid>(type: "uniqueidentifier", nullable: false, defaultValueSql: "NEWID()"),
                    user_id = table.Column<string>(type: "nvarchar(100)", maxLength: 100, nullable: false),
                    title = table.Column<string>(type: "nvarchar(255)", maxLength: 255, nullable: false, defaultValue: "New conversation"),
                    scope = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    message_count = table.Column<int>(type: "int", nullable: false),
                    site_name = table.Column<string>(type: "nvarchar(100)", maxLength: 100, nullable: true),
                    created_at = table.Column<DateTime>(type: "datetime2", nullable: false, defaultValueSql: "GETUTCDATE()"),
                    last_message_at = table.Column<DateTime>(type: "datetime2", nullable: false, defaultValueSql: "GETUTCDATE()"),
                    is_active = table.Column<bool>(type: "bit", nullable: false, defaultValue: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_chat_sessions", x => x.session_id);
                });

            migrationBuilder.CreateTable(
                name: "chat_messages",
                columns: table => new
                {
                    message_id = table.Column<Guid>(type: "uniqueidentifier", nullable: false, defaultValueSql: "NEWID()"),
                    session_id = table.Column<Guid>(type: "uniqueidentifier", nullable: false),
                    role = table.Column<string>(type: "nvarchar(10)", maxLength: 10, nullable: false),
                    content = table.Column<string>(type: "nvarchar(max)", nullable: false),
                    sql_query = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    response_type = table.Column<string>(type: "nvarchar(20)", maxLength: 20, nullable: true),
                    chart_type = table.Column<string>(type: "nvarchar(20)", maxLength: 20, nullable: true),
                    chart_data = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    data_source = table.Column<string>(type: "nvarchar(100)", maxLength: 100, nullable: true),
                    date_range = table.Column<string>(type: "nvarchar(100)", maxLength: 100, nullable: true),
                    model_used = table.Column<string>(type: "nvarchar(50)", maxLength: 50, nullable: true),
                    execution_time_ms = table.Column<int>(type: "int", nullable: true),
                    token_count = table.Column<int>(type: "int", nullable: true),
                    created_at = table.Column<DateTime>(type: "datetime2", nullable: false, defaultValueSql: "GETUTCDATE()")
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_chat_messages", x => x.message_id);
                    table.CheckConstraint("CK_chat_messages_role", "role IN ('user', 'assistant')");
                    table.ForeignKey(
                        name: "FK_chat_messages_chat_sessions_session_id",
                        column: x => x.session_id,
                        principalTable: "chat_sessions",
                        principalColumn: "session_id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "chat_feedback",
                columns: table => new
                {
                    feedback_id = table.Column<Guid>(type: "uniqueidentifier", nullable: false, defaultValueSql: "NEWID()"),
                    message_id = table.Column<Guid>(type: "uniqueidentifier", nullable: false),
                    user_id = table.Column<string>(type: "nvarchar(100)", maxLength: 100, nullable: false),
                    rating = table.Column<int>(type: "int", nullable: false),
                    comment = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    created_at = table.Column<DateTime>(type: "datetime2", nullable: false, defaultValueSql: "GETUTCDATE()")
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_chat_feedback", x => x.feedback_id);
                    table.CheckConstraint("CK_chat_feedback_rating", "rating IN (-1, 1)");
                    table.ForeignKey(
                        name: "FK_chat_feedback_chat_messages_message_id",
                        column: x => x.message_id,
                        principalTable: "chat_messages",
                        principalColumn: "message_id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "user_pins",
                columns: table => new
                {
                    pin_id = table.Column<Guid>(type: "uniqueidentifier", nullable: false, defaultValueSql: "NEWID()"),
                    message_id = table.Column<Guid>(type: "uniqueidentifier", nullable: false),
                    user_id = table.Column<string>(type: "nvarchar(100)", maxLength: 100, nullable: false),
                    created_at = table.Column<DateTime>(type: "datetime2", nullable: false, defaultValueSql: "GETUTCDATE()")
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_user_pins", x => x.pin_id);
                    table.ForeignKey(
                        name: "FK_user_pins_chat_messages_message_id",
                        column: x => x.message_id,
                        principalTable: "chat_messages",
                        principalColumn: "message_id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "IX_chat_feedback_message_id",
                table: "chat_feedback",
                column: "message_id");

            migrationBuilder.CreateIndex(
                name: "IX_chat_messages_session",
                table: "chat_messages",
                columns: new[] { "session_id", "created_at" });

            migrationBuilder.CreateIndex(
                name: "IX_chat_sessions_user",
                table: "chat_sessions",
                columns: new[] { "user_id", "last_message_at" },
                descending: new[] { false, true });

            migrationBuilder.CreateIndex(
                name: "UQ_user_pins_message_user",
                table: "user_pins",
                columns: new[] { "message_id", "user_id" },
                unique: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "chat_feedback");

            migrationBuilder.DropTable(
                name: "user_pins");

            migrationBuilder.DropTable(
                name: "chat_messages");

            migrationBuilder.DropTable(
                name: "chat_sessions");
        }
    }
}
