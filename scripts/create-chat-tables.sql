-- =============================================================================
-- Script: create-chat-tables.sql
-- Purpose: Create the chat history user and tables for the HAWAQM AI service.
--          EF Core migrations will apply these automatically on startup,
--          but this script can be run manually for DBA review/approval.
-- Run as: sysadmin or db_owner on the target database
-- =============================================================================

USE [AQMSADPHCINDOOR]
GO

-- ── 1. Create SQL login and user for chat history ─────────────────────────
IF NOT EXISTS (SELECT 1 FROM sys.server_principals WHERE name = 'hawaqm_ai_chat')
BEGIN
    CREATE LOGIN [hawaqm_ai_chat]
        WITH PASSWORD = N'CHANGE_ME_STRONG_PASSWORD_HERE',
             CHECK_EXPIRATION = OFF,
             CHECK_POLICY = ON;
    PRINT 'Login hawaqm_ai_chat created.';
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.database_principals WHERE name = 'hawaqm_ai_chat')
BEGIN
    CREATE USER [hawaqm_ai_chat] FOR LOGIN [hawaqm_ai_chat];
    PRINT 'User hawaqm_ai_chat created.';
END
GO

-- ── 2. Create chat_sessions table ─────────────────────────────────────────
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'chat_sessions')
BEGIN
    CREATE TABLE dbo.chat_sessions (
        session_id      UNIQUEIDENTIFIER NOT NULL CONSTRAINT PK_chat_sessions PRIMARY KEY DEFAULT NEWID(),
        user_id         NVARCHAR(100) NOT NULL,
        title           NVARCHAR(255) NOT NULL DEFAULT 'New conversation',
        scope           NVARCHAR(MAX) NULL,       -- JSON: {site_id, period, pollutants}
        message_count   INT NOT NULL DEFAULT 0,
        site_name       NVARCHAR(100) NULL,
        created_at      DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        last_message_at DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        is_active       BIT NOT NULL DEFAULT 1
    );

    CREATE INDEX IX_chat_sessions_user
        ON dbo.chat_sessions (user_id, last_message_at DESC);

    PRINT 'chat_sessions table created.';
END
ELSE
    PRINT 'chat_sessions table already exists.';
GO

-- ── 3. Create chat_messages table ─────────────────────────────────────────
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'chat_messages')
BEGIN
    CREATE TABLE dbo.chat_messages (
        message_id       UNIQUEIDENTIFIER NOT NULL CONSTRAINT PK_chat_messages PRIMARY KEY DEFAULT NEWID(),
        session_id       UNIQUEIDENTIFIER NOT NULL
                            CONSTRAINT FK_chat_messages_session
                            REFERENCES dbo.chat_sessions(session_id) ON DELETE CASCADE,
        role             VARCHAR(10) NOT NULL CONSTRAINT CK_chat_messages_role CHECK (role IN ('user', 'assistant')),
        content          NVARCHAR(MAX) NOT NULL,
        sql_query        NVARCHAR(MAX) NULL,
        response_type    VARCHAR(20) NULL,     -- 'text', 'table', 'chart'
        chart_type       VARCHAR(20) NULL,     -- 'line', 'bar', 'area', 'scatter'
        chart_data       NVARCHAR(MAX) NULL,   -- JSON array
        data_source      NVARCHAR(100) NULL,
        date_range       NVARCHAR(100) NULL,
        model_used       VARCHAR(50) NULL,
        execution_time_ms INT NULL,
        token_count      INT NULL,
        created_at       DATETIME2 NOT NULL DEFAULT GETUTCDATE()
    );

    CREATE INDEX IX_chat_messages_session
        ON dbo.chat_messages (session_id, created_at);

    PRINT 'chat_messages table created.';
END
ELSE
    PRINT 'chat_messages table already exists.';
GO

-- ── 4. Create chat_feedback table ─────────────────────────────────────────
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'chat_feedback')
BEGIN
    CREATE TABLE dbo.chat_feedback (
        feedback_id  UNIQUEIDENTIFIER NOT NULL CONSTRAINT PK_chat_feedback PRIMARY KEY DEFAULT NEWID(),
        message_id   UNIQUEIDENTIFIER NOT NULL
                        CONSTRAINT FK_chat_feedback_message
                        REFERENCES dbo.chat_messages(message_id) ON DELETE CASCADE,
        user_id      NVARCHAR(100) NOT NULL,
        rating       INT NOT NULL CONSTRAINT CK_chat_feedback_rating CHECK (rating IN (-1, 1)),
        comment      NVARCHAR(MAX) NULL,
        created_at   DATETIME2 NOT NULL DEFAULT GETUTCDATE()
    );

    PRINT 'chat_feedback table created.';
END
ELSE
    PRINT 'chat_feedback table already exists.';
GO

-- ── 5. Create user_pins table ─────────────────────────────────────────────
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'user_pins')
BEGIN
    CREATE TABLE dbo.user_pins (
        pin_id      UNIQUEIDENTIFIER NOT NULL CONSTRAINT PK_user_pins PRIMARY KEY DEFAULT NEWID(),
        message_id  UNIQUEIDENTIFIER NOT NULL
                        CONSTRAINT FK_user_pins_message
                        REFERENCES dbo.chat_messages(message_id) ON DELETE CASCADE,
        user_id     NVARCHAR(100) NOT NULL,
        created_at  DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        CONSTRAINT UQ_user_pins_message_user UNIQUE (message_id, user_id)
    );

    PRINT 'user_pins table created.';
END
ELSE
    PRINT 'user_pins table already exists.';
GO

-- ── 6. Grant permissions to hawaqm_ai_chat ────────────────────────────────
GRANT SELECT, INSERT, UPDATE, DELETE ON dbo.chat_sessions TO [hawaqm_ai_chat];
GRANT SELECT, INSERT, UPDATE, DELETE ON dbo.chat_messages TO [hawaqm_ai_chat];
GRANT SELECT, INSERT, UPDATE, DELETE ON dbo.chat_feedback TO [hawaqm_ai_chat];
GRANT SELECT, INSERT, UPDATE, DELETE ON dbo.user_pins     TO [hawaqm_ai_chat];
PRINT 'Permissions granted to hawaqm_ai_chat.';
GO

PRINT 'Chat tables setup complete.';
