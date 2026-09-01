-- =============================================================================
-- Script: create-readonly-user.sql
-- Purpose: Create the hawaqm_ai_readonly SQL user with SELECT-only access
--          to the air quality tables used by the HAWAQM AI Chat Service.
-- Run as: sysadmin or db_owner on the air quality database
-- =============================================================================

USE [AQMSADPHCINDOOR]
GO

-- Create login (if using SQL Auth)
IF NOT EXISTS (SELECT 1 FROM sys.server_principals WHERE name = 'hawaqm_ai_readonly')
BEGIN
    CREATE LOGIN [hawaqm_ai_readonly]
        WITH PASSWORD = N'CHANGE_ME_STRONG_PASSWORD_HERE',
             CHECK_EXPIRATION = OFF,
             CHECK_POLICY = ON;
    PRINT 'Login hawaqm_ai_readonly created.';
END
ELSE
BEGIN
    PRINT 'Login hawaqm_ai_readonly already exists.';
END
GO

-- Create user in the database
IF NOT EXISTS (SELECT 1 FROM sys.database_principals WHERE name = 'hawaqm_ai_readonly')
BEGIN
    CREATE USER [hawaqm_ai_readonly] FOR LOGIN [hawaqm_ai_readonly];
    PRINT 'User hawaqm_ai_readonly created.';
END
ELSE
BEGIN
    PRINT 'User hawaqm_ai_readonly already exists.';
END
GO

-- Run grant-readonly-access.sql next to grant SELECT on all actual tables.
PRINT 'hawaqm_ai_readonly setup complete. Now run grant-readonly-access.sql.';
