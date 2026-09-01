-- =============================================================================
-- Script: grant-readonly-access.sql
-- Purpose: Grant SELECT-only access on ALL tables in AQMSADPHCINDOOR to the
--          hawaqm_ai_readonly user (already created). Run as sysadmin or db_owner.
-- =============================================================================

USE [AQMSADPHCINDOOR]
GO

-- Verify the user exists before granting
IF NOT EXISTS (SELECT 1 FROM sys.database_principals WHERE name = 'hawaqm_ai_readonly')
BEGIN
    RAISERROR('User hawaqm_ai_readonly does not exist. Run create-readonly-user.sql first.', 16, 1);
    RETURN;
END
GO

-- ── Grant SELECT on every table ──────────────────────────────────────────────
GRANT SELECT ON dbo.AmbientData                         TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.AmbientParameters                   TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.AmbientStations                     TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.Device_Alarms                       TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.DeviceCompliance                    TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.DMN_Alarms                          TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.DMN_Customer_Alarms                 TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.DMN_Devices                         TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.DMN_Flags                           TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.DMN_Mold_Parameters                 TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.DMN_Mold_tests                      TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.DMN_Parameters                      TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.DMN_Permissions                     TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.DMN_Permissions_DOH                 TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.DMN_StationGrouping                 TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.DMN_Stations                        TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.License                             TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.LogTable                            TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.MST_Devices_Drivers                 TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.MST_Devices_Model                   TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.NewDeviceData                       TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.Parameter_Alarms                    TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.Parameter_Conversion                TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.ParameterAverages                   TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.ParameterAveragesMonth              TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.ParameterAveragesYear               TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.ParameterReadings                   TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.ParameterReadingsHistory            TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.ParameterReadingsSiteInActiveOffload TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.Parameters_Excedence_Values         TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.Pollutents_config                   TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.Regions                             TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.ReportedUnits                       TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.RolePermission                      TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.RoleStations                        TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.Sectors                             TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.SubSectors                          TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.UserLoginHistory                    TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.Users                               TO [hawaqm_ai_readonly];
GRANT SELECT ON dbo.UsersGroups                         TO [hawaqm_ai_readonly];

-- Also grant on the chat tables (created by create-chat-tables.sql) — read-only access only
-- The hawaqm_ai_chat user gets full CRUD; hawaqm_ai_readonly gets SELECT for auditability
IF OBJECT_ID('dbo.chat_sessions', 'U') IS NOT NULL
BEGIN
    GRANT SELECT ON dbo.chat_sessions TO [hawaqm_ai_readonly];
    GRANT SELECT ON dbo.chat_messages TO [hawaqm_ai_readonly];
    GRANT SELECT ON dbo.chat_feedback TO [hawaqm_ai_readonly];
    GRANT SELECT ON dbo.user_pins     TO [hawaqm_ai_readonly];
    PRINT 'SELECT granted on chat tables.';
END

PRINT 'SELECT granted on all 40 tables to hawaqm_ai_readonly.';
GO

-- ── Belt-and-suspenders: deny all writes at the schema level ─────────────────
-- This catches any future tables added to dbo schema automatically
DENY INSERT, UPDATE, DELETE, EXECUTE ON SCHEMA::dbo TO [hawaqm_ai_readonly];
PRINT 'INSERT/UPDATE/DELETE/EXECUTE denied at schema level.';
GO

PRINT 'Done. hawaqm_ai_readonly has SELECT-only access to all tables.';
