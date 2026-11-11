BEGIN IMMEDIATE;

DROP TABLE IF EXISTS auth_attempts;
DROP TABLE IF EXISTS admin_audit_logs;
DROP TABLE IF EXISTS activity_events;
DROP TABLE IF EXISTS planner_reports;
DROP TABLE IF EXISTS membership_applications;
DROP TABLE IF EXISTS password_reset_tokens;
DROP TABLE IF EXISTS sessions;
DROP TABLE IF EXISTS users;
DELETE FROM schema_migrations WHERE version = 1;

COMMIT;
