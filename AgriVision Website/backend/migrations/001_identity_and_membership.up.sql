BEGIN IMMEDIATE;

CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    username TEXT UNIQUE COLLATE NOCASE,
    display_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin', 'superadmin')),
    account_status TEXT NOT NULL DEFAULT 'active' CHECK (account_status IN ('active', 'suspended')),
    membership_status TEXT NOT NULL DEFAULT 'none' CHECK (membership_status IN ('none', 'pending', 'approved', 'rejected')),
    premium_since TEXT,
    saved_preferences_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    last_login_at TEXT
);

CREATE TABLE sessions (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    revoked_at TEXT,
    ip_hash TEXT,
    user_agent TEXT NOT NULL DEFAULT ''
);
CREATE INDEX sessions_user_active ON sessions(user_id, revoked_at, expires_at);

CREATE TABLE password_reset_tokens (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    consumed_at TEXT
);
CREATE INDEX password_reset_user ON password_reset_tokens(user_id, consumed_at, expires_at);

CREATE TABLE membership_applications (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    plan_code TEXT NOT NULL CHECK (plan_code = 'field-plus'),
    message TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
    decision_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    decision_note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    decided_at TEXT
);
CREATE INDEX membership_status_created ON membership_applications(status, created_at DESC);
CREATE INDEX membership_user_created ON membership_applications(user_id, created_at DESC);

CREATE TABLE planner_reports (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    inputs_json TEXT NOT NULL,
    report_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX planner_reports_user_created ON planner_reports(user_id, created_at DESC);

CREATE TABLE activity_events (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX activity_user_created ON activity_events(user_id, created_at DESC);
CREATE INDEX activity_created ON activity_events(created_at DESC);

CREATE TABLE admin_audit_logs (
    id INTEGER PRIMARY KEY,
    actor_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    target_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    action TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    ip_hash TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX admin_audit_created ON admin_audit_logs(created_at DESC);
CREATE INDEX admin_audit_target ON admin_audit_logs(target_user_id, created_at DESC);
CREATE TRIGGER admin_audit_no_update
BEFORE UPDATE ON admin_audit_logs
BEGIN
    SELECT RAISE(ABORT, 'administrative audit entries are immutable');
END;
CREATE TRIGGER admin_audit_no_delete
BEFORE DELETE ON admin_audit_logs
BEGIN
    SELECT RAISE(ABORT, 'administrative audit entries are immutable');
END;

CREATE TABLE auth_attempts (
    id INTEGER PRIMARY KEY,
    scope TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    attempted_at TEXT NOT NULL
);
CREATE INDEX auth_attempt_scope_key_time ON auth_attempts(scope, key_hash, attempted_at DESC);

INSERT INTO schema_migrations(version, name, applied_at)
VALUES (1, 'identity_and_membership', strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));

COMMIT;
