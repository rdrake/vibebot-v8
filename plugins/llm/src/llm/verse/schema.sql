PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER NOT NULL,
    applied_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL CHECK (kind IN ('avatar','npc','place','faction','item')),
    name       TEXT NOT NULL,
    summary    TEXT NOT NULL DEFAULT '',
    status     TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','retired')),
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entities_kind ON entities(kind, status);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);

CREATE TABLE IF NOT EXISTS attributes (
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    key       TEXT NOT NULL,
    value     TEXT NOT NULL,
    PRIMARY KEY (entity_id, key)
);
CREATE INDEX IF NOT EXISTS idx_attributes_kv ON attributes(key, value);

CREATE TABLE IF NOT EXISTS relations (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    from_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    to_id   INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    kind    TEXT NOT NULL,
    note    TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_id, kind);
CREATE INDEX IF NOT EXISTS idx_relations_to   ON relations(to_id, kind);

CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    summary    TEXT NOT NULL,
    entity_ids TEXT NOT NULL DEFAULT '[]',
    source     TEXT NOT NULL CHECK (source IN ('avatar','loom','crosspoll','operator','llm'))
);
CREATE INDEX IF NOT EXISTS idx_events_ts     ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_source ON events(source);

CREATE TABLE IF NOT EXISTS avatar_link (
    entity_id INTEGER PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
    nick      TEXT NOT NULL,
    account   TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_avatar_link_nick    ON avatar_link(nick);
CREATE UNIQUE INDEX IF NOT EXISTS idx_avatar_link_account ON avatar_link(account) WHERE account IS NOT NULL;
-- Case-insensitive seek path for the opt-in nick fallback. The binary-unique
-- idx_avatar_link_nick above cannot serve `nick = ? COLLATE NOCASE`, so without
-- this index that lookup degrades to a full SCAN while the write lock is held.
-- Non-unique: case-insensitive uniqueness already belongs to the binary index.
CREATE INDEX IF NOT EXISTS idx_avatar_link_nick_nocase ON avatar_link(nick COLLATE NOCASE);

CREATE TABLE IF NOT EXISTS proposals (
    id          TEXT PRIMARY KEY,
    created_at  REAL NOT NULL,
    cycle_id    TEXT NOT NULL,
    op          TEXT NOT NULL CHECK (op IN ('add_event','set_attribute','add_relation','add_entity','crosspoll_seed','update_entity','set_status','edit_event','delete_event','delete_relation','set_pinned')),
    payload     TEXT NOT NULL,
    confidence  REAL NOT NULL,
    provenance  TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','approved','rejected')),
    reviewer    TEXT,
    reviewed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals(status, created_at);

CREATE TABLE IF NOT EXISTS entity_alias (
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    alias     TEXT NOT NULL COLLATE NOCASE,
    PRIMARY KEY (entity_id, alias)
);
CREATE INDEX IF NOT EXISTS idx_entity_alias_alias ON entity_alias(alias COLLATE NOCASE);

CREATE TABLE IF NOT EXISTS event_actor (
    event_id  INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (event_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_event_actor_entity ON event_actor(entity_id, event_id);
