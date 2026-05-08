PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER NOT NULL,
    applied_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS crosspoll_seeds (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_channel  TEXT NOT NULL,
    summary         TEXT NOT NULL,
    payload         TEXT NOT NULL DEFAULT '{}',
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_crosspoll_seeds_created ON crosspoll_seeds(created_at);

CREATE TABLE IF NOT EXISTS crosspoll_consumptions (
    seed_id      INTEGER NOT NULL REFERENCES crosspoll_seeds(id) ON DELETE CASCADE,
    dest_channel TEXT NOT NULL,
    consumed_at  REAL NOT NULL,
    proposal_id  TEXT NOT NULL,
    PRIMARY KEY (seed_id, dest_channel)
);
CREATE INDEX IF NOT EXISTS idx_crosspoll_consumptions_dest ON crosspoll_consumptions(dest_channel, consumed_at);
