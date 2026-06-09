import os
import psycopg2
import psycopg2.extras
from datetime import datetime
from config import STARTING_COUNT

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "dbname":   os.getenv("DB_NAME",     "cctv_counting"),
    "user":     os.getenv("DB_USER",     "ferbos"),
    "password": os.getenv("DB_PASSWORD", "cctv_ferbos_2024"),
}

SCHEMA = "cctv"

class DataPersistence:
    def __init__(self):
        self._conn = None
        self._session_id = None
        self._current_count = 0
        self._connect()
        self._init_db()
        self._load_or_create_session()

    def _connect(self):
        try:
            self._conn = psycopg2.connect(**DB_CONFIG)
            self._conn.autocommit = False
            print(f"✓ Connected to PostgreSQL")
            print(f"  Host: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
            print(f"  Database: {DB_CONFIG['dbname']}")
        except psycopg2.OperationalError as e:
            print(f"\n❌ PostgreSQL Connection Failed!")
            print(f"   Error: {e}")
            raise

    def _init_db(self):
        with self._conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA}.session (
                    id              SERIAL PRIMARY KEY,
                    starting_count  INTEGER     NOT NULL,
                    current_count   INTEGER     NOT NULL,
                    total_entries   INTEGER     NOT NULL DEFAULT 0,
                    total_exits     INTEGER     NOT NULL DEFAULT 0,
                    session_start   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_update     TIMESTAMPTZ
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA}.events (
                    id          SERIAL PRIMARY KEY,
                    session_id  INTEGER     NOT NULL REFERENCES {SCHEMA}.session(id),
                    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    event_type  VARCHAR(5)  NOT NULL CHECK (event_type IN ('entry', 'exit')),
                    delta       SMALLINT    NOT NULL,
                    count_after INTEGER     NOT NULL
                )
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_events_session_id
                ON {SCHEMA}.events (session_id)
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA}.realtime_metrics (
                    time        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    roi_count   INTEGER,
                    occupancy   INTEGER,
                    fps         FLOAT,
                    wifi_signal INTEGER
                )
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_realtime_metrics_time
                ON {SCHEMA}.realtime_metrics (time DESC)
            """)
        self._conn.commit()
        print(f"✓ Database schema '{SCHEMA}' initialized")

    def _load_or_create_session(self):
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"SELECT * FROM {SCHEMA}.session ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()

        if row and row["starting_count"] == STARTING_COUNT:
            self._session_id = row["id"]
            self._current_count = row["current_count"]
            print(f"\n✓ Resumed session #{self._session_id}")
            print(f"  Current count: {row['current_count']}")
            print(f"  Total entries: {row['total_entries']}")
            print(f"  Total exits: {row['total_exits']}")
        else:
            if row:
                print("\n⚠ Starting count changed - creating new session")
            else:
                print(f"\n✓ Starting fresh with count: {STARTING_COUNT}")
            with self._conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {SCHEMA}.session
                        (starting_count, current_count, session_start)
                    VALUES (%s, %s, NOW())
                    RETURNING id
                """, (STARTING_COUNT, STARTING_COUNT))
                self._session_id = cur.fetchone()[0]
            self._current_count = STARTING_COUNT
            self._conn.commit()
            print(f"✓ New session created: #{self._session_id}")

    def _get_session(self):
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"SELECT * FROM {SCHEMA}.session WHERE id = %s", (self._session_id,))
            return dict(cur.fetchone())

    def _update_session(self, delta_count, delta_entries, delta_exits):
        with self._conn.cursor() as cur:
            cur.execute(f"""
                UPDATE {SCHEMA}.session
                SET current_count = current_count + %s,
                    total_entries = total_entries + %s,
                    total_exits   = total_exits + %s,
                    last_update   = NOW()
                WHERE id = %s
            """, (delta_count, delta_entries, delta_exits, self._session_id))
        self._conn.commit()

    def _log_events(self, event_type, delta, count):
        current = self._current_count
        direction = 1 if event_type == "entry" else -1
        base = current - direction * (count - 1)
        rows = [
            (self._session_id, event_type, delta, base + direction * i)
            for i in range(count)
        ]
        with self._conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                f"""INSERT INTO {SCHEMA}.events
                    (session_id, event_type, delta, count_after)
                    VALUES %s""",
                rows,
            )
            cur.execute(f"""
                DELETE FROM {SCHEMA}.events
                WHERE session_id = %s
                  AND id NOT IN (
                      SELECT id FROM {SCHEMA}.events
                      WHERE session_id = %s
                      ORDER BY id DESC
                      LIMIT 1000
                  )
            """, (self._session_id, self._session_id))
        self._conn.commit()

    def add_entries(self, count):
        self._current_count += count
        self._update_session(delta_count=count, delta_entries=count, delta_exits=0)
        self._log_events("entry", delta=1, count=count)

    def add_exits(self, count):
        self._current_count -= count
        self._update_session(delta_count=-count, delta_entries=0, delta_exits=count)
        self._log_events("exit", delta=-1, count=count)

    def get_current_count(self):
        return self._current_count

    def get_summary(self):
        row = self._get_session()
        return {
            'starting_count': row["starting_count"],
            'current_count':  self._current_count,
            'total_entries':  row["total_entries"],
            'total_exits':    row["total_exits"],
            'net_change':     self._current_count - row["starting_count"],
            'session_start':  str(row["session_start"]),
            'last_update':    str(row["last_update"]) if row["last_update"] else None,
        }

    def get_history(self, limit=100):
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT occurred_at AS timestamp,
                       event_type  AS type,
                       delta,
                       count_after
                FROM {SCHEMA}.events
                WHERE session_id = %s
                ORDER BY id DESC
                LIMIT %s
            """, (self._session_id, limit))
            return [dict(row) for row in cur.fetchall()]

    def update_realtime_metrics(self, roi_count, occupancy, fps, wifi_signal):
        """Write a snapshot every ~1s for Grafana time-series panels."""
        try:
            with self._conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {SCHEMA}.realtime_metrics
                        (roi_count, occupancy, fps, wifi_signal)
                    VALUES (%s, %s, %s, %s)
                """, (roi_count, occupancy, fps, wifi_signal))
                cur.execute(f"""
                    DELETE FROM {SCHEMA}.realtime_metrics
                    WHERE time < NOW() - INTERVAL '24 hours'
                """)
            self._conn.commit()
        except Exception as e:
            print(f"⚠ Metrics write failed: {e}")

    def save_data(self):
        pass

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
            print("✓ PostgreSQL connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
