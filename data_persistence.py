# """
# Data Persistence for Line Crossing Counts INI PAKE JSON
# """
# import os
# import json
# from datetime import datetime
# from config import DATA_SAVE_FILE, STARTING_COUNT


# class DataPersistence:   
#     def __init__(self, filename=DATA_SAVE_FILE):
#         self.filename = filename
#         self.data = {
#             'starting_count': STARTING_COUNT,
#             'current_count': STARTING_COUNT,
#             'total_entries': 0,
#             'total_exits': 0,
#             'session_start': None,
#             'last_update': None,
#             'history': []  # Log of all crossing events
#         }
#         self.load_data()
    
#     def load_data(self):
#         if os.path.exists(self.filename):
#             try:
#                 with open(self.filename, 'r') as f:
#                     loaded_data = json.load(f)
#                     # Keep the loaded data but update starting count if changed in config
#                     if loaded_data.get('starting_count') == STARTING_COUNT:
#                         self.data = loaded_data
#                         print(f"✓ Loaded existing data from {self.filename}")
#                         print(f"  Current count: {self.data['current_count']}")
#                         print(f"  Total entries: {self.data['total_entries']}")
#                         print(f"  Total exits: {self.data['total_exits']}")
#                     else:
#                         print(f"⚠ Starting count changed, resetting data")
#                         self.data['starting_count'] = STARTING_COUNT
#                         self.data['current_count'] = STARTING_COUNT
#                         self.save_data()
#             except Exception as e:
#                 print(f"⚠ Error loading data: {e}")
#                 self.save_data()
#         else:
#             print(f"✓ Starting fresh with count: {STARTING_COUNT}")
#             self.data['session_start'] = datetime.now().isoformat()
#             self.save_data()
    
#     def save_data(self):
#         """Save current data to file"""
#         self.data['last_update'] = datetime.now().isoformat()
#         try:
#             with open(self.filename, 'w') as f:
#                 json.dump(self.data, f, indent=2)
#         except Exception as e:
#             print(f"⚠ Error saving data: {e}")
    
#     def add_entries(self, count):
#         """Add entry events"""
#         self.data['current_count'] += count
#         self.data['total_entries'] += count
        
#         # Log events
#         for _ in range(count):
#             event = {
#                 'timestamp': datetime.now().isoformat(),
#                 'type': 'entry',
#                 'delta': 1,
#                 'count_after': self.data['current_count']
#             }
#             self.data['history'].append(event)
        
#         # Keep only last 1000 events
#         if len(self.data['history']) > 1000:
#             self.data['history'] = self.data['history'][-1000:]
    
#     def add_exits(self, count):
#         """Add exit events"""
#         self.data['current_count'] -= count
#         self.data['total_exits'] += count
        
#         # Log events
#         for _ in range(count):
#             event = {
#                 'timestamp': datetime.now().isoformat(),
#                 'type': 'exit',
#                 'delta': -1,
#                 'count_after': self.data['current_count']
#             }
#             self.data['history'].append(event)
        
#         # Keep only last 1000 events
#         if len(self.data['history']) > 1000:
#             self.data['history'] = self.data['history'][-1000:]
    
#     def get_current_count(self):
#         """Get current count"""
#         return self.data['current_count']
    
#     def get_summary(self):
#         """Get summary statistics"""
#         return {
#             'starting_count': self.data['starting_count'],
#             'current_count': self.data['current_count'],
#             'total_entries': self.data['total_entries'],
#             'total_exits': self.data['total_exits'],
#             'net_change': self.data['current_count'] - self.data['starting_count'],
#             'session_start': self.data['session_start'],
#             'last_update': self.data['last_update']
#         }

"""
Data Persistence for Line Crossing Counts - PostgreSQL Version

Stores counting data in PostgreSQL database instead of JSON files.
Compatible with Odoo deployments and provides better scalability.
"""
import os
import psycopg2
import psycopg2.extras
from datetime import datetime
from config import STARTING_COUNT


# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================
# Reads from environment variables for flexibility
# Set these in your environment or .env file
DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "dbname":   os.getenv("DB_NAME",     "cctv_counting"),
    "user":     os.getenv("DB_USER",     "ferbos"),
    "password": os.getenv("DB_PASSWORD", "cctv_ferbos_2024"),
}

SCHEMA = "cctv"


class DataPersistence:
    """
    PostgreSQL-based data persistence for people counting
    
    Same API as JSON version for drop-in compatibility
    """
    
    def __init__(self):
        self._conn = None
        self._session_id = None
        self._connect()
        self._init_db()
        self._load_or_create_session()

    # ========================================================================
    # DATABASE CONNECTION
    # ========================================================================

    def _connect(self):
        """Establish connection to PostgreSQL database"""
        try:
            self._conn = psycopg2.connect(**DB_CONFIG)
            self._conn.autocommit = False  # Manual transaction control
            print(f"✓ Connected to PostgreSQL")
            print(f"  Host: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
            print(f"  Database: {DB_CONFIG['dbname']}")
        except psycopg2.OperationalError as e:
            print(f"\n❌ PostgreSQL Connection Failed!")
            print(f"   Error: {e}")
            print(f"\n   Please check:")
            print(f"   1. PostgreSQL is running")
            print(f"   2. Database '{DB_CONFIG['dbname']}' exists")
            print(f"   3. Credentials are correct")
            print(f"   4. Connection settings in environment variables")
            print(f"\n   See POSTGRESQL_SETUP.md for help")
            raise

    # ========================================================================
    # SCHEMA & TABLE INITIALIZATION
    # ========================================================================

    def _init_db(self):
        """Create schema and tables if they don't exist"""
        with self._conn.cursor() as cur:
            # Create dedicated schema
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")

            # Session table - stores counting sessions
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

            # Events table - stores individual crossing events
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

            # Index for fast event queries
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_events_session_id
                ON {SCHEMA}.events (session_id)
            """)

        self._conn.commit()
        print(f"✓ Database schema '{SCHEMA}' initialized")

    # ========================================================================
    # SESSION MANAGEMENT
    # ========================================================================

    def _load_or_create_session(self):
        """Resume latest session or create new one"""
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Get most recent session
            cur.execute(f"""
                SELECT * FROM {SCHEMA}.session
                ORDER BY id DESC
                LIMIT 1
            """)
            row = cur.fetchone()

        if row and row["starting_count"] == STARTING_COUNT:
            # Resume existing session
            self._session_id = row["id"]
            print(f"\n✓ Resumed session #{self._session_id}")
            print(f"  Current count: {row['current_count']}")
            print(f"  Total entries: {row['total_entries']}")
            print(f"  Total exits: {row['total_exits']}")
        else:
            # Create new session
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
            
            self._conn.commit()
            print(f"✓ New session created: #{self._session_id}")

    def _get_session(self):
        """Get current session data"""
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"SELECT * FROM {SCHEMA}.session WHERE id = %s",
                (self._session_id,)
            )
            return dict(cur.fetchone())

    def _update_session(self, delta_count, delta_entries, delta_exits):
        """Update session counters"""
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

    # ========================================================================
    # EVENT LOGGING
    # ========================================================================

    def _log_events(self, event_type, delta, count):
        """Log individual crossing events"""
        session = self._get_session()
        current = session["current_count"]
        direction = 1 if event_type == "entry" else -1

        # Calculate count_after for each event
        base = current - direction * (count - 1)
        rows = [
            (self._session_id, event_type, delta, base + direction * i)
            for i in range(count)
        ]

        with self._conn.cursor() as cur:
            # Batch insert events
            psycopg2.extras.execute_values(
                cur,
                f"""INSERT INTO {SCHEMA}.events
                    (session_id, event_type, delta, count_after)
                    VALUES %s""",
                rows,
            )

            # Keep only last 1000 events per session
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

    # ========================================================================
    # PUBLIC API (Compatible with JSON version)
    # ========================================================================

    def add_entries(self, count):
        """Record entry events"""
        self._update_session(delta_count=count, delta_entries=count, delta_exits=0)
        self._log_events("entry", delta=1, count=count)

    def add_exits(self, count):
        """Record exit events"""
        self._update_session(delta_count=-count, delta_entries=0, delta_exits=count)
        self._log_events("exit", delta=-1, count=count)

    def get_current_count(self):
        """Get current count"""
        return self._get_session()["current_count"]

    def get_summary(self):
        """Get summary statistics - same format as JSON version"""
        row = self._get_session()
        return {
            'starting_count': row["starting_count"],
            'current_count':  row["current_count"],
            'total_entries':  row["total_entries"],
            'total_exits':    row["total_exits"],
            'net_change':     row["current_count"] - row["starting_count"],
            'session_start':  str(row["session_start"]),
            'last_update':    str(row["last_update"]) if row["last_update"] else None,
        }

    def get_history(self, limit=100):
        """Get recent events (bonus feature not in JSON version)"""
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

    def save_data(self):
        """No-op for compatibility with JSON version (auto-saves to DB)"""
        pass

    def close(self):
        """Close database connection"""
        if self._conn and not self._conn.closed:
            self._conn.close()
            print("✓ PostgreSQL connection closed")

    # ========================================================================
    # CONTEXT MANAGER SUPPORT
    # ========================================================================

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False