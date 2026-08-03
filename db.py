"""
db.py - all PostgreSQL access for the SQL Insight Assistant.

Kept separate from app.py so database logic and LLM/chat logic can evolve
independently (e.g. swapping drivers or adding a second database source
without touching the LangGraph/Chainlit code).
"""

import os                                   # to read environment variables (APP_DB_URL)
import pandas as pd                         # DataFrame support for visualization and analysis
from sqlalchemy import create_engine, text  # create_engine = connection pool, text() = safe raw SQL wrapper
from sqlalchemy.engine import Engine        # type hints only

# ---------------------------------------------------------------------------
# Schema cache
# ---------------------------------------------------------------------------
# A plain module-level dict acts as our cache. It lives in memory for as
# long as the Python process runs. We deliberately keep this simple (no
# Redis, no file cache) because for an MVP the schema is small and rarely
# changes while the app is running - re-reading it from information_schema
# on every single user question would add latency and DB load for no benefit.
_schema_cache: dict[str, str] = {}


def get_engine() -> Engine:
    """
    Build a SQLAlchemy engine from the APP_DB_URL environment variable.

    Expected format:
        postgresql+psycopg2://<user>:<password>@<host>:<port>/<database>

    Why SQLAlchemy instead of calling psycopg2 directly?
    SQLAlchemy gives us connection pooling, a `text()` helper for safe
    parameterised/raw SQL, and a driver-agnostic API - so swapping Postgres
    for another database later only means changing the URL, not the code.
    """
    database_url = os.environ["APP_DB_URL"]  # KeyError here fails fast with a clear message if unset
    # pool_pre_ping=True makes SQLAlchemy test a pooled connection before
    # handing it back out, which avoids cryptic "connection closed" errors
    # after the DB has been idle for a while (common with cloud Postgres).
    return create_engine(database_url, pool_pre_ping=True)


def load_schema(engine: Engine, force_refresh: bool = False) -> str:
    """
    Return a compact, LLM-friendly description of every table and column in
    the appropriate schema, e.g.: "public", "sales", etc.

        products(id integer, name text, price numeric)
        orders(id integer, product_id integer, quantity integer, order_date date)

    The result is cached in `_schema_cache` after the first call so that
    later questions in the same run reuse it instantly instead of hitting
    the database again. Pass force_refresh=True if the schema changed and
    you want to re-read it (e.g. after adding a table).
    """
    if not force_refresh and "schema_text" in _schema_cache:
        return _schema_cache["schema_text"]  # cache hit — skip the DB round-trip entirely

    # information_schema.columns is part of the ANSI SQL standard, so this
    # query works on any Postgres database without needing special
    # extensions or admin privileges.
    query = text(
        """
        SELECT table_schema, table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name, ordinal_position
        """
    )

    with engine.connect() as conn:            # opens a pooled connection, auto-closes on exit
        rows = conn.execute(query).fetchall()  # one round-trip fetches all tables + columns at once

    # Group the flat (table, column, type) rows into {table: [columns]} so
    # we can print one line per table instead of one line per column -
    # this keeps the prompt we later send to the LLM short and cheap.
    tables: dict[str, list[str]] = {}
    for table_schema, table_name, column_name, data_type in rows:
        qualified_name = f"{table_schema}.{table_name}"
        tables.setdefault(qualified_name, []).append(f"{column_name} {data_type}")

    schema_text = "\n".join(
        f"{table}({', '.join(columns)})" for table, columns in tables.items()
    )

    _schema_cache["schema_text"] = schema_text  # store for next call
    return schema_text


def run_sql_dataframe(engine: Engine, sql: str, row_limit: int = 50) -> tuple[pd.DataFrame, str]:
    """
    Execute a read-only SQL query and return both a DataFrame and a text table.
    The DataFrame is used for visualization and structured analysis.
    """
    # Prefix check only - not a full SQL parser, so a query like
    # "select 1; drop table x;" would pass this. In production this should
    # be paired with a database role that has read-only permissions, so
    # this check is a fast first line of defense, not the only one.
    normalized = sql.strip().lower()
    if not (normalized.startswith("select") or normalized.startswith("with")):
        raise ValueError("Only SELECT queries are allowed for safety.")

    with engine.connect() as conn:
        result = conn.execute(text(sql))
        columns = list(result.keys())
        rows = result.fetchmany(row_limit)

    df = pd.DataFrame(rows, columns=columns)

    if df.empty:
        text_result = "Query ran successfully but returned no rows."
    else:
        text_result = df.to_string(index=False)

    return df, text_result
