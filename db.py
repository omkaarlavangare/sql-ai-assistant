"""PostgreSQL access for the SQL Insight Assistant."""

import os  # Read environment variables.
import pandas as pd  # DataFrame support for analysis and charts.
from sqlalchemy import create_engine, text  # Database connection helpers.
from sqlalchemy.engine import Engine  # Type hints.

# Simple in-memory schema cache.
_schema_cache: dict[str, str] = {}


def get_engine() -> Engine:
    """Create a SQLAlchemy engine from the configured database URL."""
    database_url = os.environ["APP_DB_URL"]
    return create_engine(database_url, pool_pre_ping=True)


def load_schema(engine: Engine, force_refresh: bool = False) -> str:
    """Return a compact schema description for the LLM."""
    if not force_refresh and "schema_text" in _schema_cache:
        return _schema_cache["schema_text"]
    query = text(
        """
        SELECT table_schema, table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name, ordinal_position
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    # Group column rows by table for a compact schema string.
    tables: dict[str, list[str]] = {}
    for table_schema, table_name, column_name, data_type in rows:
        qualified_name = f"{table_schema}.{table_name}"
        tables.setdefault(qualified_name, []).append(f"{column_name} {data_type}")

    schema_text = "\n".join(
        f"{table}({', '.join(columns)})" for table, columns in tables.items()
    )

    _schema_cache["schema_text"] = schema_text
    return schema_text


def run_sql_dataframe(engine: Engine, sql: str, row_limit: int = 50) -> tuple[pd.DataFrame, str]:
    """Execute a read-only SQL query and return a DataFrame plus text output."""
    # This is a basic safety check, not a full SQL parser.
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
