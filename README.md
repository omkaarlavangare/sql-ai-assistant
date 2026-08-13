# SQL Insight Assistant

SQL Insight Assistant is an AI-powered, natural-language-to-SQL analytics copilot for PostgreSQL. It helps user ask business questions such as "top 5 products by sales" and receive a safe SQL query, executed results, a plain-English insight, and an interactive Plotly chart inside a modern Chainlit chat experience.

Built with **LangGraph** for resilient multi-step workflows, **LangChain + langchain-anthropic** for Claude-powered SQL generation, **Plotly** for interactive visualizations, and **Chainlit** for a polished conversational UI.

## How it works

```
User question
     │
     ▼
 write_sql  ──────────────► Claude writes a SELECT query using the
     │                      cached DB schema as context
     ▼
 execute_sql  ─── error, < 2 tries ───► back to write_sql (self-correct)
     │
     └─ success (or out of retries)
     ▼
 generate_viz_spec ──────► LLM selects a simple Plotly chart spec
     │                      based on the query results
     ▼
 generate_insight ───────► Claude turns the raw rows into a plain-English
     │                      answer and insight summary
     ▼
 Answer shown in chat with an interactive Plotly visualisation
```

This is implemented as a LangGraph graph in `app.py`, because a plain
"prompt → SQL → run" pipeline breaks the moment the LLM writes SQL with a
typo or a wrong column name - which happens often. The graph loops back to
regenerate the SQL (capped at 2 attempts) when execution fails, instead of
just crashing.

The database schema (table + column names) is read once via
`information_schema.columns` and cached in memory (see `db.py`) so it isn't
re-fetched on every single question. The workflow also generates an
interactive Plotly chart when the result set is suitable for visualization,
making the assistant useful for both analytics answers and data storytelling.

## Why this project stands out

- AI-powered natural language analytics for PostgreSQL and SQL workflows
- Safe, read-only query execution with an extra defensive guard for destructive prompts
- Multi-step LangGraph orchestration that retries failed SQL instead of failing fast
- Interactive Plotly visualisations for charts, trends, and comparisons
- Lightweight, extensible MVP architecture designed for rapid experimentation
- Structured JSONL logging for debugging workflow failures and final run summaries

## Logging and observability

The app now records workflow activity as structured JSONL entries instead of mixing plain-text markers into the file. This keeps logs machine-readable and makes it easier to inspect failures or audit a completed run.

### Log envelope

Every record follows this schema:

```json
{
  "record_type": "step" | "run_summary",
  "run_id": "<uuid>",
  "timestamp": "<ISO8601 UTC>",
  "schema_version": "1.0",
  "status": "success" | "error",
  "duration_ms": 123,
  "error": null | "<error message>",
  "payload": { ... }
}
```

### Behavior

- Failed workflow steps log a single `step` record with the step name, timing, and error message.
- Successful step executions do not emit success logs.
- Each completed request writes exactly one final `run_summary` record containing the original question, SQL, attempts, structured rows, chart metadata, generated response, and per-step timings.
- The log file remains valid JSONL: no plain-text separators or `=== RUN START ===` markers are written.

Example final summary record:

```json
{
  "record_type": "run_summary",
  "run_id": "d5c6d7d4-1a9d-4d8d-a81a-6b7efccf0d21",
  "timestamp": "2026-08-14T12:34:56.789Z",
  "schema_version": "1.0",
  "status": "success",
  "duration_ms": 9854,
  "error": null,
  "payload": {
    "question": "Top 5 products by sales",
    "sql_query": "SELECT product_name, SUM(total_sales) ...",
    "attempts": 1,
    "row_count": 5,
    "rows": [
      {"product_name": "Laptop Pro", "total_sales": 86936},
      {"product_name": "Mobile X", "total_sales": 74132}
    ],
    "viz_spec": {
      "chart_type": "bar",
      "x_column": "product_name",
      "y_column": "total_sales",
      "title": "Sales by Product"
    },
    "chart_created": true,
    "generated_response": "The top products by sales were ...",
    "step_timings": {
      "write_sql": 120,
      "execute_sql": 70,
      "generate_viz_spec": 45,
      "generate_insight": 4200
    }
  }
}
```

## Project structure

```
.
├── app.py             # LangGraph workflow + Chainlit chat handlers
├── db.py               # DB engine, schema caching, safe SQL execution
├── requirements.txt      # pinned, verified-compatible dependencies
├── .env.example            # template for your API key + DB connection string
└── .gitignore
```

Only two Python files on purpose - everything else is config. This keeps
the whole thing readable in one sitting while still being easy to grow
(see "Next steps" below).

## Setup (using [uv](https://docs.astral.sh/uv/))

```bash
# 1. Create and activate a virtual environment
uv venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
uv add -r requirements.txt

# 3. Configure secrets
cp .env.example .env
# then edit .env with your ANTHROPIC_API_KEY and `APP_DB_URL` (SQLAlchemy connection string)

# 4. Run the app
chainlit run app.py -w
```

`-w` enables auto-reload while you edit the code. Chainlit will open the
chat UI at `http://localhost:8000`.

## Try it

Once connected, ask things like:

- "top 5 products by sales"
- "which customers spent the most in January 2011"
- "monthly revenue trend in the year 2011"
- "show me a chart of revenue by month"
- "compare sales across regions"

The assistant will show you the generated SQL (for transparency), an
interactive Plotly chart when appropriate, and a plain-English answer that
helps turn raw database results into actionable business insight.

## Safety notes for this MVP

- `db.py` only allows statements starting with `SELECT`/`WITH` - this stops
  the LLM (or a malicious prompt) from running `DELETE`/`DROP`/etc.
- For real production use, also connect with a **read-only database role**
  as a second line of defense, in addition to the query-prefix check above.
- Query results are capped at 50 rows before being summarized, to keep
  LLM costs and latency predictable.

## Next steps (ideas to extend this MVP)

- **Dashboard creation**: add a feature to build interactive dashboards from query results.
- **RAG integration**: implement retrieval-augmented generation to answer questions using documents and internal knowledge.
- **Planner agent**: add a planner agent that decides whether to use SQL, RAG, or both for a given question.
- **Read-only DB role**: create a dedicated Postgres user with only
  `SELECT` grants and point `APP_DB_URL` at it.
