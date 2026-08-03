# SQL Insight Assistant

A minimal chat assistant that answers natural-language analytics questions
(e.g. *"top 5 products by sales"*) by generating SQL against your PostgreSQL
database, running it, and explaining the result in plain English.

Built with **LangGraph** (workflow/state machine), **LangChain +
langchain-anthropic** (Claude access), and **Chainlit** (chat UI).

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
 generate_insight ──────► Claude turns the raw rows into a plain-English
     │                    answer
     ▼
 Answer shown in chat
```

This is implemented as a LangGraph graph in `app.py`, because a plain
"prompt → SQL → run" pipeline breaks the moment the LLM writes SQL with a
typo or a wrong column name - which happens often. The graph loops back to
regenerate the SQL (capped at 2 attempts) when execution fails, instead of
just crashing.

The database schema (table + column names) is read once via
`information_schema.columns` and cached in memory (see `db.py`) so it isn't
re-fetched on every single question.

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

The assistant will show you the generated SQL (for transparency) followed
by a plain-English answer.

## Safety notes for this MVP

- `db.py` only allows statements starting with `SELECT`/`WITH` — this stops
  the LLM (or a malicious prompt) from running `DELETE`/`DROP`/etc.
- For real production use, also connect with a **read-only database role**
  as a second line of defense, in addition to the query-prefix check above.
- Query results are capped at 50 rows before being summarized, to keep
  LLM costs and latency predictable.

## Next steps (ideas to extend this MVP)

- **Streaming**: stream the SQL/answer tokens to Chainlit as they're
  generated, instead of waiting for the full response.
- **Multi-table joins**: pass foreign key relationships (from
  `information_schema.table_constraints`) into the schema text so Claude
  can write accurate JOINs.
- **Charts**: pass `sql_result` to a plotting library and render a chart
  as a Chainlit element alongside the text answer.
- **Conversation memory**: let follow-up questions ("now break that down
  by region") reference the previous query/result.
- **Read-only DB role**: create a dedicated Postgres user with only
  `SELECT` grants and point `APP_DB_URL` at it.
