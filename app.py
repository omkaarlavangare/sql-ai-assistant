"""SQL Insight Assistant app with a LangGraph workflow."""

import json  # Parse the visualization spec.
import logging  # Log workflow progress and evaluation events.
import os  # Read environment variables.
import re  # Strip markdown fences from model SQL output.
import time  # Measure request duration.
import uuid  # Generate a stable run id for each interaction.
from datetime import date, datetime, time as dt_time
from decimal import Decimal
from pathlib import Path
from typing import Any, TypedDict, cast  # Type hints for the graph state.

import chainlit as cl  # Chat UI.
import pandas as pd  # DataFrame support for analysis and charts.
import plotly.graph_objects as go  # Interactive charts for Chainlit.
from dotenv import load_dotenv  # Load values from a local .env file when present.
from langchain_anthropic import ChatAnthropic  # Claude wrapper.
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END  # State-machine builder.

from db import get_engine, load_schema, run_sql_dataframe  # Database helpers.

load_dotenv()

logger = logging.getLogger("sql_insight_assistant")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

LOG_PATH: Path = Path(
    os.environ.get(
        "ASSISTANT_LOG_PATH",
        str(Path(__file__).resolve().parent / "logs" / "assistant_runs.jsonl"),
    )
)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Shared objects reused across requests.
engine = None  # Created lazily so helpers can be imported safely.


def get_app_engine():
    """Create the shared database engine the first time it is needed."""
    global engine
    if engine is None:
        engine = get_engine()
    return engine


# Initialize the LLM.
llm = ChatAnthropic(
    model_name=os.environ.get("CLAUDE_MODEL", "claude-sonnet-5"),
    timeout=60,
    max_retries=2,
    stop=["END_OF_RESPONSE"],
    streaming=True,
)


def _normalize_text_content(content: object) -> str:
    """Convert model content blocks into plain text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                block_type = item.get("type")
                if block_type in ("thinking", "redacted_thinking"):
                    continue  # skip internal reasoning blocks entirely
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                block_type = getattr(item, "type", None)
                if block_type in ("thinking", "redacted_thinking"):
                    continue
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    return str(content)

# Reject obvious destructive requests before calling the model.
_DESTRUCTIVE_KEYWORDS = re.compile(
    r"\b(delete|drop|truncate|remove|destroy|update|insert|alter)\b",
    flags=re.IGNORECASE,
)


def _is_destructive_request(text: str) -> bool:
    """Cheap keyword check to block obviously destructive requests early."""
    return bool(_DESTRUCTIVE_KEYWORDS.search(text))


def _utc_timestamp() -> str:
    """Return an ISO timestamp for logging and persistence."""
    return datetime.utcnow().isoformat() + "Z"


def _append_jsonl_record(record: dict[str, Any]) -> None:
    """Append a structured record to a JSONL log file."""
    try:
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str, ensure_ascii=False))
            handle.write("\n")
    except Exception as exc:
        logger.exception("Failed to write assistant log record: %s", exc)


def _log_error_step(run_id: str, step_name: str, duration_ms: int, error: str) -> None:
    """Persist a single error log record for a failed workflow step."""
    record = {
        "record_type": "step",
        "run_id": run_id,
        "timestamp": _utc_timestamp(),
        "schema_version": "1.0",
        "step": step_name,
        "status": "error",
        "duration_ms": duration_ms,
        "error": error,
        "payload": {},
    }
    _append_jsonl_record(record)
    logger.error("workflow_step_failed step=%s run_id=%s error=%s", step_name, run_id, error)


def _log_run_summary(state: Any, duration_ms: int, chart_created: bool, insight_duration_ms: int | None = None) -> None:
    """Persist exactly one structured summary record for the full interaction."""
    dataframe = state.get("dataframe")
    row_count = len(dataframe) if dataframe is not None else 0
    rows = []
    if dataframe is not None:
        rows = [_json_safe(row) for row in dataframe.head(50).to_dict(orient="records")]

    step_timings: dict[str, int] = {}
    for step_result in state.get("step_results", []) or []:
        if not isinstance(step_result, dict):
            continue
        step_name = step_result.get("step")
        step_duration = step_result.get("duration_ms")
        if isinstance(step_name, str) and isinstance(step_duration, (int, float)):
            step_timings[step_name] = int(step_duration)

    if insight_duration_ms is not None:
        step_timings["generate_insight"] = int(insight_duration_ms)

    error_value = state.get("error") or None
    status = "error" if bool(error_value) and row_count == 0 else "success"
    record = {
        "record_type": "run_summary",
        "run_id": state.get("run_id"),
        "timestamp": _utc_timestamp(),
        "schema_version": "1.0",
        "status": status,
        "duration_ms": duration_ms,
        "error": error_value,
        "payload": {
            "question": state.get("question", ""),
            "sql_query": state.get("sql_query", ""),
            "attempts": state.get("attempts", 0),
            "row_count": row_count,
            "rows": rows,
            "viz_spec": state.get("viz_spec", {}),
            "chart_created": chart_created,
            "generated_response": state.get("final_answer", ""),
            "step_timings": step_timings,
        },
    }
    _append_jsonl_record(record)
    logger.info("persisted_run_summary run_id=%s status=%s", record["run_id"], record["status"])


# Graph state shared across nodes.
class GraphState(TypedDict):
    """State passed between graph nodes."""
    question: str      # the user's original natural-language question
    chat_history: list[dict[str, str]]  # the conversation history
    schema: str         # cached DB schema text, used as LLM context
    sql_query: str       # the SQL the LLM most recently generated
    sql_result: str       # text table of query results (empty if it errored)
    dataframe: pd.DataFrame | None  # DataFrame used for visualization and analysis
    viz_spec: dict[str, Any]       # structured chart specification returned by the LLM
    error: str              # last execution error, or "" if none
    attempts: int             # how many times we've tried to write working SQL
    final_answer: str          # the natural-language answer shown to the user
    run_id: str              # unique id for logging and evaluation
    started_at: str          # timestamp when the request started
    step_results: list[dict[str, Any]]  # workflow steps captured for audit/evaluation
    row_count: int           # number of rows returned from the executed query
    chart_created: bool      # whether a Plotly chart was rendered


# Graph nodes.
def write_sql(state: GraphState) -> dict:
    """Turn the question and schema into SQL."""
    start_time = time.perf_counter()
    try:
        history_context = ""
        chat_history = state.get("chat_history") or []
        if chat_history:
            turns = "\n".join(
                f"Q: {turn['question']}\nA: {turn['answer']}" for turn in chat_history
            )
            history_context = (
                "\n\nHere is the recent conversation for context. The user's new "
                "question may refer back to it (e.g. 'now filter by...', 'summarize "
                "the last messages'):\n"
                f"{turns}"
            )

        error_context = ""
        if state.get("error"):
            # Reuse the failed query and error to help the retry recover.
            error_context = (
                f"\n\nYour previous attempt failed.\n"
                f"Previous SQL: {state['sql_query']}\n"
                f"Error: {state['error']}\n"
                f"Please write a corrected query."
            )

        system_prompt = (
            "You are a PostgreSQL expert. Using ONLY the tables/columns in the "
            "schema below, write ONE valid PostgreSQL SELECT query that answers "
            "the user's question. Use the schema-qualified table names exactly as "
            "shown. Reply with the raw SQL only - no markdown code fences, no "
            "explanation.\n\n"
            f"Database schema:\n{state['schema']}"
            f"{history_context}"
            f"{error_context}"
        )

        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=state["question"])]
        )
        content = _normalize_text_content(response.content)

        # Strip any markdown fences from the generated SQL.
        sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE).strip()
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        state.setdefault("step_results", []).append({"step": "write_sql", "duration_ms": duration_ms})
        return {"sql_query": sql, "attempts": state.get("attempts", 0) + 1}
    except Exception as exc:
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        state.setdefault("step_results", []).append({"step": "write_sql", "duration_ms": duration_ms})
        _log_error_step(state["run_id"], "write_sql", duration_ms, str(exc))
        raise


def execute_sql(state: GraphState) -> dict:
    """Run the generated SQL and capture the result or error."""
    start_time = time.perf_counter()
    try:
        df, result = run_sql_dataframe(get_app_engine(), state["sql_query"])
        row_count = len(df)
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        state.setdefault("step_results", []).append({"step": "execute_sql", "duration_ms": duration_ms})
        return {"sql_result": result, "dataframe": df, "error": "", "row_count": row_count}
    except Exception as exc:  # Any database error should trigger the retry path.
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        state.setdefault("step_results", []).append({"step": "execute_sql", "duration_ms": duration_ms})
        _log_error_step(state["run_id"], "execute_sql", duration_ms, str(exc))
        return {"error": str(exc)}


def _json_safe(value: Any) -> Any:
    """Convert database-specific values into JSON-friendly Python values."""
    if isinstance(value, Decimal):
        exponent = value.as_tuple().exponent
        if isinstance(exponent, int) and exponent >= 0:
            return int(value)
        return float(value)

    if isinstance(value, (datetime, date, dt_time)):
        return value.isoformat()

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if pd.isna(value):
        return None

    return value


def _describe_dataframe(df: pd.DataFrame) -> str:
    """Create a compact summary of the DataFrame for the LLM."""
    if df.empty:
        return "The dataframe is empty."

    sample_rows = [_json_safe(row) for row in df.head(5).to_dict(orient="records")]

    return json.dumps(
        {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "row_count": len(df),
            "sample_rows": sample_rows,
        },
        indent=2,
    )


def _validate_viz_spec(spec: Any, columns: list[str]) -> dict[str, Any]:
    """Validate and normalize the chart spec returned by the LLM."""
    if not isinstance(spec, dict):
        spec = {}

    chart_type = spec.get("chart_type", "bar")
    if chart_type not in {"bar", "line", "scatter"}:
        chart_type = "bar"

    x_column = spec.get("x_column", "")
    if x_column not in columns:
        x_column = columns[0] if columns else ""

    y_column = spec.get("y_column", "")
    if y_column not in columns:
        y_column = columns[-1] if len(columns) > 1 else (columns[0] if columns else "")

    title = spec.get("title") or f"{y_column} by {x_column}"

    return {
        "chart_type": chart_type,
        "x_column": x_column,
        "y_column": y_column,
        "title": title,
    }


def _build_plotly_chart(df: pd.DataFrame, spec: dict[str, Any]):
    """Render a Plotly chart from a validated visualization spec."""
    chart_type = spec.get("chart_type", "bar")
    x_column = spec.get("x_column", "")
    y_column = spec.get("y_column", "")

    # Keep long labels readable on the axis.
    x_values = df[x_column]
    if x_values.dtype == object:
        x_values = x_values.astype(str).apply(lambda s: s if len(s) <= 20 else s[:17] + "...")

    accent_color = "#6C5CE7"

    if chart_type == "line":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_values, y=df[y_column], mode="lines+markers", name=y_column,
            line=dict(color=accent_color, width=3),
            marker=dict(size=7, color=accent_color),
        ))
    elif chart_type == "scatter":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_values, y=df[y_column], mode="markers", name=y_column,
            marker=dict(size=10, color=accent_color, line=dict(width=1, color="white")),
        ))
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x_values, y=df[y_column], name=y_column,
            marker=dict(color=accent_color, line=dict(width=0)),
            text=df[y_column],
            texttemplate="%{text:,.0f}",
            textposition="outside",
        ))

    fig.update_layout(
        title=dict(
            text=spec.get("title", "Result"),
            font=dict(size=20, family="Arial, sans-serif", color="#2D3436"),
        ),
        xaxis_title=x_column.replace("_", " ").title(),
        yaxis_title=y_column.replace("_", " ").title(),
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=13, color="#2D3436"),
        margin=dict(l=60, r=30, t=70, b=100),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, tickangle=-30),
        yaxis=dict(showgrid=True, gridcolor="#EDEDED", zeroline=False),
        showlegend=False,
        bargap=0.3,
    )
    return fig


def generate_viz_spec(state: GraphState) -> dict:
    """Ask the model for a structured visualization specification."""
    start_time = time.perf_counter()
    try:
        df = state.get("dataframe")
        if df is None:
            return {"viz_spec": {"chart_type": "bar", "x_column": "", "y_column": "", "title": "Result"}}

        prompt = (
            "You are a helpful data assistant. Given the dataframe description below, "
            "choose the best simple chart for the results. Return ONLY valid JSON with this structure:\n"
            "{\n  \"chart_type\": \"bar|line|scatter\",\n  \"x_column\": \"column_name\",\n  \"y_column\": \"column_name\",\n  \"title\": \"short plain English title\"\n}\n"
            "Rules:\n- Use only columns present in the dataframe.\n- Keep the title short and easy to understand.\n- Prefer bar charts for comparisons.\n- Prefer line charts for trends over time.\n- Prefer scatter charts for relationships.\n\n"
            f"Dataframe description:\n{_describe_dataframe(df)}"
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        content = _normalize_text_content(response.content)

        try:
            raw_spec = json.loads(content)
        except Exception:
            raw_spec = {}

        validated = _validate_viz_spec(raw_spec, list(df.columns))
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        state.setdefault("step_results", []).append({"step": "generate_viz_spec", "duration_ms": duration_ms})
        return {"viz_spec": validated}
    except Exception as exc:
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        state.setdefault("step_results", []).append({"step": "generate_viz_spec", "duration_ms": duration_ms})
        _log_error_step(state["run_id"], "generate_viz_spec", duration_ms, str(exc))
        raise


def route_after_execute(state: GraphState) -> str:
    """Decide whether to retry SQL generation or continue."""
    if state.get("error") and state["attempts"] < 2:
        return "retry"
    return "continue"


def _build_insight_prompt(state: GraphState) -> str:
    """Build the prompt used to generate the final insight."""
    return (
        "You are a data analyst. Using the SQL query and its result below, "
        "answer the user's original question in clear, plain English. "
        "Reference concrete numbers from the result where relevant.\n\n"
        f"Question: {state['question']}\n"
        f"SQL query: {state['sql_query']}\n"
        f"Result:\n{state['sql_result']}"
    )


def generate_insight(state: GraphState) -> dict:
    """Turn the query result into a plain-English answer."""
    start_time = time.perf_counter()
    if state.get("error"):
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        state.setdefault("step_results", []).append({"step": "generate_insight", "duration_ms": duration_ms})
        _log_error_step(state["run_id"], "generate_insight", duration_ms, state.get("error", ""))
        return {
            "final_answer": (
                "I couldn't run a working SQL query for that question. "
                f"Last error: {state['error']}"
            )
        }

    duration_ms = int((time.perf_counter() - start_time) * 1000)
    state.setdefault("step_results", []).append({"step": "generate_insight", "duration_ms": duration_ms})
    return {"final_answer": ""}  # signals on_message to stream the real answer


# Build and compile the graph once at import time.
builder = StateGraph(GraphState)
builder.add_node("write_sql", write_sql)
builder.add_node("execute_sql", execute_sql)
builder.add_node("generate_viz_spec", generate_viz_spec)
builder.add_node("generate_insight", generate_insight)

builder.set_entry_point("write_sql")
builder.add_edge("write_sql", "execute_sql")
builder.add_conditional_edges(
    "execute_sql",
    route_after_execute,
    {"retry": "write_sql", "continue": "generate_viz_spec"},
)
builder.add_edge("generate_viz_spec", "generate_insight")
builder.add_edge("generate_insight", END)

graph = builder.compile()


# Chainlit UI hooks.

# To use these, remove the await cl.Messages from on_chat_start, as they 
# @cl.set_starters
# async def set_starters(user: cl.User | None = None, chat_profile: str | None = None):
#     """Suggested prompts shown as clickable buttons on the first screen."""
#     return [
#         cl.Starter(
#             label="Top 5 products by sales",
#             message="Top 5 products by sales",
#         ),
#         cl.Starter(
#             label="How is the business doing?",
#             message="How is the business doing?",
#         ),
#         cl.Starter(
#             label="Delete all tables",
#             message="Delete all tables",
#         ),
#     ]

@cl.on_chat_start
async def on_chat_start():
    """Initialize the session and load the database schema."""
    await cl.Message(content="Connecting to the database and reading its schema...").send()
    schema = load_schema(get_app_engine())  # first call hits the DB; later calls reuse the cache in db.py
    cl.user_session.set("schema", schema)  # stash per-session so on_message can read it back
    cl.user_session.set("chat_history", []) # conversational memory, reset per session
    await cl.Message(
        content="Ready! Ask me an insight question, e.g. **\"Top 5 products by sales\"**."
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Run the LangGraph workflow for each user message."""
    if _is_destructive_request(message.content):
        await cl.Message(
            content=(
                "🚫 This assistant is read-only and can't delete, modify, or "
                "insert data. I can only answer questions using SELECT queries."
            )
        ).send()
        return
    schema = cl.user_session.get("schema", "")
    schema_text = str(schema or "")
    chat_history = cl.user_session.get("chat_history") or []

    started_at = _utc_timestamp()
    initial_state: GraphState = {
        "question": message.content,
        "chat_history": chat_history,
        "schema": schema_text,
        "sql_query": "",
        "sql_result": "",
        "dataframe": None,
        "viz_spec": {},
        "error": "",
        "attempts": 0,
        "final_answer": "",
        "run_id": str(uuid.uuid4()),
        "started_at": started_at,
        "step_results": [],
        "row_count": 0,
        "chart_created": False,
    }

    start_time = time.perf_counter()

    # Run the graph and return the final merged state.
    final_state = cast(GraphState, await graph.ainvoke(initial_state))

    # Show the SQL that was actually run.
    await cl.Message(content=f"```sql\n{final_state['sql_query']}\n```").send()

    dataframe = final_state.get("dataframe")
    chart_created = False
    if dataframe is not None:
        fig = _build_plotly_chart(dataframe, final_state.get("viz_spec", {}))
        await cl.Message(
            content="Here is the generated chart:",
            elements=[cl.Plotly(name="chart", figure=fig, display="inline")],
        ).send()
        chart_created = True

    insight_duration_ms: int | None = None
    if final_state.get("final_answer"):
        # Use the final answer directly when the error path is taken.
        await cl.Message(content=final_state["final_answer"]).send()
    else:
        # Stream the answer token-by-token for a smoother experience.
        streamed_msg = cl.Message(content="")
        await streamed_msg.send()

        insight_start = time.perf_counter()
        try:
            insight_prompt = _build_insight_prompt(final_state)
            async for chunk in llm.astream([HumanMessage(content=insight_prompt)]):
                token = _normalize_text_content(chunk.content)
                if token:
                    await streamed_msg.stream_token(token)

            await streamed_msg.update()
            final_state["final_answer"] = streamed_msg.content
            insight_duration_ms = int((time.perf_counter() - insight_start) * 1000)
        except Exception as exc:
            insight_duration_ms = int((time.perf_counter() - insight_start) * 1000)
            final_state.setdefault("step_results", []).append({"step": "generate_insight", "duration_ms": insight_duration_ms})
            _log_error_step(final_state["run_id"], "generate_insight_stream", insight_duration_ms, str(exc))
            fallback_answer = f"I ran the query but couldn't generate the summary text: {exc}"
            final_state["final_answer"] = fallback_answer
            streamed_msg.content = fallback_answer
            await streamed_msg.update()

    final_state["chart_created"] = chart_created
    final_state["row_count"] = len(dataframe) if dataframe is not None else 0
    final_state["step_results"] = final_state.get("step_results", [])
    duration_ms = int((time.perf_counter() - start_time) * 1000)
    _log_run_summary(final_state, duration_ms, chart_created, insight_duration_ms)

    # Save this turn to conversational memory.
    chat_history.append({
        "question": message.content,
        "answer": final_state["final_answer"],
    })
    cl.user_session.set("chat_history", chat_history[-5:])

