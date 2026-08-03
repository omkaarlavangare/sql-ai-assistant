"""
app.py - the SQL Insight Assistant.

Flow for every user question (implemented as a LangGraph graph below):

    Understand user question --> write_sql --> execute_sql --(error, < 2 attempts) --> write_sql with error feedback [retry loop]
                                    |
                       (success, or out of retries)--> generate_insight --> END

Why LangGraph instead of a simple linear chain?
A plain "prompt -> SQL -> run -> prompt" pipeline breaks the moment the LLM
writes SQL with a typo or references a column that doesn't exist - which
happens often in practice. LangGraph lets us express that as an explicit
state machine: on failure, loop back to `write_sql` with the error message
attached, so the LLM can self-correct, instead of just crashing.
"""

import json                                          # parse the structured visualization spec
import os                                            # read env vars
import re                                            # strip markdown fences from LLM SQL output
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, TypedDict, cast             # type hints for the graph state

import chainlit as cl                                  # chat UI
import pandas as pd                                    # DataFrame support for visualization and analysis
import plotly.graph_objects as go                     # interactive charts for Chainlit
from dotenv import load_dotenv                          # loads ANTHROPIC_API_KEY / APP_DB_URL from .env
from langchain_anthropic import ChatAnthropic            # Claude wrapper
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END              # the state-machine builder

from db import get_engine, load_schema, run_sql_dataframe  # our database helpers

load_dotenv()  # populate os.environ from a local .env file, if present

# ---------------------------------------------------------------------------
# Shared objects — created once at import time, reused for every user message
# ---------------------------------------------------------------------------
engine = None  # lazily created so helpers can be imported and tested safely


def get_app_engine():
    """Create the shared database engine the first time it is needed."""
    global engine
    if engine is None:
        engine = get_engine()
    return engine


# Initialize the LLM  
llm = ChatAnthropic(
    model_name=os.environ.get("CLAUDE_MODEL", "claude-sonnet-5"),  # default to claude-sonnet-5 if not set
    timeout=60,                                                              # seconds to wait before giving up
    max_retries=2,                                                           # retry up to 2 times on network errors or timeouts
    stop=["END_OF_RESPONSE"],                                              # optional stop sequence to prevent the model from generating beyond the expected output
    streaming=True,                                                        # enables token-by-token output via .astream()
)


def _normalize_text_content(content: object) -> str:
    """
    Convert Anthropic/LangChain content blocks into plain text.

    Claude may return multiple block types in one response - e.g. a
    "thinking" block (internal reasoning) followed by a "text" block (the
    actual answer). We only want the text blocks; thinking blocks must be
    skipped entirely, otherwise their raw dict/signature payload leaks into
    whatever we're building (SQL, JSON, the final answer, etc).
    """
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

# Keywords that indicate the user is asking for a write/destructive operation
# rather than a read/insight question. Checked against the raw user message
# (not generated SQL) so we can short-circuit before calling the LLM at all.
_DESTRUCTIVE_KEYWORDS = re.compile(
    r"\b(delete|drop|truncate|remove|destroy|update|insert|alter)\b",
    flags=re.IGNORECASE,
)


def _is_destructive_request(text: str) -> bool:
    """Cheap keyword check to block obviously destructive requests early."""
    return bool(_DESTRUCTIVE_KEYWORDS.search(text))


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    """
    The data that flows between nodes in the graph. Each node reads what it
    needs from this dict and returns a partial dict of updates - LangGraph
    merges those updates into the running state automatically.
    """
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


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------
def write_sql(state: GraphState) -> dict:
    """Ask Claude to turn the question (+ schema, + any past error) into SQL."""
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
        # Feeding the previous failed query + its error back to the LLM is
        # what lets it self-correct on the retry instead of repeating the
        # same mistake.
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

    # Claude sometimes wraps SQL in ```sql ... ``` even when told not to;
    # this strips that formatting and gets clean SQL.
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE).strip()

    return {"sql_query": sql, "attempts": state.get("attempts", 0) + 1}


def execute_sql(state: GraphState) -> dict:
    """Run the generated SQL against Postgres; capture success or error."""
    try:
        df, result = run_sql_dataframe(get_app_engine(), state["sql_query"])
        return {"sql_result": result, "dataframe": df, "error": ""}
    except Exception as exc:                        # any DB error should trigger our retry path
        return {"error": str(exc)}


def _json_safe(value: Any) -> Any:
    """Convert database-specific values into JSON-friendly Python values."""
    if isinstance(value, Decimal):
        exponent = value.as_tuple().exponent
        if isinstance(exponent, int) and exponent >= 0:
            return int(value)
        return float(value)

    if isinstance(value, (datetime, date, time)):
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

    # Truncate long category labels so they don't overflow the axis
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
    """Ask the LLM for a structured visualization specification."""
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
    return {"viz_spec": validated}


def route_after_execute(state: GraphState) -> str:
    """
    Conditional edge: decide whether to retry SQL generation or move on.
    Capped at 2 attempts total so a persistently-wrong question can't loop
    forever and rack up LLM calls.
    """
    if state.get("error") and state["attempts"] < 2:
        return "retry"
    return "continue"


def _build_insight_prompt(state: GraphState) -> str:
    """Shared prompt builder, used by both the graph node and the streaming call in on_message."""
    return (
        "You are a data analyst. Using the SQL query and its result below, "
        "answer the user's original question in clear, plain English. "
        "Reference concrete numbers from the result where relevant.\n\n"
        f"Question: {state['question']}\n"
        f"SQL query: {state['sql_query']}\n"
        f"Result:\n{state['sql_result']}"
    )


def generate_insight(state: GraphState) -> dict:
    """
    Turn the raw query result into a plain-English answer.

    On success, the actual LLM call is deferred to on_message so the
    response can be streamed token-by-token into the Chainlit UI. This node
    only handles the error case (no streaming needed - it's a fixed string).
    """
    if state.get("error"):
        # We exhausted retries - be transparent about the failure rather
        # than making something up.
        return {
            "final_answer": (
                "I couldn't run a working SQL query for that question. "
                f"Last error: {state['error']}"
            )
        }

    return {"final_answer": ""}  # signals on_message to stream the real answer


# ---------------------------------------------------------------------------
# Build and compile the graph once at import time
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Chainlit UI hooks
# ---------------------------------------------------------------------------

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
    """Runs once when a user opens the chat: read + cache the DB schema."""
    await cl.Message(content="Connecting to the database and reading its schema...").send()
    schema = load_schema(get_app_engine())  # first call hits the DB; later calls reuse the cache in db.py
    cl.user_session.set("schema", schema)  # stash per-session so on_message can read it back
    cl.user_session.set("chat_history", []) # conversational memory, reset per session
    await cl.Message(
        content="Ready! Ask me an insight question, e.g. **\"Top 5 products by sales\"**."
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Runs on every user message: execute the LangGraph workflow end-to-end."""
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
    }

    # ainvoke runs the whole graph (including any retry loop) and returns
    # the final merged state.
    final_state = cast(GraphState, await graph.ainvoke(initial_state))

    # Show the SQL that was actually run - transparency matters for a tool
    # that's making database queries on the user's behalf.
    await cl.Message(content=f"```sql\n{final_state['sql_query']}\n```").send()

    dataframe = final_state.get("dataframe")
    if dataframe is not None:
        fig = _build_plotly_chart(dataframe, final_state.get("viz_spec", {}))
        await cl.Message(
            content="Here is the generated chart:",
            elements=[cl.Plotly(name="chart", figure=fig, display="inline")],
        ).send()

    if final_state.get("final_answer"):
            # Error path: generate_insight already produced the full text, no streaming needed.
            await cl.Message(content=final_state["final_answer"]).send()
    else:
        # Success path: stream the answer token-by-token for better perceived responsiveness.
        streamed_msg = cl.Message(content="")
        await streamed_msg.send()

        insight_prompt = _build_insight_prompt(final_state)
        async for chunk in llm.astream([HumanMessage(content=insight_prompt)]):
            token = _normalize_text_content(chunk.content)
            if token:
                await streamed_msg.stream_token(token)

        await streamed_msg.update()
        final_state["final_answer"] = streamed_msg.content

    # Save this turn into conversational memory, capped to the last 5 turns
    chat_history.append({
        "question": message.content,
        "answer": final_state["final_answer"],
    })
    cl.user_session.set("chat_history", chat_history[-5:])

