import os

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_core.runnables import RunnableConfig

from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends.context_hub import ContextHubBackend

from agent.tools import TOOLS
from context import CONTEXT_HUB_REPO, get_prompt
from utils.streaming import iter_text
from utils.models import MODEL_CONFIG, model

# AGENTS.md is the agent's system prompt — pulled fresh from LangSmith
# Context Hub at module import.
# Seed source: utils/context_hub.py (`_SEED_AGENTS_MD`), pushed to Context Hub by
# `scripts/setup.py` (`push_agents_md()`). A prompt fix can be applied BOTH as a
# PR to that seed AND to the live Context Hub.
SYSTEM_PROMPT = get_prompt()


def _model_id() -> str:
    """The model actually passed to init_chat_model, for run metadata."""
    # Never hardcode a second default here — it drifts from the real model call
    # and the traced metadata then contradicts the LLM span.
    return MODEL_CONFIG["model"]


# Bounded cardinality: `topic` metadata is charted per-topic, so it is always one
# of these labels, never raw user question text.
_TOPICS = (
    "langgraph",
    "langsmith",
    "middleware",
    "tracing",
    "installation",
    "deployment",
    "evaluation",
)
_OFF_DOMAIN = "off_domain"


def _resolve_topic(question: str) -> str:
    """Resolve the question to the concept the agent looks up, or off_domain."""
    text = question.lower()
    return next((topic for topic in _TOPICS if topic in text), _OFF_DOMAIN)


def _run_metadata(thread_id: str | None = None, topic: str | None = None) -> dict:
    metadata = {
        "demo": "true",
        "demo_type": "chat-lc-lite",
        "model": _model_id(),
        "environment": os.getenv("CHAT_LANGCHAIN_LITE_ENV", "development"),
    }
    if thread_id:
        metadata["thread_id"] = thread_id
    if topic:
        metadata["topic"] = topic
    revision_id = os.getenv("LANGSMITH_REVISION_ID") or os.getenv("LANGCHAIN_REVISION_ID")
    if revision_id:
        metadata["revision_id"] = revision_id
    # Only set user_id when a real identifier exists — an empty string looks
    # populated to filters but partitions nothing.
    user_id = os.getenv("CHAT_LANGCHAIN_LITE_USER_ID")
    if user_id:
        metadata["user_id"] = user_id
    return metadata


# The Context Hub-backed filesystem holds the agent's OWN context (AGENTS.md,
# playbooks) — it is a read-only reference, NOT a user-delivery channel.
_READONLY_FS_TOOLS = {"ls", "read_file", "glob", "grep"}


def _readonly_context_hub_fs() -> FilesystemMiddleware:
    fs = FilesystemMiddleware(backend=ContextHubBackend(CONTEXT_HUB_REPO))
    fs.tools = [t for t in fs.tools if t.name in _READONLY_FS_TOOLS]
    return fs


def build_agent():
    agent = create_agent(
        # temperature=0 for deterministic, reproducible demo behavior — the
        # intentional bugs (tone, scope, truncation) come from the prompt and
        # max_tokens, not sampling, so pinning temperature keeps traces consistent.
        model=model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[_readonly_context_hub_fs()],
    )
    # langgraph.json serves this factory directly, so the LangGraph Platform path
    # never sees _config() — bind the same metadata to the graph so both entry
    # paths record the same keys.
    return agent.with_config(metadata=_run_metadata())


def _config(thread_id: str | None = None, topic: str | None = None) -> RunnableConfig:
    return RunnableConfig(
        run_name="chat-lc-lite-demo",
        metadata=_run_metadata(thread_id, topic),
        tags=["engine-demo", CONTEXT_HUB_REPO],
    )


def _user_msg(question: str) -> dict:
    return {"messages": [{"role": "user", "content": question}]}


def invoke_agent(question: str, thread_id: str | None = None) -> dict:
    """Run the agent once. Returns {output, tools_called, messages}."""
    result = build_agent().invoke(
        _user_msg(question), _config(thread_id, _resolve_topic(question))
    )
    output = next(
        (m.content for m in reversed(result["messages"])
         if isinstance(getattr(m, "content", None), str) and m.content),
        "",
    )
    tools_called = [m.name for m in result["messages"] if isinstance(m, ToolMessage)]
    return {"output": output, "tools_called": tools_called, "messages": result["messages"]}


def stream_agent(question: str, thread_id: str | None = None):
    """Stream the agent's response text as it's generated."""
    for chunk, _meta in build_agent().stream(
        _user_msg(question),
        _config(thread_id, _resolve_topic(question)),
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessageChunk):
            yield from iter_text(chunk)
