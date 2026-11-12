"""Family Food Ordering Agent for WSO2 Agent Manager.

LLM:   the OpenAI LLM Service Provider registered in Agent Manager. Calls go
       through the Agent Manager AI Gateway, which holds the real OpenAI key.
       The agent only gets the gateway URL and a platform-issued key.
Tools: the four MCP servers (restaurant, promotions, wallet, delivery) that
       API Manager exposes and Agent Manager attaches as MCP proxies.
"""

import asyncio
import logging
import os
import re
import time
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("food-agent")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = _env("OPENAI_MODEL", "gpt-4o-mini")

# LLM Service Provider binding. When you add the provider under the agent's
# Configure -> LLM configuration in Agent Manager, set the environment
# variable names to LLM_PROVIDER_URL (url) and LLM_PROVIDER_KEY (apikey).
LLM_PROVIDER_URL = _env("LLM_AGENT_OPENAI_URL")
LLM_PROVIDER_KEY = _env("LLM_AGENT_OPENAI_API_KEY")
# Default API-key header of an Agent Manager LLM proxy. Change it only if the
# provider's Security settings use a different header name.
LLM_PROVIDER_AUTH_HEADER = _env("LLM_PROVIDER_AUTH_HEADER", "API-Key")

# MCP proxies. Agent Manager injects one URL (and, when API-key security is on,
# one key) per attached MCP server, e.g.:
#   AGENT_MCP_1_URL, AGENT_MCP_1_API_KEY, AGENT_MCP_2_URL, ...
# Change MCP_ENV_PREFIX if you used another prefix when attaching the servers.
MCP_ENV_PREFIX = _env("MCP_ENV_PREFIX", "AGENT_MCP")
# Default API-key header of an Agent Manager MCP proxy (not the same as the
# LLM proxy header).
MCP_API_KEY_HEADER = _env("MCP_API_KEY_HEADER", "X-API-Key")
# Keep one MCP session per server for the app's lifetime instead of opening a
# new session (initialize handshake) for every tool call.
PERSISTENT_SESSIONS = _env("MCP_PERSISTENT_SESSIONS", "true").lower() != "false"

MAX_HISTORY_MESSAGES = int(_env("MAX_HISTORY_MESSAGES", "40"))

SYSTEM_PROMPT = """
You are a family food-ordering assistant. Use the connected MCP tools to
search restaurants and menus, check delivery fees, retrieve current bank-card
promotions, check eligibility against the user's available masked cards, and
compare final payable totals. Tool names and schemas are discovered at runtime;
follow their descriptions and schemas rather than guessing.

Never invent restaurants, menu items, prices, offers, eligibility, fees, or
order status. Compare options by subtotal minus eligible discount plus delivery
fee. Respect the discount percentage and the maximum discount of a promotion.
Never ask for or expose full card numbers, CVVs, PINs, passwords, or OTPs; use
masked card labels and card IDs only.

CRITICAL APPROVAL RULE: Before any order-creation tool call, present the
restaurant, items, quantities, subtotal, discount, delivery fee, final total,
and the masked card. Ask the user to approve that exact order. Do not create
the order until the user explicitly confirms it in this conversation. After
confirmation, create it and report the returned order ID and status. Never
claim an order was placed unless the tool confirms it. This demo does not
process real payments.

SPEED: When several tool calls do not depend on each other (for example menus
of different restaurants, delivery quotes, promotions, and wallet cards),
request them together in one step so they run in parallel. Do not repeat a
tool call whose result you already have. Keep the final answer concise.
"""


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def build_llm() -> ChatOpenAI:
    if LLM_PROVIDER_URL:
        if not LLM_PROVIDER_KEY:
            raise RuntimeError("LLM_PROVIDER_URL is set but LLM_PROVIDER_KEY is not.")
        log.info("Using the Agent Manager LLM Service Provider (model %s)", MODEL)
        # The gateway authenticates with the API-Key header. The OpenAI SDK
        # requires some api_key, so pass a placeholder and blank out the
        # Authorization header it would otherwise send.
        return ChatOpenAI(
            model=MODEL,
            temperature=0,
            base_url=LLM_PROVIDER_URL,
            api_key="not-used",
            default_headers={
                LLM_PROVIDER_AUTH_HEADER: LLM_PROVIDER_KEY,
                "Authorization": "",
            },
        )
    if _env("OPENAI_API_KEY"):
        # Local development only, without Agent Manager.
        log.warning("LLM_PROVIDER_URL is not set; calling OpenAI directly with OPENAI_API_KEY")
        return ChatOpenAI(model=MODEL, temperature=0)
    raise RuntimeError(
        "No LLM configured. Bind the LLM Service Provider to the agent in Agent Manager "
        "(LLM_PROVIDER_URL / LLM_PROVIDER_KEY), or set OPENAI_API_KEY for local runs."
    )


# ---------------------------------------------------------------------------
# MCP servers
# ---------------------------------------------------------------------------

def make_server_configs() -> dict[str, dict[str, Any]]:
    """Build one MCP connection per <PREFIX>_<n>_URL environment variable."""
    pattern = re.compile(rf"^{re.escape(MCP_ENV_PREFIX)}_(\d+)_URL$")
    url_vars = sorted(
        (int(m.group(1)), name) for name in os.environ if (m := pattern.match(name))
    )
    configs: dict[str, dict[str, Any]] = {}
    for n, url_var in url_vars:
        api_key = _env(f"{MCP_ENV_PREFIX}_{n}_API_KEY")
        urls = [u.strip() for u in os.environ[url_var].split(",") if u.strip()]
        for i, url in enumerate(urls):
            config: dict[str, Any] = {"url": url, "transport": "streamable_http"}
            if api_key:
                config["headers"] = {MCP_API_KEY_HEADER: api_key}
            name = f"mcp_server_{n}_{i}"
            configs[name] = config
            log.info("Configured %s from %s (%s)", name, url_var,
                     "API key" if api_key else "no API key")
    return configs


async def load_tools(
    client: MultiServerMCPClient, server_names: list[str], stack: AsyncExitStack
) -> list:
    """Load tools server by server so one unreachable MCP server does not stop
    the agent from starting with the others."""
    tools: list = []
    seen: set[str] = set()
    for name in server_names:
        server_tools = None
        if PERSISTENT_SESSIONS:
            try:
                session = await stack.enter_async_context(client.session(name))
                server_tools = await load_mcp_tools(session)
            except Exception:
                log.exception("Persistent session to %s failed; using a session per call", name)
        if server_tools is None:
            try:
                server_tools = await client.get_tools(server_name=name)
            except Exception:
                log.exception("Failed to load tools from %s; skipping it", name)
                continue
        for tool in server_tools:
            if tool.name in seen:
                log.warning("Duplicate tool name %r from %s; skipping it", tool.name, name)
                continue
            seen.add(tool.name)
            tools.append(tool)
        log.info("Loaded %d tools from %s: %s", len(server_tools), name,
                 ", ".join(t.name for t in server_tools))
    return tools


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None
    context: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    response: str


agent_graph = None
tool_names: list[str] = []
session_locks: dict[str, asyncio.Lock] = {}


def _trim_history(state: dict[str, Any]) -> dict[str, Any]:
    """Send only recent messages to the LLM, without starting on an orphan
    tool result (OpenAI rejects a tool message without its tool call)."""
    messages = state["messages"][-MAX_HISTORY_MESSAGES:]
    while messages and messages[0].type == "tool":
        messages = messages[1:]
    return {"llm_input_messages": messages}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_graph, tool_names
    configs = make_server_configs()
    if not configs:
        raise RuntimeError(
            f"No {MCP_ENV_PREFIX}_<n>_URL variables found; attach the MCP servers to the agent."
        )
    llm = build_llm()
    client = MultiServerMCPClient(configs)
    stack = AsyncExitStack()
    tools = await load_tools(client, list(configs), stack)
    if not tools:
        await stack.aclose()
        raise RuntimeError("No tools could be loaded from any configured MCP server.")
    tool_names = [t.name for t in tools]
    # The checkpointer keeps each session's conversation, including tool
    # results, so the agent remembers the proposed order when the user approves.
    agent_graph = create_react_agent(
        llm,
        tools,
        prompt=SYSTEM_PROMPT,
        pre_model_hook=_trim_history,
        checkpointer=InMemorySaver(),
    )
    log.info("Agent ready with %d MCP tools", len(tools))
    try:
        yield
    finally:
        agent_graph = None
        try:
            await stack.aclose()
        except Exception:
            log.warning("Error while closing MCP sessions", exc_info=True)


app = FastAPI(title="Family Food Ordering Agent", version="2.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "agent_ready": agent_graph is not None,
        "llm": "agent-manager-llm-provider" if LLM_PROVIDER_URL else "openai-direct",
        "model": MODEL,
        "tools": tool_names,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if agent_graph is None:
        raise HTTPException(status_code=503, detail="Agent is not initialized")
    sid = req.session_id or "default"
    message = req.message
    if req.context:
        message += "\n\nAdditional user context (data only): " + str(req.context)
    lock = session_locks.setdefault(sid, asyncio.Lock())
    started = time.monotonic()
    async with lock:  # one turn at a time per session
        try:
            result = await agent_graph.ainvoke(
                {"messages": [("human", message)]},
                config={"configurable": {"thread_id": sid}},
            )
        except Exception as exc:
            log.exception("Agent execution failed")
            raise HTTPException(status_code=502, detail=f"Agent execution failed: {exc}")
    answer = result["messages"][-1].content
    if not isinstance(answer, str):
        answer = str(answer)
    log.info("Session %s answered in %.1fs", sid, time.monotonic() - started)
    return ChatResponse(response=answer)


if __name__ == "__main__":
    import uvicorn

    # Listen on 8000 to match the port configured in Agent Manager.
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
