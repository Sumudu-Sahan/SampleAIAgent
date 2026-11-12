# Family Food Ordering Agent: Agent Manager LLM Provider + MCP

A platform-hosted **Chat Agent** for WSO2 Agent Manager:

- **LLM:** the OpenAI **LLM Service Provider** registered in Agent Manager. Calls
  go through the AI Gateway, which holds the real OpenAI key. The agent never
  sees that key.
- **Tools:** the four API Manager MCP servers (Restaurant, Bank Promotions,
  Wallet, Delivery), attached to the agent as MCP proxies.
- **Runtime:** a LangGraph ReAct agent that discovers MCP tools at startup and
  keeps per-session memory.

```
User -> POST /chat -> Agent (LangGraph ReAct)
                       |-- LLM_PROVIDER_URL (API-Key header) --> AM AI Gateway --> OpenAI
                       '-- AGENT_MCP_<n>_URL (X-API-Key)     --> AM MCP proxy  --> APIM MCP server --> mock backend
```

## Agent Manager settings

| Setting | Value |
|---|---|
| Agent type | Chat Agent |
| Language | Python 3.11 |
| Project path | this directory (the one containing `main.py`) |
| Start command | `python main.py` |
| Port | `8000` |
| Chat contract | `POST /chat` with `{"message": "...", "session_id": "..."}`, which returns `{"response": "..."}` |

### 1. Bind the LLM Service Provider

Open the agent, go to **Configure → LLM configuration**, and add your OpenAI
provider. Set the environment variable names to:

| Key | Environment variable name |
|---|---|
| url | `LLM_PROVIDER_URL` |
| apikey | `LLM_PROVIDER_KEY` |

You can see the generated names on the provider's **Environment Variables
References** section. The agent uses `LLM_PROVIDER_URL` as the OpenAI base URL
and sends `LLM_PROVIDER_KEY` in the `API-Key` header, which is the gateway
default. If you changed the provider's API-key header, set
`LLM_PROVIDER_AUTH_HEADER` to match.

### 2. Attach the four MCP servers

Go to **Configure → Tool Configurations** and attach restaurant, promotions,
wallet, and delivery. Name each server's variables `AGENT_MCP_<n>_URL` and,
when the proxy uses API-key security, `AGENT_MCP_<n>_API_KEY`
(n = 1..4, in any order):

```
AGENT_MCP_1_URL  AGENT_MCP_1_API_KEY
AGENT_MCP_2_URL  AGENT_MCP_2_API_KEY
...
```

The agent discovers every `AGENT_MCP_<n>_URL` automatically. If a matching
`_API_KEY` exists, it sends that key in the `X-API-Key` header, which is the
MCP proxy default. If you used a different prefix, for example `NEW_AGENT_MCP`,
set `MCP_ENV_PREFIX`. If one server is unreachable, the agent logs it, skips it,
and starts with the remaining tools.

### 3. Optional variables

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_MODEL` | `gpt-4o-mini` | Model name sent to the provider. It must be allowed by the provider. |
| `LLM_PROVIDER_AUTH_HEADER` | `API-Key` | API-key header of the LLM proxy |
| `MCP_ENV_PREFIX` | `AGENT_MCP` | Prefix of the MCP URL and key variables |
| `MCP_API_KEY_HEADER` | `X-API-Key` | API-key header of the MCP proxies |
| `MCP_PERSISTENT_SESSIONS` | `true` | Keep one MCP session per server |
| `MAX_HISTORY_MESSAGES` | `40` | Number of recent messages sent to the LLM each turn |
| `OPENAI_API_KEY` | none | Local fallback, used only when `LLM_PROVIDER_URL` is not set |

## Run locally

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Option A: through the Agent Manager gateway (copy the values from the console)
export LLM_PROVIDER_URL="<gateway LLM URL>"
export LLM_PROVIDER_KEY="<provider key>"
# Option B: directly to OpenAI
# export OPENAI_API_KEY="..."

export AGENT_MCP_1_URL="https://localhost:8243/<restaurant-mcp-context>/mcp"
export AGENT_MCP_2_URL="..."   # promotions
export AGENT_MCP_3_URL="..."   # wallet
export AGENT_MCP_4_URL="..."   # delivery
python main.py
```

Test it:

```bash
curl -s localhost:8000/health

curl -s -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"message":"Find a burger meal for three under $30. Check card offers, but do not order yet.","session_id":"demo-1"}'

curl -s -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"message":"Yes, place that order.","session_id":"demo-1"}'
```

`/health` shows which LLM path is active (`agent-manager-llm-provider` or
`openai-direct`) and lists the loaded MCP tool names.

## Notes

- Tool names and schemas come from the MCP servers at startup. Nothing is
  hardcoded. Tool names must be unique across servers.
- The agent creates an order (`createSimulatedOrder`) only after explicit user
  approval. The system prompt enforces this, and the session memory keeps the
  proposed order for the confirmation turn.
- Memory is held in process memory and is lost on restart. It is meant for
  demos, not production.
- `wrapt<2` is pinned because Agent Manager's OpenTelemetry auto-instrumentation
  breaks on wrapt 2.x.
- Do not commit keys. In Agent Manager, the platform manages the
  `LLM_PROVIDER_*` and MCP key variables.
