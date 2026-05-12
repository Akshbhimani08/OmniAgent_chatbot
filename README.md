# OmniAgent — Multi-Agent RAG & MCP Chatbot

A production-grade, multi-tool AI chatbot powered by LangGraph, featuring RAG over PDFs, live web search, real-time stock prices, and remote MCP tool integration — all in a Streamlit UI with persistent conversation memory.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-green?logo=langchain)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/LLM-Groq-orange)](https://groq.com/)
[![FastMCP](https://img.shields.io/badge/MCP-FastMCP-purple)](https://github.com/jlowin/fastmcp)
[![FAISS](https://img.shields.io/badge/VectorDB-FAISS-blue)](https://faiss.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)

---

## Problem Statement

Building a truly useful AI assistant requires more than a single LLM — it needs the ability to search the web, read uploaded documents, fetch live financial data, and call external services, all while remembering context across conversations. OmniAgent solves this by wiring multiple tools into a single LangGraph agent with persistent SQLite-backed memory, accessible through a clean Streamlit chat interface.

---

## Architecture

```
User (Streamlit UI)
        │
        │  HumanMessage / AIMessage stream
        ▼
 LangGraph StateGraph  (chat_node ↔ tools node)
        │
        ├── DuckDuckGo Search      (real-time web)
        ├── Alpha Vantage          (live stock prices)
        ├── FAISS RAG Engine       (per-thread PDF Q&A)
        └── Remote MCP Client     (FastMCP over HTTP)
                │
                │  Streamable HTTP / SSE
                ▼
        Remote MCP Server(s)   (expense tracker tools, …)

 Persistence: AsyncSqliteSaver → chatbot_database.db
 Embeddings:  HuggingFace sentence-transformers/all-MiniLM-L6-v2
```

---

## Features

| Feature | Detail |
|---------|--------|
| 🔍 **Web Search** | DuckDuckGo live search via LangChain community tool |
| 📈 **Stock Prices** | Alpha Vantage `GLOBAL_QUOTE` endpoint by ticker symbol |
| 📄 **PDF RAG** | Per-thread FAISS index — upload any PDF and ask questions |
| 💸 **Expense Tracker** | Remote MCP server over Streamable HTTP |
| 🗂️ **Persistent Memory** | SQLite checkpointer — conversations survive restarts |
| 🖥️ **Streamlit UI** | Multi-chat sidebar, live token streaming, tool usage indicators |

---

## Tools

### Built-in Tools

| Tool | Source | Description |
|------|--------|-------------|
| `duckduckgo_search` | LangChain Community | Real-time web search |
| `get_stock_price` | Alpha Vantage API | Fetch latest stock quote by symbol |
| `rag_tool` | FAISS + HuggingFace | Answer questions from an uploaded PDF |

### MCP Tools (Remote)

| Tool | Server | Transport |
|------|--------|-----------|
| Expense Tracker tools | `splendid-gold-dingo.fastmcp.app` | Streamable HTTP |

### MCP Tools (Local — `mcp_server.py`)

---

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/Akshbhimani08/OmniAgent_chatbot.git
cd omniagent

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
HUGGINGFACEHUB_API_TOKEN=your_hf_token
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
```

### 3. (Optional) Run the Local MCP Server

```bash
python mcp_server.py
#Tools MCP server starts at http://localhost:8000/sse
```

### 4. Launch the Chatbot

```bash
streamlit run chatbot_frontend.py
```

---

## Project Structure

```
omniagent/
├── chatbot_frontend.py     # Streamlit UI — chat interface, sidebar, streaming
├── chatbot_backend.py      # LangGraph graph, tools, PDF ingestion, async loop
├── mcp_server.py           # Local FastMCP server with tools
├── chatbot_database.db     # SQLite persistence (auto-created)
├── .env                    # API keys (not committed)
└── requirements.txt
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `langgraph` | Agent graph orchestration & checkpointing |
| `langchain-groq` | Groq LLM integration |
| `langchain-community` | DuckDuckGo search, FAISS, PDF loader |
| `langchain-huggingface` | HuggingFace embedding endpoint |
| `langchain-mcp-adapters` | Connect LangGraph to MCP servers |
| `fastmcp` | Local MCP server framework |
| `streamlit` | Frontend UI |
| `faiss-cpu` | Vector similarity search |
| `aiosqlite` | Async SQLite for checkpointer |
| `python-dotenv` | Environment variable management |
| `requests` | Alpha Vantage HTTP calls |

---

## How It Works

### LangGraph Agent Loop

The graph has two nodes in a loop:

```
START → chat_node → (tools_condition) → tools → chat_node → … → END
```

`chat_node` calls the LLM with all tools bound. If the model decides to use a tool, `tools_condition` routes to the `ToolNode`; the tool result is appended to state and the loop continues until the model returns a final answer with no tool calls.

### Per-Thread RAG

Each conversation thread gets its own FAISS index. When a PDF is uploaded via the sidebar, `ingest_pdf()` chunks it with `RecursiveCharacterTextSplitter`, embeds with `sentence-transformers/all-MiniLM-L6-v2`, and stores the retriever keyed by `thread_id`. The `rag_tool` reads the thread ID from `RunnableConfig` at call time — so different chats can have different documents loaded simultaneously.

### Async Architecture

Streamlit runs synchronously. The LangGraph graph and SQLite checkpointer are async. The solution is a **dedicated background event loop** running in a daemon thread:

```python
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()
```

All async calls (`astream`, `aget_state`, `alist`) are submitted with `asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP).result()`, keeping the Streamlit main thread unblocked.

### Double-Run Prevention

A common Streamlit pitfall is `st.rerun()` firing the agent a second time for the same user message. This is prevented by a `processed_inputs` set in session state, keyed by `thread_id:message_count:input_text`. The key is added **before** the agent runs, so any rerun triggered mid-stream hits the guard and calls `st.stop()` immediately.

---

## JSON-RPC 2.0 — The Data Transport Layer

The Model Context Protocol uses **[JSON-RPC 2.0](https://www.jsonrpc.org/specification)** as its underlying data transport layer. Every message exchanged between an MCP client (`langchain-mcp-adapters`) and an MCP server (`mcp_server.py` or the remote expense server) is a JSON-RPC 2.0 envelope.

### What is JSON-RPC 2.0?

JSON-RPC 2.0 is a stateless, lightweight remote procedure call (RPC) protocol encoded in JSON. It defines a strict message structure so any client can call any server method without knowing the underlying transport (HTTP, WebSocket, SSE, stdio).

### Message Structure

**Request** (client → server):
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "add",
    "arguments": { "a": 12.5, "b": 7.3 }
  }
}
```

**Success Response** (server → client):
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      { "type": "text", "text": "19.8" }
    ]
  }
}
```

**Error Response** (server → client):
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32602,
    "message": "Invalid params",
    "data": "Field 'b' is required"
  }
}
```

**Notification** (fire-and-forget, no `id`, no response expected):
```json
{
  "jsonrpc": "2.0",
  "method": "notifications/progress",
  "params": { "progressToken": "abc123", "progress": 50, "total": 100 }
}
```

### Key Rules

| Rule | Detail |
|------|--------|
| `"jsonrpc": "2.0"` | **Required** on every message |
| `id` | Present on requests; must be echoed in the matching response. Omit for notifications |
| `method` | String naming the procedure (e.g. `tools/call`, `tools/list`) |
| `params` | Optional object or array of arguments |
| `result` **xor** `error` | A response carries exactly one of these, never both |

### Standard Error Codes

| Code | Meaning |
|------|---------|
| `-32700` | Parse error — invalid JSON received |
| `-32600` | Invalid request — missing required fields |
| `-32601` | Method not found |
| `-32602` | Invalid params |
| `-32603` | Internal error |
| `-32000` to `-32099` | Server-defined application errors (e.g. divide by zero) |

### How MCP Uses JSON-RPC 2.0 in This Project

`langchain-mcp-adapters` (`MultiServerMCPClient`) wraps every LangChain tool call in a `tools/call` JSON-RPC request and sends it over the configured transport:

- **`mcp_server.py`** — local server using **SSE** transport (`http://localhost:8000/sse`)
- **Remote expense server** — using **Streamable HTTP** transport

FastMCP handles all serialisation, routing, and error wrapping on the server side — you define plain Python functions decorated with `@mcp.tool()` and the framework generates the JSON-RPC layer automatically.

> **Spec**: https://www.jsonrpc.org/specification

---

## Roadmap

- [ ] Add `search_arxiv` MCP tool integration for research paper lookup
- [ ] Multi-PDF support per thread (index multiple documents simultaneously)
- [ ] Voice input via Whisper API
- [ ] Agent memory summarisation for very long threads
- [ ] Docker Compose setup (frontend + local MCP server)
- [ ] User authentication and per-user thread isolation
- [ ] Export conversation as PDF or Markdown
