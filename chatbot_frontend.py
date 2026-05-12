import streamlit as st
from chatbot_backend import chatbot, llm, retreive_all_threads, _ASYNC_LOOP, ingest_pdf, thread_document_metadata
import streamlit.components.v1 as components
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
import asyncio
import queue


def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state["thread_id"])
    st.session_state["message_history"] = []
    # ✅ FIX: clear processed_inputs on new chat so guard resets cleanly
    st.session_state["processed_inputs"] = set()


def add_thread(thread_id):
    if "chat_threads" not in st.session_state:
        st.session_state["chat_threads"] = []
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    try:
        async def _get():
            return await chatbot.aget_state(config={"configurable": {"thread_id": thread_id}})
        result = asyncio.run_coroutine_threadsafe(_get(), _ASYNC_LOOP).result()
        return result.values["messages"]
    except:
        return []


def chat_name(thread_id):
    if "chat_names" not in st.session_state:
        st.session_state["chat_names"] = {}

    if thread_id in st.session_state["chat_names"]:
        return st.session_state["chat_names"][thread_id]

    messages = load_conversation(thread_id)
    temp_message = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
        else:
            role = "ai"
        temp_message.append({"role": role, "content": message.content})

    prompt = f"""You are a chat title generator. Given the message history below, reply with ONLY a short 2-5 word title summarizing the conversation. 
Do NOT explain, do NOT use markdown, do NOT add any extra text — just the title itself.
If the message history is empty, reply with exactly: start the conversation

Message history: {temp_message}"""

    name = llm.invoke(prompt).content.strip()
    st.session_state["chat_names"][thread_id] = name
    return name


# ── Session state init ──────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retreive_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

# ✅ FIX: use a set of processed input IDs instead of a nullable last_input string.
# A set survives st.rerun() correctly — once an input is added, it stays marked
# as processed across reruns, so the agent never fires twice for the same message.
if "processed_inputs" not in st.session_state:
    st.session_state["processed_inputs"] = set()

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
selected_thread = None

CONFIG = {"configurable": {"thread_id": thread_key}, "run_name": "chat-turn"}

# ── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.title("OmniAgent — Multi-Agent RAG & MCP Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()


# ── Available Tools Panel ────────────────────────────────────────────────
st.sidebar.markdown("**AVAILABLE TOOLS**")
st.sidebar.markdown("""
<style>
.tool-card {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 9px 11px;
    border-radius: 10px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 7px;
}
.tool-icon {
    font-size: 18px;
    line-height: 1;
    margin-top: 2px;
    flex-shrink: 0;
}
.tool-body { flex: 1; min-width: 0; }
.tool-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    flex-wrap: wrap;
}
.tool-name {
    font-size: 13.5px;
    font-weight: 600;
    color: #e2e8f0;
    white-space: nowrap;
}
.tool-badge {
    font-size: 10px;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 99px;
    white-space: nowrap;
    letter-spacing: 0.03em;
}
.badge-duck   { background: rgba(251,191, 36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3);  }
.badge-alpha  { background: rgba( 52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3);  }
.badge-rag    { background: rgba(167,139,250,0.15); color: #a78bfa; border: 1px solid rgba(167,139,250,0.3); }
.badge-local  { background: rgba( 56,189,248,0.15); color: #38bdf8; border: 1px solid rgba(56,189,248,0.3);  }
.badge-remote { background: rgba(251,113, 94,0.15); color: #fb7065; border: 1px solid rgba(251,113,94,0.3);  }
.tool-desc {
    font-size: 12px;
    color: #94a3b8;
    margin-top: 3px;
    line-height: 1.4;
}
</style>

<div class="tool-card">
  <div class="tool-icon">🔍</div>
  <div class="tool-body">
    <div class="tool-header">
      <span class="tool-name">Web Search</span>
      <span class="tool-badge badge-duck">DuckDuckGo</span>
    </div>
    <div class="tool-desc">Search the web for real-time info, news, and facts</div>
  </div>
</div>

<div class="tool-card">
  <div class="tool-icon">📈</div>
  <div class="tool-body">
    <div class="tool-header">
      <span class="tool-name">Stock Price</span>
      <span class="tool-badge badge-alpha">Alpha Vantage</span>
    </div>
    <div class="tool-desc">Fetch live stock prices by ticker symbol</div>
  </div>
</div>

<div class="tool-card">
  <div class="tool-icon">📄</div>
  <div class="tool-body">
    <div class="tool-header">
      <span class="tool-name">PDF Q&amp;A</span>
      <span class="tool-badge badge-rag">RAG Engine</span>
    </div>
    <div class="tool-desc">Ask questions about your uploaded document</div>
  </div>
</div>

<div class="tool-card">
  <div class="tool-icon">💸</div>
  <div class="tool-body">
    <div class="tool-header">
      <span class="tool-name">Expense Tracker</span>
      <span class="tool-badge badge-remote">MCP Remote</span>
    </div>
    <div class="tool-desc">Track and manage expenses</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.header("My Conversations")

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(chat_name(thread_id)), key=str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        temp_message = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            else:
                role = "ai"
            temp_message.append({"role": role, "content": message.content})
        st.session_state["message_history"] = temp_message

        doc_meta = thread_document_metadata(thread_key)
        if doc_meta:
            st.caption(
                f"Document indexed: {doc_meta.get('filename')} "
                f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
            )

# ── Active button highlight ───────────────────────────────────────────────
active_name = chat_name(st.session_state["thread_id"])

st.markdown("""
<style>
section[data-testid="stSidebar"] div[data-testid="stButton"] button {
    width: 100% !important;
    border-radius: 8px !important;
    transition: border 0.2s ease, box-shadow 0.2s ease, background 0.2s ease !important;
}
</style>
""", unsafe_allow_html=True)

components.html(f"""
<script>
(function() {{
    const ACTIVE_NAME = {repr(active_name)};

    function applyHighlight() {{
        const sidebar = window.parent.document.querySelector('section[data-testid="stSidebar"]');
        if (!sidebar) return;

        sidebar.querySelectorAll('div[data-testid="stButton"] button').forEach(btn => {{
            const text = btn.innerText || btn.textContent || "";

            if (text.trim() === ACTIVE_NAME.trim()) {{
                btn.style.border      = "1.5px solid rgba(56, 189, 248, 0.85)";
                btn.style.background  = "rgba(14, 165, 233, 0.12)";
                btn.style.boxShadow   = "0 0 12px rgba(56, 189, 248, 0.4), inset 0 0 8px rgba(56, 189, 248, 0.06)";
                btn.style.color       = "#7dd3fc";
                btn.style.fontWeight  = "600";
            }} else {{
                btn.style.border      = "";
                btn.style.background  = "";
                btn.style.boxShadow   = "";
                btn.style.color       = "";
                btn.style.fontWeight  = "";
            }}
        }});
    }}

    applyHighlight();
    const observer = new MutationObserver(applyHighlight);
    observer.observe(window.parent.document.body, {{ subtree: true, childList: true }});
}})();
</script>
""", height=0)

# ── Render existing chat history ──────────────────────────────────────────
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        if message.get("tools"):
            with st.status("✅ Tool finished", state="complete", expanded=False):
                st.write(f"Tools used: {', '.join(message['tools'])}")
        st.markdown(message["content"], unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────────────────
user_input = st.chat_input("Type Here...")

if user_input:
    # ✅ FIX: generate a stable key for this exact input submission.
    # We combine the input text with the current message count so that sending
    # the same text twice in the same thread still gets processed both times,
    # but a st.rerun() mid-stream never re-triggers the agent.
    input_key = f"{thread_key}:{len(st.session_state['message_history'])}:{user_input}"

    if input_key in st.session_state["processed_inputs"]:
        # Already handled this exact submission — skip silently.
        st.stop()

    # Mark as processed BEFORE running the agent so that if a rerun fires
    # mid-stream, the guard above catches it immediately.
    st.session_state["processed_inputs"].add(input_key)

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("ai"):
        placeholder = st.empty()
        ai_message = ""
        tool_names = []

        token_queue = queue.Queue()

        async def _stream_tokens(q: queue.Queue):
            async for chunk, metadata in chatbot.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(chunk, ToolMessage):
                    q.put(("tool", getattr(chunk, "name", "tool")))
                elif isinstance(chunk, AIMessage) and chunk.content:
                    q.put(("token", chunk.content))
            q.put(("done", None))

        asyncio.run_coroutine_threadsafe(_stream_tokens(token_queue), _ASYNC_LOOP)

        while True:
            item = token_queue.get()
            kind, data = item
            if kind == "token":
                ai_message += data
                placeholder.markdown(ai_message + "▌")
            elif kind == "tool":
                tool_names.append(data)
            elif kind == "done":
                placeholder.markdown(ai_message)
                break

        if tool_names:
            with st.status("✅ Tool finished", state="complete", expanded=False):
                st.write(f"Tools used: {', '.join(tool_names)}")

    st.session_state["message_history"].append({
        "role": "ai",
        "content": ai_message,
        "tools": tool_names,
    })

    # ✅ FIX: delete the chat name cache for the current thread so it regenerates
    # on next render, but do NOT call st.rerun() — the response is already
    # rendered live via streaming, so rerun is not needed and was the root cause
    # of the second agent invocation.
    current = st.session_state["thread_id"]
    if current in st.session_state.get("chat_names", {}):
        del st.session_state["chat_names"][current]

    # ✅ REMOVED: st.rerun() — this was the primary trigger for the double-run.
    # The sidebar chat name will update naturally on the user's next interaction.
    # If you need the sidebar name to update immediately, use st.rerun() ONLY
    # after confirming processed_inputs contains the input_key (guard is set).