from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict, Literal
from dotenv import load_dotenv
from typing import Annotated
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # ← async checkpointer
from langchain_mcp_adapters.client import MultiServerMCPClient
import operator
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import aiosqlite
import requests
import asyncio
import threading
from typing import Optional,Dict,Any
import os
import tempfile
from langchain_core.runnables import RunnableConfig

load_dotenv()

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

llm = ChatGroq(model="openai/gpt-oss-20b")

from langgraph.graph.message import add_messages

search_tool = DuckDuckGoSearchRun(region="us-en")


from langchain_huggingface import HuggingFaceEndpointEmbeddings

embed_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",  # or any embedding model
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN") 
)

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks,embedding=embed_model)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Use this when the user asks questions about the uploaded document.
    """
    thread_id = config.get("configurable", {}).get("thread_id")
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }
    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]
    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


# ── CHANGED: async initializer that builds the graph with MCP tools ──
async def build_graph():
    mcp_tools = []

    # ── Remote expense server ──
    try:
        remote_client = MultiServerMCPClient({
            "expense": {
                "url": "https://splendid-gold-dingo.fastmcp.app/mcp",
                "transport": "streamable_http",
            }
        })
        remote_tools = await remote_client.get_tools()
        mcp_tools += remote_tools
        print(f"✅ Remote MCP — {len(remote_tools)} tools loaded")
    except BaseException as e:
        print(f"⚠️ Remote MCP failed: {e}")

    print(f"📦 Total MCP tools: {len(mcp_tools)}")

    all_tools = [search_tool, get_stock_price , rag_tool] + mcp_tools
    llm_with_tools = llm.bind_tools(all_tools)

    def chat_node(state: ChatState, config: RunnableConfig):
        messages = state["messages"]
        thread_id = str(config.get("configurable", {}).get("thread_id", ""))

        has_doc = thread_id in _THREAD_RETRIEVERS
        doc_hint = (
            f"A PDF document has been uploaded for this chat (thread_id={thread_id})."
            f"When the user asks about the document, ALWAYS call rag_tool with query=<user question> and thread_id='{thread_id}'."
            if has_doc else
            "No PDF has been uploaded yet for this chat."
        )

        system = SystemMessage(content=f"""You are a helpful multi-utility assistant.

    {doc_hint}

    Never ask for clarification if the answer can be found using a tool. Always try tools first.
    """)

        response = llm_with_tools.invoke([system] + messages)
        return {"messages": [response]}

    tool_node = ToolNode(all_tools)

    aconn = await aiosqlite.connect("chatbot_database.db")
    checkpointer = AsyncSqliteSaver(conn=aconn)

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    return graph.compile(checkpointer=checkpointer), checkpointer


# ── CHANGED: run build_graph on the dedicated async loop ──
chatbot, checkpointer = asyncio.run_coroutine_threadsafe(
    build_graph(), _ASYNC_LOOP
).result()


# ✅ Fixed — run async list on the dedicated loop
def retreive_all_threads():
    async def _list():
        all_threads = set()
        async for checkpoint in checkpointer.alist(None):
            all_threads.add(checkpoint.config["configurable"]["thread_id"])
        return list(all_threads)
    return asyncio.run_coroutine_threadsafe(_list(), _ASYNC_LOOP).result()


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})

