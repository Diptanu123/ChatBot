from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

import requests
import streamlit as st

# --------------------------------------------------
# ENV (Streamlit Secrets will override this)
# --------------------------------------------------
load_dotenv()

# Get API key from Streamlit secrets or environment
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
ALPHA_VANTAGE_KEY = st.secrets.get("ALPHA_VANTAGE_KEY", os.getenv("ALPHA_VANTAGE_KEY", "C9PE94QUEW9VWGFM"))

# --------------------------------------------------
# LLM
# --------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
)

# --------------------------------------------------
# Local embeddings (NO API)
# --------------------------------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = get_embeddings()

# --------------------------------------------------
# In-memory thread stores
# --------------------------------------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: Optional[str]):
    if not thread_id:
        return None
    return _THREAD_RETRIEVERS.get(str(thread_id))

# --------------------------------------------------
# PDF ingestion
# --------------------------------------------------
def ingest_pdf(file_bytes: bytes, thread_id: str, filename: str) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        path = f.name

    try:
        loader = PyPDFLoader(path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(docs)

        store = FAISS.from_documents(chunks, embeddings)
        retriever = store.as_retriever(search_kwargs={"k": 4})

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename,
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return _THREAD_METADATA[str(thread_id)]

    finally:
        try:
            os.remove(path)
        except OSError:
            pass

# --------------------------------------------------
# Tools
# --------------------------------------------------
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Basic arithmetic calculator.
    
    Args:
        first_num: The first number
        second_num: The second number
        operation: Operation to perform - 'add', 'sub', 'mul', or 'div'
    
    Returns:
        Dictionary with 'result' key or 'error' key
    """
    if operation == "add":
        return {"result": first_num + second_num}
    if operation == "sub":
        return {"result": first_num - second_num}
    if operation == "mul":
        return {"result": first_num * second_num}
    if operation == "div":
        if second_num == 0:
            return {"error": "Division by zero"}
        return {"result": first_num / second_num}
    return {"error": "Invalid operation"}

@tool
def get_stock_price(symbol: str) -> dict:
    """Get current stock price for a given symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    
    Returns:
        Dictionary with stock price information
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}"
    )
    try:
        response = requests.get(url, timeout=10)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """Retrieve information from the uploaded PDF for the current conversation.
    
    Args:
        query: The search query to find relevant information
        config: The configuration containing thread_id
    
    Returns:
        Dictionary with answer and source filename
    """
    thread_id = config.get("configurable", {}).get("thread_id")

    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {"error": "No PDF indexed for this chat. Please upload a PDF first."}

    try:
        docs = retriever.invoke(query)
        return {
            "answer": "\n\n".join(d.page_content for d in docs),
            "source": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
        }
    except Exception as e:
        return {"error": f"Error retrieving information: {str(e)}"}


tools = [calculator, get_stock_price, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# --------------------------------------------------
# State
# --------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# --------------------------------------------------
# Node
# --------------------------------------------------
def chat_node(state: ChatState, config=None):
    thread_id = None
    if config and "configurable" in config:
        thread_id = config["configurable"].get("thread_id")

    has_pdf = thread_id and _get_retriever(thread_id) is not None
    pdf_info = ""
    if has_pdf:
        meta = _THREAD_METADATA.get(str(thread_id), {})
        pdf_info = f"\nA PDF document '{meta.get('filename', 'unknown')}' has been uploaded. Use rag_tool to answer questions about it."

    system = SystemMessage(
        content=(
            "You are a helpful AI assistant with access to tools.\n"
            f"Current thread_id: {thread_id}\n"
            f"{pdf_info}\n"
            "When users ask about documents, uploaded files, or PDFs, use the rag_tool.\n"
            "For calculations, use the calculator tool.\n"
            "For stock prices, use the get_stock_price tool.\n"
            "Always be helpful and friendly."
        )
    )

    response = llm_with_tools.invoke(
        [system, *state["messages"]],
        config=config,
    )
    return {"messages": [response]}

tool_node = ToolNode(tools)

# --------------------------------------------------
# Persistence - Initialize database properly
# --------------------------------------------------
@st.cache_resource
def get_checkpointer():
    db_path = os.path.join(tempfile.gettempdir(), "chatbot.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn), conn

checkpointer, conn = get_checkpointer()

# --------------------------------------------------
# Graph
# --------------------------------------------------
@st.cache_resource
def build_graph():
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    return graph.compile(checkpointer=checkpointer)

chatbot = build_graph()

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def retrieve_all_threads():
    threads = set()
    try:
        for cp in checkpointer.list(None):
            cfg = cp.config.get("configurable")
            if cfg and "thread_id" in cfg:
                threads.add(cfg["thread_id"])
    except Exception as e:
        st.error(f"Error retrieving threads: {e}")
    return list(threads)

def thread_document_metadata(thread_id: str):
    return _THREAD_METADATA.get(str(thread_id), {})

def delete_thread(thread_id: str) -> bool:
    """Delete a thread from the database and memory."""
    try:
        cursor = conn.cursor()
        
        # Delete from checkpoints table
        cursor.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?",
            (thread_id,)
        )
        
        # Also try the JSON extract method as fallback
        cursor.execute(
            "DELETE FROM checkpoints "
            "WHERE json_extract(config, '$.configurable.thread_id') = ?",
            (thread_id,),
        )
        
        # Delete from writes table if it exists
        try:
            cursor.execute(
                "DELETE FROM writes WHERE thread_id = ?",
                (thread_id,)
            )
        except sqlite3.OperationalError:
            pass  # writes table might not exist
        
        conn.commit()
        
        # Clean up memory
        _THREAD_RETRIEVERS.pop(str(thread_id), None)
        _THREAD_METADATA.pop(str(thread_id), None)
        
        return True
        
    except Exception as e:
        st.error(f"Error deleting thread: {e}")
        return False