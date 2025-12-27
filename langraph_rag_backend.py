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

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

# Use Groq instead of Google Gemini
from langchain_groq import ChatGroq

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

import requests
import streamlit as st

# --------------------------------------------------
# ENV (Streamlit Secrets will override this)
# --------------------------------------------------
load_dotenv()

# Get Groq API key from Streamlit secrets or environment
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    if not GROQ_API_KEY:
        st.error("⚠️ GROQ_API_KEY not found! Get one free at https://console.groq.com")
        st.stop()
except Exception as e:
    st.error(f"Error loading secrets: {e}")
    GROQ_API_KEY = None

ALPHA_VANTAGE_KEY = st.secrets.get("ALPHA_VANTAGE_KEY", os.getenv("ALPHA_VANTAGE_KEY", "C9PE94QUEW9VWGFM"))

# --------------------------------------------------
# LLM with Groq (FREE & FAST!)
# --------------------------------------------------
try:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  # Free, powerful model
        temperature=0.7,
        groq_api_key=GROQ_API_KEY,
        max_tokens=2048,
    )
    print("✅ Groq LLM initialized successfully")
except Exception as e:
    st.error(f"Error initializing LLM: {e}")
    st.error("Please check your GROQ_API_KEY in Streamlit secrets")
    st.stop()

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

# --------------------------------------------------
# State
# --------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# --------------------------------------------------
# Node Functions
# --------------------------------------------------
def chat_node(state: ChatState, config: RunnableConfig = None):
    """Main chat node that processes messages."""
    print(f"🔵 chat_node called with {len(state['messages'])} messages")
    
    thread_id = None
    if config and "configurable" in config:
        thread_id = config["configurable"].get("thread_id")
        print(f"📌 Thread ID: {thread_id}")

    has_pdf = thread_id and _get_retriever(thread_id) is not None
    pdf_info = ""
    if has_pdf:
        meta = _THREAD_METADATA.get(str(thread_id), {})
        pdf_info = f"\nA PDF document '{meta.get('filename', 'unknown')}' has been uploaded. Use rag_tool to answer questions about it."
        print(f"📄 PDF available: {meta.get('filename')}")

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

    try:
        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(tools)
        
        # Invoke LLM
        print("🤖 Invoking Groq LLM...")
        response = llm_with_tools.invoke(
            [system, *state["messages"]],
            config=config,
        )
        print(f"✅ LLM responded successfully")
        
        return {"messages": [response]}
        
    except Exception as e:
        print(f"❌ Error in chat_node: {e}")
        import traceback
        print(traceback.format_exc())
        error_msg = AIMessage(content=f"I apologize, but I encountered an error: {str(e)}. Please try again.")
        return {"messages": [error_msg]}


def should_continue(state: ChatState):
    """Determine if we should continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If there are tool calls, continue to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"🔧 Tool calls found: {len(last_message.tool_calls)}")
        return "tools"
    
    # Otherwise, end
    print("✅ No tool calls, ending")
    return "end"


# --------------------------------------------------
# Persistence
# --------------------------------------------------
@st.cache_resource
def get_checkpointer():
    db_path = os.path.join(tempfile.gettempdir(), "chatbot.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    print(f"✅ Database initialized at: {db_path}")
    return saver, conn

checkpointer, conn = get_checkpointer()

# --------------------------------------------------
# Graph
# --------------------------------------------------
def build_graph():
    """Build the graph without caching to avoid stale state."""
    print("🔨 Building graph...")
    
    # Create tool node
    tool_node = ToolNode(tools)
    
    # Create graph
    graph = StateGraph(ChatState)
    
    # Add nodes
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    # Add edges
    graph.add_edge(START, "chat_node")
    
    # Add conditional edges with proper routing
    graph.add_conditional_edges(
        "chat_node",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # Tools always go back to chat
    graph.add_edge("tools", "chat_node")

    compiled = graph.compile(checkpointer=checkpointer)
    print("✅ Graph compiled successfully")
    return compiled

# Build graph on module load
chatbot = build_graph()
print("🎯 Groq Chatbot ready")

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def retrieve_all_threads():
    """Retrieve all unique thread IDs from the database."""
    threads = set()
    try:
        cursor = conn.cursor()
        
        # Check if checkpoints table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'")
        if not cursor.fetchone():
            return list(threads)
        
        # Get all columns in checkpoints table
        cursor.execute("PRAGMA table_info(checkpoints)")
        columns = {col[1] for col in cursor.fetchall()}
        
        # LangGraph stores thread_id in the thread_id column (newer versions)
        # or in checkpoint_ns (older versions)
        if 'thread_id' in columns:
            cursor.execute("SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id IS NOT NULL")
            for row in cursor.fetchall():
                if row[0]:
                    threads.add(row[0])
        elif 'checkpoint_ns' in columns:
            cursor.execute("SELECT DISTINCT checkpoint_ns FROM checkpoints WHERE checkpoint_ns IS NOT NULL AND checkpoint_ns != ''")
            for row in cursor.fetchall():
                if row[0]:
                    threads.add(row[0])
        
        print(f"📋 Retrieved {len(threads)} threads from database")
        
    except Exception as e:
        print(f"Error retrieving threads: {e}")
        import traceback
        print(traceback.format_exc())
    
    return list(threads)

def thread_document_metadata(thread_id: str):
    return _THREAD_METADATA.get(str(thread_id), {})

def delete_thread(thread_id: str) -> bool:
    """Delete a thread from the database and memory."""
    try:
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Delete from checkpoints table
        if 'checkpoints' in tables:
            cursor.execute("PRAGMA table_info(checkpoints)")
            columns = {col[1] for col in cursor.fetchall()}
            
            # Try thread_id column first (newer LangGraph)
            if 'thread_id' in columns:
                cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
                print(f"🗑️ Deleted from checkpoints using thread_id: {thread_id}")
            
            # Also try checkpoint_ns (older LangGraph)
            if 'checkpoint_ns' in columns:
                cursor.execute("DELETE FROM checkpoints WHERE checkpoint_ns = ?", (thread_id,))
                print(f"🗑️ Deleted from checkpoints using checkpoint_ns: {thread_id}")
        
        # Delete from writes table if it exists
        if 'writes' in tables:
            cursor.execute("PRAGMA table_info(writes)")
            write_columns = {col[1] for col in cursor.fetchall()}
            
            if 'thread_id' in write_columns:
                cursor.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
            
            if 'checkpoint_ns' in write_columns:
                cursor.execute("DELETE FROM writes WHERE checkpoint_ns = ?", (thread_id,))
        
        conn.commit()
        
        # Clean up memory
        _THREAD_RETRIEVERS.pop(str(thread_id), None)
        _THREAD_METADATA.pop(str(thread_id), None)
        
        print(f"✅ Thread {thread_id} deleted successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error deleting thread {thread_id}: {e}")
        import traceback
        print(traceback.format_exc())
        conn.rollback()
        return False