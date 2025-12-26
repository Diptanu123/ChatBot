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

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from langchain_google_genai import ChatGoogleGenerativeAI

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

# Get API key from Streamlit secrets or environment
try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
    if not GOOGLE_API_KEY:
        st.error("⚠️ GOOGLE_API_KEY not found! Please add it to Streamlit secrets.")
        st.stop()
except Exception as e:
    st.error(f"Error loading secrets: {e}")
    GOOGLE_API_KEY = None

ALPHA_VANTAGE_KEY = st.secrets.get("ALPHA_VANTAGE_KEY", os.getenv("ALPHA_VANTAGE_KEY", "C9PE94QUEW9VWGFM"))

# --------------------------------------------------
# LLM - Use stable model
# --------------------------------------------------
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Changed to stable model
        temperature=0.7,
        google_api_key=GOOGLE_API_KEY,
        convert_system_message_to_human=True,  # Important for Gemini
    )
    # Test the connection
    test_response = llm.invoke("test")
    st.success("✅ LLM initialized successfully!")
except Exception as e:
    st.error(f"❌ Error initializing LLM: {e}")
    st.error("Please check your GOOGLE_API_KEY")
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
def calculator(first_num: float, second_num: float, operation: str) -> str:
    """Basic arithmetic calculator.
    
    Args:
        first_num: The first number
        second_num: The second number
        operation: Operation to perform - 'add', 'sub', 'mul', or 'div'
    
    Returns:
        String with the result
    """
    try:
        if operation == "add":
            return f"Result: {first_num + second_num}"
        if operation == "sub":
            return f"Result: {first_num - second_num}"
        if operation == "mul":
            return f"Result: {first_num * second_num}"
        if operation == "div":
            if second_num == 0:
                return "Error: Division by zero"
            return f"Result: {first_num / second_num}"
        return "Error: Invalid operation"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price for a given symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    
    Returns:
        String with stock price information
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}"
    )
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if "Global Quote" in data:
            price = data["Global Quote"].get("05. price", "N/A")
            return f"Stock price for {symbol}: ${price}"
        return f"Could not fetch price for {symbol}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def rag_tool(query: str, config: RunnableConfig) -> str:
    """Retrieve information from the uploaded PDF for the current conversation.
    
    Args:
        query: The search query to find relevant information
        config: The configuration containing thread_id
    
    Returns:
        String with the answer from the PDF
    """
    try:
        thread_id = config.get("configurable", {}).get("thread_id")
        
        retriever = _get_retriever(thread_id)
        if retriever is None:
            return "Error: No PDF indexed for this chat. Please upload a PDF first."

        docs = retriever.invoke(query)
        answer = "\n\n".join(d.page_content for d in docs)
        source = _THREAD_METADATA.get(str(thread_id), {}).get("filename", "Unknown")
        
        return f"From document '{source}':\n\n{answer}"
    except Exception as e:
        return f"Error retrieving information: {str(e)}"


tools = [calculator, get_stock_price, rag_tool]

# --------------------------------------------------
# State
# --------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# --------------------------------------------------
# Nodes
# --------------------------------------------------
def should_continue(state: ChatState):
    """Determine if we should continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If there are no tool calls, we're done
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "end"
    return "continue"

def chat_node(state: ChatState, config: RunnableConfig = None):
    """Main chat node that processes messages."""
    try:
        thread_id = None
        if config and "configurable" in config:
            thread_id = config["configurable"].get("thread_id")

        has_pdf = thread_id and _get_retriever(thread_id) is not None
        pdf_info = ""
        if has_pdf:
            meta = _THREAD_METADATA.get(str(thread_id), {})
            pdf_info = f"\n\nA PDF document '{meta.get('filename', 'unknown')}' has been uploaded. You can use the rag_tool to answer questions about it."

        # Create system message
        system_prompt = (
            "You are a helpful AI assistant with access to tools.\n"
            f"{pdf_info}\n\n"
            "Available tools:\n"
            "- calculator: For math operations\n"
            "- get_stock_price: For stock prices\n"
            "- rag_tool: To search the uploaded PDF\n\n"
            "When users ask about documents or PDFs, use the rag_tool.\n"
            "Always be helpful, friendly, and concise."
        )
        
        system_message = SystemMessage(content=system_prompt)
        
        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(tools)
        
        # Invoke LLM
        response = llm_with_tools.invoke(
            [system_message] + state["messages"],
            config=config,
        )
        
        return {"messages": [response]}
        
    except Exception as e:
        error_msg = f"I apologize, but I encountered an error: {str(e)}"
        return {"messages": [AIMessage(content=error_msg)]}

# Create tool node
tool_node = ToolNode(tools)

# --------------------------------------------------
# Persistence
# --------------------------------------------------
@st.cache_resource
def get_checkpointer():
    """Initialize SQLite checkpointer."""
    try:
        db_path = os.path.join(tempfile.gettempdir(), "chatbot.db")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        return saver, conn
    except Exception as e:
        st.error(f"Error initializing checkpointer: {e}")
        raise

checkpointer, conn = get_checkpointer()

# --------------------------------------------------
# Graph
# --------------------------------------------------
@st.cache_resource
def build_graph():
    """Build the LangGraph state graph."""
    try:
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("agent", chat_node)
        workflow.add_node("tools", tool_node)
        
        # Add edges
        workflow.add_edge(START, "agent")
        
        # Conditional edge from agent
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )
        
        # Edge from tools back to agent
        workflow.add_edge("tools", "agent")
        
        return workflow.compile(checkpointer=checkpointer)
        
    except Exception as e:
        st.error(f"Error building graph: {e}")
        raise

chatbot = build_graph()

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def retrieve_all_threads():
    """Retrieve all thread IDs from the database."""
    threads = set()
    try:
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'")
        if cursor.fetchone():
            cursor.execute("PRAGMA table_info(checkpoints)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'checkpoint_ns' in columns:
                cursor.execute("SELECT DISTINCT checkpoint_ns FROM checkpoints WHERE checkpoint_ns IS NOT NULL")
                for row in cursor.fetchall():
                    if row[0]:
                        threads.add(row[0])
    except Exception as e:
        st.error(f"Error retrieving threads: {e}")
    return list(threads)

def thread_document_metadata(thread_id: str):
    """Get document metadata for a thread."""
    return _THREAD_METADATA.get(str(thread_id), {})

def delete_thread(thread_id: str) -> bool:
    """Delete a thread from the database and memory."""
    try:
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        if 'checkpoints' in tables:
            cursor.execute("PRAGMA table_info(checkpoints)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'checkpoint_ns' in columns:
                cursor.execute(
                    "DELETE FROM checkpoints WHERE checkpoint_ns = ?",
                    (thread_id,)
                )
            
            if 'parent_checkpoint_ns' in columns:
                cursor.execute(
                    "DELETE FROM checkpoints WHERE parent_checkpoint_ns LIKE ?",
                    (f"%{thread_id}%",)
                )
        
        if 'writes' in tables:
            cursor.execute("PRAGMA table_info(writes)")
            write_columns = [col[1] for col in cursor.fetchall()]
            
            if 'checkpoint_ns' in write_columns:
                cursor.execute(
                    "DELETE FROM writes WHERE checkpoint_ns = ?",
                    (thread_id,)
                )
        
        conn.commit()
        
        _THREAD_RETRIEVERS.pop(str(thread_id), None)
        _THREAD_METADATA.pop(str(thread_id), None)
        
        return True
        
    except Exception as e:
        st.error(f"Error deleting thread: {e}")
        conn.rollback()
        return False