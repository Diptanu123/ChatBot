from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool

from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

import requests

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# LLM (chat only)
# --------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# --------------------------------------------------
# LOCAL embeddings (NO API / NO quota)
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------
# Thread stores
# --------------------------------------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: str):
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
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        store = FAISS.from_documents(chunks, embeddings)
        retriever = store.as_retriever(search_kwargs={"k": 4})

        _THREAD_RETRIEVERS[thread_id] = retriever
        _THREAD_METADATA[thread_id] = {
            "filename": filename,
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return _THREAD_METADATA[thread_id]

    finally:
        os.remove(path)

# --------------------------------------------------
# Tools
# --------------------------------------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform basic arithmetic."""
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
    """Get stock price from Alpha Vantage."""
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    return requests.get(url).json()

@tool
def rag_tool(query: str, thread_id: str) -> dict:
    """
    Retrieve information from the uploaded PDF.
    thread_id is ALWAYS provided automatically.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {"error": "No PDF indexed"}

    docs = retriever.invoke(query)
    return {
        "answer": "\n\n".join(d.page_content for d in docs),
        "source": _THREAD_METADATA[thread_id]["filename"]
    }

tools = [search_tool, calculator, get_stock_price, rag_tool]
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
    thread_id = config["configurable"]["thread_id"]

    system = SystemMessage(
        content=(
            "You are a helpful assistant.\n"
            f"The current thread_id is: {thread_id}\n"
            "If the user asks anything about the uploaded PDF, "
            "you MUST call rag_tool using this thread_id.\n"
            "NEVER ask the user for thread_id."
        )
    )

    response = llm_with_tools.invoke(
        [system, *state["messages"]],
        config=config
    )
    return {"messages": [response]}

tool_node = ToolNode(tools)

# --------------------------------------------------
# Persistence
# --------------------------------------------------
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# --------------------------------------------------
# Graph
# --------------------------------------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def retrieve_all_threads():
    return list({
        cp.config["configurable"]["thread_id"]
        for cp in checkpointer.list(None)
        if cp.config.get("configurable")
    })

def thread_document_metadata(thread_id: str):
    return _THREAD_METADATA.get(thread_id, {})

def delete_thread(thread_id: str) -> bool:
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM checkpoints "
        "WHERE json_extract(config, '$.configurable.thread_id') = ?",
        (thread_id,)
    )
    conn.commit()

    _THREAD_RETRIEVERS.pop(thread_id, None)
    _THREAD_METADATA.pop(thread_id, None)
    return cursor.rowcount > 0
