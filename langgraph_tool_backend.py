# backend.py

from typing import TypedDict, Annotated, List
import sqlite3
import requests
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# 1. LLM (Gemini)
# --------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# --------------------------------------------------
# 2. Tools (NO DuckDuckGo ❌)
# --------------------------------------------------

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform basic arithmetic: add, sub, mul, div"""
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
    """Fetch stock price using Alpha Vantage"""
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    return requests.get(url, timeout=10).json()


tools = [calculator, get_stock_price]
llm_with_tools = llm.bind_tools(tools)

# --------------------------------------------------
# 3. State
# --------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# --------------------------------------------------
# 4. Node
# --------------------------------------------------
def chat_node(state: ChatState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

# --------------------------------------------------
# 5. Persistence
# --------------------------------------------------
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# --------------------------------------------------
# 6. Graph
# --------------------------------------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# --------------------------------------------------
# 7. Helper
# --------------------------------------------------
def retrieve_all_threads():
    return list({
        cp.config["configurable"]["thread_id"]
        for cp in checkpointer.list(None)
        if cp.config.get("configurable")
    })
