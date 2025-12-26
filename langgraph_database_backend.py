from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3
import json

load_dotenv()

# -------------------
# 1. LLM
# -------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# -------------------
# 2. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 3. Node
# -------------------
def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# -------------------
# 4. Checkpointer
# -------------------
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 5. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 6. Helpers
# -------------------
def retrieve_all_threads():
    """Return all unique thread IDs."""
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

# def delete_thread(thread_id: str) -> bool:
#     """
#     Permanently delete a conversation (thread) from the SQLite database.
#     """
#     try:
#         cursor = conn.cursor()

#         cursor.execute(
#             """
#             DELETE FROM checkpoints
#             WHERE json_extract(config, '$.configurable.thread_id') = ?
#             """,
#             (thread_id,),
#         )

#         deleted = cursor.rowcount
#         conn.commit()

#         return deleted > 0

#     except Exception as e:
#         print("Delete error:", e)
#         return False
