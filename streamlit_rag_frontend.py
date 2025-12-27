import uuid
import streamlit as st
import os
import time
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langraph_rag_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
    delete_thread,
)

# =========================== Page Config ===========================
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# =========================== Utilities ===========================

def generate_thread_id():
    return str(uuid.uuid4())


def extract_text(content):
    """Extract text from message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    text_parts.append(block["text"])
                elif "type" in block and block["type"] == "text" and "text" in block:
                    text_parts.append(block["text"])
        return "".join(text_parts)
    return str(content)


def add_thread(thread_id):
    """Add thread to session state if not already present."""
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    """Create a new chat thread."""
    new_id = generate_thread_id()
    st.session_state["thread_id"] = new_id
    add_thread(new_id)
    st.session_state["message_history"] = []
    st.session_state["processing"] = False


def load_conversation(thread_id):
    """Load conversation history for a thread."""
    try:
        state = chatbot.get_state(
            config={"configurable": {"thread_id": thread_id}}
        )
        return state.values.get("messages", [])
    except Exception as e:
        st.error(f"Error loading conversation: {e}")
        return []


# ======================= Session Initialization ===================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

# Always refresh threads from database on page load
# This ensures past conversations persist after refresh
if "chat_threads" not in st.session_state or st.session_state.get("force_reload_threads", False):
    st.session_state["chat_threads"] = retrieve_all_threads()
    st.session_state["force_reload_threads"] = False

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

if "processing" not in st.session_state:
    st.session_state["processing"] = False

add_thread(st.session_state["thread_id"])

# If current thread has no messages, try to load from database
if not st.session_state["message_history"]:
    messages = load_conversation(st.session_state["thread_id"])
    if messages:
        cleaned = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                cleaned.append(
                    {"role": "user", "content": extract_text(msg.content)}
                )
            elif isinstance(msg, AIMessage):
                text = extract_text(msg.content)
                if text:
                    cleaned.append(
                        {"role": "assistant", "content": text}
                    )
        st.session_state["message_history"] = cleaned

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

# ============================ Sidebar ============================

st.sidebar.title("📄 PDF Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key[:8]}...`")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.session_state["force_reload_threads"] = True
    st.rerun()

# Refresh button to manually reload threads
if st.sidebar.button("🔄 Refresh Conversations", use_container_width=True):
    st.session_state["chat_threads"] = retrieve_all_threads()
    st.sidebar.success("✅ Refreshed!")
    st.rerun()

# PDF Upload Section
st.sidebar.subheader("📤 Upload PDF")

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"✅ Using `{latest_doc['filename']}`\n"
        f"📊 {latest_doc['chunks']} chunks • {latest_doc['documents']} pages"
    )
else:
    st.sidebar.info("ℹ️ No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF for this chat",
    type=["pdf"],
    key="pdf_uploader"
)

if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"✓ `{uploaded_pdf.name}` already indexed.")
    else:
        with st.spinner("📄 Indexing PDF..."):
            try:
                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                )
                thread_docs[uploaded_pdf.name] = summary
                st.sidebar.success("✅ PDF indexed successfully!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Error indexing PDF: {e}")


# ====================== Past Conversations ======================

st.sidebar.subheader("🕘 Past Conversations")

threads = list(reversed(st.session_state["chat_threads"]))

if not threads:
    st.sidebar.info("No past conversations yet.")
else:
    st.sidebar.caption(f"📊 {len(threads)} conversation(s) found")
    
    for tid in threads:
        display_id = f"{tid[:8]}..."
        
        col1, col2 = st.sidebar.columns([5, 1])

        if col1.button(display_id, key=f"load-{tid}", use_container_width=True):
            st.session_state["thread_id"] = tid
            st.session_state["processing"] = False
            
            # Load conversation
            messages = load_conversation(tid)
            cleaned = []
            
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    cleaned.append(
                        {"role": "user", "content": extract_text(msg.content)}
                    )
                elif isinstance(msg, AIMessage):
                    text = extract_text(msg.content)
                    if text:
                        cleaned.append(
                            {"role": "assistant", "content": text}
                        )
            
            st.session_state["message_history"] = cleaned
            st.session_state["ingested_docs"].setdefault(tid, {})
            st.rerun()

        if col2.button("🗑", key=f"delete-{tid}"):
            with st.spinner("Deleting..."):
                if delete_thread(tid):
                    # Remove from session state
                    if tid in st.session_state["chat_threads"]:
                        st.session_state["chat_threads"].remove(tid)
                    st.session_state["ingested_docs"].pop(tid, None)

                    # If deleting current thread, create new one
                    if tid == thread_key:
                        reset_chat()
                        st.session_state["force_reload_threads"] = True

                    st.sidebar.success("✅ Deleted")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Delete failed")

# ============================ Main UI ============================

st.title("🤖 AI Chatbot")
st.caption("Ask questions about your documents or anything else!")

# Check if API key is configured
if not st.secrets.get("GROQ_API_KEY") and not os.getenv("GROQ_API_KEY"):
    st.error("⚠️ **Groq API Key not configured!**")
    st.info("""
    **To fix this:**
    1. Go to [Groq Console](https://console.groq.com)
    2. Create a FREE API key (no credit card needed!)
    3. In Streamlit Cloud: Go to App Settings → Secrets
    4. Add: `GROQ_API_KEY = "your-key-here"`
    5. Restart the app
    
    **Why Groq?** Super fast, generous free tier (14,400 requests/day), no credit card required!
    """)
    st.stop()

# Display message history
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input(
    "Ask about your document or anything else...",
    disabled=st.session_state.get("processing", False)
)

# ============================ Chat Handling ============================

if user_input and not st.session_state.get("processing", False):
    st.session_state["processing"] = True
    
    # Add user message
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare config
    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    # Generate AI response
    with st.chat_message("assistant"):
        try:
            full_response = ""
            placeholder = st.empty()
            thinking_placeholder = st.empty()
            
            # Show animated thinking indicator
            thinking_dots = ["🤔 Thinking", "🤔 Thinking.", "🤔 Thinking..", "🤔 Thinking..."]
            dot_index = 0
            thinking_placeholder.markdown(thinking_dots[0])
            
            event_count = 0
            has_response = False
            last_update = time.time()
            
            # Use stream_mode="values" for reliable streaming
            stream = chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="values",
            )
            
            for event in stream:
                event_count += 1
                
                # Animate thinking dots
                if not has_response and time.time() - last_update > 0.3:
                    dot_index = (dot_index + 1) % len(thinking_dots)
                    thinking_placeholder.markdown(thinking_dots[dot_index])
                    last_update = time.time()
                
                # Get the messages from the state
                messages = event.get("messages", [])
                
                if messages:
                    last_msg = messages[-1]
                    
                    # Only process AI messages
                    if isinstance(last_msg, AIMessage):
                        # Extract text from the message
                        text = extract_text(last_msg.content)
                        
                        # Update only if we have new content
                        if text and text != full_response:
                            # Clear thinking indicator once we get response
                            if not has_response:
                                thinking_placeholder.empty()
                                has_response = True
                            
                            full_response = text
                            placeholder.markdown(full_response + "▌")
            
            # Remove cursor and show final response
            thinking_placeholder.empty()
            if full_response:
                placeholder.markdown(full_response)
            else:
                # Fallback if no response was generated
                full_response = "Hello! How can I help you today?"
                placeholder.markdown(full_response)

            # Add to history
            st.session_state["message_history"].append(
                {"role": "assistant", "content": full_response}
            )

            # Show document info if available
            doc_meta = thread_document_metadata(thread_key)
            if doc_meta:
                st.caption(
                    f"📄 `{doc_meta['filename']}` • "
                    f"{doc_meta['chunks']} chunks • "
                    f"{doc_meta['documents']} pages"
                )

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.error("**Full error details:**")
            st.code(str(e))
            import traceback
            st.code(traceback.format_exc())
            
            # Remove the failed user message
            if st.session_state["message_history"][-1]["role"] == "user":
                st.session_state["message_history"].pop()
    
    st.session_state["processing"] = False
    st.rerun()