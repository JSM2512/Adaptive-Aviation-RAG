# app.py
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import HumanMessage
from src.graph import app
from src.utils import ensure_env_vars, ensure_directories
import config
load_dotenv()
# Ensure setup
ensure_env_vars()
ensure_directories()

# Streamlit config
st.set_page_config(
    page_title=config.PAGE_TITLE,
    layout=config.PAGE_LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("✈️ Adaptive RAG Chat - Aviation Reports")
st.markdown("Ask questions about aviation accidents from NTSB reports (2023-2024)")

# Sidebar
with st.sidebar:
    st.header("📋 Session Management")
    
    thread_id = st.text_input(
        "Session ID:",
        value=st.session_state.thread_id or "",
        placeholder="e.g., user_123",
        help="Unique identifier for conversation history"
    )
    
    if thread_id and thread_id != st.session_state.thread_id:
        st.session_state.thread_id = thread_id
        st.session_state.messages = []
        st.rerun()
    
    if st.session_state.thread_id:
        st.success(f"✅ Connected: {st.session_state.thread_id}")
        
        if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.success("History cleared")
    else:
        st.warning("⚠️ Enter Session ID to continue")

# Main content
if st.session_state.thread_id:
    # Chat display
    st.subheader(f"💬 Chat - Session: {st.session_state.thread_id}")
    
    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
    
    # Chat input
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input(
            "Your question:",
            placeholder="e.g., What happened in Bell helicopter accident?",
            key="question_input",
            label_visibility="collapsed"
        )
    with col2:
        send_button = st.button("Send", type="primary", use_container_width=True)
    
    # Process question
    if send_button and question:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.spinner("🔄 Analyzing question..."):
            try:
                # Invoke the graph
                result = app.invoke(
                    {
                        "question": question,
                        "messages": [HumanMessage(content=""+question)],
                        "documents": [],
                        "generation": ""
                    },
                    config={"configurable": {"thread_id": st.session_state.thread_id}}
                )
                
                generation = result["generation"]
                documents = result["documents"]
                
                # Add assistant message to chat
                st.session_state.messages.append({"role": "assistant", "content": generation})
                
                # Display answer
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.messages.pop()  # Remove failed user message
    
    # Display source documents in sidebar
    with st.sidebar:
        st.divider()
        st.header("📚 References")
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            with st.expander("View source documents", expanded=False):
                try:
                    # Get last result's documents
                    if 'result' in locals() and result.get("documents"):
                        for i, doc in enumerate(result["documents"], 1):
                            with st.expander(f"Document {i}"):
                                st.write(f"**Source:** {doc.metadata.get('source_file', 'Unknown')}")
                                st.write(doc.page_content[:500] + "...")
                except:
                    pass

else:
    st.info("👈 **Step 1:** Enter a Session ID in the sidebar\n\n**Step 2:** Ask your question")