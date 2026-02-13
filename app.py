import streamlit as st
import json
import os
from ccore.retrieval import retrieve_context, generate_answer, summarize_document, analyze_image
from ccore.indexing import create_embeddings, save_to_faiss
from ccore.ingestion import extract_text_from_page
import fitz
# 1. Page Configuration
st.set_page_config(page_title="My Unique RAG", page_icon="", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    .stSidebar {
        background-color: #f9f9f9;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# 2. Sidebar for controls
with st.sidebar:
    st.title("🎛️ Control Center")
    st.markdown("---")
    
    # Section 1: Document Management
    st.header("📂 Document Base")
    uploaded_file = st.file_uploader("Upload New PDF", type=["pdf"], help="Upload a PDF to replace the current knowledge base.")
    
    if uploaded_file:
        # Ensure directories exist
        os.makedirs("data", exist_ok=True)
        os.makedirs("vector_db", exist_ok=True)
        
        # Save file
        pdf_path = os.path.join("data", "uploaded_doc.pdf")
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Run Ingestion Pipeline with status updates
        chunks = []
        doc = fitz.open(pdf_path)
        num_pages = len(doc)

        with st.status("🔄 Processing & Indexing Document...", expanded=True) as status:
            for i, page in enumerate(doc):
                status.update(label=f"📄 Processing page {i + 1}/{num_pages}...")
                text, ocr_used = extract_text_from_page(page)
                if ocr_used:
                    status.update(label=f"🤖 Performing OCR on page {i + 1}/{num_pages} (this might be slow)...")
                chunks.append(text)

            if not any(c.strip() for c in chunks):
                status.update(label="⚠️ No text extracted! PDF might be empty or fully scanned.", state="error")
                st.stop()

            status.update(label="🧠 Generating embeddings...")
            embeddings = create_embeddings(chunks)

            status.update(label="💾 Saving to knowledge base...")
            with open("data/chunks.json", "w") as f:
                json.dump(chunks, f)
            save_to_faiss(embeddings, metadata=None)

            status.update(label="✅ Knowledge Base Updated!", state="complete", expanded=False)
        
        st.rerun()

    st.markdown("---")
    
    # Section 2: Image Analysis
    st.header("🖼️ Image Analysis")
    uploaded_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], help="Upload an image to get a detailed description.")
    
    if uploaded_img:
        # Check if we haven't processed this file yet
        if "last_image" not in st.session_state or st.session_state.last_image != uploaded_img.name:
            with st.spinner("👁️ Analyzing image..."):
                st.session_state.messages.append({"role": "user", "content": f"**[Uploaded Image]** {uploaded_img.name}"})
                
                image_bytes = uploaded_img.getvalue()
                description = analyze_image(image_bytes)
                
                st.session_state.messages.append({"role": "assistant", "content": f"**Image Analysis:**\n\n{description}"})
                st.session_state.last_image = uploaded_img.name
                st.rerun()

    st.markdown("---")

    # Section 2: Chat Controls
    st.header("💬 Chat Actions")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    if st.button("📝 Summarize Document", use_container_width=True):
        with st.spinner("📄 Reading document and summarizing..."):
            if os.path.exists("data/chunks.json"):
                with open("data/chunks.json", "r") as f:
                    all_chunks = json.load(f)
                
                if not all_chunks or all(c.strip() == "" for c in all_chunks):
                    st.error("⚠️ Document has no text to summarize.")
                else:
                    try:
                        summary = summarize_document(all_chunks)
                        st.session_state.messages.append({"role": "assistant", "content": f"**📄 Document Summary:**\n\n{summary}"})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Summary failed: {str(e)}")

            else:
                st.error("No document data found. Please run setup.py.")
    
    st.markdown("---")

    # Section 3: Advanced Settings
    with st.expander("⚙️ Advanced Settings", expanded=False):
        temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.7, help="Higher values make the AI more creative, lower values more focused.")
        max_tokens = st.slider("Max Response Length", 256, 4096, 1024, help="Maximum number of words the AI can generate.")

    # Section 4: Stats
    if os.path.exists("data/chunks.json"):
        with open("data/chunks.json", "r") as f:
            chunk_count = len(json.load(f))
        st.caption(f"📊 **System Stats:** {chunk_count} text chunks indexed.")
    
    st.info("🚀 Powered by **Groq** & **Llama 3**")

# 3. Main Header
st.title("🧠 Intelligent Document Assistant")
st.caption("🚀 Powered by Groq & Llama 3 | Ask me anything about your PDF")

# Initialize chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! 👋 I'm ready to help. Ask me anything about your documents."}]

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Optional: If it was an image upload message, show the image in the sidebar or expander? 
        # For now, we just keep the text log.

# ... inside your chat input block ...
if query := st.chat_input("Ask me anything..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 1. Find the best context
    with st.spinner("🧠 Thinking and searching..."):
        if not os.path.exists("vector_db/index.faiss"):
            st.error("Error: Knowledge base not found. Please run 'python setup.py' first.")
            st.stop()
            
        indices = retrieve_context(query)
        
        # Load the actual text chunks saved during setup
        if os.path.exists("data/chunks.json"):
            with open("data/chunks.json", "r") as f:
                all_chunks = json.load(f)
            retrieved_context = [all_chunks[i] for i in indices if i < len(all_chunks)]
        else:
            retrieved_context = ["Error: Knowledge base not found. Please run setup.py first."]
        
    # 2. Get the answer
    response = generate_answer(query, retrieved_context, temperature, max_tokens)
    
    with st.chat_message("assistant"):
        st.markdown(response)
        # Feature: Show sources
        with st.expander("📚 View Source Context"):
            for i, chunk in enumerate(retrieved_context):
                st.info(f"**Source {i+1}:** {chunk}")
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})