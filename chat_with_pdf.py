import streamlit as st
import os
import tempfile

from session_4_rag_backend import (
    setup_api_key,
    upload_pdf,
    parse_pdf,
    create_document_chunks,
    init_embedding_model,
    store_embeddings,
    query_with_full_context
)

st.set_page_config(
    page_title="RAG Chatbot with Gemini",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

def main():
    with st.sidebar:
        st.title("RAG Chatbot")
        st.subheader("Configuration")

        api_key = st.text_input("Enter Gemini API Key:", type="password")
        if api_key and st.button("Set API Key"):
            setup_api_key(api_key)
            st.success("API Key set successfully!")

        st.divider()

        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

        if uploaded_files and st.button("Process Documents"):
            process_documents(uploaded_files)

        if st.session_state.processed_files:
            st.subheader("Processed Documents")
            for file in st.session_state.processed_files:
                st.write(f"- {file}")

        st.divider()

        with st.expander("Advanced Options"):
            st.slider("Number of chunks to retrieve (k)", min_value=1, max_value=10, value=3, key="k_value")
            st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1, key="temperature")

    st.title("Retrieval Augmented Generation Chatbot")

    if st.session_state.vectorstore is None:
        st.info("Please upload and process documents to start chatting.")
        with st.expander("How to use this app"):
            st.markdown("""
            1. Enter your Gemini API Key in the sidebar  
            2. Upload one or more PDF documents  
            3. Click "Process Documents" to analyze them  
            4. Ask questions about the documents in the chat  
            """)
    else:
        display_chat()
        user_query = st.chat_input("Ask a question about your documents...")
        if user_query:
            handle_user_query(user_query)

def process_documents(uploaded_files):
    try:
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        if st.session_state.embedding_model is None:
            status_text.text("Initializing embedding model...")
            st.session_state.embedding_model = init_embedding_model()
            if st.session_state.embedding_model is None:
                st.sidebar.error("Failed to initialize embedding model.")
                return

        all_chunks = []
        processed_file_names = []

        for i, uploaded_file in enumerate(uploaded_files):
            progress_bar.progress((i + 1) / len(uploaded_files))
            status_text.text(f"Processing {uploaded_file.name}...")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                pdf_path = tmp_file.name

            pdf_file = upload_pdf(pdf_path)
            text = parse_pdf(pdf_file)
            if not text:
                st.sidebar.warning(f"Failed to extract text from {uploaded_file.name}")
                continue

            chunks = create_document_chunks(text)
            if not chunks:
                st.sidebar.warning(f"Failed to create chunks from {uploaded_file.name}")
                continue

            for chunk in chunks:
                all_chunks.append({"content": chunk, "source": uploaded_file.name})

            processed_file_names.append(uploaded_file.name)
            os.unlink(pdf_path)

        if all_chunks:
            texts = [chunk["content"] for chunk in all_chunks]
            vectorstore = store_embeddings(
                 st.session_state.embedding_model,
                texts,
                persist_directory="./streamlit_chroma_db",
                metadatas=metadatas
            )
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.processed_files = processed_file_names
                status_text.text("Processing complete!")
                st.sidebar.success("Documents processed successfully.")
            else:
                st.sidebar.error("Failed to create vector database.")
        else:
            st.sidebar.error("No valid chunks found in documents.")

        progress_bar.empty()
        status_text.empty()
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")

def handle_user_query(query):
    if st.session_state.vectorstore is None:
        st.error("Please process documents before asking questions.")
        return

    st.session_state.conversation.append({"role": "user", "content": query})
    thinking = st.empty()
    thinking.info("🤔 Thinking...")

    try:
        response, context, _ = query_with_full_context(
            query,
            st.session_state.vectorstore,
            k=st.session_state.k_value,
            temperature=st.session_state.temperature
        )
        st.session_state.conversation.append({
            "role": "assistant",
            "content": response,
            "context": context
        })
        thinking.empty()
        display_chat()
    except Exception as e:
        thinking.empty()
        st.session_state.conversation.append({"role": "assistant", "content": f"Error: {str(e)}"})
        display_chat()

def display_chat():
    for message in st.session_state.conversation:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                if "context" in message and message["context"]:
                    with st.expander("View source context"):
                        st.text(message["context"])

if __name__ == "__main__":
    main()
