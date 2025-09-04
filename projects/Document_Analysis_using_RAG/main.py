import streamlit as st
import asyncio
import tempfile
import os
from pathlib import Path
from dotenv import load_dotenv

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.settings import Settings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.response_synthesizers import CompactAndRefine
import nest_asyncio

# Apply nest_asyncio for Streamlit compatibility
nest_asyncio.apply()

# Load the environment variables from the .env file
load_dotenv()

# Configure API key
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]


@st.cache_resource
def initialize_models():
    """Initialize LLM and embedding models"""
    llm = GoogleGenAI(model="models/gemini-2.0-flash-exp")
    embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004")
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    return llm, embed_model

class DocumentChatApp:
    def __init__(self):
        self.llm, self.embed_model = initialize_models()
        self.index = None
        self.documents = []

    async def ingest_documents(self, file_paths, original_names):
        """Ingest uploaded documents with original file names"""
        # Create mapping from temp file paths to original names
        file_mapping = {file_path: original_name for file_path, original_name in zip(file_paths, original_names)}
        
        documents = SimpleDirectoryReader(input_files=file_paths).load_data()
        
        # Update metadata with original file names for all documents
        for doc in documents:
            # Get the source file path from metadata
            source_file = doc.metadata.get('file_path', '')
            
            # Find matching original name
            for temp_path, original_name in file_mapping.items():
                if temp_path in source_file or source_file in temp_path:
                    doc.metadata['file_name'] = original_name
                    doc.metadata['original_file_name'] = original_name
                    break
        
        self.documents = documents
        self.index = VectorStoreIndex.from_documents(documents=documents)
        return len(documents)

    async def query_documents(self, query, top_k=3):
        """Query documents and return response with sources"""
        if not self.index:
            return None, []
        
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = await retriever.aretrieve(query)
        
        # Generate response
        synthesizer = CompactAndRefine(streaming=False)
        response = await synthesizer.asynthesize(query, nodes=nodes)
        
        # Extract source information
        sources = []
        for node in nodes:
            source_info = {
                'content': node.text[:200] + "...",
                'score': node.score,
                'metadata': node.metadata
            }
            sources.append(source_info)
        
        return response, sources

# Initialize the app
if 'chat_app' not in st.session_state:
    st.session_state.chat_app = DocumentChatApp()

st.title("📄 Document Chat")
st.write("Upload documents and chat with them using Gemini AI")

# Sidebar for document upload
with st.sidebar:
    st.header("📁 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx', 'md']
    )
    
    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            # Save uploaded files temporarily
            temp_files = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_files.append(tmp_file.name)
            
            # Process documents
            try:
                # Get original file names
                original_names = [f.name for f in uploaded_files]
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                doc_count = loop.run_until_complete(
                    st.session_state.chat_app.ingest_documents(temp_files, original_names)
                )
                st.session_state.documents_loaded = True
            except Exception as e:
                st.error(f"Error processing documents: {e}")
            finally:
                # Clean up temp files
                for temp_file in temp_files:
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
    
    # Document status
    if hasattr(st.session_state, 'documents_loaded'):
        st.success("📚 Documents ready for chat!")
    else:
        st.info("👆 Upload documents to start chatting")

# Main chat interface
if hasattr(st.session_state, 'documents_loaded'):
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📖 Sources & References"):
                    for i, source in enumerate(message["sources"]):
                        st.write(f"**Source {i+1}** (Relevance: {source['score']:.3f})")
                        st.write(f"📄 **File:** {source['metadata'].get('file_name', 'Unknown')}")
                        if 'page_label' in source['metadata']:
                            st.write(f"📄 **Page:** {source['metadata']['page_label']}")
                        st.write(f"**Content:** {source['content']}")
                        st.divider()

    # Chat input
    if prompt := st.chat_input("Ask about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response, sources = loop.run_until_complete(
                        st.session_state.chat_app.query_documents(prompt)
                    )
                    
                    if response:
                        st.markdown(str(response))
                        
                        # Display sources
                        if sources:
                            with st.expander("📖 Sources & References"):
                                for i, source in enumerate(sources):
                                    st.write(f"**Source {i+1}** (Relevance: {source['score']:.3f})")
                                    st.write(f"📄 **File:** {source['metadata'].get('file_name', 'Unknown')}")
                                    if 'page_label' in source['metadata']:
                                        st.write(f"📄 **Page:** {source['metadata']['page_label']}")
                                    st.write(f"**Content:** {source['content']}")
                                    st.divider()
                        
                        # Save to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": str(response),
                            "sources": sources
                        })
                    else:
                        st.error("No response generated")
                        
                except Exception as e:
                    st.error(f"Error: {e}")

else:
    st.info("👈 Please upload and process documents first to start chatting!")

# Footer
st.markdown("---")
st.markdown("*Powered by Gemini AI and LlamaIndex*")