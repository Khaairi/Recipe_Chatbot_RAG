import streamlit as st
from rag_handler import RAGHandlerOllamaReqDocling

# Page Config
st.set_page_config(page_title="Chef Bot RAG", layout="wide")

# Initialize Session State
if "rag" not in st.session_state:
    st.session_state.rag = RAGHandlerOllamaReqDocling()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Setup
with st.sidebar:
    st.header("📚 Cookbook Knowledge Base")
    uploaded_file = st.file_uploader("Upload a Recipe PDF", type="pdf")
    
    if uploaded_file and st.button("Process Cookbook"):
        with st.spinner("Chef is reading the cookbook..."):
            status = st.session_state.rag.process_pdf(uploaded_file)
            if status == "success":
                st.success("Cookbook learned! You can now ask questions.")
            elif status == "exists":
                st.info(f"'{uploaded_file.name}' is already in the cookbook. Ready to ask!")
            else:
                st.error("Failed to process the cookbook.")

# Main Chat Interface
st.title("👨‍🍳 Chef Bot (Recipe RAG)")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if "sources" in message:
            with st.expander("📚 Source Recipes Used"):
                for source in message["sources"]:
                    st.caption(f"Page: {source['page']}")
                    st.text(source['content'])

# User Input
if prompt := st.chat_input("Ask about a recipe..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate Response
    try:
        with st.spinner("Chef is thinking..."):
            answer, sources = st.session_state.rag.query(prompt)

            clean_sources = []
            for doc in sources:
                clean_sources.append({
                    "page": doc.metadata.get("page", "Unknown"),
                    "content": doc.page_content
                })
            
            # Display assistant message
            with st.chat_message("assistant"):
                st.markdown(answer)

                with st.expander("📚 Source Recipes Used"):
                    for idx, src in enumerate(clean_sources):
                        st.markdown(f"**Source {idx+1} (Page {src['page']})**")
                        st.info(src['content'])
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": clean_sources
            })
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Did you upload a cookbook PDF yet?")