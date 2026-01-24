import os
import tempfile
from typing import Optional, List
from langchain_community.document_loaders import PyPDFLoader
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_qdrant import QdrantVectorStore
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from database import get_db_instance
from config import app_config, api_config
import requests
import json
# from langchain_ollama import ChatOllama

class RAGHandlerGemini:
    """Handles RAG operations for document-based question answering."""
    
    def __init__(self):
        """
        Initialize RAG handler.
        
        Args:
            gemini_api_key: API key for Google Gemini
            collection_name: Name for Qdrant collection
        """
        self.gemini_key = api_config.gemini_key
        self.collection_name = app_config.collection_name
        self.chain = None
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=app_config.embedding_model
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=app_config.chunk_size,
            chunk_overlap=app_config.chunk_overlap
        )
    
    def process_pdf(self, uploaded_file) -> bool:
        """
        Process uploaded PDF file and create RAG chain.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            True if processing successful, False otherwise
        """
        try:
            db = get_db_instance()
            real_filename = uploaded_file.name

            # Create new collection
            db.create_collection_if_not_exists(collection_name=self.collection_name)

            if db.document_exists(self.collection_name, real_filename):
                self._create_chain()
                return "exists"

            # Save temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # Load and split document
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            for doc in documents:
                doc.metadata["source"] = real_filename

            splits = self.text_splitter.split_documents(documents)
            
            # Setup vector store
            self._add_to_vector_store(splits)
            
            # Create RAG chain
            self._create_chain()
            
            # Cleanup
            os.remove(tmp_path)
            
            return "success"
            
        except Exception as e:
            raise Exception(f"Failed to process PDF: {str(e)}")
    
    def _add_to_vector_store(self, documents):
        """Setup Qdrant vector store with documents."""
        db = get_db_instance()
        
        # Create and populate vector store
        if db.type == "qdrant":
            vectorstore = QdrantVectorStore(
                client=db.client,
                embedding=self.embeddings,
                collection_name=self.collection_name
            )
            # Add documents to Qdrant
            vectorstore.add_documents(documents=documents)
            
        elif db.type == "chroma":
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=db.persist_dir
            )
        
        self.retriever = vectorstore.as_retriever()
    
    def _create_chain(self):
        """Create the RAG chain for question answering."""
        if not hasattr(self, 'retriever'):
            self._initialize_retriever_from_existing()

        system_prompt = (
            "You are a helpful cooking assistant. "
            "Use the provided recipe context to answer the user's culinary questions. "
            "You must strictly follow the output format below for every recipe."
            "\n\n"
            "If the user asks 'what can I cook with [ingredients]?', check the recipes strictly. "
            "Provide a detailed and helpful answer, including: Recipe name and description, COMPLETE Ingredients list, Step-by-step instructions"
            "If the answer is not in the cookbook, politely say you don't have that recipe and offer general cooking advice if applicable. "
            "Format your answer with clear headings for Ingredients and Instructions."
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        llm = ChatGoogleGenerativeAI(
            model=app_config.llm_model,
            google_api_key=self.gemini_key,
            temperature=0.1
        )
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        self.chain = create_retrieval_chain(self.retriever, question_answer_chain)
    
    def query(self, question: str) -> str:
        """
        Query the RAG system with a question.
        
        Args:
            question: Question to ask about the document
            
        Returns:
            Answer from the RAG system
        """
        if not self.chain:
            raise ValueError("RAG chain not initialized. Process a PDF first.")
        
        response = self.chain.invoke({"input": question})
        return response["answer"], response["context"]

    def _initialize_retriever_from_existing(self):
        """Reconnects to DB if we skipped processing."""
        db = get_db_instance()
        if db.type == "qdrant":
            vectorstore = QdrantVectorStore(
                client=db.client, 
                embedding=self.embeddings, 
                collection_name=self.collection_name
            )
        else:
            vectorstore = Chroma(
                embedding_function=self.embeddings,
                collection_name=self.collection_name, 
                persist_directory=db.persist_dir
            )
        self.retriever = vectorstore.as_retriever()
    
    def is_ready(self) -> bool:
        """Check if RAG system is ready to answer questions."""
        return self.chain is not None
    
    def get_document_name(self) -> Optional[str]:
        """Get the name of the currently loaded document."""
        return self.current_document
    
class RAGHandlerOllama:
    """Handles RAG operations for document-based question answering."""
    
    def __init__(self):
        """
        Initialize RAG handler.
        
        Args:
            gemini_api_key: API key for Google Gemini
            collection_name: Name for Qdrant collection
        """
        self.collection_name = app_config.collection_name
        self.chain = None
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=app_config.embedding_model
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=app_config.chunk_size,
            chunk_overlap=app_config.chunk_overlap
        )
    
    def process_pdf(self, uploaded_file) -> bool:
        """
        Process uploaded PDF file and create RAG chain.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            True if processing successful, False otherwise
        """
        try:
            db = get_db_instance()
            real_filename = uploaded_file.name

            db.create_collection_if_not_exists(collection_name=self.collection_name)

            if db.document_exists(self.collection_name, real_filename):
                self._create_chain()
                return "exists"
            # Save temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # Load and split document
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            for doc in documents:
                doc.metadata["source"] = real_filename

            splits = self.text_splitter.split_documents(documents)
            
            # Setup vector store
            self._add_to_vector_store(splits)
            
            # Create RAG chain
            self._create_chain()
            
            # Cleanup
            os.remove(tmp_path)
            
            return "success"
            
        except Exception as e:
            raise Exception(f"Failed to process PDF: {str(e)}")
    
    def _add_to_vector_store(self, documents):
        """Setup Qdrant vector store with documents."""
        db = get_db_instance()
        
        # Create and populate vector store
        if db.type == "qdrant":
            vectorstore = QdrantVectorStore(
                client=db.client,
                embedding=self.embeddings,
                collection_name=self.collection_name
            )
            # Add documents to Qdrant
            vectorstore.add_documents(documents=documents)
            
        elif db.type == "chroma":
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=db.persist_dir
            )
        
        self.retriever = vectorstore.as_retriever()
    
    def _create_chain(self):
        """Create the RAG chain for question answering."""
        if not hasattr(self, 'retriever'):
            self._initialize_retriever_from_existing()

        system_prompt = (
            "You are a helpful cooking assistant. "
            "Use the provided recipe context to answer the user's culinary questions. "
            "You must strictly follow the output format below for every recipe."
            "\n\n"
            "If the user asks 'what can I cook with [ingredients]?', check the recipes strictly. "
            "Provide a detailed and helpful answer, including: Recipe name and description, COMPLETE Ingredients list, Step-by-step instructions"
            "If the answer is not in the cookbook, politely say you don't have that recipe and offer general cooking advice if applicable. "
            "Format your answer with clear headings for Ingredients and Instructions."
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        llm = ChatOllama(
            model=app_config.ollama_model,
            base_url=api_config.ollama_base_url,
            temperature=0.1
        )
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        self.chain = create_retrieval_chain(self.retriever, question_answer_chain)
    
    def query(self, question: str) -> str:
        """
        Query the RAG system with a question.
        
        Args:
            question: Question to ask about the document
            
        Returns:
            Answer from the RAG system
        """
        if not self.chain:
            raise ValueError("RAG chain not initialized. Process a PDF first.")
        
        response = self.chain.invoke({"input": question})
        return response["answer"], response["context"]
    
    def _initialize_retriever_from_existing(self):
        """Reconnects to DB if we skipped processing."""
        db = get_db_instance()
        if db.type == "qdrant":
            vectorstore = QdrantVectorStore(
                client=db.client, 
                embedding=self.embeddings, 
                collection_name=self.collection_name
            )
        else:
            vectorstore = Chroma(
                embedding_function=self.embeddings,
                collection_name=self.collection_name, 
                persist_directory=db.persist_dir
            )
        self.retriever = vectorstore.as_retriever()
    
    def is_ready(self) -> bool:
        """Check if RAG system is ready to answer questions."""
        return self.chain is not None
    
    def get_document_name(self) -> Optional[str]:
        """Get the name of the currently loaded document."""
        return self.current_document
    
class RAGHandlerOllamaReq:
    """Handles RAG operations for document-based question answering."""
    
    def __init__(self):
        """
        Initialize RAG handler.
        
        Args:
            gemini_api_key: API key for Google Gemini
            collection_name: Name for Qdrant collection
        """
        self.collection_name = app_config.collection_name
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=app_config.embedding_model
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=app_config.chunk_size,
            chunk_overlap=app_config.chunk_overlap
        )
    
    def process_pdf(self, uploaded_file) -> str:
        """
        Process uploaded PDF file and create RAG chain.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            True if processing successful, False otherwise
        """
        try:
            db = get_db_instance()
            real_filename = uploaded_file.name

            db.create_collection_if_not_exists(collection_name=self.collection_name)

            if db.document_exists(self.collection_name, real_filename):
                return "exists"

            # Save temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # Load and split document
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            for doc in documents:
                doc.metadata["source"] = real_filename

            splits = self.text_splitter.split_documents(documents)
            
            # Setup vector store
            self._add_to_vector_store(splits)
            
            # Cleanup
            os.remove(tmp_path)
            
            return "success"
            
        except Exception as e:
            raise Exception(f"Failed to process PDF: {str(e)}")
    
    def _add_to_vector_store(self, documents):
        """Setup Qdrant vector store with documents."""
        db = get_db_instance()
        
        # Create and populate vector store
        if db.type == "qdrant":
            vectorstore = QdrantVectorStore(
                client=db.client,
                embedding=self.embeddings,
                collection_name=self.collection_name
            )
            # Add documents to Qdrant
            vectorstore.add_documents(documents=documents)
            
        elif db.type == "chroma":
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=db.persist_dir
            )
        
        self.retriever = vectorstore.as_retriever()

    def query(self, question: str):
        """
        Query the RAG system with a question.
        
        Args:
            question: Question to ask about the document
            
        Returns:
            Answer from the RAG system
        """
        if not hasattr(self, 'retriever'):
            self._initialize_retriever_from_existing()

        source_docs = self.retriever.invoke(question)

        context_text = "\n\n---\n\n".join([doc.page_content for doc in source_docs])

        system_prompt = (
            "You are a helpful cooking assistant. "
            "Use the provided recipe context to answer the user's culinary questions. "
            "You must strictly follow the output format below for every recipe."
            "\n\n"
            "If the user asks 'what can I cook with [ingredients]?', check the recipes strictly. "
            "Provide a detailed and helpful answer, including: Recipe name and description, COMPLETE Ingredients list, Step-by-step instructions"
            "If the answer is not in the cookbook, politely say you don't have that recipe and offer general cooking advice if applicable. "
            "Format your answer with clear headings for Ingredients and Instructions."
        )

        user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}"

        payload = {
            "model": app_config.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }

        try:
            response = requests.post(
                api_config.ollama_url, 
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result_json = response.json()
            answer = result_json["message"]["content"]
            
            return answer, source_docs

        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama. Is the app running?", []
        except Exception as e:
            return f"Error generating answer: {str(e)}", []
        
    def _initialize_retriever_from_existing(self):
        """Reconnects to DB if we skipped processing."""
        db = get_db_instance()
        if db.type == "qdrant":
            vectorstore = QdrantVectorStore(
                client=db.client, 
                embedding=self.embeddings, 
                collection_name=self.collection_name
            )
        else:
            vectorstore = Chroma(
                embedding_function=self.embeddings,
                collection_name=self.collection_name, 
                persist_directory=db.persist_dir
            )
        self.retriever = vectorstore.as_retriever()

class RAGHandlerOllamaReqDocling:
    """Handles RAG operations for document-based question answering."""
    
    def __init__(self):
        """
        Initialize RAG handler.
        
        Args:
            gemini_api_key: API key for Google Gemini
            collection_name: Name for Qdrant collection
        """
        self.collection_name = app_config.collection_name
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=app_config.embedding_model
        )
        
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Title"),
                ("##", "Section"),
                ("###", "Subsection"),
            ]
        )

        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=app_config.chunk_size,
            chunk_overlap=app_config.chunk_overlap
        )
    
    def process_pdf(self, uploaded_file) -> str:
        """
        Process uploaded PDF file and create RAG chain.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            True if processing successful, False otherwise
        """
        try:
            db = get_db_instance()
            real_filename = uploaded_file.name

            db.create_collection_if_not_exists(collection_name=self.collection_name)

            if db.document_exists(self.collection_name, real_filename):
                return "exists"

            # Save temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # Load and split document
            loader = DoclingLoader(
                file_path=tmp_path,
                export_type=ExportType.MARKDOWN
            )
            docs = loader.load()
            final_splits = []
            for doc in docs:
                md_splits = self.markdown_splitter.split_text(doc.page_content)
                
                for split in md_splits:
                    split.metadata["source"] = real_filename
                    if "dl_meta" in doc.metadata:
                        split.metadata["doc_meta"] = doc.metadata["dl_meta"]

                rec_splits = self.recursive_splitter.split_documents(md_splits)
                final_splits.extend(rec_splits)
            
            # Setup vector store
            self._add_to_vector_store(final_splits)
            
            # Cleanup
            os.remove(tmp_path)
            
            return "success"
            
        except Exception as e:
            raise Exception(f"Failed to process PDF: {str(e)}")
    
    def _add_to_vector_store(self, documents):
        """Setup Qdrant vector store with documents."""
        db = get_db_instance()
        
        # Create and populate vector store
        if db.type == "qdrant":
            vectorstore = QdrantVectorStore(
                client=db.client,
                embedding=self.embeddings,
                collection_name=self.collection_name
            )
            # Add documents to Qdrant
            vectorstore.add_documents(documents=documents)
            
        elif db.type == "chroma":
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=db.persist_dir
            )
        
        self.retriever = vectorstore.as_retriever()

    def query(self, question: str):
        """
        Query the RAG system with a question.
        
        Args:
            question: Question to ask about the document
            
        Returns:
            Answer from the RAG system
        """
        if not hasattr(self, 'retriever'):
            self._initialize_retriever_from_existing()

        source_docs = self.retriever.invoke(question)

        context_text = "\n\n---\n\n".join([doc.page_content for doc in source_docs])

        system_prompt = (
            "You are a helpful cooking assistant. "
            "Use the provided recipe context to answer the user's culinary questions. "
            "You must strictly follow the output format below for every recipe."
            "\n\n"
            "If the user asks 'what can I cook with [ingredients]?', check the recipes strictly. "
            "Provide a detailed and helpful answer, including: Recipe name and description, COMPLETE Ingredients list, Step-by-step instructions"
            "If the answer is not in the cookbook, politely say you don't have that recipe and offer general cooking advice if applicable. "
            "Format your answer with clear headings for Ingredients and Instructions."
        )

        user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}"

        payload = {
            "model": app_config.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }

        try:
            response = requests.post(
                api_config.ollama_url, 
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result_json = response.json()
            answer = result_json["message"]["content"]
            
            return answer, source_docs

        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama. Is the app running?", []
        except Exception as e:
            return f"Error generating answer: {str(e)}", []
        
    def _initialize_retriever_from_existing(self):
        """Reconnects to DB if we skipped processing."""
        db = get_db_instance()
        if db.type == "qdrant":
            vectorstore = QdrantVectorStore(
                client=db.client, 
                embedding=self.embeddings, 
                collection_name=self.collection_name
            )
        else:
            vectorstore = Chroma(
                embedding_function=self.embeddings,
                collection_name=self.collection_name, 
                persist_directory=db.persist_dir
            )
        self.retriever = vectorstore.as_retriever()
    