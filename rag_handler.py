import os
import tempfile
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
            # Save temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # Load and split document
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            splits = self.text_splitter.split_documents(documents)
            
            # Setup vector store
            self._setup_vector_store(splits)
            
            # Create RAG chain
            self._create_chain()
            
            # Cleanup
            os.remove(tmp_path)
            
            return True
            
        except Exception as e:
            raise Exception(f"Failed to process PDF: {str(e)}")
    
    def _setup_vector_store(self, documents):
        """Setup Qdrant vector store with documents."""
        db = get_db_instance()
        # Clear existing collection
        try:
            db.del_collection(collection_name=self.collection_name)
        except:
            pass
        
        # Create new collection
        db.create_collection_if_not_exists(collection_name=self.collection_name)
        
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
            # Save temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # Load and split document
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            splits = self.text_splitter.split_documents(documents)
            
            # Setup vector store
            self._setup_vector_store(splits)
            
            # Create RAG chain
            self._create_chain()
            
            # Cleanup
            os.remove(tmp_path)
            
            return True
            
        except Exception as e:
            raise Exception(f"Failed to process PDF: {str(e)}")
    
    def _setup_vector_store(self, documents):
        """Setup Qdrant vector store with documents."""
        db = get_db_instance()
        # Clear existing collection
        try:
            db.del_collection(collection_name=self.collection_name)
        except:
            pass
        
        # Create new collection
        db.create_collection_if_not_exists(collection_name=self.collection_name)
        
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
    
    def process_pdf(self, uploaded_file) -> bool:
        """
        Process uploaded PDF file and create RAG chain.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            True if processing successful, False otherwise
        """
        try:
            # Save temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # Load and split document
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            splits = self.text_splitter.split_documents(documents)
            
            # Setup vector store
            self._setup_vector_store(splits)
            
            # Cleanup
            os.remove(tmp_path)
            
            return True
            
        except Exception as e:
            raise Exception(f"Failed to process PDF: {str(e)}")
    
    def _setup_vector_store(self, documents):
        """Setup Qdrant vector store with documents."""
        db = get_db_instance()
        # Clear existing collection
        try:
            db.del_collection(collection_name=self.collection_name)
        except:
            pass
        
        # Create new collection
        db.create_collection_if_not_exists(collection_name=self.collection_name)
        
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

    def query(self, question: str) -> str:
        """
        Query the RAG system with a question.
        
        Args:
            question: Question to ask about the document
            
        Returns:
            Answer from the RAG system
        """

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
    