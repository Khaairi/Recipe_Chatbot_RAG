# ChefBot RAG: Multimodal Recipe Assistant
**ChefBot** is an intelligent cooking assistant powered by **Retrieval-Augmented Generation (RAG)**. Unlike generic AI, this bot "learns" from your specific PDF cookbooks to give accurate culinary advice.

It features **Multimodal capabilities**, allowing you to upload photos of ingredients, which the bot identifies to suggest recipes from your knowledge base.

## Key Features

* **RAG (Retrieval-Augmented Generation):** Upload any PDF cookbook, and the bot will answer questions based strictly on *that* document.
* **Multimodal Input:** Upload photos of your fridge or pantry. The bot uses Computer Vision (Gemini Vision or Local LLaVA) to detect ingredients and suggest recipes.
* **Smart Parsing with Docling:** Uses **Docling** to intelligently parse PDFs, preserving document layout, tables, and ingredient lists better than standard text loaders.
* **Robust Database Fallback:** Tries to connect to **Qdrant Cloud** first. If the internet is down or the key is missing, it automatically falls back to a local **ChromaDB**, ensuring the app never crashes.
* **Hybrid AI Support:**
    * **Cloud:** Google Gemini Flash (High speed, huge context).
    * **Local:** Ollama (Llama 3.1 and LLaVA) for privacy.

## Tech Stack

* **Frontend:** Streamlit
* **Orchestration:** LangChain
* **Document Parsing:** Docling
* **Vector Database:** Qdrant (Cloud) + ChromaDB (Local Fallback)
* **LLM:** Google Gemini Flash / Ollama
* **Vision:** Gemini Vision / LLaVA

## Getting Started
### 1. Clone the Repository
```bash
git clone https://github.com/Khaairi/Cinebot.git
cd Recipe_Chatbot_Multimodal_RAG
```
### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Setup Environment Variables
Create a file named .env in the root folder and add your keys:
```bash
# Google Gemini (Required for Cloud LLM & Vision). Leave blank if use ollama
GEMINI_KEY="your_google_api_key_here"

# Qdrant Cloud (Optional - Will use local ChromaDB if empty)
QDRANT_URL="your_cluster_url_qdrant_here"
QDRANT_API_KEY="your_qdrant_key_here"
```
### 5. Setup Local AI (Optional)
If you want to run completely locally using **Ollama**:
1. Download [Ollama](https://ollama.com/download/windows)
2. Pull the models:
```bash
# For Text/Chat
ollama pull llama3.1
# For Vision (Image analysis)
ollama pull llava
```
### 6. Run the Application
```bash
streamlit run app.py
```

## Project Structure

```bash
Recipe_Chatbot_Multimodal_RAG/
├── app.py               # Main Streamlit UI application
├── rag_handler.py       # Core RAG logic (Loading, Splitting, Retrieval)
├── vision_handler.py    # Computer Vision logic (Ingredient Detection)
├── database.py          # DB connection manager (Handles Qdrant/Chroma switch)
├── config.py            # Centralized configuration
├── .env                 # API Keys
└── requirements.txt     # Python dependencies
```
