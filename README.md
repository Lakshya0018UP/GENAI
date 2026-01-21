# GenAI Knowledge Assistant 

A Retrieval-Augmented Generation (RAG) assistant built with LangChain, Google Gemini, and ChromaDB. This tool allows users to chat with their PDF documents and receive fact-grounded answers with source citations.

##  Features
- **Fact-Grounded Responses:** Only answers based on the provided PDF context.
- **Source Attribution:** Shows exactly which file and page number the information came from.
- **Persistent Storage:** Uses ChromaDB to store document embeddings for fast retrieval.
- **Modern UI:** Clean, web-based chat interface built with Streamlit.

## Tech Stack
- **LLM:** Google Gemini (Generative AI)
- **Framework:** LangChain
- **Vector Database:** ChromaDB
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
- **Frontend:** Streamlit

##  Prerequisites
- Python 3.10+
- A Google Gemini API Key

##  Installation & Setup
1. Clone the repository:
   ```bash
   git clone <https://github.com/Lakshya0018UP/GENAI>
   cd GENAI

## 2. Create `.env` file

```GOOGLE_API_KEY=your_api_key_here```

## 3. Install Dependencies

```uv add -r requirements.txt```

## 4.Place your pdf in the data folderand run 

```uv run ingest.py```

## 5. Run the Application

```streamlit run app.py```
