# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# Model config
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0

# Retriever config
VECTORSTORE_PATH = "aviation_reports_index"
VECTORSTORE_ALLOW_DANGEROUS = True
SEMANTIC_K = 5
BM25_K = 5
ENSEMBLE_WEIGHTS = [0.6, 0.4]

# Database config
DB_PATH = "aviation_rag.db"

# Streamlit config
PAGE_TITLE = "Aviation RAG Chat"
PAGE_LAYOUT = "wide"