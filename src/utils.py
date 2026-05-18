# src/utils.py
import os
from pathlib import Path

def ensure_env_vars():
    """Ensure required environment variables are set"""
    required_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")

def ensure_directories():
    """Ensure required directories exist"""
    Path("aviation_reports_index").mkdir(exist_ok=True)