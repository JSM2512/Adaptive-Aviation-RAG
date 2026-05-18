# src/web_search.py
from langchain_community.tools.tavily_search import TavilySearchResults
import config

def setup_web_search():
    """Setup web search tool"""
    web_search_tool = TavilySearchResults(k=3)
    return web_search_tool

web_search_tool = setup_web_search()