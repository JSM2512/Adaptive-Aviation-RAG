# src/routers.py
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import config

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

def setup_question_router():
    """Setup question router"""
    llm = ChatOpenAI(model=config.MODEL_NAME, temperature=config.TEMPERATURE)
    structured_llm_router = llm.with_structured_output(RouteQuery)
    
    system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains NTSB (National Transportation Safety Board) aviation accident reports from 2023-2024 in the United States.
Reports include incident details, aircraft information, investigation findings, and accident numbers (e.g., DCA26WA202).
Use the vectorstore for questions about:
- Specific NTSB accident reports or report numbers
- Aviation accidents in the US
- Aircraft incident details and investigations
- Safety findings and recommendations
Otherwise, use web-search for current news or information outside the vectorstore."""
    
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ])
    
    question_router = route_prompt | structured_llm_router
    return question_router

question_router = setup_question_router()