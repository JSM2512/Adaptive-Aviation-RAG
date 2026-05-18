# src/graders.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import config

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

def setup_retrieval_grader():
    """Setup document relevance grader"""
    llm = ChatOpenAI(model=config.MODEL_NAME, temperature=config.TEMPERATURE)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    
    system = """You are a grader assessing relevance of a retrieved document to a user question. 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \\n\\n {document} \\n\\n User question: {question}"),
    ])
    
    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader

def setup_hallucination_grader():
    """Setup hallucination grader"""
    llm = ChatOpenAI(model=config.MODEL_NAME, temperature=config.TEMPERATURE)
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)
    
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Set of facts: \\n\\n {documents} \\n\\n LLM generation: {generation}"),
    ])
    
    hallucination_grader = hallucination_prompt | structured_llm_grader
    return hallucination_grader

def setup_answer_grader():
    """Setup answer quality grader"""
    llm = ChatOpenAI(model=config.MODEL_NAME, temperature=config.TEMPERATURE)
    structured_llm_grader = llm.with_structured_output(GradeAnswer)
    
    system = """You are a grader assessing whether an answer addresses / resolves a question 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""
    
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "User question: \\n\\n {question} \\n\\n LLM generation: {generation}"),
    ])
    
    answer_grader = answer_prompt | structured_llm_grader
    return answer_grader

retrieval_grader = setup_retrieval_grader()
hallucination_grader = setup_hallucination_grader()
answer_grader = setup_answer_grader()