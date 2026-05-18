# src/generators.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import config

def setup_rag_chain():
    """Setup RAG generation chain"""
    
    prompt = ChatPromptTemplate.from_template(
        """You are an expert aviation accident investigator assistant. 
Your role is to provide comprehensive, detailed answers about NTSB aviation accident investigations.

IMPORTANT: Provide detailed, thorough answers. Include:
- Key facts and timeline
- Aircraft information
- Contributing factors
- Safety findings and recommendations
- Investigation details
- Any other relevant information from the context

Use the following pieces of retrieved context to answer the question comprehensively.
If you don't know specific details, say so clearly.

Context: {context}

Question: {question}

Provide a detailed and comprehensive answer:"""
    )
    
    llm = ChatOpenAI(
        model_name=config.MODEL_NAME, 
        temperature=config.TEMPERATURE,
        max_tokens=1000  
    )
    
    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain

rag_chain = setup_rag_chain()