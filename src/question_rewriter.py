# src/question_rewriter.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import config

def setup_question_rewriter():
    """Setup question rewriter for query optimization"""
    llm = ChatOpenAI(model=config.MODEL_NAME, temperature=config.TEMPERATURE)
    
    system = """You a question re-writer that converts an input question to a better version that is optimized 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    
    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Here is the initial question: \\n\\n {question} \\n Formulate an improved question."),
    ])
    
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    return question_rewriter

question_rewriter = setup_question_rewriter()