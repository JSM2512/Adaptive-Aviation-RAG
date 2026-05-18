# src/graph.py
from typing import List, Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START

from .retriever import retriever
from .routers import question_router
from .graders import retrieval_grader, hallucination_grader, answer_grader
from .generators import rag_chain
from .question_rewriter import question_rewriter
from .web_search import web_search_tool
from langchain_core.documents import Document

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    generation: str
    documents: List[str]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    retry_count: int

# ==================== NODES ====================

def retrieve(state):
    """Retrieve documents"""
    print("---RETRIEVE---")
    question = state["question"]
    retry_count = state.get("retry_count", 0)

    if state["messages"]:
        history_context = "\n".join([f"{m.type}: {m.content}" for m in state["messages"][-4:]])
        rewritten_q = question_rewriter.invoke({
            "question": f"Previous context: {history_context}\n\nNew question: {question}"
        })
        question = rewritten_q

    documents = retriever.invoke(question)
    return {
        "documents": documents, 
        "question": question,
        "retry_count": retry_count
    }

def generate(state):
    """Generate answer"""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    retry_count = state.get("retry_count", 0)

    generation = rag_chain.invoke({
        "context": documents, 
        "question": question
    })

    return {
        "documents": documents, 
        "question": question, 
        "generation": generation, 
        "messages": [HumanMessage(content=question), AIMessage(content=generation)],
        "retry_count": retry_count
    }

def grade_documents(state):
    """Determines whether the retrieved documents are relevant to the question."""
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    retry_count = state.get("retry_count", 0)

    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({
            "question": question, 
            "document": d.page_content
        })
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")

    return {
        "documents": filtered_docs, 
        "question": question,
        "retry_count": retry_count
    }

def transform_query(state):
    """Transform the query to produce a better question."""
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    retry_count = state.get("retry_count", 0) + 1
    
    print(f"---RETRY COUNT: {retry_count}/1---")
    
    better_question = question_rewriter.invoke({"question": question})
    return {
        "documents": documents, 
        "question": better_question,
        "retry_count": retry_count
    }

def web_search(state):
    """Web search based on the re-phrased question."""
    print("---WEB SEARCH---")
    question = state["question"]
    retry_count = state.get("retry_count", 0)

    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    return {
        "documents": web_results, 
        "question": question,
        "retry_count": retry_count
    }

# ==================== EDGES ====================

def route_question(state):
    """Route question to web search or RAG."""
    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_generate(state):
    """Determines whether to generate an answer, or re-generate a question."""
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    retry_count = state.get("retry_count", 0)

    print(f"---CURRENT RETRY COUNT: {retry_count}---")

    if not filtered_documents:
        if retry_count >= 1:
            print("---MAX RETRIES REACHED (1), GENERATING ANSWER---")
            return "generate"
        else:
            print("---NO RELEVANT DOCS, RETRYING QUERY TRANSFORMATION---")
            return "transform_query"
    else:
        print("---DOCUMENTS FOUND, GENERATING---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """Determines whether the generation is grounded in the document and answers question."""
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({
        "documents": documents, 
        "generation": generation
    })
    grade = score.binary_score

    if grade == "yes":
        print("---GENERATION IS GROUNDED---")
        score = answer_grader.invoke({
            "question": question, 
            "generation": generation
        })
        grade = score.binary_score
        if grade == "yes":
            print("---GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---GENERATION DOES NOT ADDRESS QUESTION---")
            return "useful"  
    else:
        print("---GENERATION NOT GROUNDED, BUT ACCEPTING---")
        return "useful"  

# ==================== BUILD GRAPH ====================

def build_graph():
    """Build and compile the LangGraph workflow"""
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("web_search", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    # Build graph
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "useful": END,
            "not useful": END,
            "not supported": END,
        },
    )

    # Compile with checkpointer
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    
    return app

# Initialize graph
app = build_graph()