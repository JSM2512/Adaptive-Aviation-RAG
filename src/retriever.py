# src/retriever.py
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import config

def setup_retrievers():
    """Setup hybrid retriever combining semantic and BM25 search"""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        config.VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=config.VECTORSTORE_ALLOW_DANGEROUS
    )
    
    semantic_retriever = vectorstore.as_retriever(
        search_kwargs={"k": config.SEMANTIC_K}
    )
    
    aviation_docs = list(vectorstore.docstore._dict.values())
    
    bm25_retriever = BM25Retriever.from_documents(aviation_docs)
    bm25_retriever.k = config.BM25_K
    
    retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=config.ENSEMBLE_WEIGHTS
    )
    
    return retriever

retriever = setup_retrievers()