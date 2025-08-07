"""Agentic RAG system using LangGraph.

This agent can:
1. Decide whether to retrieve documents
2. Retrieve relevant information from indexed documents
3. Generate answers based on retrieved context
4. Grade the quality of retrieved documents
5. Decide whether to re-retrieve or answer
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, TypedDict

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END


class Configuration(TypedDict):
    """Configuration for the agentic RAG system."""
    
    model: str
    temperature: float
    max_tokens: int
    retrieval_threshold: float


@dataclass
class AgentState:
    """State for the agentic RAG agent."""
    
    question: str = ""
    messages: List[BaseMessage] = field(default_factory=list)
    documents: List[Document] = field(default_factory=list)
    answer: str = ""
    needs_retrieval: bool = True
    retrieval_attempts: int = 0
    max_retrieval_attempts: int = 2


class RAGSystem:
    """RAG system with document loading and retrieval capabilities."""
    
    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.documents_loaded = False
    
    async def initialize_documents(self):
        """Load and index documents from URLs."""
        if self.documents_loaded:
            return
        
        urls = [
            "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
            "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
            "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
        ]
        
        try:
            # Load documents
            loader = WebBaseLoader(urls)
            docs = await asyncio.to_thread(loader.load)
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)
            
            # Create vector store
            embeddings = OpenAIEmbeddings()
            self.vectorstore = await asyncio.to_thread(
                FAISS.from_documents, splits, embeddings
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
            self.documents_loaded = True
            
        except Exception as e:
            print(f"Error loading documents: {e}")
            # Fallback to mock retriever for testing
            self.retriever = None
    
    async def retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents."""
        if not self.documents_loaded:
            await self.initialize_documents()
        
        if self.retriever:
            try:
                docs = await asyncio.to_thread(self.retriever.invoke, query)
                return docs
            except Exception as e:
                print(f"Error during retrieval: {e}")
                return []
        else:
            # Mock documents for testing
            return [
                Document(
                    page_content=f"Mock document content related to: {query}",
                    metadata={"source": "test", "title": "Test Document"}
                )
            ]


# Global RAG system instance
rag_system = RAGSystem()


async def decide_retrieval(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Decide whether to retrieve documents or use existing ones."""
    
    configuration = config.get("configurable", {})
    model_name = configuration.get("model", "gpt-3.5-turbo")
    
    llm = ChatOpenAI(model=model_name, temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at deciding whether a question needs document retrieval.
        
        Look at the question and determine if it requires specific information from documents about:
        - Reward hacking in AI systems
        - Hallucination in language models  
        - Diffusion models for video generation
        
        Respond with exactly 'yes' if retrieval is needed, 'no' if it's a general question."""),
        ("human", "Question: {question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = await chain.ainvoke({"question": state.question})
        needs_retrieval = result.strip().lower() == "yes"
    except:
        needs_retrieval = True  # Default to retrieval if uncertain
    
    return {
        "needs_retrieval": needs_retrieval
    }


async def retrieve_documents(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Retrieve relevant documents based on the question."""
    
    if state.retrieval_attempts >= state.max_retrieval_attempts:
        return {"documents": state.documents}
    
    docs = await rag_system.retrieve_documents(state.question)
    
    return {
        "documents": docs,
        "retrieval_attempts": state.retrieval_attempts + 1,
        "needs_retrieval": False
    }


async def grade_documents(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Grade the relevance of retrieved documents."""
    
    if not state.documents:
        return {"needs_retrieval": True}
    
    configuration = config.get("configurable", {})
    model_name = configuration.get("model", "gpt-3.5-turbo")
    
    llm = ChatOpenAI(model=model_name, temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a grader assessing relevance of retrieved documents to a user question.
        
        Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.
        
        Document: {document}
        Question: {question}
        
        Respond with exactly 'yes' if relevant, 'no' if not relevant."""),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    relevant_docs = []
    
    for doc in state.documents[:3]:  # Grade top 3 docs
        try:
            result = await chain.ainvoke({
                "document": doc.page_content[:500],
                "question": state.question
            })
            if result.strip().lower() == "yes":
                relevant_docs.append(doc)
        except:
            relevant_docs.append(doc)  # Include if grading fails
    
    # If no relevant docs and haven't exceeded attempts, try retrieval again
    needs_more_retrieval = (
        len(relevant_docs) == 0 and 
        state.retrieval_attempts < state.max_retrieval_attempts
    )
    
    return {
        "documents": relevant_docs if relevant_docs else state.documents,
        "needs_retrieval": needs_more_retrieval
    }


async def generate_answer(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Generate an answer using retrieved documents."""
    
    configuration = config.get("configurable", {})
    model_name = configuration.get("model", "gpt-3.5-turbo")
    temperature = configuration.get("temperature", 0.7)
    max_tokens = configuration.get("max_tokens", 500)
    
    llm = ChatOpenAI(
        model=model_name, 
        temperature=temperature, 
        max_tokens=max_tokens
    )
    
    # Create context from documents
    if state.documents:
        context = "\n\n".join([
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in state.documents[:3]
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that answers questions based on provided context.
            
            Use the following context to answer the question. If the answer cannot be found in the context,
            say so clearly. Always cite which source you're using for your answer.
            
            Context:
            {context}"""),
            ("human", "{question}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        
        try:
            answer = await chain.ainvoke({
                "context": context,
                "question": state.question
            })
        except Exception as e:
            answer = f"I apologize, but I encountered an error generating the answer: {str(e)}"
    
    else:
        # No documents available
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Answer the question to the best of your ability."),
            ("human", "{question}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        
        try:
            answer = await chain.ainvoke({"question": state.question})
        except Exception as e:
            answer = f"I apologize, but I encountered an error: {str(e)}"
    
    # Update messages
    new_messages = state.messages + [
        HumanMessage(content=state.question),
        AIMessage(content=answer)
    ]
    
    return {
        "answer": answer,
        "messages": new_messages
    }


def should_retrieve(state: AgentState) -> Literal["retrieve_documents", "generate_answer"]:
    """Router function to decide next step."""
    if state.needs_retrieval and state.retrieval_attempts < state.max_retrieval_attempts:
        return "retrieve_documents"
    return "generate_answer"


def should_grade(state: AgentState) -> Literal["grade_documents", "generate_answer"]:
    """Router function after retrieval."""
    if state.documents and state.retrieval_attempts <= state.max_retrieval_attempts:
        return "grade_documents" 
    return "generate_answer"


# Build the graph
workflow = StateGraph(AgentState, config_schema=Configuration)

# Add nodes
workflow.add_node("decide_retrieval", decide_retrieval)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate_answer", generate_answer)

# Add edges
workflow.add_edge("__start__", "decide_retrieval")
workflow.add_conditional_edges(
    "decide_retrieval",
    should_retrieve,
    {
        "retrieve_documents": "retrieve_documents",
        "generate_answer": "generate_answer"
    }
)
workflow.add_conditional_edges(
    "retrieve_documents",
    should_grade,
    {
        "grade_documents": "grade_documents",
        "generate_answer": "generate_answer"
    }
)
workflow.add_conditional_edges(
    "grade_documents",
    should_retrieve,
    {
        "retrieve_documents": "retrieve_documents",
        "generate_answer": "generate_answer"
    }
)
workflow.add_edge("generate_answer", END)

# Compile the graph
graph = workflow.compile(name="Agentic RAG System")