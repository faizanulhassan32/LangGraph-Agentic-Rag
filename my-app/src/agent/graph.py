from __future__ import annotations

import asyncio
import os
from typing import List, TypedDict, Any, Dict, Literal

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

class Configuration(TypedDict):
    model: str
    temperature: float
    max_tokens: int

class MultiHopState(TypedDict):
    original_question: str
    decomposed_questions: List[str]
    intermediate_answers: List[str]
    retrieved_docs: List[Document]
    rewritten_query: str
    current_question_index: int
    final_answer: str

class RAGSystem:
    def __init__(self):
        self.vectorstore = Chroma(
            persist_directory="./chroma_db_hotpotqa",
            embedding_function=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

    async def retrieve_documents(self, query: str) -> List[Document]:
        try:
            print(f"🔍 RETRIEVAL: Query = '{query}'")
            docs = await asyncio.to_thread(self.retriever.invoke, query)
            print(f"🔍 RETRIEVAL: Found {len(docs)} documents")
            return docs
        except Exception as e:
            print(f"❌ RETRIEVAL ERROR: {e}")
            return []

rag_system = RAGSystem()

async def decompose_question(state: MultiHopState, config: RunnableConfig) -> Dict[str, Any]:
    print("---NODE: Decomposing Question---")
    print(f"📝 INPUT: {state['original_question']}")
    
    configuration = config.get("configurable", {})
    llm = ChatOpenAI(model=configuration.get("model", "gpt-4o-mini-2024-07-18"), temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        """You are an expert at breaking down complex questions into a sequence of simpler questions.
        The user will provide a complex question, and you must generate a JSON object containing a list of sub-questions.
        Each sub-question should be answerable on its own and build upon the previous one.

        Example:
        Question: "Which movie, directed by the person who directed 'Jaws', stars an actor born in the same city as that director?"
        Output:
        {{
            "questions": [
                "Who directed the movie 'Jaws'?",
                "In which city was the director of 'Jaws' born?",
                "Which major actor was born in that same city?",
                "Which movie directed by the director of 'Jaws' stars that actor?"
            ]
        }}

        Now, decompose the following question:
        Question: {question}
        """
    )
    
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    
    result = await chain.ainvoke({"question": state["original_question"]})
    print(f"🎯 DECOMPOSED: {len(result['questions'])} questions")
    for i, q in enumerate(result['questions']):
        print(f"  {i+1}. {q}")
    
    return {
        "decomposed_questions": result["questions"],
        "current_question_index": 0,
        "intermediate_answers": [],
    }

async def rewrite_query(state: MultiHopState, config: RunnableConfig) -> Dict[str, Any]:
    print("---NODE: Rewriting Query---")
    current_index = state["current_question_index"]
    current_question = state["decomposed_questions"][current_index]
    print(f"📍 CURRENT Q{current_index+1}: {current_question}")

    if state["intermediate_answers"]:
        context = "\n".join(f"- {ans}" for ans in state["intermediate_answers"])
        
        configuration = config.get("configurable", {})
        llm = ChatOpenAI(model=configuration.get("model", "gpt-4o-mini-2024-07-18"), temperature=0)
        
        prompt = ChatPromptTemplate.from_template(
            """You are a query rewriting expert. Your task is to rewrite a follow-up question to be self-contained,
            using the context of previous answers.

            Given these facts:
            {context}

            Rewrite the following question:
            Question: {question}
            """
        )
        chain = prompt | llm | StrOutputParser()
        rewritten_question = await chain.ainvoke({"context": context, "question": current_question})
        print(f"✏️ REWRITTEN: {rewritten_question}")
        return {"rewritten_query": rewritten_question}
        
    print("✏️ REWRITTEN: No rewrite needed (first question)")
    return {"rewritten_query": current_question}


async def retrieve_documents(state: MultiHopState, config: RunnableConfig) -> Dict[str, Any]:
    print("---NODE: Retrieving Documents---")
    query = state.get("rewritten_query", state["decomposed_questions"][0])
    docs = await rag_system.retrieve_documents(query)
    return {"retrieved_docs": docs}

async def generate_intermediate_answer(state: MultiHopState, config: RunnableConfig) -> Dict[str, Any]:
    print("---NODE: Generating Intermediate Answer---")
    context = "\n\n".join([doc.page_content for doc in state["retrieved_docs"]])
    current_question = state["decomposed_questions"][state["current_question_index"]]
    
    print(f"💭 ANSWERING: {current_question}")
    print(f"📄 CONTEXT LENGTH: {len(context)} chars")
    
    configuration = config.get("configurable", {})
    llm = ChatOpenAI(model=configuration.get("model", "gpt-4o-mini-2024-07-18"), temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        """You are an expert answer generator. Based *only* on the provided context,
        answer the following question as concisely as possible. Do not add any extra information.
        If the answer is not in the context, state that clearly.

        Context:
        {context}

        Question:
        {question}
        """
    )
    chain = prompt | llm | StrOutputParser()
    answer = await chain.ainvoke({"context": context, "question": current_question})
    
    print(f"✅ ANSWER: {answer}")
    
    new_answers = state["intermediate_answers"] + [f"The answer to '{current_question}' is: {answer}"]
    
    return {
        "intermediate_answers": new_answers,
        "current_question_index": state["current_question_index"] + 1,
    }

async def generate_final_answer(state: MultiHopState, config: RunnableConfig) -> Dict[str, Any]:
    print("---NODE: Generating Final Answer---")
    context = "\n".join(state["intermediate_answers"])
    original_question = state["original_question"]
    
    print(f"🏁 SYNTHESIZING: {len(state['intermediate_answers'])} intermediate answers")

    configuration = config.get("configurable", {})
    llm = ChatOpenAI(
        model=configuration.get("model", "gpt-4o-mini-2024-07-18"),
        temperature=configuration.get("temperature", 0.7),
        max_tokens=configuration.get("max_tokens", 500)
    )
    
    prompt = ChatPromptTemplate.from_template(
        """You are a master synthesist. You have been provided with a series of facts that were gathered
        to answer a complex user question. Combine these facts into a single, clear, and comprehensive answer.

        Original Question: {original_question}

        Facts Gathered:
        {context}

        Final Answer:
        """
    )
    chain = prompt | llm | StrOutputParser()
    final_answer = await chain.ainvoke({"context": context, "original_question": original_question})
    
    print(f"🎉 FINAL ANSWER: {final_answer}")
    
    return {"final_answer": final_answer}

def should_continue(state: MultiHopState) -> Literal["rewrite_query", "generate_final_answer"]:
    print("---EDGE: Deciding to Continue---")
    current = state["current_question_index"]
    total = len(state["decomposed_questions"])
    print(f"🚦 PROGRESS: {current}/{total} questions answered")
    
    if current < total:
        print("➡️ CONTINUE: More questions to answer")
        return "rewrite_query"
    print("🏁 FINISH: All questions answered")
    return "generate_final_answer"


workflow = StateGraph(MultiHopState, config_schema=Configuration)

workflow.add_node("decompose_question", decompose_question)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("generate_intermediate_answer", generate_intermediate_answer)
workflow.add_node("generate_final_answer", generate_final_answer)

workflow.set_entry_point("decompose_question")
workflow.add_edge("decompose_question", "rewrite_query")
workflow.add_edge("rewrite_query", "retrieve_documents")
workflow.add_edge("retrieve_documents", "generate_intermediate_answer")

workflow.add_conditional_edges(
    "generate_intermediate_answer",
    should_continue,
    {
        "rewrite_query": "rewrite_query",
        "generate_final_answer": "generate_final_answer"
    }
)
workflow.add_edge("generate_final_answer", END)

graph = workflow.compile(name="multi-hop-agent")