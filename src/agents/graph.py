from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain.schema import HumanMessage, AIMessage
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
from config.settings import settings

# Load model and pipeline
os.environ["HF_TOKEN"] = settings.HF_TOKEN
tokenizer = AutoTokenizer.from_pretrained("lastmass/Qwen3_Medical_GRPO", trust_remote_code=True, token=os.environ.get("HF_TOKEN"))
model = AutoModelForCausalLM.from_pretrained("lastmass/Qwen3_Medical_GRPO", trust_remote_code=True, token=os.environ.get("HF_TOKEN"), torch_dtype=torch.float16)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=settings.MAX_NEW_TOKENS, temperature=settings.TEMPERATURE, device=0 if torch.cuda.is_available() else -1)
llm = HuggingFacePipeline(pipeline=pipe)

# State Definition
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    documents: List[str]
    generation: str
    critique: str
    is_meta: bool

# Nodes (Agents)
def retrieval_agent(state: AgentState):
    query = state["messages"][-1].content
    state["is_meta"] = query.lower().startswith("synthesize")
    docs = retriever.invoke(query) if not state["is_meta"] else []
    documents = [d.page_content for d in docs]
    return {"documents": documents, "is_meta": state["is_meta"]}

def generation_agent(state: AgentState):
    if state["is_meta"]:
        return {"generation": "", "messages": []}
    prompt = ChatPromptTemplate.from_template(
        "You are an expert on Parkinson's disease. Provide a factual answer to the question based only on the provided context.\n"
        "Question: {question}\n"
        "Context: {context}\n"
        "Answer:"
    )
    chain = (
        {"context": lambda x: "\n".join(state["documents"]), "question": lambda x: state["messages"][-1].content}
        | prompt
        | llm
        | StrOutputParser()
    )
    generation = chain.invoke({})
    return {"generation": generation, "messages": [AIMessage(content=generation)]}

def critique_agent(state: AgentState):
    if state["is_meta"]:
        return {"critique": "OK"}
    prompt = ChatPromptTemplate.from_template(
        "Critique the following answer for hallucinations or inaccuracies regarding Parkinson's disease.\n"
        "Question: {question}\n"
        "Context: {context}\n"
        "Answer: {answer}\n"
        "Critique: Provide a brief assessment (e.g., 'OK' if accurate, or note specific issues)."
    )
    chain = (
        {"context": lambda x: "\n".join(state["documents"]), 
         "question": lambda x: state["messages"][-2].content, 
         "answer": lambda x: state["generation"]}
        | prompt
        | llm
        | StrOutputParser()
    )
    critique = chain.invoke({})
    return {"critique": critique}

def synthesis_agent(state: AgentState):
    if state["is_meta"] and not state.get("generation"):
        generation = state["messages"][-1].content.split("Generation:")[1].split("Critique:")[0].strip().split("Answer:")[-1].strip()
        critique = state["messages"][-1].content.split("Critique:")[1].split("Final Answer:")[0].strip()
    else:
        generation = state["generation"]
        critique = state["critique"]
    prompt = ChatPromptTemplate.from_template(
        "Synthesize the final answer based on the generation and critique, ensuring a clear and concise response.\n"
        "Generation: {generation}\n"
        "Critique: {critique}\n"
        "Final Answer:"
    )
    chain = (
        {"generation": generation, "critique": critique}
        | prompt
        | llm
        | StrOutputParser()
    )
    final = chain.invoke({})
    return {"messages": [AIMessage(content=f"Final Answer: {final} Assistant:")]}

# Graph Setup
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieval_agent)
workflow.add_node("generate", generation_agent)
workflow.add_node("critique", critique_agent)
workflow.add_node("synthesize", synthesis_agent)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "critique")
workflow.add_edge("critique", "synthesize")
workflow.add_edge("synthesize", END)

app = workflow.compile()