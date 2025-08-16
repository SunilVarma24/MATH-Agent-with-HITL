# Retrieve Documents
import asyncio
from src.chroma_index import retriever

async def retrieve(state):
    print("📥 Retrieving from Knowledge Base...")
    question = state["question"]
    if state.get("generation") and "I'm sorry" in state["generation"]:
        return state
    try:
        docs = await asyncio.to_thread(retriever.get_relevant_documents, question)
        return {**state, "documents": docs}
    except Exception as e:
        print(f"Document retrieval error: {e}")
        return {**state, "documents": []}