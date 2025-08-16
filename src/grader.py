# Document Grader

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from src.config import eval_llm

grader_prompt = ChatPromptTemplate.from_template("""
You are a document relevance grader.

Question:
{question}

Documents:
{documents}

Your task is to:
1. Identify which documents are relevant to the question.
2. Determine if at least one of the relevant documents is sufficient to answer the question.
3. Output a JSON with:
- "relevant_documents": [list of indices of relevant docs],
- "is_sufficient": true/false

Return your output as JSON only.
""")

parser = JsonOutputParser()

def grade_documents(inputs):
    question = inputs["question"]
    documents = [doc.page_content for doc in inputs["documents"]]
    return {"question": question, "documents": documents}
document_grader_chain = RunnableLambda(grade_documents) | grader_prompt | eval_llm | parser

async def document_grader(state):
    if not state["documents"]:
        return {**state, "web_search_needed": "Yes"}
    result = await document_grader_chain.ainvoke({"question": state["question"], "documents": state["documents"]})
    relevant_indices = result.get("relevant_documents", [])
    is_sufficient = result.get("is_sufficient", False)
    relevant_docs = [state["documents"][i] for i in relevant_indices]
    print(f"🗂️ Document Grader Result: {len(relevant_docs)} relevant docs found. ✅ Sufficient: {is_sufficient}")
    return {
        **state,
        "documents": relevant_docs,
        "web_search_needed": "No" if is_sufficient else "Yes"
    }