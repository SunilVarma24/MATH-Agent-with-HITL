# Input Guardrails
# This guardrail checks if the question is math-related.
# If not, it returns a message indicating that only math questions are allowed.

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import llm, eval_llm

input_guard_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a mathematics teaching assistant. Only answer questions related to mathematics. 
    If not a math query, say: '❌ I'm sorry, I can only help with math questions.'"""),
    ("human", "{question}")
])

input_guard_chain = input_guard_prompt | llm | StrOutputParser()

async def input_guardrails(state):
    print("🧠 Running Input Guardrails: Checking if the question is math-related...")
    question = state["question"]
    response = await input_guard_chain.ainvoke({"question": question})

    # If it's not math-related
    if "❌ I'm sorry, I can only help with math questions." in response:
        #return {**state, "generation": response, "documents": []}
        return {**state, "generation": response, "should_end": True}
    
    # Otherwise continue with an empty generation
    return {**state, "generation": ""}



# Output Guardrails
# This is the final check to ensure the solution is correct and complete.

async def output_guardrails(state):
    raw = state.get("raw_solution", "")
    if not raw:
        return state

    check_prompt = ChatPromptTemplate.from_messages([
        ("system", 
        "You are a mathematics expert responsible for validating the accuracy and completeness of step-by-step solutions.\n\n"
        "Given a math question and a proposed solution:\n"
        "1. If the solution is entirely correct, includes all necessary steps, and clearly explains the reasoning, return the solution exactly as-is.\n"
        "2. If the solution is incorrect, missing key steps, or lacks clear logic, rewrite it completely with detailed, correct, step-by-step reasoning.\n\n"
        "Do not include any additional commentary or notes. Return only the corrected (or confirmed) solution."),
        
        ("human", 
        "Question:\n{question}\n\nProposed Solution:\n{raw_solution}")
    ])

    chain = check_prompt | eval_llm | StrOutputParser()
    validated = await chain.ainvoke({
        "question": state["question"],
        "raw_solution": raw
    })

    # Place the final solution into 'generation'
    return {**state, "generation": validated}