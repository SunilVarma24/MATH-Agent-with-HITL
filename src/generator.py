# Final Generation

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import llm

async def generate(state):
    # If we already have a solution, do nothing
    if state.get("raw_solution") and not state.get("human_feedback"):
            return state

    # Build context from KB docs + web results
    docs = "\n\n".join(doc.page_content for doc in state.get("documents", []))
    web = state.get("web_results", "")
    feedback = state.get("human_feedback", "")
    prev_solution = state.get("raw_solution", "")

    context = docs + "\n\n" + web

    # Create and invoke the chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
        "You are a expert mathematics professor helping students understand the math problems clearly.\n\n"
        "Your task is to:\n"
        "1. Understand the math question.\n"
        "2. Use the provided context to identify key formulas, definitions, or values.\n"
        "3. Solve the question step-by-step using clear reasoning and intermediate steps.\n"
        "4. Show all formulas and substitutions, and explain your logic.\n"
        "5. If human feedback and previous answer are provided, follow the human feedback carefully to improve or correct your previous answer.\n\n"
        "Note: Do not skip steps, and ensure clarity for students learning maths."),
        
        ("human", 
        """Previous Response:\n{previous_answer_block}\n\n
        Human Feedback:\n{feedback_block}\n\n
        Context:\n{context}\n\n
        Question:\n{question}""")
    ])

    # Add feedback & previous answer if any
    feedback_block = f"Human Feedback: {feedback}" if feedback else ""
    previous_answer_block = f"Previous Answer:\n{prev_solution}\n\n" if prev_solution else ""

    chain = prompt | llm | StrOutputParser()
    raw = await chain.ainvoke({"context": context, "question": state["question"], 
                        "feedback_block": feedback_block, 
                        "previous_answer_block": previous_answer_block})

    # Store the raw solution separately, leave generation empty for now
    return {**state, "previous_solution": prev_solution, "raw_solution": raw, "generation": "", "human_feedback": ""}
