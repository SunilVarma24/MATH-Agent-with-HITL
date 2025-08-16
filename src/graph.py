# State Graph

from langgraph.graph import StateGraph, END
from src.graph_state import GraphState
from src.guardrails import input_guardrails, output_guardrails
from src.retriever import retrieve
from src.grader import document_grader
from src.web_search import web_search, assess_web_results
from src.generator import generate
from src.human_review import human_in_the_loop
from src.stop import stop

# =========================================================
# Agent Workflow Graph

graph_A = StateGraph(GraphState)

graph_A.add_node("input_guardrails", input_guardrails)
graph_A.add_node("retrieve", retrieve)
graph_A.add_node("grade", document_grader)
graph_A.add_node("web_search", web_search)
graph_A.add_node("assess_web_results", assess_web_results)
graph_A.add_node("generate", generate)
graph_A.add_node("output_guardrails", output_guardrails)
graph_A.add_node("stop", stop)
graph_A.add_edge("stop", END)

graph_A.set_entry_point("input_guardrails")

graph_A.add_conditional_edges(
    "input_guardrails",
    lambda state: "end" if state.get("should_end", False) else "continue",
    {
        "end": "stop",
        "continue": "retrieve"
    }
)

graph_A.add_edge("retrieve", "grade")
graph_A.add_conditional_edges(
    "grade", 
    lambda state: "No" if state["web_search_needed"] else "Yes",
    {
        "Yes": "generate",     # Use retrieved docs
        "No": "web_search"     # Perform web search if no relevant doc
    }
)
graph_A.add_edge("web_search", "assess_web_results")
graph_A.add_conditional_edges(
    "assess_web_results",
    lambda state: "end" if state.get("should_end", False) else "continue",
    {
        "end": "stop",
        "continue": "generate" 
    }
)

graph_A.add_edge("generate", "output_guardrails")
graph_A.add_edge("output_guardrails", END)


workflow_A = graph_A.compile()

# Expose the compiled workflow for use in main.py
def get_agent_workflow_A():
    return workflow_A

# =========================================================
# Human Review Workflow

# Human Review Graph
graph_B = StateGraph(GraphState)

graph_B.add_node("generate", generate)
graph_B.add_node("output_guardrails", output_guardrails)
graph_B.add_node("human_review", human_in_the_loop)
graph_B.add_node("stop", stop)

graph_B.set_entry_point("human_review")

# Add conditional edge from human_review back to generate if feedback provided
graph_B.add_conditional_edges(
    "human_review",
    lambda state: "regenerate" if state.get("human_feedback") else "end",
    {
        "regenerate": "generate",
        "end": "stop"
    }
)
graph_B.add_edge("generate", "output_guardrails")
graph_B.add_edge("output_guardrails", "human_review")
graph_B.add_edge("stop", END)

workflow_B = graph_B.compile()

# Expose the compiled workflow for use in main.py
def get_agent_workflow_B():
    return workflow_B