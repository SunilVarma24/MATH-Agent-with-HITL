# Ask Function to invoke the workflow

from uuid import uuid4
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.utils import latex_to_text
from src.graph import get_agent_workflow_A, get_agent_workflow_B
from src.save_feedback import save_feedback

# Initialize FastAPI app
app = FastAPI(
    title="Agentic RAG - Math Agent with Human Feedback based Learning",
    description="An API for solving math problems using a RAG agent with human-in-the-loop feedback.",
    version="1.0.0"
)

agent_workflow_A = get_agent_workflow_A()
agent_workflow_B = get_agent_workflow_B()

# Define the request model
class QuestionRequest(BaseModel):
    question: str

class HumanFeedbackRequest(BaseModel):
    human_feedback: str = ""
    rating: int = 0
    approval: str

# In-memory session state to store workflow states
session_states = {}

latest_session_id = None

@app.post("/ask")
async def ask(question: QuestionRequest):
    global latest_session_id

    # Initialize the workflow with an empty state
    initial_state = {
        "question": question.question,
        "generation": "",
        "documents": [],
        "web_search_needed": "",
        "web_results": "",
        "web_search_sufficient": "",
        "human_feedback": "",
        "should_end": False,
        "previous_solution": "",
        "raw_solution": "",
        "rating": 0,
        "approval": ""

    }

    try:
        # First pass through the workflow
        result = await agent_workflow_A.ainvoke(initial_state)

        session_id = str(uuid4())[:8]
        session_states[session_id] = result
        latest_session_id = session_id

        # Convert the generation to plain text for display
        if result.get("generation"):
            result["generation"] = latex_to_text(result["generation"])
        
        # Return the final generation result
        return {
            "generation": result["generation"],
            "session_id": session_id
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.post("/human_review")
async def human_review(feedback: HumanFeedbackRequest, session_id: str = None):
    global latest_session_id

    if session_id is None:
        session_id = latest_session_id

    state = session_states.get(session_id)
    if not state:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

        # Save feedback IMMEDIATELY with the correct rating
    if feedback.approval.lower() == "yes":
        # For acceptance
        save_feedback(
            state["question"], 
            state.get("generation", ""), 
            "", 
            rating=feedback.rating
        )

        # Return directly without running workflow_B
        return {"status": "completed", "message": "Workflow ended."}
    else:
        # For feedback/regeneration
        save_feedback(
            state["question"], 
            state.get("generation", ""), 
            feedback.human_feedback, 
            rating=feedback.rating
        )

    # Update state for feedback
    state["human_feedback"] = feedback.human_feedback
    state["rating"] = feedback.rating
    state["approval"] = feedback.approval

    # Ensure previous_solution and raw_solution are set for regeneration
    if feedback.human_feedback:
        state["previous_solution"] = state.get("generation", "")
        state["raw_solution"] = state.get("generation", "")

    try:
        result = await agent_workflow_B.ainvoke(state)

        # Persist updated workflow state for further iterations
        session_states[session_id] = result

        if result.get("generation"):
            result["generation"] = latex_to_text(result["generation"])
            return {"generation": result["generation"], "status": "regenerated"}
        else:
            return {"status": "completed", "message": "Workflow ended."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Run using: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)