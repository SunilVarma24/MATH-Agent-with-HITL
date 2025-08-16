# Human-in-the-Loop Feedback Function

async def human_in_the_loop(state):
    approval = state.get("approval", "")
    
    if approval.strip().lower() == "yes":
        return {**state, "human_feedback": "", "previous_solution": state["raw_solution"]}
    
    feedback = state.get("human_feedback", "")
    
    # If we have feedback, process it and reset approval for next iteration
    if feedback:
        # Here we can process the feedback as needed
        return {**state, "human_feedback": feedback, "previous_solution": state["raw_solution"], 
                "raw_solution": "", "generation": ""}
    
    # If no feedback but approval is "no", we're in the middle of processing
    # Let the workflow continue
    return state