# stop() to end the workflow
# This function is called when the workflow need to end at intermediate state.

async def stop(state):
    print("=========================================================")
    print(state["generation"])
    return state