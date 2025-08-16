# Web Search Tool

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import web_search_tool, eval_llm

async def web_search(state):
    if state.get("web_search_needed") == "Yes":
        print("🌐 No sufficient documents. Performing Web Search...")
        
        try:
            # Perform web search
            # results = web_search_tool.invoke({"query": state["question"]})
            response = await web_search_tool.search(
                state["question"],
                search_depth='advanced',
                max_tokens=10000
            )

            # Format the results similar to TavilySearchResults format
            results = []
            for result in response.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0)
                })

            # Check if results are empty
            if not results:
                return {**state, "web_results": "", "web_search_sufficient": "No"}
            
            # Combine contents of the search results
            web_context = "\n\n".join(r["content"] for r in results)
            
            # Add this web search assessment step
            return {**state, "web_results": web_context, "web_search_sufficient": "Unknown"}
        
        except Exception as e:
            print(f"Web search error: {e}")
            return {**state, "web_results": "", "web_search_sufficient": "No"}
        
    return {**state, "web_results": "", "web_search_sufficient": "NotNeeded"}


# Web Results Assessment

async def assess_web_results(state):
    # Skip if web search wasn't needed
    if state["web_search_sufficient"] == "NotNeeded":
        return state
    
    # Skip if web results are already known to be insufficient
    if state["web_search_sufficient"] == "No":
        return {**state, "generation": "❌ I'm sorry, I couldn't find any reliable information online to answer your question."}
    
    try:
        # Define the assessment prompt
        assess_prompt = ChatPromptTemplate.from_messages([
            ("system", 
            "You are a mathematics expert evaluating whether web search results contain sufficient information to solve a math problem."
            "Your task is to determine if the provided web content has relevant mathematical information to answer the question."
            "Return ONLY 'Yes' if the content is sufficient, or 'No' if it lacks necessary information."),
            ("human", 
            "Question: {question}\n\nWeb Content:\n{web_results}")
        ])
        
        assessment_chain = assess_prompt | eval_llm | StrOutputParser()
        result = await assessment_chain.ainvoke({
            "question": state["question"],
            "web_results": state["web_results"]
        })
        
        print(f"🔍 Web Results Assessment: {result}")
        
        if "No" in result:
            return {**state, 
                    "web_search_sufficient": "No", 
                    "generation": "❌ I'm sorry, I couldn't find any reliable information online to answer your question.",
                    "should_end": True}
        
        return {**state, "web_search_sufficient": "Yes", "should_end": False}
    
    except Exception as e:
        print(f"Web results assessment error: {e}")
        return {**state, 
                "web_search_sufficient": "No", 
                "generation": "❌ I'm sorry, there was an error assessing the web search results.",
                "should_end": True}    