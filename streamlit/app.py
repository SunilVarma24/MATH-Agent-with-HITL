# 1. Import Required Libraries
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from IPython.display import Image, display
from typing import List
from typing_extensions import TypedDict
import os
import json
from pylatexenc.latex2text import LatexNodes2Text
import re
import tempfile
from pathlib import Path
from datetime import datetime
import time
import random

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Graph State
class GraphState(TypedDict):
    question: str
    raw_solution: str
    generation: str
    documents: List[Document]
    web_search_needed: str
    web_search_sufficient: str
    web_results: str
    human_feedback: str
    should_end: bool

# LaTeX Cleaning Functions
def clean_latex(latex_str):
    latex_str = latex_str.replace('\\\\', '\\')
    latex_str = re.sub(r"\\\[|\\\]", "", latex_str)
    latex_str = re.sub(r"\$+", "", latex_str)
    return latex_str

def latex_to_text(latex_str):
    cleaned = clean_latex(latex_str)
    return LatexNodes2Text().latex_to_text(cleaned)

# Dataset Loader with Cleaning
def load_math_dataset(uploaded_files):
    """Loads JSON files from uploaded files, cleans LaTeX, and returns LangChain Documents."""
    documents = []
    temp_dir = tempfile.TemporaryDirectory()
    
    for file in uploaded_files:
        if file.name.endswith(".json"):
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
                
            try:
                with open(temp_filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Extract and clean problem and solution text
                problem = data.get("problem", "")
                solution = data.get("solution", "")
                level = data.get("level", "")
                qtype = data.get("type", "")
                clean_problem = latex_to_text(problem.strip())
                clean_solution = latex_to_text(solution.strip())
                content = f"Problem:\n{clean_problem}\n\nSolution:\n{clean_solution}"
                metadata = {
                    "level": level,
                    "type": qtype
                }
                documents.append(Document(page_content=content, metadata=metadata))
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")
    st.write(f"Loaded {len(documents)} documents.")
    return documents

def initialize_components():
    """Initialize all components and store in session state"""
    if not st.session_state.initialized:
        st.session_state.embedding_model = HuggingFaceEmbeddings(
                                            model_name="BAAI/bge-small-en",
                                            model_kwargs={"device": "cpu"},
                                            encode_kwargs={"normalize_embeddings": True}
                                        )
        st.session_state.llm = ChatGoogleGenerativeAI(
                                    model="gemini-2.0-flash",
                                    temperature=0,
                                )
        
        st.session_state.eval_llm = ChatGoogleGenerativeAI(
                                    model="gemini-1.5-pro",
                                    temperature=0,
                                )
        st.session_state.tv_search = TavilySearchResults(max_results=3, search_depth='advanced')

        if os.path.exists("./math_db"):
            st.session_state.chroma_db = Chroma(
                embedding_function=st.session_state.embedding_model,
                persist_directory="./math_db",
                collection_name="math_knowledge"
            )
            st.write("✅ Chroma index loaded from disk.")
        else:
            st.session_state.chroma_db = None
            st.write("⚠️ No existing database found. Need to create an index.")
            
        st.session_state.initialized = True


def configure_retriever():
    """Configure the document retriever with the current Chroma DB"""
    if st.session_state.chroma_db:
        st.session_state.retriever = st.session_state.chroma_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.8 
            }
        )
        return True
    return False

# Input Guardrails
def input_guardrails(state):
    with st.status("🧠 Checking if the question is math-related..."):
        question = state["question"]
        input_guard_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a mathematics teaching assistant. Only answer questions related to mathematics. If not a math query, say: 'I'm sorry, I can only help with math questions.'"),
            ("human", "{question}")
        ])
        
        input_guard_chain = input_guard_prompt | st.session_state.llm | StrOutputParser()
        response = input_guard_chain.invoke({"question": question})
        
        # If it's not math-related
        if "I'm sorry, I can only help with math questions." in response:
            return {**state, "generation": response, "should_end": True}
        
    # Otherwise continue with an empty generation
    return {**state, "generation": ""}

# Retrieve Documents
def retrieve(state):
    with st.status("📥 Retrieving from Knowledge Base..."):
        question = state["question"]
        if state.get("generation") and "I'm sorry" in state["generation"]:
            return state
        
        if st.session_state.retriever:
            docs = st.session_state.retriever.get_relevant_documents(question)
            st.write(f"Found {len(docs)} potentially relevant documents")
            return {**state, "documents": docs}
        else:
            st.warning("No retriever configured.")
            return {**state, "documents": []}

# Document Grader
def document_grader(state):
    with st.status("🗂️ Grading document relevance..."):
        if not state["documents"]:
            st.write("No documents found, will need web search")
            return {**state, "web_search_needed": "Yes"}
        
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
            
        document_grader_chain = RunnableLambda(grade_documents) | grader_prompt | st.session_state.eval_llm | parser

        result = document_grader_chain.invoke({"question": state["question"], "documents": state["documents"]})
        relevant_indices = result.get("relevant_documents", [])
        is_sufficient = result.get("is_sufficient", False)
        relevant_docs = [state["documents"][i] for i in relevant_indices if i < len(state["documents"])]
        
        st.write(f"{len(relevant_docs)} relevant docs found. Sufficient: {is_sufficient}")
        
        return {
            **state,
            "documents": relevant_docs,
            "web_search_needed": "No" if is_sufficient else "Yes"
        }

# Web Search
def web_search(state):
    if state.get("web_search_needed") == "Yes":
        with st.status("🌐 Performing Web Search..."):
            time.sleep(1)  # Give UI time to update
            # Perform web search
            try:
                results = st.session_state.tv_search.invoke({"query": state["question"]})
                
                # Check if results are empty
                if not results:
                    st.write("No web results found")
                    return {**state, "web_results": "", "web_search_sufficient": "No"}
                
                # Combine contents of the search results
                web_context = "\n\n".join(r["content"] for r in results)
                st.write(f"Found {len(results)} web results")
                
                return {**state, "web_results": web_context, "web_search_sufficient": "Unknown"}
            except Exception as e:
                st.error(f"Web search error: {str(e)}")
                return {**state, "web_results": "", "web_search_sufficient": "No"}
    
    return {**state, "web_results": "", "web_search_sufficient": "NotNeeded"}
        
# Web Results Assessment
def assess_web_results(state):
    # Skip if web search wasn't needed
    if state["web_search_sufficient"] == "NotNeeded":
        return state
    
    # Skip if web results are already known to be insufficient
    if state["web_search_sufficient"] == "No":
        return {**state, "generation": "Sorry, I couldn't find any reliable information online to answer your question.", "should_end": True}
    
    with st.status("🔍 Assessing Web Results..."):
        # Define the assessment prompt
        assess_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a mathematics expert evaluating whether web search results contain sufficient information to solve a math problem."
             "Your task is to determine if the provided web content has relevant mathematical information to answer the question."
             "Return ONLY 'Yes' if the content is sufficient, or 'No' if it lacks necessary information."),
            ("human", 
             "Question: {question}\n\nWeb Content:\n{web_results}")
        ])
        
        assessment_chain = assess_prompt | st.session_state.llm | StrOutputParser()
        result = assessment_chain.invoke({
            "question": state["question"],
            "web_results": state["web_results"]
        })
        
        st.write(f"Web Results Assessment: {result}")
        
        if "No" in result:
            return {**state, 
                    "web_search_sufficient": "No", 
                    "generation": "Sorry, I couldn't find any reliable information online to answer your question.",
                    "should_end": True}
        
    return {**state, "web_search_sufficient": "Yes", "should_end": False}

def generate(state):
    with st.status("🧮 Generating Solution..."):
        # If we already have a solution _and_ no new feedback, skip
        if state.get("raw_solution") and not state.get("human_feedback"):
            return state

        docs = "\n\n".join(d.page_content for d in state["documents"])
        context = docs + "\n\n" + state["web_results"]
        fb = state.get("human_feedback", "")
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
            "You are a expert mathematics professor helping students understand the math problems clearly.\n\n"
            "Your task is to:\n"
            "1. Understand the math question.\n"
            "2. Use the provided context to identify key formulas, definitions, or values.\n"
            "3. Solve the question step-by-step using clear reasoning and intermediate steps.\n"
            "4. Show all formulas and substitutions, and explain your logic.\n"
            "5. End with a final boxed answer like: \\boxed{{your_final_answer}}.\n\n"
            "Do not skip steps, and ensure clarity for someone learning math."),

            ("human", f"Human Feedback: {fb}\nContext:\n{context}\n\nQuestion:\n{state['question']}")
        ])
        chain = prompt | st.session_state.llm | StrOutputParser()
        raw = chain.invoke({"question": state["question"], "context": context})
        return {**state, "raw_solution": raw, "generation": ""}

# Output Guardrails
def output_guardrails(state):
    with st.status("✅ Validating Solution..."):
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

        chain = check_prompt | st.session_state.eval_llm | StrOutputParser()
        validated = chain.invoke({
            "question": state["question"],
            "raw_solution": raw
        })

        # Place the final solution into 'generation'
        return {**state, "generation": validated}
    
# Save feedback function
def save_feedback(question, solution, feedback, rating=None):
    feedback_file = Path("./feedback_data/feedback_log.json")
    feedback_file.parent.mkdir(exist_ok=True, parents=True)

    # Use readable datetime format
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = {
        "timestamp": timestamp,
        "question": question,
        "solution": solution,
        "feedback": feedback,
        "rating": rating
    }

    # Load existing data if file exists
    if feedback_file.exists():
        with open(feedback_file, "r") as f:
            try:
                feedback_data = json.load(f)
            except json.JSONDecodeError:
                feedback_data = []
    else:
        feedback_data = []

    # Append the new entry
    feedback_data.append(entry)

    # Save back to the file
    with open(feedback_file, "w") as f:
        json.dump(feedback_data, f, indent=2)
    
    st.success(f"Feedback saved!")

def human_in_the_loop(state):
    # Initialize session state variables if not present
    if "feedback_iterations" not in st.session_state:
        st.session_state.feedback_iterations = 0
    if "phase" not in st.session_state:
        st.session_state.phase = "rating"
    if "processing_feedback" not in st.session_state:
        st.session_state.processing_feedback = False
    
    # Get current iteration and phase
    it = st.session_state.get("feedback_iterations", 0)
    phase = st.session_state.get("phase", "rating")
    
    # Create truly unique keys for widgets
    if "widget_id" not in st.session_state:
        st.session_state.widget_id = str(random.randint(10000, 99999))
    wid = st.session_state.widget_id
    suf = f"_iter{it}_{wid}"
    
    # Display solution information
    st.subheader("🔎 Generated Solution")
    if it > 0:
        st.info(f"Iteration {it} - Solution based on your feedback")
    st.markdown(f"**Question:** {state['question']}")
    st.markdown(state["generation"])
    st.write("---")
    
    # Skip feedback process for non-math questions
    low = state["generation"].lower() if state["generation"] else ""
    if "only help with math" in low or "couldn't find any reliable information" in low:
        st.session_state.submitted = False
        return {**state, "human_feedback": ""}
    
    # Maximum iterations check
    if st.session_state.feedback_iterations >= 3:
        st.warning("Maximum feedback iterations (3) reached. Final solution saved.")
        save_feedback(state["question"], state["generation"], "Max iterations reached", rating=None)
        reset_feedback_state()
        return {**state, "human_feedback": ""}
    
    # STEP 1: Rating phase
    if phase == "rating":
        rating = st.slider("⭐ Rate this solution:", 1, 5, 3, key=f"rating{suf}")
        if st.button("Submit Rating", key=f"submit{suf}"):
            st.session_state.current_rating = rating
            st.session_state.phase = "approval"
            st.rerun()
        return state
    
    # STEP 2: Approval phase
    if phase == "approval":
        st.write(f"Rating: {st.session_state.current_rating} ⭐")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Accept", key=f"accept{suf}"):
                save_feedback(
                    state["question"], state["generation"], "", rating=st.session_state.current_rating
                )
                st.success("Solution accepted!")
                reset_feedback_state()
                return {**state, "human_feedback": ""}
        with c2:
            if st.button("❌ Request Changes", key=f"request{suf}"):
                st.session_state.phase = "feedback"
                st.rerun()
        return state
    
    # STEP 3: Feedback and regeneration phase
    if phase == "feedback":
        fb = st.text_area("✏️ Enter your feedback:", key=f"fb{suf}")
        submit_fb = st.button("Submit Feedback", key=f"fb_submit{suf}")
        
        if submit_fb and fb.strip():
            # Set processing flag to prevent multiple submissions
            st.session_state.processing_feedback = True
            
            # Save the current state and feedback for use after rerun
            st.session_state.current_feedback = fb
            st.session_state.last_question = state["question"]
            st.session_state.last_generation = state["generation"]
            
            # Save feedback record
            save_feedback(
                state["question"], state["generation"], fb, rating=st.session_state.current_rating
            )
            
            # Create a new state with feedback
            new_state = GraphState(
                question=state["question"],
                human_feedback=fb,
                raw_solution="",
                generation="",
                documents=state.get("documents", []),
                web_search_needed=state.get("web_search_needed", ""),
                web_search_sufficient=state.get("web_search_sufficient", ""),
                web_results=state.get("web_results", ""),
                should_end=False
            )
            
            # Store this temporary state to show a placeholder while processing
            st.session_state.temp_state = new_state
            
            try:
                with st.spinner("🧮 Generating new solution with your feedback..."):
                    # Resume graph from 'generate'
                    math_agent = build_math_agent()
                    new_state = math_agent.invoke(new_state)
                
                # Increment feedback iteration
                st.session_state.feedback_iterations += 1
                st.session_state.phase = "rating"
                st.session_state.widget_id = str(random.randint(10000, 99999))
                st.session_state.processing_feedback = False
                
                # Update the current state with the new solution
                st.session_state.current_state = new_state
                return new_state
            except Exception as e:
                st.error(f"Error regenerating solution: {str(e)}")
                st.session_state.processing_feedback = False
                
                # If error occurs, show the previous state
                return state
        
        # If user hasn't submitted feedback yet, show the current state
        return state

def reset_feedback_state():
    """Reset all feedback-related session state variables"""
    st.session_state.phase = "rating"
    st.session_state.feedback_iterations = 0
    st.session_state.submitted = False
    st.session_state.processing_feedback = False
    st.session_state.current_state = None
    st.session_state.widget_id = str(random.randint(10000, 99999))

def stop(state):
    """End the workflow"""
    return state

def build_math_agent():
    """Build the MATH Agent workflow"""
    agent = StateGraph(GraphState)

    agent.add_node("input_guardrails", input_guardrails)
    agent.add_node("retrieve", retrieve)
    agent.add_node("grade", document_grader)
    agent.add_node("web_search", web_search)
    agent.add_node("assess_web_results", assess_web_results)
    agent.add_node("generate", generate)
    agent.add_node("output_guardrails", output_guardrails)
    agent.add_node("human_review", human_in_the_loop)
    agent.add_node("stop", stop)
    agent.add_edge("stop", END)

    agent.set_entry_point("input_guardrails")

    agent.add_conditional_edges(
        "input_guardrails",
        lambda state: "end" if state.get("should_end", False) else "continue",
        {
            "end": "stop",
            "continue": "retrieve"
        }
    )

    agent.add_edge("retrieve", "grade")
    agent.add_conditional_edges(
        "grade", 
        lambda state: "No" if state["web_search_needed"] else "Yes",
        {
            "Yes": "generate",     # Use retrieved docs
            "No": "web_search"     # Perform web search if no relevant doc
        }
    )
    agent.add_edge("web_search", "assess_web_results")
    agent.add_conditional_edges(
        "assess_web_results",
        lambda state: "end" if state.get("should_end", False) else "continue",
        {
            "end": "stop",
            "continue": "generate" 
        }
    )
    agent.add_edge("generate", "output_guardrails")
    agent.add_edge("output_guardrails", "human_review")

    agent.add_conditional_edges(
        "human_review",
        lambda state: "regenerate" if state["human_feedback"] else "complete",
        {
            "regenerate": "generate",
            "complete": END
        }
    )

    return agent.compile()

def main():
    
    st.set_page_config(page_title="MATH Agent", page_icon="🧮", layout="wide")
    st.title("🧮 MATH Agent: Advanced Mathematical Problem Solver")
    st.markdown("""This application uses AI to solve mathematical problems step-by-step, drawing on both a knowledge base and web search when needed.""")
    
    initialize_components()
    retriever_configured = configure_retriever()
    st.session_state.web_search_tool = st.session_state.tv_search
    
    tab1, tab2 = st.tabs(["Ask Questions", "View Feedback"])
    
    with tab1:
        st.header("Ask Your Math Question")
        
        if retriever_configured:
            st.success("✅ Knowledge base is ready for queries.")
        else:
            st.warning("⚠️ No knowledge base available. The agent will rely on web search.")
        
        # Initialize state variables if needed
        if "current_state" not in st.session_state:
            st.session_state.current_state = None
        if "is_generating" not in st.session_state:
            st.session_state.is_generating = False
        if "widget_id" not in st.session_state:
            st.session_state.widget_id = str(random.randint(10000, 99999))
        
        # Question submission form
        with st.form(key="query_form"):
            user_query = st.text_input("Ask Your Math Question", key="query_input")
            submit_button = st.form_submit_button("Solve", disabled=st.session_state.get("is_generating", False))
        
        # Process new question submission
        if submit_button and user_query.strip():
            # Reset feedback process when a new question is submitted
            reset_feedback_state()
            
            st.session_state.is_generating = True
            with st.spinner("Solving your question..."):
                math_agent = build_math_agent()
                initial_state = GraphState(
                    question=user_query,
                    raw_solution="",
                    generation="",
                    documents=[],
                    web_search_needed="",
                    web_search_sufficient="",
                    web_results="",
                    human_feedback="",
                    should_end=False
                )
                result = math_agent.invoke(initial_state)
            
            st.session_state.current_state = result
            st.session_state.is_generating = False
        
        # Process and display current solution with feedback loop
        if st.session_state.current_state:
            # Show processing indicator if feedback is being processed
            if st.session_state.get("processing_feedback", False):
                st.info("Processing your feedback and generating a new solution...")
                if st.session_state.get("temp_state"):
                    # Show a placeholder with the previous solution and the feedback
                    temp_state = st.session_state.temp_state
                    st.subheader("Previous Solution")
                    st.markdown(f"**Question:** {temp_state['question']}")
                    st.markdown(st.session_state.get("last_generation", ""))
                    st.write("---")
                    st.subheader("Your Feedback")
                    st.write(st.session_state.get("current_feedback", ""))
            else:
                # Normal flow - show the solution with feedback options
                final_state = human_in_the_loop(st.session_state.current_state)
                st.session_state.current_state = final_state
    
    # Feedback records tab
    with tab2:
        st.header("Feedback Records")
        feedback_file = Path("./feedback_data/feedback_log.json")
        if feedback_file.exists():
            try:
                with open(feedback_file, "r") as f:
                    feedback_data = json.load(f)
                if feedback_data:
                    st.write(f"Found {len(feedback_data)} feedback records.")
                    json_str = json.dumps(feedback_data, indent=2)
                    st.download_button("Download Feedback Data", json_str, "math_agent_feedback.json", "application/json")
                    for i, entry in enumerate(reversed(feedback_data)):
                        with st.expander(f"Entry {len(feedback_data) - i}: {entry['timestamp']}"):
                            st.write(f"**Question:** {entry['question']}")
                            st.write(f"**Solution:** {entry['solution']}")
                            if entry.get("rating"):
                                st.write(f"**Rating:** {'⭐' * entry['rating']}")
                            else:
                                st.write("**Rating:** Not provided")
                            st.write(f"**Feedback:** {entry.get('feedback', 'None (Accepted)')}")
                else:
                    st.info("No feedback records found yet.")
            except Exception as e:
                st.error(f"Error loading feedback: {str(e)}")
        else:
            st.info("No feedback records found yet.")

if __name__ == "__main__":
    main()