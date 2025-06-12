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
from IPython.display import Image, display, Markdown
from typing import List
from typing_extensions import TypedDict
import os
import json
from pylatexenc.latex2text import LatexNodes2Text
from dotenv import load_dotenv
import re
import tempfile
from pathlib import Path
from datetime import datetime
import time
import uuid

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Graph State
class GraphState(TypedDict):
    question: str
    raw_solution: str
    previous_solution: str
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
                                            model_name="BAAI/bge-small-en-v1.5",
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

        # Get the absolute path to the directory containing this file
        BASE_DIR = Path(__file__).resolve().parent

        # Path to the folder that contains chroma.sqlite3
        CHROMA_DIR = BASE_DIR.parent / 'vectorstore' / 'math_db'

        if CHROMA_DIR.exists():
            st.session_state.chroma_db = Chroma(
                embedding_function=st.session_state.embedding_model,
                persist_directory=str(CHROMA_DIR),
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
        
        assessment_chain = assess_prompt | st.session_state.eval_llm | StrOutputParser()
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
        # If we already have a solution *and* no new feedback, skip
        if state.get("raw_solution") and not state.get("human_feedback"):
            return state
        docs = "\n\n".join(d.page_content for d in state["documents"])
        context = docs + "\n\n" + state["web_results"]
        feedback = state.get("human_feedback", "")
        prev_solution = state.get("raw_solution", "")

        prompt = ChatPromptTemplate.from_messages([
            ("system", 
            "You are a expert mathematics professor helping students understand the math problems clearly.\n\n"
            "Your task is to:\n"
            "1. Understand the math question.\n"
            "2. Use the provided context to identify key formulas, definitions, or values.\n"
            "3. Solve the question step-by-step using clear reasoning and intermediate steps.\n"
            "4. Show all formulas and substitutions, and explain your logic.\n"
            "5. End with a final boxed answer like: \\boxed{{your_final_answer}}.\n"
            "6. If human feedback and previous answer are provided, follow the human feedback carefully to improve or correct your previous answer.\n\n"
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

        chain = prompt | st.session_state.llm | StrOutputParser()
        raw = chain.invoke({"context": context, "question": state["question"], 
                            "feedback_block": feedback_block, 
                            "previous_answer_block": previous_answer_block})
        return {**state, "previous_solution": prev_solution, "raw_solution": raw, "generation": "", "human_feedback": ""}

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
    # Get base directory
    BASE_DIR = Path(__file__).resolve().parent

    # Construct full path to feedback_data/feedback_log.json
    feedback_file = BASE_DIR.parent / "feedback_data" / "feedback_log.json"

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


def initialize_feedback_session():
    """Initialize feedback session with unique identifiers"""
    if "feedback_session_id" not in st.session_state:
        st.session_state.feedback_session_id = str(uuid.uuid4())
    if "feedback_iterations" not in st.session_state:
        st.session_state.feedback_iterations = 0
    if "phase" not in st.session_state:
        st.session_state.phase = "rating"
    if "processing_feedback" not in st.session_state:
        st.session_state.processing_feedback = False
    # Add this new flag to prevent multiple calls
    if "initial_display_done" not in st.session_state:
        st.session_state.initial_display_done = False

def reset_feedback_state():
    """Reset all feedback-related session state variables"""
    st.session_state.phase = "rating"
    st.session_state.feedback_iterations = 0
    st.session_state.processing_feedback = False
    st.session_state.current_state = None
    st.session_state.initial_display_done = False  # Reset this flag too
    # Generate new session ID for completely fresh start
    st.session_state.feedback_session_id = str(uuid.uuid4())
    # Clear any stored temporary states
    if "temp_state" in st.session_state:
        del st.session_state.temp_state
    if "current_rating" in st.session_state:
        del st.session_state.current_rating
    if "current_feedback" in st.session_state:
        del st.session_state.current_feedback

def human_in_the_loop(state):
    # Initialize feedback session
    initialize_feedback_session()
    
    # Get current iteration and phase
    it = st.session_state.get("feedback_iterations", 0)
    phase = st.session_state.get("phase", "rating")
    session_id = st.session_state.get("feedback_session_id", "default")
    
    # Create unique keys based on session ID and iteration
    def get_unique_key(base_key):
        return f"{base_key}_{session_id}_iter{it}_{phase}"  # Added phase to make it more unique
    
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
        return {**state, "human_feedback": ""}
    
    # Maximum iterations check
    if st.session_state.feedback_iterations >= 3:
        st.warning("Maximum feedback iterations (3) reached. Final solution saved.")
        save_feedback(state["question"], state["generation"], "Max iterations reached", rating=None)
        reset_feedback_state()
        return {**state, "human_feedback": ""}
    
    # STEP 1: Rating phase
    if phase == "rating":
        rating = st.slider("⭐ Rate this solution:", 1, 5, 3, key=get_unique_key("rating"))
        if st.button("Submit Rating", key=get_unique_key("submit_rating")):
            st.session_state.current_rating = rating
            st.session_state.phase = "approval"
            st.session_state.initial_display_done = True
            st.rerun()
        return state
    
    # STEP 2: Approval phase
    if phase == "approval":
        st.write(f"Rating: {st.session_state.current_rating} ⭐")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Accept", key=get_unique_key("accept")):
                save_feedback(
                    state["question"], state["generation"], "", rating=st.session_state.current_rating
                )
                st.success("Solution accepted!")
                reset_feedback_state()
                return {**state, "human_feedback": "", "previous_solution": state["raw_solution"],}
        with c2:
            if st.button("❌ Request Changes", key=get_unique_key("request_changes")):
                st.session_state.phase = "feedback"
                st.rerun()
        return state
    
    # STEP 3: Feedback and regeneration phase
    if phase == "feedback":
        fb = st.text_area("✏️ Enter your feedback:", key=get_unique_key("feedback_text"))
        submit_fb = st.button("Submit Feedback", key=get_unique_key("submit_feedback"))
        
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
                previous_solution=state["raw_solution"],
                raw_solution="",
                generation="",
                documents=state.get("documents", []),
                web_search_needed=state.get("web_search_needed", ""),
                web_search_sufficient=state.get("web_search_sufficient", ""),
                web_results=state.get("web_results", ""),
                should_end=False
            )
            
            try:
                with st.spinner("🧮 Generating new solution with your feedback..."):
                    # Resume graph from 'generate'
                    math_agent = build_math_agent()
                    # Run the regeneration process
                    new_state = generate(new_state)
                    new_state = output_guardrails(new_state)
                
                # Increment feedback iteration and reset phase
                st.session_state.feedback_iterations += 1
                st.session_state.phase = "rating"
                st.session_state.processing_feedback = False
                
                # Update the current state with the new solution
                st.session_state.current_state = new_state
                st.rerun()
                
            except Exception as e:
                st.error(f"Error regenerating solution: {str(e)}")
                st.session_state.processing_feedback = False
                return state
        
        # If user hasn't submitted feedback yet, show the current state
        return state

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
            st.session_state.initial_display_done = False  # Reset this flag for the new question
            st.rerun()  # Force a rerun to properly display the feedback interface
        
        # Process and display current solution with feedback loop
        if st.session_state.current_state and not st.session_state.get("is_generating", False):
            # Show processing indicator if feedback is being processed
            if st.session_state.get("processing_feedback", False):
                st.info("Processing your feedback and generating a new solution...")
            else:
                # Normal flow - show the solution with feedback options
                # Only call human_in_the_loop if we're not in the initial processing phase
                if st.session_state.get("initial_display_done", False) or st.session_state.get("phase", "rating") != "rating":
                    final_state = human_in_the_loop(st.session_state.current_state)
                    st.session_state.current_state = final_state
                else:
                    # First time display - just show the interface without processing
                    human_in_the_loop(st.session_state.current_state)
                    st.session_state.initial_display_done = True
    
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