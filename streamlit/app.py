# app.py (Streamlit frontend for FastAPI backend)

import streamlit as st
import requests
import json
from pathlib import Path

# Define the FastAPI endpoint
FASTAPI_URL = "http://127.0.0.1:8000/ask"
FASTAPI_HUMAN_REVIEW_URL = "http://127.0.0.1:8000/human_review"

def main():
    st.set_page_config(page_title="MATH Agent", page_icon="🧮", layout="wide")
    st.title("🧮 Agentic RAG: MATH Agent - Advanced Mathematical Problem Solver")
    st.markdown("""This application uses AI to solve mathematical problems step-by-step, drawing on both a knowledge base and web search when needed.""")

    tab1, tab2 = st.tabs(["Ask Questions", "View Feedback"])

    with tab1:
        st.header("Ask your Math Question")
        st.markdown("""
        Enter your math problem below. The agent will attempt to solve it step-by-step, 
        using both its knowledge base and web search if necessary.
        """)

        # Input box
        user_input = st.text_area(
            "Enter your math problem:",
            placeholder="e.g., What is the integral of x^2?"
        )

        # Initialize session state
        if "solution" not in st.session_state:
            st.session_state.solution = ""
        if "session_id" not in st.session_state:
            st.session_state.session_id = ""
        if "review_phase" not in st.session_state:
            st.session_state.review_phase = "none"
        if "current_rating" not in st.session_state:
            st.session_state.current_rating = 3
        if "feedback_text" not in st.session_state:
            st.session_state.feedback_text = ""
        if "iteration_count" not in st.session_state:
            st.session_state.iteration_count = 0
        if "is_regenerated" not in st.session_state:
            st.session_state.is_regenerated = False

        # Generate button
        if st.button("Solve Problem"):
            if not user_input.strip():
                st.warning("Please enter a prompt.")
            else:
                st.info("Sending prompt to FastAPI backend...")
                
                # Reset session state for new question
                st.session_state.solution = ""
                st.session_state.session_id = ""
                st.session_state.review_phase = "none"
                st.session_state.current_rating = 3
                st.session_state.feedback_text = ""
                st.session_state.iteration_count = 0
                st.session_state.is_regenerated = False

                try:
                    response = requests.post(FASTAPI_URL, json={"question": user_input})

                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.session_id = result.get("session_id")
                        st.session_state.solution = result.get("generation", "")

                        # Check if it is not a math question
                        if "I'm sorry, I can only help with math questions." in st.session_state.solution:
                            st.error(st.session_state.solution)
                        elif "I'm sorry, I couldn't find any reliable information online to answer your question." in st.session_state.solution:
                            st.error(st.session_state.solution)
                        else:
                            st.session_state.review_phase = "rating"
                            st.success("✅ Response received:")
                            st.rerun()

                    else:
                        st.error(f"🚨 Error {response.status_code}: {response.json().get('error', 'Unknown error')}")
                except Exception as e:
                    st.error(f"❌ Exception occurred: {str(e)}")

        # Display current solution if exists
        if st.session_state.solution and "I'm sorry" not in st.session_state.solution:
            st.markdown("### Current Solution:")

            # Show regeneration message if applicable
            if st.session_state.is_regenerated and st.session_state.iteration_count > 0:
                st.info(f"🔄 **Solution regenerated (Iteration {st.session_state.iteration_count})! Please review the new solution.**")


            st.markdown(st.session_state.solution)
            st.markdown("---")

            # PHASE 1: Rating
            if st.session_state.review_phase == "rating":
                st.subheader("Step 1: Rate the Solution")
                rating = st.slider("⭐ Rate this solution (1-5):", 1, 5, st.session_state.current_rating)
                
                if st.button("Submit Rating"):
                    st.session_state.current_rating = rating
                    st.session_state.review_phase = "approval"
                    st.rerun()

            # PHASE 2: Approval
            elif st.session_state.review_phase == "approval":
                st.subheader("Step 2: Accept or Request Changes")
                st.write(f"**Your Rating:** {st.session_state.current_rating} ⭐")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("✅ Accept Solution"):
                        # Submit acceptance
                        payload = {
                            "human_feedback": "",
                            "rating": st.session_state.current_rating,
                            "approval": "Yes"
                        }
                        try:
                            response = requests.post(
                                FASTAPI_HUMAN_REVIEW_URL,
                                params={"session_id": st.session_state.session_id},
                                json=payload
                            )
                            if response.status_code == 200:
                                st.success("✅ Solution accepted! Feedback saved.")
                                st.session_state.review_phase = "none"
                                st.rerun()
                            else:
                                st.error("Failed to submit acceptance.")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

                with col2:
                    if st.button("❌ Request Changes"):
                        st.session_state.review_phase = "feedback"
                        st.rerun()

            # PHASE 3: Feedback
            elif st.session_state.review_phase == "feedback":
                st.subheader("Step 3: Provide Feedback")
                st.write(f"**Your Rating:** {st.session_state.current_rating} ⭐")
                
                feedback_text = st.text_area(
                    "Please provide your feedback or corrections:",
                    value=st.session_state.feedback_text,
                    height=100
                )
                
                if st.button("Submit Feedback & Regenerate"):
                    if not feedback_text.strip():
                        st.warning("Please provide feedback before submitting.")
                    else:
                        st.session_state.feedback_text = feedback_text
                        
                        payload = {
                            "human_feedback": feedback_text,
                            "rating": st.session_state.current_rating,
                            "approval": "No"
                        }
                        
                        try:
                            st.info("Regenerating solution based on your feedback...")
                            response = requests.post(
                                FASTAPI_HUMAN_REVIEW_URL,
                                params={"session_id": st.session_state.session_id},
                                json=payload
                            )

                            if response.status_code == 200:
                                result = response.json()
                                new_solution = result.get("generation", "")
                                
                                if new_solution and new_solution != st.session_state.solution:
                                    # Replace old solution with new one
                                    st.session_state.solution = new_solution
                                    # Reset for next iteration
                                    st.session_state.review_phase = "rating"
                                    st.session_state.current_rating = 3
                                    st.session_state.feedback_text = ""
                                    st.session_state.iteration_count += 1
                                    st.session_state.is_regenerated = True
                                    st.rerun()
                                else:
                                    st.warning("No new solution generated. Please try different feedback.")
                            else:
                                st.error(f"Error {response.status_code}: {response.json().get('error', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"Exception: {str(e)}")


    with tab2:
        st.header("View Feedback")
        st.markdown("""
        This section will display feedback from users on the solutions provided by the MATH Agent.
        """)

        feedback_file = Path("./feedback_data/feedback_log.json")

        if feedback_file.exists():
            with feedback_file.open("r") as f:
                feedback_data = json.load(f)
            if feedback_data:
                st.write(f"Found {len(feedback_data)} feedback records.")
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
        else:
            st.info("Feedback log file does not exist. No feedback has been recorded yet.")

if __name__ == "__main__":
    main()