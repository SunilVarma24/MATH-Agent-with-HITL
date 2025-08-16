from datetime import datetime
import json
from pathlib import Path

# Function to save feedback to disk
def save_feedback(question, solution, feedback, rating=None):
    # Get base directory (streamlit/)
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

    #print(f"📝 Feedback logged at {timestamp}")