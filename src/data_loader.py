# Read the JSON files from the dataset

import os
import json
from langchain.schema import Document
from src.utils import latex_to_text

# Dataset Loader with Cleaning
def load_math_dataset(base_path):
    """
    Loads JSON files from topic folders, cleans LaTeX, and returns LangChain Documents.
    """
    documents = []

    for topic_folder in os.listdir(base_path):
        topic_path = os.path.join(base_path, topic_folder)
        
        if not os.path.isdir(topic_path):
            continue

        for file_name in os.listdir(topic_path):
            if file_name.endswith(".json"):
                file_path = os.path.join(topic_path, file_name)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
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
                    print(f"Error reading {file_path}: {e}")

    print(f"Loaded {len(documents)} documents.")
    return documents