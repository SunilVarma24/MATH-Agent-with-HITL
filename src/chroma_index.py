# Create or Load Chroma Index

from pathlib import Path
from langchain.vectorstores import Chroma
from src.data_loader import load_math_dataset
from src.config import embedding_model

# Get the absolute path to the directory containing this file
BASE_DIR = Path(__file__).resolve().parent

# Path to the folder that contains chroma.sqlite3
CHROMA_DIR = BASE_DIR.parent / 'vectorstore' / 'math_db'

# Path to the folder that contains chroma.sqlite3
DATASET_DIR = BASE_DIR.parent / 'data' / 'MATH'

# Load the dataset from the specified path
all_documents = load_math_dataset(DATASET_DIR)

if CHROMA_DIR.exists():
    chroma_db = Chroma(
    embedding_function=embedding_model,
    persist_directory=str(CHROMA_DIR),
    collection_name="math_database"
    )
    print("Chroma index loaded from disk.")
else:
    chroma_db = Chroma.from_documents(
    all_documents,
    embedding=embedding_model,
    persist_directory=str(CHROMA_DIR),
    collection_name="math_database"
    )
    print("Chroma Index Created and Saved to Disk.")

# Retriever with Similarity Threshold
retriever = chroma_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.8 
    }
)