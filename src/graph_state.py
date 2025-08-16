# Graph State

from typing import TypedDict, List
from langchain.schema import Document

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
    rating: int
    approval: str