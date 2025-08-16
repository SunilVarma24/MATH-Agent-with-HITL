from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from tavily import AsyncTavilyClient

# Load environment variables
load_dotenv()

# Initialize LLM and Embedding Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0
)

eval_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

#web_search_tool = TavilySearchResults(max_results=3, search_depth='advanced', max_tokens=10000)
# Initialize async Tavily client
web_search_tool = AsyncTavilyClient()