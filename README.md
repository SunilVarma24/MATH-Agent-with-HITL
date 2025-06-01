# Agentic RAG: Math Agent

## Introduction  
The **Agentic RAG: Math Agent with Feedback based Learning** is designed to simulate a human math professor that:  
- Understands a student's question  
- Retrieves relevant examples  
- Performs web search if necessary  
- Generates a **detailed, step-by-step explanation**  
- Improves responses based on human feedback  

The system guarantees **safe, accurate, and domain-specific reasoning**, particularly for mathematics.


## How It Works  

### 1. AI Gateway with Guardrails  
- **Input Guardrails**: A Gemini 2.0 Flash model classifies whether the input is **math-related**. If not, it politely rejects the query.  
- **Output Guardrails**: Gemini 1.5 Pro validates that the final answer is complete, logically correct, and step-by-step. If not, it regenerates the output.

### 2. Knowledge Base Creation  
- A dataset of **12,000 math questions** across topics (Algebra, Geometry, Number Theory, etc.) is used.  
- The dataset is embedded using the **`bge-small-en-v1.5`** model.  
- Stored in a **ChromaDB Vector Store** with indexed retrieval based on similarity threshold.

### 3. Retrieval and Grading Pipeline  
- A **similarity-based retriever** fetches highly relevant documents.  
- A **document grader** (Gemini 1.5 Pro) evaluates if the retrieved context is:  
  - Relevant to the query  
  - Sufficient to generate a correct response  
- If context is insufficient → triggers **Web Search**.

### 4. Web Search Pipeline  
- Uses **Tavily** to fetch web results.  
- Results are validated to ensure they contain relevant **mathematical content**.  
- If no good source is found → responds with a polite message, avoiding hallucinations.

### 5. Response Generation  
- Uses all available and approved context to generate a **step-by-step explanation** using Gemini 2.0 Flash.

### 6. Output Validation (Output Guardrail)  
- The generated response is **validated** by Gemini 1.5 Pro for:  
  - Mathematical correctness  
  - Completeness  
  - Logical clarity  
- If invalid → regenerated until valid.

### 7. Human-in-the-Loop (HITL) Feedback Loop  
- User is prompted to **rate** the answer from 1 to 5.  
- If rated poorly:  
  - User provides **feedback**  
  - The system regenerates a better response using both feedback and original output  
  - Loop continues until user accepts  
- Accepted responses are stored with the rating for future model tuning.

## Installation  

To run the app locally:  
```bash
pip install -r requirements.txt