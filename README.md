# Human-in-the-Loop: Feedback Based Learning - Math Agent

## Introduction  
The goal is to design a **math teaching agent** that simplifies complex math problems and generates **clear step-by-step solutions**. This intelligent agent first checks its **preloaded math knowledge base** and, if needed, falls back to the web for information. For safety and accuracy, the system incorporates **AI guardrails** and **HITL validation**.

## Architecture Overview  

### 1. AI Gateway & Guardrails  
- Integrated **input/output guardrails** to validate that all questions and responses are educational and math-focused.  
- Blocks irrelevant or unsafe queries.  

### 2. Knowledge Base Creation  
- Preprocessed math questions are stored in **ChromaDB** using vector embeddings (`bge-small-en-v1.5`).  
- If a user query matches existing content, the system retrieves and responds using the **Gemini 1.5 Flash** model.  

### 3. Web Search Fallback  
- If no relevant documents are found in the knowledge base, the query is rewritten and a **web search is performed via Tavily**.  
- If no reliable online content is available, the system **avoids hallucinating answers**.  

### 4. Human-in-the-Loop Feedback  
- The final step involves a **feedback loop** where human users rate or correct responses.  
- These corrections can be used for **improving** system accuracy over time.  

## Installation  

To run the app locally:  
```bash
pip install -r requirements.txt