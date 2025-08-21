# Agentic RAG: Math Agent

## Introduction  
The **Agentic RAG: Math Agent with Feedback based Learning** is designed to simulate a human math professor that:  
- Understands a student's question  
- Retrieves relevant examples  
- Performs web search if necessary  
- Generates a **detailed, step-by-step explanation**  
- Improves responses based on human feedback  

The system guarantees **safe, accurate, and domain-specific reasoning**, particularly for mathematics.

## Features
- **Interactive Learning**: Engages students with questions and hints.
- **Step-by-Step Solutions**: Breaks down complex problems into manageable steps.
- **Web Search Capabilities**: Retrieves up-to-date information from the web.
- **Feedback Loop**: Learns from user interactions to improve future responses.

## How It Works  

### 1. AI Gateway with Guardrails  
- **Input Guardrails**: A Gemini 2.0 Flash model classifies whether the input is **math-related**. If not, it politely rejects the query.  
- **Output Guardrails**: Gemini 2.5 Flash validates that the final answer is complete, logically correct, and step-by-step. If not, it regenerates the output.

### 2. Knowledge Base Creation  
- A dataset of **12,000 math questions** across topics (Algebra, Geometry, Number Theory, etc.) is used.  
- The dataset is embedded using the **`bge-small-en-v1.5`** model.  
- Stored in a **ChromaDB Vector Store** with indexed retrieval based on similarity threshold.

### 3. Retrieval and Grading Pipeline  
- A **similarity-based retriever** fetches highly relevant documents.  
- A **document grader** (Gemini 2.5 Flash) evaluates if the retrieved context is:  
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
- The generated response is **validated** by Gemini 2.5 Flash for:  
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
```

## Running the Application

After installing the dependencies, you can start the backend and frontend as follows:

### 1. Start the FastAPI Backend
```bash
uvicorn main:app --reload
```
This will launch the API server at [http://127.0.0.1:8000](http://127.0.0.1:8000).

### 2. Start the Streamlit Frontend
```bash
streamlit run streamlit/app.py
```
This will open the interactive Math Agent UI in your browser.

---
**Note:**  
- Make sure both the backend and frontend are running for full functionality.
- The default configuration assumes both run on your local machine.

## Cloud Deployment
To deploy the application on Azure, we build a Docker image and push the image to Azure Container Registry (ACR) and then run the image in a Azure Container Instance (ACI).
### 0. start.sh
- This script is used to start the FastAPI and Streamlit applications concurrently. It ensures that both services are running and accessible.
- Create a Dockerfile in the project root directory.
- Create a docker-compose.yml file to manage the services.

### 1. Build the Docker Image
```bash
docker build -t myacrmathagent.azurecr.io/myapp:latest .
```
### 2. Create a Resource Group
```bash
az group create --name myResourceGroup --location eastus
```
### 3. Create an Azure Container Registry
```bash
az acr create --resource-group myResourceGroup --name myacrmathagent --sku Basic --admin-enabled true
```
### 4. Login to Azure Container Registry
```bash
az acr login --name myacrmathagent
```
### 5. Push the Docker Image to Azure Container Registry
```bash
docker push myacrmathagent.azurecr.io/myapp:latest
```
### 6. Create Azure Container Instance
```bash
az container create --resource-group myResourceGroup --name myContainer --image myacrmathagent.azurecr.io/myapp:latest --registry-login-server myacrmathagent.azurecr.io --registry-username <your-username> --registry-password <your-password> --os-type Linux --ports 8000 8501 --cpu 4 --memory 8 --environment-variables OPENAI_API_KEY=your_openai_api_key TAVILY_API_KEY=your_tavily_api_key --dns-name-label mathagent --ip-address public
```
### 7. Get the IP and DNS
```bash
az container show --resource-group myResourceGroup --name myContainer --query "{IP: ipAddress.ip, DNS:ipAddress.fqdn}" -o table
```
### 8. Access the Application
* FastAPI API: `http://<your-public-ip>:8000/docs`
* Streamlit UI: `http://<your-public-ip>:8501`
### 9. Stop the Application
```bash
az container stop --resource-group myResourceGroup --name myContainer
```
### 10. Delete the Container
```bash
az container delete --resource-group myResourceGroup --name myContainer --yes
```
### 11. Delete the Resource Group
```bash
az group delete --name myResourceGroup --yes --no-wait
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Conclusion
The Agentic RAG: Math Agent provides a robust framework for interactive math problem solving, leveraging advanced AI techniques and human feedback to continuously improve its performance. By following the setup instructions, you can deploy the system locally and start exploring its capabilities.