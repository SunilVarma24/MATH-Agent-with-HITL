#!/bin/bash

# Start FastAPI app in the background
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit app
streamlit run streamlit/app.py --server.port=8501 --server.address=0.0.0.0
