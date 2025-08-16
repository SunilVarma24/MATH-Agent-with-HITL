# Use a stable Python version that matches your local environment
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy only requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Now copy the rest of the app
COPY . .

# Expose ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# Add a startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Run the FastAPI app and Streamlit app
CMD ["/start.sh"]