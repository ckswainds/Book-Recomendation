FROM python:3.10-slim

# Create working dir
WORKDIR /app

# Copy requirements first (better caching for Docker)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (only required files)
COPY src/ ./src/
COPY app.py .
COPY templates/ ./templates/
COPY static/ ./static/
COPY data/ ./data/

# Make "src" discoverable as a package
ENV PYTHONPATH="/app"

# Expose Streamlit port for Hugging Face Spaces
EXPOSE 7860

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
