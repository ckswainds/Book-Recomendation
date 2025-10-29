# # Use official Python image
# FROM python:3.10-slim

# # Metadata labels (best practice)
# LABEL Name="book_recommender" \
#       Version="1.0" \
#       Description="Book Recommendation System using ML & FastAPI" \
#       Maintainer="Chandan Kumar Swain"

# # Set working directory
# WORKDIR /app

# # Copy project files
# COPY . /app

# COPY data/interim/modified_books.csv /app/data/interim/modified_books.csv
# COPY data/interim/modified_papers.csv /app/data/interim/modified_papers.csv

# # Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Expose port for the app (Hugging Face / Render expects 7860)
# EXPOSE 7860

# # Run the FastAPI app
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

FROM python:3.10-slim

WORKDIR /app

# Copy ONLY src first (avoids nesting issues)
COPY src/ /app/src/

# Copy everything else (app.py / data / requirements.txt)
COPY . /app/
COPY templates/ /app/templates/
COPY static/ /app/static/
# Make "src" discoverable as a package
ENV PYTHONPATH="/app"

RUN pip install --no-cache-dir -r requirements.txt

# EXPOSE 7860
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
EXPOSE 7860
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
# CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
CMD ["bash", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & sleep 5 && streamlit run streamlit_app.py --server.port 7860 --server.address 0.0.0.0"]


# CMD ["bash", "-c", "uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 & sleep 5 && streamlit run streamlit_app.py --server.port 7860 --server.address 0.0.0.0"]
