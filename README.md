

# 📚 AI Book & Research Paper Recommender

![Project Banner](https://img.shields.io/badge/ML-Content--Based-blue) ![Python](https://img.shields.io/badge/Python-3.11-green) ![FastAPI](https://img.shields.io/badge/FastAPI-API-red) ![DVC](https://img.shields.io/badge/DVC-Versioning-orange)

A **content-based recommendation system** that suggests top **books and research papers** based on user queries.
The project integrates multiple components — **data ingestion, preprocessing, feature extraction, and recommendation** — into a reproducible pipeline.

---

## 🚀 Overview

This project helps users discover relevant academic and technical resources by leveraging **semantic similarity** between user queries and textual data from books and papers.

* 📘 **Book data** fetched from **Google Books API**
* 📄 **Research paper data** fetched from **Semantic Scholar API**
* 🧠 Uses **Sentence Transformers** to compute semantic embeddings
* ⚙️ Implements **data version control (DVC)** for reproducibility
* 🐳 Deployable using **Docker**, compatible with **Vercel / Render**
* 💾 Pipelines for ingestion, cleaning, feature building, and model inference

---

## 🧩 Features

* 🔎 Recommend **top N books and papers** for any query (AI, ML, NLP, or Electronics-related).
* 🧠 **Sentence Transformer embeddings** for deep semantic understanding.
* 🧾 Metadata-aware ranking using title, authors, publisher, year, and citations.
* ⚡ **FastAPI-based backend** and **responsive web frontend** (HTML + TailwindCSS).
* 🔄 **DVC-tracked pipelines** ensure reproducibility of every experiment stage.
* 🧱 **Modular architecture** for extending recommendation logic or adding new models easily.
* 📦 **Config-driven pipeline** (YAML-based) for flexible experimentation.

---

## 🧠 Approach to Solution

1. **Data Ingestion** – Collected raw data from **Google Books API** and **Semantic Scholar API** for AI, ML, NLP, and Electronics-related topics.
2. **Preprocessing** – Cleaned and standardized metadata (author, publisher, descriptions). Unknown fields were labeled as `"Unknown"`.
3. **Feature Engineering** – Combined textual data (title, author, abstract, description) into a single representation.
4. **Embeddings** – Generated high-dimensional embeddings using **Sentence Transformers** to capture semantic meaning.
5. **Similarity Computation** – Calculated **cosine similarity** between user query embeddings and resource embeddings.
6. **Ranking & Recommendation** – Returned top N matches sorted by similarity scores.
7. **Versioning & Reproducibility** – Managed datasets, intermediate files, and trained models using **DVC**.
8. **Deployment** – Containerized the FastAPI app with **Docker**, ready for deployment on **Streamlit/HuggingFace**.

---

## 🧩 MLOps & Pipeline Design

| Component                    | Description                                                                           |
| ---------------------------- | ------------------------------------------------------------------------------------- |
| **DVC**                      | Used for data and model versioning, ensuring reproducibility and experiment tracking. |
| **Modular Pipeline**         | Separate stages for ingestion, cleaning, feature building, training, and evaluation.  |
| **YAML Configuration**       | Dynamic config-driven system for changing topics, paths, and model parameters.        |
| **Logging & Error Handling** | Custom logger for tracing pipeline execution and runtime issues.                      |
| **CI/CD Ready**              | Compatible with GitHub Actions for automated data validation and deployment.          |

---

## 🧱 Project Structure

```
BOOK-RECOMMENDATION/
├── LICENSE
├── README.md
├── Makefile                     # Useful commands: make data / make train
├── configs                      # Config files (YAML for models and ingestion)
│   ├── books_topics.yaml
│   └── papers_topics.yaml
│
├── data                         # Data versioned by DVC
│   ├── raw                      # Raw data from APIs
│   ├── interim                  # Cleaned/merged data
│   ├── processed                # Final datasets ready for modeling
│   └── external                 # Third-party or sample data
│
├── models                       # Stored model embeddings or trained models
│
├── src
│   ├── data                     # Data engineering scripts
│   │   ├── ingestion.py         # Fetches data from APIs
│   │   ├── cleaning.py          # Handles missing values, standardizes fields
│   │   ├── build_features.py    # Creates embeddings and features
│   │   ├── splitting.py         # Handles dataset splitting
│   │   └── validation.py        # Validates pipeline data integrity
│   │
│   ├── models                   # ML logic for recommendation
│   │   ├── model.py             # Embedding + Similarity logic
│   │   ├── train.py             # Embedding generation
│   │   ├── predict.py           # Top N recommendation
│   │   ├── preprocessing.py     # Text preprocessing utilities
│   │   └── hyperparameter_tuning.py
│   │
│   └── visualization
│       ├── evaluation.py        # Evaluate ranking performance
│       └── exploration.py       # Exploratory plots for data analysis
│
├── app.py                       # FastAPI entry point
├── Dockerfile                   # For containerized deployment
├── dvc.yaml                     # Defines the pipeline stages
├── dvc.lock                     # Tracks file versions
├── requirements.txt
└── setup.py
```

---

## ⚙️ Tech Stack

* **Backend:** FastAPI
* **Model:** Sentence Transformers (`all-MiniLM-L6-v2`)
* **Data Management:** DVC
* **Frontend:** TailwindCSS + HTML Templates
* **Deployment:** Docker + Render/Vercel
* **APIs Used:** Google Books API, Semantic Scholar API

---

## 🧰 Installation

```bash
git clone https://github.com/yourusername/Book-Recommendation.git
cd Book-Recommendation
pip install -r requirements.txt
```

### Run locally

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

---

## 🪣 Using Google Drive as DVC Remote

Add remote:

```bash
dvc remote add -d gdrive gdrive://<folder-id>
dvc push
```

---

## 🐳 Docker Setup

```bash
docker build -t book_recommendation .
docker run -p 7860:7860 book_recommendation
```

---

## 📈 Future Enhancements

* 🧩 Integrate **hybrid filtering** (content + collaborative).
* 📚 Add **dynamic topic expansion** using LLMs.
* 🔄 Auto-update new papers/books from APIs.
* ☁️ Deploy with **CI/CD** using GitHub Actions + DVC Remote.
* 🧮 Store embeddings in **vector database** (FAISS / Pinecone).

---