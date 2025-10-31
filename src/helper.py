import os
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

# ✅ Load HF token securely (from Hugging Face Secrets or .env)
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token, add_to_git_credential=False)

# ✅ Load model once (not on every request)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", use_auth_token=hf_token)


def get_query_embedding(query: str):
    """
    Returns vector embedding for a query string using SentenceTransformer model.

    Args:
        query (str): User entered text/query

    Returns:
        list: Embedding vector (list of floats)
    """
    embedding = model.encode(query).tolist()
    return embedding
