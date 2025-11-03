from huggingface_hub import InferenceClient
from huggingface_hub import login
import numpy as np
import os


token = os.getenv("HF_TOKEN") 
login(token)
client = InferenceClient(token=token)
def get_query_embedding(query: str):
    response = client.feature_extraction(
        model="sentence-transformers/all-MiniLM-L6-v2",
        text=query
    )
    return np.array(response)