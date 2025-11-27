import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import requests

# -------------------------------
# 1. Load JSON data
# -------------------------------
with open("experiences.json", "r") as f:
    experiences = json.load(f)

with open("writing_samples.json", "r") as f:
    writing_samples = json.load(f)

# -------------------------------
# 2. Embedding function (local HF model)
# -------------------------------
hf_model = SentenceTransformer("all-MiniLM-L6-v2")

def local_embedding_fn(texts):
    return hf_model.encode(texts, convert_to_numpy=True).tolist()

# -------------------------------
# 3. ChromaDB Setup
# -------------------------------
client = chromadb.Client()  # local in-memory DB

# Create or load collections
try:
    exp_collection = client.get_collection("experiences")
except:
    exp_collection = client.create_collection("experiences")

try:
    writing_collection = client.get_collection("writing_samples")
except:
    writing_collection = client.create_collection("writing_samples")

# Populate DB if empty
if exp_collection.count() == 0:
    exp_collection.add(
        documents=[e["description"] for e in experiences],
        ids=[e["id"] for e in experiences],
        embeddings=local_embedding_fn([e["description"] for e in experiences])
    )

if writing_collection.count() == 0:
    writing_collection.add(
        documents=[w["content"] for w in writing_samples],
        ids=[w["id"] for w in writing_samples],
        embeddings=local_embedding_fn([w["content"] for w in writing_samples])
    )

# -------------------------------
# 4. Retrieval function
# -------------------------------
def retrieve_relevant(question, top_exp=3, top_write=2):
    exps = exp_collection.query(
        query_texts=[question],
        n_results=top_exp,
        include=["documents"]
    )
    samples = writing_collection.query(
        query_texts=[question],
        n_results=top_write,
        include=["documents"]
    )

    exp_texts = exps["documents"][0]
    writing_texts = samples["documents"][0]

    return exp_texts, writing_texts

# -------------------------------
# 5. Generate answers via cloud-hosted model
# -------------------------------
def generate_answers_cloud(question, exp_texts, writing_texts, n_answers=2):
    context = (
        "Relevant Experiences:\n" +
        "\n".join(exp_texts) +
        "\n\nWriting Samples:\n" +
        "\n".join(writing_texts)
    )

    payload = {
        "question": question,
        "context": context,
        "n_answers": n_answers
    }

    # Replace with your cloud model URL
    response = requests.post("https://your-model-server.com/generate", json=payload)
    response.raise_for_status()  # raise error if request fails

    return response.json()["answers"]

# -------------------------------
# 6. Example usage
# -------------------------------
if __name__ == "__main__":
    question = "Tell us about a time you solved a technical problem "

    exp_texts, writing_texts = retrieve_relevant(question)
    print( "Exp Tests: ")
    for i , text in enumerate(exp_texts , 1):
        print(f"{i}: {text}\n")

    print("Writing texts:")
    for i , text in enumerate(writing_texts , 1): 
        print(f"{i} : {text}\n")