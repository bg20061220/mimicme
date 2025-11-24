import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os


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

# Create OR load collections
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
        ids=[e["id"] for e in experiences]
    )

if writing_collection.count() == 0:
    writing_collection.add(
        documents=[w["content"] for w in writing_samples],
        ids=[w["id"] for w in writing_samples]
    )


# -------------------------------
# 4. OpenAI Client
# -------------------------------
clientai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -------------------------------
# 5. Retrieval function
# -------------------------------
def retrieve_relevant(question, top_exp=3, top_write=2):
    exps = exp_collection.query(
        query_texts=[question],
        n_results=top_exp
    )
    samples = writing_collection.query(
        query_texts=[question],
        n_results=top_write
    )

    exp_texts = exps["documents"][0]
    writing_texts = samples["documents"][0]

    return exp_texts, writing_texts


# -------------------------------
# 6. Answer Generation (OpenAI API v1.33.0)
# -------------------------------
def generate_answers(question, exp_texts, writing_texts, n_answers=3):
    context = (
        "Relevant Experiences:\n" +
        "\n".join(exp_texts) +
        "\n\nWriting Samples:\n" +
        "\n".join(writing_texts)
    )

    prompt = (
        f"Use the writing style and experiences below to answer the question.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        f"Generate {n_answers} unique responses."
    )

    response = clientai.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    outputs = response.output_text.split("\n")

    # Clean: keep only non-empty lines
    answers = [line.strip() for line in outputs if line.strip()]

    return answers[:n_answers]


# -------------------------------
# 7. Run Example
# -------------------------------
question = "Tell us about a time you learned a new skill"

exp_texts, writing_texts = retrieve_relevant(question)
answers = generate_answers(question, exp_texts, writing_texts)

for i, ans in enumerate(answers, 1):
    print(f"\nANSWER {i}:\n{ans}\n")
