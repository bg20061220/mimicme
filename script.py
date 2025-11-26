import faiss 
import numpy as np 
import google.generativeai as genai 
from sentence_transformers import SentenceTransformer
import json 

genai.configure(api_key="REMOVED")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # If JSON is an array of objects, join their text fields
    all_text = []
    for entry in data:
        if isinstance(entry, dict):
            text = " ".join([str(v) for v in entry.values()])
            all_text.append(text)
        else:
            all_text.append(str(entry))

    return all_text

data = load_json("experiences.json")
chunk_size = 512 
chunks = [data[i:i + chunk_size] for i in range (0 , len(data) , chunk_size)]

embed_model = SentenceTransformer("all-miniLM-L6-v2")

embeddings = np.array([embed_model.encode(chunk) for chunk in chunks ])

d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

def retrieve_context(query , k = 2 ) : 
    query_embedding = embed_model.encode([query])
    __, indices = index.search(np.array(query_embedding), k )
    return "\n".join([chunks[i] for i in indices[0]])

def answer_question(question) : 
    context = retrieve_context(question)

    prompt = f"Use the following context to answer the question: \n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text if response else "No Answer FOund "


question1 = " Tell us about a time you learned a skill"
answer = answer_question(question1)
print("Answer:" , answer) 