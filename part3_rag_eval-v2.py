import json
import numpy as np
import re
import faiss
from sentence_transformers import SentenceTransformer

# Load policies from knowledge_base.json
def load_knowledge_base(filepath="knowledge_base.json"):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Could not locate knowledge_base.json file.")

policies_data = load_knowledge_base()
documents = [p["content"] for p in policies_data]
doc_ids = [p["id"] for p in policies_data]

# RAG Evaluation Queries and Ground-Truth IDs
queries_eval = [
    {
        "query": "I ordered a pair of shoes but they are too tight. How many days do I have to return them?",
        "ground_truth": [1]  # Apparel & Footwear Return Window (Doc 1)
    },
    {
        "query": "I paid for my order in cash at the door. If I return it, how do I get my money back and how long does it take?",
        "ground_truth": [5]  # COD Refund Timelines (Doc 5)
    },
    {
        "query": "I live in New Delhi. How fast will my Flipkart Assured orders arrive?",
        "ground_truth": [7]  # Metro Cities Delivery SLA (Doc 7)
    },
    {
        "query": "Does the delivery person pack the shirt for me during pickup, and what happens if I miss them?",
        "ground_truth": [9, 10]  # Item Condition (Doc 9) and Attempts/Packing (Doc 10)
    },
    {
        "query": "My laptop arrived with a cracked screen. Can I get a full cash refund on it?",
        "ground_truth": [2]  # Electronics Return Window (Doc 2)
    }
]

# =====================================================================
# PATH A: Neural FAISS RAG (Standard Online Mode)
# =====================================================================
print("Initialize neural RAG...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents, show_progress_bar=False)
embeddings = np.array(embeddings).astype("float32")

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def search_rag(query, top_k=3):
    query_vector = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(documents):
            results.append({
                "doc_id": doc_ids[idx],
                "text": documents[idx],
                "score": float(dist) # Euclidean distance
            })
    return results

# =====================================================================
# Evaluation Loop
# =====================================================================
rag_mode_str = "NEURAL FAISS (Dense)"
print(f"=== RAG Retrieval Performance [Mode: {rag_mode_str}] ===")

all_recalls = []
all_precisions = []

for q_idx, q in enumerate(queries_eval):
    query_text = q["query"]
    ground_truth = q["ground_truth"]
    
    retrieved = search_rag(query_text, top_k=3)
    retrieved_ids = [r["doc_id"] for r in retrieved]
    
    relevant_retrieved = [rid for rid in retrieved_ids if rid in ground_truth]
    p_3 = len(relevant_retrieved) / 3.0
    r_3 = len(relevant_retrieved) / len(ground_truth)
    
    all_precisions.append(p_3)
    all_recalls.append(r_3)
    
    print(f"\nQuery {q_idx + 1}: '{query_text}'")
    print(f"Ground Truth IDs: {ground_truth}")
    print(f"Retrieved Top-3 IDs: {retrieved_ids}")
    print(f"Precision@3: {p_3:.4f} | Recall@3: {r_3:.4f}")

mean_precision = np.mean(all_precisions)
mean_recall = np.mean(all_recalls)
print(f"\nMean Precision@3: {mean_precision:.4%}")
print(f"Mean Recall@3: {mean_recall:.4%}")
