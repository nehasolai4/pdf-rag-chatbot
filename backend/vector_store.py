import faiss
import numpy as np

def create_vector_store(embeddings):
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

def search(index,query_embedding,chunks,k=3):
    query_embedding = np.array([query_embedding]).astype("float32")
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding,k)
    results=[]

    for i,score in zip(indices[0],scores[0]):
        results.append({
            "chunk":chunks[i],
            "score":float(score)
        })
    return results