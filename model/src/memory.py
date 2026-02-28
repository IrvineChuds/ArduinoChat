from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

MODEL = "BAAI/bge-small-en"

def main():
    model = SentenceTransformer("BAAI/bge-small-en")

    documents = [
        "User is a college student",
        "User wants to build AI applications",
        "User likes Jazz",
        "User is 30 years old",
        "User is male",
        "User loves cats more than dogs",
        "User loves songs by Michael Jackson",
        "User doesn't like drinking",
        "User was born in China",
        "User was raised in Canada"
    ]

    print("embedding...")
    embeddings = model.encode(["Represent this user memory for searching relevant passages: " + document for document in documents], normalize_embeddings=True)

    dim = embeddings.shape[1]

    print(f"dim: {dim}")

    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings))

    queries = ["Give me music recommendations", "User was raised in Canada"]

    query_embedding = model.encode(
        ["Represent this user memory for searching relevant passages: " + q for q in queries],
        normalize_embeddings=True
    )

    lims, scores, indices = index.range_search(np.array(query_embedding), 0.9)

    print(lims)

    for i in range(len(lims) - 1):
        retrieved_docs = [documents[i] for i in indices[lims[i]:lims[i+1]]]

        context = "\n".join(retrieved_docs)

        print("Context:\n", context)
    

if __name__ == "__main__":
    main()