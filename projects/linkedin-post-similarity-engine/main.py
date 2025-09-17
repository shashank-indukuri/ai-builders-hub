from fastembed import TextEmbedding
import numpy as np

# Professional LinkedIn posts for similarity analysis
documents: list[str] = [
    "Just launched my new AI startup focused on revolutionizing customer service with LLMs",
    "Excited to share my journey from software engineer to AI product manager at Google",
    "Looking for talented ML engineers to join our team building the future of healthcare AI",
    "Thrilled to announce our Series A funding round led by Andreessen Horowitz",
    "Published my latest research on transformer architectures in computer vision",
]

# Initialize embedding model for semantic search
embedding_model = TextEmbedding()
print("FastEmbed model ready for LinkedIn content analysis!")

# Generate embeddings for semantic similarity
embeddings_list = list(embedding_model.embed(documents))
print(
    f"Generated {len(embeddings_list)} embeddings with {len(embeddings_list[0])} dimensions each"
)


# Find most similar posts
def find_similar_posts(query: str, top_k: int = 2):
    query_embedding = list(embedding_model.embed([query]))[0]

    similarities = []
    for i, doc_embedding in enumerate(embeddings_list):
        # Calculate cosine similarity
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        similarities.append((similarity, documents[i]))

    # Sort by similarity and return top results
    similarities.sort(reverse=True)
    return similarities[:top_k]


# Example: Find posts similar to a startup query
query = "Building an AI company with venture capital funding"
print(f"\nQuery: '{query}'")
print("\nMost similar LinkedIn posts:")

for similarity, post in find_similar_posts(query):
    print(f"   Similarity: {similarity:.3f} - {post[:60]}...")
