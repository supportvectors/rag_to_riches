from qdrant_client import QdrantClient, models
from rich import print

# 1️⃣  Launch an embedded instance (on-disk)
client = QdrantClient(path="qdrant_db")
collection_name = "demo"

# 2️⃣  To delete a collection, you can use delete_collection.
# For ensuring a clean state, recreate_collection is often more useful.
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)
print(f"Collection '{collection_name}' recreated successfully.")

# 3️⃣  Let us get the text-only embeddings

texts = [
    "The cow jumped over the moon",
    "The happy cow mooed",
    "The dog chased the cat",
    "Cars are fast",
    "The house by the lake is beautiful",
    "The milky way galaxy is a spiral galaxy",
    "To err is human, to forgive is divine"
]

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(texts)
print(embeddings.shape)

similarity = util.cos_sim(embeddings, embeddings)

print(similarity)

# 4️⃣  Populate the Qdrant collection with embeddings
print("Populating Qdrant collection with embeddings...")

# Prepare points for insertion
points = []
for i, (text, embedding) in enumerate(zip(texts, embeddings)):
    point = models.PointStruct(
        id=i,
        vector=embedding.tolist(),  # Convert numpy array to list
        payload={"text": text}
    )
    points.append(point)

# Insert points into the collection
client.upsert(
    collection_name=collection_name,
    points=points
)

print(f"Successfully inserted {len(points)} points into collection '{collection_name}'")

# Verify the insertion by checking collection info
collection_info = client.get_collection(collection_name)
print(f"Collection '{collection_name}' now contains {collection_info.points_count} points")

# 5️⃣  Now, let us search for the most similar text to "The cows were grazing in the meadow "

query_text = "The cows were grazing in the meadow"
print(f"Query text: {query_text}")
query_embedding = model.encode(query_text)
print(query_embedding.shape)
results = client.search(
    collection_name=collection_name,
    query_vector=query_embedding.tolist(),
    limit=3
)

print(results)





