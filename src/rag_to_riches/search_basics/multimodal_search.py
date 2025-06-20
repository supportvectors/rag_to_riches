from qdrant_client import QdrantClient, models
import numpy as np
import torch
import open_clip
from PIL import Image

# 1️⃣  Launch an embedded instance (on-disk)
client = QdrantClient(path="qdrant_db")
collection_name = "demo_multimodal"

# 2️⃣  Check if collection exists and create if not
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

# SigLIP models typically have 768-dimensional embeddings
client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
)
print(f"Collection '{collection_name}' created successfully.")


# 3️⃣  Load SigLIP model from open_clip
print("Loading SigLIP model...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-SigLIP2', pretrained='webli')
tokenizer = open_clip.get_tokenizer('ViT-B-16-SigLIP2')

# Set model to evaluation mode
model.eval()

# Load the actual car image
car_image = Image.open("data/small_image_collection/car.jpg")
print(f"Loaded car image: {car_image.size} pixels, mode: {car_image.mode}")

# Test multiple text descriptions
text_descriptions = [
    "a red vintage car",
    "a car",
    "an automobile",
    "a vehicle",
    "a red car"
]

print("\nTesting different text descriptions:")
best_similarity = -1
best_text = ""
all_embeddings = []

for i, text_desc in enumerate(text_descriptions):
    # Preprocess image and text
    image_input = preprocess(car_image).unsqueeze(0)
    text_input = tokenizer([text_desc])

    # Generate embeddings
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Calculate similarity
    similarity = torch.cosine_similarity(text_features, image_features)
    print(f"  '{text_desc}': {similarity.item():.4f}")
    
    # Track best match
    if similarity.item() > best_similarity:
        best_similarity = similarity.item()
        best_text = text_desc
        # Convert to numpy for Qdrant storage
        text_emb = text_features.squeeze().numpy()
        img_emb = image_features.squeeze().numpy()

print(f"\nBest match: '{best_text}' with similarity: {best_similarity:.4f}")

# 4️⃣  Store embeddings in Qdrant
points = [
    models.PointStruct(
        id=1,
        vector=text_emb.tolist(),
        payload={"type": "text", "content": best_text}
    ),
    models.PointStruct(
        id=2,
        vector=img_emb.tolist(),
        payload={"type": "image", "description": "car image from data/small_image_collection/car.jpg"}
    )
]

client.upsert(collection_name=collection_name, points=points)
print("Embeddings stored in Qdrant successfully!")

# 5️⃣  Search for similar items
search_results = client.query_points(
    collection_name=collection_name,
    query=text_emb.tolist(),
    limit=2
)

print("\nSearch results:")
for result in search_results.points:
    print(f"ID: {result.id}, Score: {result.score:.4f}, Type: {result.payload['type']}")
    if result.payload['type'] == 'image':
        print(f"  Description: {result.payload['description']}")
    else:
        print(f"  Content: {result.payload['content']}")

# 6️⃣  Demonstrate reverse search (image to text)
print("\nReverse search (using image to find similar text):")
reverse_results = client.query_points(
    collection_name=collection_name,
    query=img_emb.tolist(),
    limit=2
)

for result in reverse_results.points:
    print(f"ID: {result.id}, Score: {result.score:.4f}, Type: {result.payload['type']}")
    if result.payload['type'] == 'text':
        print(f"  Content: {result.payload['content']}")
    else:
        print(f"  Description: {result.payload['description']}")
