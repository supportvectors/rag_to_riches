# =============================================================================
#  Filename: example_animals_usage.py
#
#  Short Description: End-to-end example of Animals class usage for loading and searching.
#
#  Creation date: 2025-01-06
#  Author: Asif Qamar
# =============================================================================

from pathlib import Path
from rag_to_riches.vectordb.embedded_vectordb import EmbeddedVectorDB
from rag_to_riches.vectordb.embedder import SimpleTextEmbedder
from rag_to_riches.corpus.animals import Animals


def main():
    """End-to-end example of Animals class usage."""
    
    print("🐾 Animals Quotes Vector Search Demo 🐾")
    print("=" * 50)
    
    # Step 1: Initialize components
    print("\n📚 Step 1: Initializing Vector Database and Embedder...")
    vector_db = EmbeddedVectorDB()
    embedder = SimpleTextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Step 2: Create Animals corpus loader
    print("🔧 Step 2: Creating Animals corpus loader...")
    animals_loader = Animals(
        vector_db=vector_db,
        embedder=embedder,
        collection_name="animals"
    )
    
    # Step 3: Load and index the animals JSONL file
    print("📂 Step 3: Loading and indexing animal quotes...")
    jsonl_path = Path("data/corpus/animals/animals.jsonl")
    
    if not jsonl_path.exists():
        print(f"❌ Error: File not found at {jsonl_path}")
        print("Please ensure the animals.jsonl file exists in the data/corpus/animals/ directory")
        return
    
    try:
        # Load and index all quotes in one call
        wisdom, point_ids = animals_loader.load_and_index(jsonl_path)
        
        print(f"✅ Successfully loaded and indexed {len(wisdom)} animal quotes!")
        print(f"📊 Indexed {len(point_ids)} points into the 'animals' collection")
        
        # Display collection statistics
        stats = animals_loader.get_collection_stats()
        print(f"\n📈 Collection Statistics:")
        print(f"   • Collection Name: {stats['collection_name']}")
        print(f"   • Points in Database: {stats['point_count']}")
        print(f"   • Loaded Quotes: {stats['loaded_quotes']}")
        print(f"   • Unique Categories: {len(stats['categories'])}")
        print(f"   • Unique Authors: {len(stats['authors'])}")
        
        # Show sample categories and authors
        print(f"\n🏷️  Sample Categories: {', '.join(stats['categories'][:3])}...")
        print(f"✍️  Sample Authors: {', '.join(stats['authors'][:5])}...")
        
    except Exception as e:
        print(f"❌ Error during loading: {e}")
        return
    
    # Step 4: Demonstrate different search capabilities
    print("\n" + "=" * 50)
    print("🔍 Step 4: Demonstrating Search Capabilities")
    print("=" * 50)
    
    # Search 1: Basic semantic search
    print("\n🔍 Search 1: Basic semantic search for 'a friendship with animals'")
    results = animals_loader.search("a friendship with animals", limit=3)
    print_search_results(results, "Basic Search")
    
    # Search 2: Search with author filter
    print("\n🔍 Search 2: Search for 'a dog is a gentleman who deserves heaven' by 'Mark Twain'")
    results = animals_loader.search("a dog is a gentleman who deserves heaven", limit=3, author="Mark Twain")
    print_search_results(results, "Author Filter")
    
    # Search 3: Search with category filter
    print("\n🔍 Search 3: Search for 'loving an animal awakens the soul' in 'Wisdom and Philosophy' category")
    results = animals_loader.search("loving an animal awakens the soul", limit=3, category="Wisdom and Philosophy")
    print_search_results(results, "Category Filter")
    
    # Search 4: Search with score threshold
    print("\n🔍 Search 4: High-confidence search for 'cats are graceful creatures' (score > 0.5)")
    results = animals_loader.search("cats are graceful creatures", limit=3, score_threshold=0.5)
    print_search_results(results, "Score Threshold")
    
    # Search 5: Combined filters
    print("\n🔍 Search 5: Search for 'the greatness of a nation is judged by how it treats animals' by Mahatma Gandhi")
    results = animals_loader.search(
        "the greatness of a nation is judged by how it treats animals", 
        limit=2, 
        author="Mahatma Gandhi"
    )
    print_search_results(results, "Combined Filters")
    
    print("\n🎉 Demo completed successfully!")
    print("💡 The Animals class provides powerful semantic search with metadata filtering!")


def print_search_results(results, search_type):
    """Helper function to format and print search results."""
    print(f"📋 {search_type} Results: {len(results)} found")
    
    if not results:
        print("   No results found.")
        return
    
    for i, result in enumerate(results, 1):
        content = result.payload.get("content", "")
        author = result.payload.get("author", "Unknown")
        category = result.payload.get("category", "Unknown")
        
        # Truncate long quotes for readability
        display_content = content if len(content) <= 100 else content[:97] + "..."
        
        print(f"   {i}. 📊 Score: {result.score:.3f}")
        print(f"      💬 Quote: \"{display_content}\"")
        print(f"      ✍️  Author: {author}")
        print(f"      🏷️  Category: {category}")
        if i < len(results):
            print()


if __name__ == "__main__":
    main()

#============================================================================================ 