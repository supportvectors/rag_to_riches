# =============================================================================
#  Filename: rag_with_animals_example.py
#
#  Short Description: Example demonstrating RAG functionality with Animals class.
#
#  Creation date: 2025-01-06
#  Author: Asif Qamar
# =============================================================================

"""
Example: RAG (Retrieval Augmented Generation) with Animals Class

This example demonstrates how to use the Animals class to:
1. Search for relevant animal quotes
2. Create RAG context for LLM input
3. Display results in a beautiful table format

The example uses the query "a friendship with animals" to show the complete workflow.
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_to_riches.corpus.animals import Animals
from rag_to_riches.vectordb.embedded_vectordb import EmbeddedVectorDB
from rag_to_riches.vectordb.embedder import SimpleTextEmbedder
from loguru import logger


def main():
    """Main function demonstrating RAG with Animals class."""
    
    print("🎯 RAG with Animals Class Example")
    print("=" * 50)
    
    # Initialize components
    print("\n1️⃣ Initializing components...")
    vector_db = EmbeddedVectorDB()
    embedder = SimpleTextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    animals = Animals(vector_db=vector_db, embedder=embedder)
    
    print("   ✅ Components initialized successfully")
    
    # Example user query
    user_query = "a friendship with animals"
    print(f"\n2️⃣ User Query: '{user_query}'")
    
    # Method 1: Search and display results
    print("\n3️⃣ Searching for relevant quotes...")
    search_results = animals.search(user_query, limit=5)
    
    # Display results in beautiful table
    print("\n4️⃣ Displaying search results:")
    animals.display_search_results(
        results=search_results,
        search_description="Animal Friendship Quotes",
        max_text_length=80
    )
    
    # Method 2: Create RAG context
    print("\n5️⃣ Creating RAG context for LLM...")
    rag_context = animals.create_rag_context(
        user_query=user_query,
        search_results=search_results
    )
    
    # Display the RAG context
    print("\n6️⃣ Generated RAG Context (LLM Input):")
    print("-" * 60)
    print(rag_context)
    print("-" * 60)
    
    # Method 3: One-step RAG context creation
    print("\n7️⃣ One-step RAG context creation:")
    one_step_context = animals.search_and_create_rag_context(
        user_query=user_query,
        limit=3,
        system_prompt=animals.SIMPLE_ANIMALS_PROMPT
    )
    
    print("   ✅ One-step context created successfully")
    print(f"   📝 Context length: {len(one_step_context)} characters")
    
    # Method 4: Filtered search with RAG
    print("\n8️⃣ Filtered search with RAG (Famous Literary Passages only):")
    filtered_context = animals.search_and_create_rag_context(
        user_query=user_query,
        limit=3,
        category="Famous Literary Passages"
    )
    
    print("   ✅ Filtered context created successfully")
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 RAG Example Completed Successfully!")
    print("\n📊 Summary:")
    print(f"   • Original search results: {len(search_results)} quotes")
    print(f"   • RAG context created: {len(rag_context)} characters")
    print(f"   • One-step context: {len(one_step_context)} characters")
    print(f"   • Filtered context: {len(filtered_context)} characters")
    
    print("\n💡 Next Steps:")
    print("   • Use the generated RAG context as input to your LLM")
    print("   • The LLM will generate thoughtful responses using the provided quotes")
    print("   • All quotes are properly attributed to their authors")
    
    return {
        "search_results": search_results,
        "rag_context": rag_context,
        "one_step_context": one_step_context,
        "filtered_context": filtered_context
    }


def demonstrate_rag_context_structure():
    """Demonstrate the structure of the generated RAG context."""
    
    print("\n🔍 RAG Context Structure Analysis")
    print("=" * 50)
    
    # Initialize components
    vector_db = EmbeddedVectorDB()
    embedder = SimpleTextEmbedder()
    animals = Animals(vector_db=vector_db, embedder=embedder)
    
    # Create a simple example
    user_query = "a friendship with animals"
    search_results = animals.search(user_query, limit=2)
    
    # Show the formatted results
    print("\n📝 Formatted Search Results:")
    formatted_results = animals.format_search_results_for_rag(search_results)
    print(formatted_results)
    
    # Show the complete context
    print("\n📋 Complete RAG Context Structure:")
    rag_context = animals.create_rag_context(user_query, search_results)
    
    # Split and display sections
    sections = rag_context.split("##")
    for i, section in enumerate(sections, 1):
        if section.strip():
            print(f"\n--- Section {i} ---")
            print(section.strip())


if __name__ == "__main__":
    try:
        # Run the main example
        results = main()
        
        # Demonstrate context structure
        demonstrate_rag_context_structure()
        
        print("\n✅ Example completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed: {str(e)}")
        print(f"❌ Example failed: {str(e)}")
        sys.exit(1)


#============================================================================================ 