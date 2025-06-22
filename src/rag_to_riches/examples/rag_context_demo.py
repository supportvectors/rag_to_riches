# =============================================================================
#  Filename: rag_context_demo.py
#
#  Short Description: Demo showing RAG context structure and LLM integration for Animals class.
#
#  Creation date: 2025-01-06
#  Author: Asif Qamar
# =============================================================================

"""
Demo: RAG Context Structure and LLM Integration for Animals Class (Real DB)

This demo shows:
1. Real search against the 'animals' collection in the vector DB
2. RAG context generation for LLM input
3. LLM integration using instructor framework with GPT-4o
4. Beautiful display of structured responses

The example uses the query "a friendship with animals" to show the complete workflow.
"""

import sys
from pathlib import Path
import time

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_to_riches.corpus.animals import Animals
from rag_to_riches.vectordb.embedded_vectordb import EmbeddedVectorDB
from rag_to_riches.vectordb.embedder import SimpleTextEmbedder
from loguru import logger


def main():
    """Main demo function using real Animals class, vector DB, and LLM integration."""
    print("🎯 RAG + LLM Demo for Animals Class (Real DB)")
    print("=" * 70)
    
    # Example user query
    user_query = "a friendship with animals"
    print(f"\n📝 User Query: '{user_query}'")
    
    # Initialize components with retry logic
    print("\n🔍 Initializing vector DB and embedder...")
    try:
        vector_db = EmbeddedVectorDB()
        print("   ✅ Vector database connected")
    except Exception as e:
        print(f"   ❌ Vector database failed: {str(e)}")
        return None
    
    # Try to initialize embedder with retry logic
    embedder = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"   🔄 Loading embedder (attempt {attempt + 1}/{max_retries})...")
            embedder = SimpleTextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
            print("   ✅ Embedder loaded successfully")
            break
        except Exception as e:
            print(f"   ⚠️  Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print("   🔄 Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("   ❌ All attempts failed. Trying with default model...")
                try:
                    embedder = SimpleTextEmbedder()  # Use default model
                    print("   ✅ Default embedder loaded successfully")
                except Exception as e2:
                    print(f"   ❌ Default embedder also failed: {str(e2)}")
                    return None
    
    if embedder is None:
        print("   ❌ Could not initialize any embedder")
        return None
    
    try:
        animals = Animals(vector_db=vector_db, embedder=embedder)
        print("   ✅ Animals class initialized successfully")
    except Exception as e:
        print(f"   ❌ Animals class initialization failed: {str(e)}")
        return None
    
    # Perform real search
    print("\n🔎 Performing real search in the animals collection...")
    try:
        search_results = animals.search(user_query, limit=5)
        print(f"   ✅ Found {len(search_results)} results")
    except Exception as e:
        print(f"   ❌ Search failed: {str(e)}")
        return None
    
    # Display results in a rich table
    print("\n📊 Search Results Table:")
    animals.display_search_results(
        results=search_results,
        search_description="Animal Friendship Quotes",
        max_text_length=80
    )
    
    # Generate RAG context for LLM
    print("\n🎯 Generating RAG Context for LLM...")
    rag_context = animals.create_rag_context(
        user_query=user_query,
        search_results=search_results
    )
    
    # Display the complete RAG context
    print("\n📋 Generated RAG Context (LLM Input):")
    print("=" * 80)
    print(rag_context)
    print("=" * 80)
    
    # Show context statistics
    print(f"\n📊 Context Statistics:")
    print(f"   • Total characters: {len(rag_context)}")
    print(f"   • Total words: {len(rag_context.split())}")
    print(f"   • Number of quotes: {len(search_results)}")
    if search_results:
        print(f"   • Average quote length: {sum(len(r.payload.get('content', '')) for r in search_results) / len(search_results):.0f} characters")
    
    # LLM Integration Demo
    print("\n" + "🤖" * 20)
    print("🤖 LLM INTEGRATION DEMO")
    print("🤖" * 20)
    
    # Demo 1: Simple LLM Response
    print("\n1️⃣ Simple LLM Response (GPT-4o):")
    print("-" * 50)
    try:
        simple_answer = animals.ask_llm_simple(
            user_query=user_query,
            limit=3,
            model="gpt-4o"
        )
        print("✅ Simple LLM Response:")
        print(simple_answer)
    except Exception as e:
        print(f"❌ Simple LLM failed: {str(e)}")
        print("   (This might be due to missing OpenAI API key or network issues)")
    
    # Demo 2: Structured LLM Response
    print("\n2️⃣ Structured LLM Response (GPT-4o with Instructor):")
    print("-" * 50)
    try:
        structured_response = animals.ask_llm(
            user_query=user_query,
            limit=5,
            model="gpt-4o"
        )
        
        print("✅ Structured LLM Response Generated!")
        print("   Displaying with rich formatting...")
        
        # Display the structured response beautifully
        animals.display_llm_response(structured_response, user_query)
        
    except Exception as e:
        print(f"❌ Structured LLM failed: {str(e)}")
        print("   (This might be due to missing OpenAI API key or network issues)")
    
    # Demo 3: Filtered LLM Query
    print("\n3️⃣ Filtered LLM Query (Famous Literary Passages only):")
    print("-" * 50)
    try:
        filtered_response = animals.ask_llm(
            user_query="What do literary authors specifically say about animal friendship?",
            limit=3,
            category="Famous Literary Passages",
            model="gpt-4o"
        )
        
        print("✅ Filtered LLM Response Generated!")
        animals.display_llm_response(filtered_response, "What do literary authors specifically say about animal friendship?")
        
    except Exception as e:
        print(f"❌ Filtered LLM failed: {str(e)}")
        print("   (This might be due to missing OpenAI API key or network issues)")
    
    # Summary
    print("\n" + "=" * 70)
    print("🎉 Complete RAG + LLM Demo Finished!")
    print("\n📊 What We Demonstrated:")
    print("   ✅ Real vector search in animals collection")
    print("   ✅ RAG context generation")
    print("   ✅ Simple LLM responses")
    print("   ✅ Structured LLM responses with instructor")
    print("   ✅ Filtered queries")
    print("   ✅ Beautiful rich display formatting")
    
    print("\n💡 Next Steps:")
    print("   • Set your OpenAI API key to enable LLM features")
    print("   • Try different queries and filters")
    print("   • Customize the response models for your needs")
    print("   • Integrate into your own applications")
    
    return {
        "search_results": search_results,
        "rag_context": rag_context,
        "user_query": user_query
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
        
        if results is None:
            print("\n❌ Demo could not complete due to initialization issues.")
            print("💡 Troubleshooting tips:")
            print("   • Check your internet connection")
            print("   • Try running the demo again (network issues can be temporary)")
            print("   • Ensure the vector database is accessible")
            print("   • Check if the animals collection exists and has data")
            sys.exit(1)
        
        # Demonstrate context structure
        demonstrate_rag_context_structure()
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"❌ Demo failed: {str(e)}")
        sys.exit(1)

#============================================================================================ 