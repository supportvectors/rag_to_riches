# =============================================================================
#  Filename: rag_facade_demo.py
#
#  Short Description: Demo of the new rag() facade method in Animals class.
#
#  Creation date: 2025-01-06
#  Author: Asif Qamar
# =============================================================================

"""
Demo of the new rag() facade method in the Animals class.

This example shows how to use the convenient rag() method that encapsulates
the entire RAG pipeline: search + LLM response in a single method call.
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
    """Main demo function using the new rag() facade method."""
    print("🎯 RAG Facade Method Demo")
    print("=" * 50)
    
    # Example user queries
    queries = [
        "What do animals teach us about friendship?",
        "How do literary authors view animal companionship?",
        "What is the wisdom about animal loyalty?"
    ]
    
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
    
    # Demo 1: Structured RAG response
    print("\n" + "="*60)
    print("1️⃣ STRUCTURED RAG RESPONSE")
    print("="*60)
    
    query = queries[0]
    print(f"\n📝 Query: '{query}'")
    
    try:
        # Use the new rag() facade method
        result = animals.rag(
            user_query=query,
            limit=5,
            response_type="structured"
        )
        
        print(f"\n✅ RAG pipeline completed successfully!")
        print(f"   📊 Found {result['query_info']['results_count']} search results")
        print(f"   🤖 Generated {result['query_info']['response_type']} LLM response")
        
        # Display the structured response
        print("\n" + "🤖"*20)
        print("🤖 LLM RESPONSE")
        print("🤖"*20)
        animals.display_llm_response(result['llm_response'], query)
        
        # Show some search result details
        print(f"\n📋 Search Results Summary:")
        print(f"   • Total results: {len(result['search_results'])}")
        if result['search_results']:
            top_result = result['search_results'][0]
            print(f"   • Top score: {top_result.score:.3f}")
            print(f"   • Top quote: '{top_result.payload.get('content', 'N/A')[:80]}...'")
        
    except Exception as e:
        print(f"   ❌ RAG pipeline failed: {str(e)}")
    
    # Demo 2: Simple RAG response
    print("\n" + "="*60)
    print("2️⃣ SIMPLE RAG RESPONSE")
    print("="*60)
    
    query = queries[1]
    print(f"\n📝 Query: '{query}'")
    
    try:
        # Use the new rag() facade method with simple response
        result = animals.rag(
            user_query=query,
            limit=3,
            response_type="simple"
        )
        
        print(f"\n✅ RAG pipeline completed successfully!")
        print(f"   📊 Found {result['query_info']['results_count']} search results")
        print(f"   🤖 Generated {result['query_info']['response_type']} LLM response")
        
        # Display the simple response
        print("\n" + "🤖"*20)
        print("🤖 LLM RESPONSE")
        print("🤖"*20)
        print(result['llm_response'])
        
    except Exception as e:
        print(f"   ❌ RAG pipeline failed: {str(e)}")
    
    # Demo 3: Filtered RAG response
    print("\n" + "="*60)
    print("3️⃣ FILTERED RAG RESPONSE")
    print("="*60)
    
    query = queries[2]
    print(f"\n📝 Query: '{query}' (filtered by 'Famous Literary Passages')")
    
    try:
        # Use the new rag() facade method with category filter
        result = animals.rag(
            user_query=query,
            limit=5,
            category="Famous Literary Passages",
            response_type="structured"
        )
        
        print(f"\n✅ RAG pipeline completed successfully!")
        print(f"   📊 Found {result['query_info']['results_count']} search results")
        print(f"   🏷️  Filtered by category: {result['query_info']['category_filter']}")
        print(f"   🤖 Generated {result['query_info']['response_type']} LLM response")
        
        # Display the structured response
        print("\n" + "🤖"*20)
        print("🤖 LLM RESPONSE")
        print("🤖"*20)
        animals.display_llm_response(result['llm_response'], query)
        
    except Exception as e:
        print(f"   ❌ RAG pipeline failed: {str(e)}")
    
    # Demo 4: Access to all RAG components
    print("\n" + "="*60)
    print("4️⃣ ACCESSING RAG COMPONENTS")
    print("="*60)
    
    query = "animal wisdom"
    print(f"\n📝 Query: '{query}'")
    
    try:
        # Use the new rag() facade method
        result = animals.rag(
            user_query=query,
            limit=3,
            response_type="structured"
        )
        
        print(f"\n✅ RAG pipeline completed successfully!")
        
        # Show all available components
        print(f"\n📋 Available RAG Components:")
        print(f"   • LLM Response: {type(result['llm_response']).__name__}")
        print(f"   • Search Results: {len(result['search_results'])} items")
        print(f"   • RAG Context: {len(result['rag_context'])} characters")
        print(f"   • Query Info: {len(result['query_info'])} fields")
        
        # Show query info details
        print(f"\n🔍 Query Information:")
        for key, value in result['query_info'].items():
            print(f"   • {key}: {value}")
        
        # Show RAG context preview
        print(f"\n📝 RAG Context Preview (first 200 chars):")
        print(f"   {result['rag_context'][:200]}...")
        
    except Exception as e:
        print(f"   ❌ RAG pipeline failed: {str(e)}")
    
    print("\n" + "="*60)
    print("🎉 RAG Facade Demo Completed!")
    print("="*60)
    
    print("\n💡 Key Benefits of the rag() method:")
    print("   ✅ Single method call for complete RAG pipeline")
    print("   ✅ Returns both LLM response and search results")
    print("   ✅ Supports both structured and simple responses")
    print("   ✅ Includes all filtering options")
    print("   ✅ Provides access to RAG context and query metadata")
    print("   ✅ Comprehensive error handling")
    
    return True


if __name__ == "__main__":
    try:
        # Run the main example
        success = main()
        
        if success is None:
            print("\n❌ Demo could not complete due to initialization issues.")
            print("💡 Troubleshooting tips:")
            print("   • Check your internet connection")
            print("   • Try running the demo again (network issues can be temporary)")
            print("   • Ensure the vector database is accessible")
            print("   • Check if the animals collection exists and has data")
            sys.exit(1)
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"❌ Demo failed: {str(e)}")
        sys.exit(1)


#============================================================================================ 