# =============================================================================
#  Filename: simple_rag_usage.py
#
#  Short Description: Simple example of using the rag() facade method.
#
#  Creation date: 2025-01-06
#  Author: Asif Qamar
# =============================================================================

"""
Simple example showing how to use the rag() facade method.

This demonstrates the minimal code needed to get a complete RAG response
from the Animals class.
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_to_riches.corpus.animals import Animals
from rag_to_riches.vectordb.embedded_vectordb import EmbeddedVectorDB
from rag_to_riches.vectordb.embedder import SimpleTextEmbedder


def main():
    """Simple example of using the rag() facade method."""
    print("🎯 Simple RAG Usage Example")
    print("=" * 40)
    
    # Initialize components
    vector_db = EmbeddedVectorDB()
    embedder = SimpleTextEmbedder()
    animals = Animals(vector_db=vector_db, embedder=embedder)
    
    # User query
    query = "What do animals teach us about love?"
    
    print(f"\n📝 Query: '{query}'")
    
    # Single method call for complete RAG pipeline
    result = animals.rag(
        user_query=query,
        limit=3,
        response_type="structured"
    )
    
    # Access the results
    llm_response = result['llm_response']
    search_results = result['search_results']
    
    print(f"\n✅ Found {len(search_results)} relevant quotes")
    print(f"🤖 LLM Answer: {llm_response.answer[:100]}...")
    print(f"💡 Key Insights: {len(llm_response.key_insights)} insights")
    
    # Display the full response
    animals.display_llm_response(llm_response, query)


if __name__ == "__main__":
    main()


#============================================================================================ 