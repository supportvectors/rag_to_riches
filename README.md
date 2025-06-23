# Rag to Riches

A simple tutorial introducing RAG (Retrieval Augmented Generation) to a broader audience.

This tutorial comprises a small corpus of animal quotes, over which we build an RAG application. In the process, we learn about:

- Embedder to convert the text to a vector
- VectorDB (Qdrant) to index the vectors for approximate k-nn search
- A generator (LLM with a system prompt) to create a coherent answer to the user's question with the given search results

## Tips

Start by studying the `animals.py` -- it contains, among other things, the `Animals` class, which includes the entire RAG implementation.

Remember, as we mentioned in the session -- this is a basic outline of a complete RAG implementation. To learn RAG deeply, register for our course at https://supportvectors.ai/courses/rag-and-ai-search-bootcamp/
