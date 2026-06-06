# =============================================================================
#  Filename: basic_rag.py
#
#  Short Description: Basic RAG starting point example for document indexing.
#
#  Creation date: 2025-01-06
#  Author: Asif Qamar
# =============================================================================

# This is a basic RAG starting point example - excluded from testing and coverage
# pragma: no cover

from openai.types.responses import Response
import warnings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from rich import print
from dotenv import load_dotenv
load_dotenv()

# Suppress specific numpy warnings that are common with llama-index when using dot products
warnings.filterwarnings("ignore", message=".*encountered in dot.*", category=RuntimeWarning)

documents = SimpleDirectoryReader("data/starter").load_data()
index = VectorStoreIndex.from_documents(documents)

# Make sure that the index is persisted for later use!

query_engine = index.as_query_engine()
response= query_engine.query("What are some key features of ancient Indus Valley civilization?")



print('\n'*10)
print('*'*100, '\n'*3, str(response.response), '\n'*3, '*'*100)

response = query_engine.query("What is self-attention in a transformer model?")

print('\n'*10)
print('*'*100, '\n'*3, str(response.response), '\n'*3, '*'*100)
