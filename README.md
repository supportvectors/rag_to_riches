# Good code with AI

This is a tutorial that helps develop a good codebase to implement RAG on a directory of documents.

### Resources

* <a href="https://glhssocialstudies.weebly.com/world-history-textbook---pdf-copy.html"> World History </a>
* <a href="https://glhssocialstudies.weebly.com/economics-textbook---pdf-copy.html"> Economics </a>

## Steps for the project

### Download the textbooks

There are pdf files at the resources above, each corresponding to a chapter of the textbook; download each of the pdfs and store it in the data/ directory. Write a python script that will do this, whose functionality should be general enough to download all the pdf links from a webpage.

### Simple RAG with Llama-Index

Now, create a simple RAG application over the documents.

### Evolve!

Gradually, evolve this application into something more elaborate.

### Steps

We will build the following components, step by step.

* A pdf-downloader to scrape PDF from an open-access textbooks website (legally)
* An indexing pipeline
* A Vector-embedding store
* The RAG pipeline