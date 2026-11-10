# PDF Chatbot using RAG

## Project Overview
An AI-powered PDF chatbot that allows users to upload and ask questions about PDF documents. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information and generate context-based answers.

## Technologies Used
- Python
- Streamlit
- LangChain
- Hugging Face Embeddings
- Pinecone Vector Database
- Ollama (Llama 3.2)

## How It Works
1. Load PDF documents using PyPDFLoader.
2. Split documents into smaller chunks.
3. Generate embeddings using all-MiniLM-L6-v2.
4. Store document embeddings in Pinecone.
5. Retrieve relevant chunks using semantic similarity search.
6. Pass the retrieved context to the Llama 3.2 model.
7. Generate answers based on the PDF content.
8. Display questions and answers through Streamlit.



