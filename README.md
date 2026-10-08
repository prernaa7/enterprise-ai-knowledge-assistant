# Enterprise AI Knowledge Assistant

A RAG-based enterprise knowledge assistant that allows users to upload documents and ask questions using natural language.

The application extracts information from enterprise documents, creates semantic embeddings, stores them in ChromaDB, retrieves relevant information for a user query, and uses Google Gemini to generate a grounded answer.

## Features

- Upload PDF, DOCX, and TXT documents
- Extract text from uploaded documents
- Split documents into smaller chunks
- Generate semantic embeddings using Sentence Transformers
- Store document embeddings in ChromaDB
- Retrieve relevant document chunks using semantic similarity
- Generate grounded answers using Google Gemini
- Prevent unsupported answers when information is not available
- Display source documents and retrieved chunks
- Generate document summaries
- FastAPI backend
- Streamlit user interface
- Dockerized API and frontend
- Persistent ChromaDB storage using Docker volumes

## RAG Workflow

```text
Document Upload
       ↓
Text Extraction
       ↓
Document Chunking
       ↓
Sentence Transformer Embeddings
       ↓
ChromaDB Vector Storage
       ↓
User Question
       ↓
Semantic Similarity Retrieval
       ↓
Relevant Document Chunks
       ↓
Google Gemini
       ↓
Grounded Answer + Sources
