import os
import faiss
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle
from pypdf import PdfReader
from .config import settings

class RAGEngine:
    def __init__(self):
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.doc_metadata = []
        self.index_path = settings.FAISS_INDEX_PATH

    def load_embedding_model(self):
        """Load the sentence transformer model for embeddings"""
        print(f"Loading embedding model {settings.EMBEDDING_MODEL}...")
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        print("Embedding model loaded successfully!")

    def chunk_documents(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split documents into chunks"""
        chunk_size = chunk_size or settings.CHUNK_SIZE
        overlap = overlap or settings.CHUNK_OVERLAP

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap

        return chunks

    def read_document(self, file_path: str) -> str:
        """Read document content based on file type"""
        file_ext = Path(file_path).suffix.lower()

        try:
            if file_ext == '.txt' or file_ext == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif file_ext == '.pdf':
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
            else:
                print(f"Unsupported file type: {file_ext}")
                return ""
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return ""

    def index_documents(self, doc_directory: str = None):
        """Index all documents in the specified directory"""
        if self.embedding_model is None:
            self.load_embedding_model()

        doc_directory = doc_directory or settings.DOCUMENTS_DIR

        if not os.path.exists(doc_directory):
            print(f"Documents directory {doc_directory} does not exist.")
            return

        print(f"Indexing documents from {doc_directory}...")

        all_chunks = []
        all_metadata = []

        # Iterate through all files
        for root, _, files in os.walk(doc_directory):
            for file in files:
                if file.startswith('.'):
                    continue

                file_path = os.path.join(root, file)
                content = self.read_document(file_path)

                if not content:
                    continue

                # Chunk the document
                chunks = self.chunk_documents(content)

                # Store chunks and metadata
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        'file': file,
                        'chunk_id': i,
                        'total_chunks': len(chunks)
                    })

        if not all_chunks:
            print("No documents found to index.")
            return

        print(f"Encoding {len(all_chunks)} chunks...")
        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))

        self.documents = all_chunks
        self.doc_metadata = all_metadata

        # Save index
        self.save_index()

        print(f"Indexed {len(all_chunks)} chunks from {len(set(m['file'] for m in all_metadata))} documents.")

    def save_index(self):
        """Save FAISS index and metadata to disk"""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        faiss.write_index(self.index, f"{self.index_path}.index")

        with open(f"{self.index_path}.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.doc_metadata
            }, f)

        print(f"Index saved to {self.index_path}")

    def load_index(self):
        """Load FAISS index and metadata from disk"""
        if self.embedding_model is None:
            self.load_embedding_model()

        try:
            self.index = faiss.read_index(f"{self.index_path}.index")

            with open(f"{self.index_path}.pkl", 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.doc_metadata = data['metadata']

            print(f"Loaded index with {len(self.documents)} chunks")
            return True
        except Exception as e:
            print(f"Could not load index: {e}")
            return False

    def retrieve_context(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve relevant document chunks for a query"""
        if self.index is None:
            if not self.load_index():
                return []

        top_k = top_k or settings.TOP_K

        # Encode query
        query_embedding = self.embedding_model.encode([query])

        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)

        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    'text': self.documents[idx],
                    'metadata': self.doc_metadata[idx],
                    'score': float(distances[0][i])
                })

        return results

# Global RAG engine instance
rag_engine = RAGEngine()
