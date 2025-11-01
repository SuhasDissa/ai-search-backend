import os
import faiss
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle
from pypdf import PdfReader
from .config import settings
import torch
import logging

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.doc_metadata = []
        self.index_path = settings.FAISS_INDEX_PATH
        self.use_gpu = torch.cuda.is_available()
        self.gpu_resources = None

        if self.use_gpu:
            logger.info("GPU is available. FAISS will use GPU acceleration.")
            self.gpu_resources = faiss.StandardGpuResources()
        else:
            logger.info("GPU not available. FAISS will use CPU.")

    def load_embedding_model(self):
        """Load the sentence transformer model for embeddings"""
        logger.info(f"Loading embedding model {settings.EMBEDDING_MODEL}...")
        device = 'cuda' if self.use_gpu else 'cpu'
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL, device=device)
        logger.info(f"Embedding model loaded successfully on {device}!")

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
                logger.warning(f"Unsupported file type: {file_ext}")
                return ""
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
            return ""

    def index_documents(self, doc_directory: str = None):
        """Index all documents in the specified directory"""
        if self.embedding_model is None:
            self.load_embedding_model()

        doc_directory = doc_directory or settings.DOCUMENTS_DIR

        if not os.path.exists(doc_directory):
            logger.warning(f"Documents directory {doc_directory} does not exist.")
            return

        logger.info(f"Indexing documents from {doc_directory}...")

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
                logger.debug(f"Document '{file}' split into {len(chunks)} chunks")

                # Store chunks and metadata
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        'file': file,
                        'chunk_id': i,
                        'total_chunks': len(chunks)
                    })

        if not all_chunks:
            logger.warning("No documents found to index.")
            return

        logger.info(f"Encoding {len(all_chunks)} chunks...")
        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)

        # Create FAISS index
        dimension = embeddings.shape[1]
        cpu_index = faiss.IndexFlatL2(dimension)

        if self.use_gpu:
            # Move index to GPU
            self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, cpu_index)
            logger.info(f"FAISS index created on GPU with dimension {dimension}")
        else:
            self.index = cpu_index
            logger.info(f"FAISS index created on CPU with dimension {dimension}")

        self.index.add(embeddings.astype('float32'))
        logger.debug(f"Added {len(embeddings)} vectors to FAISS index")

        self.documents = all_chunks
        self.doc_metadata = all_metadata

        # Save index
        self.save_index()

        num_docs = len(set(m['file'] for m in all_metadata))
        logger.info(f"Successfully indexed {len(all_chunks)} chunks from {num_docs} documents")

    def save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            logger.debug(f"Saving index to {self.index_path}")

            # If index is on GPU, move it to CPU before saving
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, f"{self.index_path}.index")
                logger.debug("Converted GPU index to CPU for saving")
            else:
                faiss.write_index(self.index, f"{self.index_path}.index")

            with open(f"{self.index_path}.pkl", 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadata': self.doc_metadata
                }, f)

            logger.info(f"Index saved successfully to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}", exc_info=True)
            raise

    def load_index(self):
        """Load FAISS index and metadata from disk"""
        if self.embedding_model is None:
            self.load_embedding_model()

        try:
            logger.info(f"Loading index from {self.index_path}")
            cpu_index = faiss.read_index(f"{self.index_path}.index")

            with open(f"{self.index_path}.pkl", 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.doc_metadata = data['metadata']

            # Move index to GPU if available
            if self.use_gpu:
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, cpu_index)
                logger.info(f"Loaded index with {len(self.documents)} chunks on GPU")
            else:
                self.index = cpu_index
                logger.info(f"Loaded index with {len(self.documents)} chunks on CPU")

            num_docs = len(set(m['file'] for m in self.doc_metadata))
            logger.info(f"Index loaded: {len(self.documents)} chunks from {num_docs} documents")
            return True
        except FileNotFoundError:
            logger.warning(f"Index files not found at {self.index_path}")
            return False
        except Exception as e:
            logger.error(f"Could not load index: {e}", exc_info=True)
            return False

    def retrieve_context(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve relevant document chunks for a query"""
        if self.index is None:
            logger.debug("Index not loaded, attempting to load...")
            if not self.load_index():
                logger.warning("No index available for retrieval")
                return []

        top_k = top_k or settings.TOP_K
        logger.debug(f"Retrieving top {top_k} chunks for query: '{query[:50]}...'")

        # Encode query
        query_embedding = self.embedding_model.encode([query])
        logger.debug(f"Query encoded to embedding of shape {query_embedding.shape}")

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

        logger.info(f"Retrieved {len(results)} relevant chunks with scores: {[r['score'] for r in results]}")
        return results

# Global RAG engine instance
rag_engine = RAGEngine()
