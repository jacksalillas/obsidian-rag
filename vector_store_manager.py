import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import logging
import uuid
from pathlib import Path
import numpy as np
# Removed: from embedding_model_manager import embedding_manager

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", 
                 collection_name: str = "obsidian_documents", 
                 chroma_path: str = "./chroma_db/data",
                 embedding_manager = None): # Added embedding_manager parameter
        """Initialize the vector store manager with ChromaDB and SentenceTransformers."""
        logger.debug(f"VectorStoreManager initialized with embedding_model_name: {embedding_model_name}")
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        self.chroma_path = Path(chroma_path)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))
        
        # Get shared embedding model instance
        if embedding_manager is None:
            raise ValueError("embedding_manager must be provided to VectorStoreManager")
        self.embedding_model = embedding_manager.get_model(embedding_model_name)
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            logger.info(f"Connected to existing ChromaDB collection: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(collection_name)
            logger.info(f"Created new ChromaDB collection: {collection_name}")
    
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], 
                     ids: Optional[List[str]] = None, batch_size: int = 100) -> None:
        """Add documents to the vector store with batching for efficiency."""
        if not documents:
            return
            
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        total_added = 0
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            
            # Generate embeddings for batch
            embeddings = self.embedding_model.encode(batch_docs, show_progress_bar=False, num_workers=0)
            logger.debug(f"Generated embeddings shape: {embeddings.shape}, first 5 values: {embeddings.flatten()[:5]}")
            
            # Convert to list format for ChromaDB
            if isinstance(embeddings, np.ndarray):
                embeddings_list = embeddings.tolist()
            else:
                embeddings_list = embeddings
            
            # Add batch to ChromaDB
            self.collection.add(
                embeddings=embeddings_list,
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
            
            total_added += len(batch_docs)
            logger.debug(f"Added batch {i//batch_size + 1}: {len(batch_docs)} documents")
        
        logger.info(f"Added {total_added} documents to vector store in {(len(documents) + batch_size - 1) // batch_size} batches")
    
    def similarity_search(self, query: str, k: int = 5, 
                         filter_dict: Optional[Dict[str, Any]] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar documents in the vector store."""
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)
        
        # Prepare query parameters
        query_params = {
            "query_embeddings": query_embedding.tolist(),
            "n_results": k,
            "include": ["documents", "metadatas", "distances"]
        }
        
        # Prepare query parameters
        query_params = {
            "query_embeddings": query_embedding.tolist(),
            "n_results": k,
            "include": ["documents", "metadatas", "distances"]
        }

        # Build the 'where' clause for filtering
        where_clause = {}
        if filter_dict:
            where_clause.update(filter_dict)
        if category:
            where_clause["category"] = category

        if where_clause:
            query_params["where"] = where_clause
        
        results = self.collection.query(**query_params)
        
        # Format results with relevance scores
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i]
                relevance_score = max(0, 1 - distance)  # Convert distance to relevance (0-1)
                
                result = {
                    "content": doc,
                    "metadata": results["metadatas"][0][i],
                    "distance": distance,
                    "relevance_score": relevance_score
                }
                formatted_results.append(result)
        
        logger.info(f"Found {len(formatted_results)} similar documents for query")
        return formatted_results
    
    def remove_documents_by_file_path(self, file_path: str) -> int:
        """Remove all documents from a specific file path."""
        try:
            # Get all documents with matching file path
            results = self.collection.get(
                where={"file_path": file_path},
                include=["metadatas"]
            )
            
            if results["ids"]:
                # Delete the documents
                self.collection.delete(ids=results["ids"])
                count = len(results["ids"])
                logger.info(f"Removed {count} documents with file path: {file_path}")
                return count
            
            return 0
        except Exception as e:
            logger.error(f"Error removing documents by file path {file_path}: {e}")
            return 0
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection."""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model_name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

    def get_file_last_modified(self, file_path: str) -> Optional[float]:
        """Get the file_last_modified timestamp for a given file_path from the vector store."""
        try:
            results = self.collection.get(
                where={
                    "file_path": file_path
                },
                include=["metadatas"]
            )
            if results and results["metadatas"] and results["metadatas"][0]:
                # Assuming all chunks for a file have the same last_modified timestamp
                return results["metadatas"][0].get('file_last_modified')
            return None
        except Exception as e:
            logger.error(f"Error getting file_last_modified for {file_path}: {e}")
            return None
    
    def add_saved_memory_document(self, memory_text: str, memory_id: str, category: Optional[str] = None) -> None:
        """Add a synthesized memory document to the vector store."""
        try:
            embedding = self.embedding_model.encode([memory_text], show_progress_bar=False).tolist()
            metadata = {"type": "saved_memory", "memory_id": memory_id}
            if category:
                metadata["category"] = category
            self.collection.add(
                embeddings=embedding,
                documents=[memory_text],
                metadatas=[metadata],
                ids=[memory_id]
            )
            logger.info(f"Added saved memory document with ID: {memory_id} (Category: {category or 'None'})")
        except Exception as e:
            logger.error(f"Error adding saved memory document: {e}")

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            # Delete the collection and recreate it
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(self.collection_name)
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")