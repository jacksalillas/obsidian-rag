import threading
from typing import Dict, Optional
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class EmbeddingModelManager:
    """Singleton manager for embedding models to avoid duplicate loading."""
    _instance = None
    _lock = threading.Lock()
    _models: Dict[str, SentenceTransformer] = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, model_name: str) -> SentenceTransformer:
        """Get or create an embedding model instance."""
        if model_name not in self._models:
            with self._lock:
                if model_name not in self._models:
                    logger.info(f"Loading embedding model: {model_name}")
                    self._models[model_name] = SentenceTransformer(model_name)
                    logger.info(f"Embedding model loaded: {model_name}")
        
        return self._models[model_name]
    
    def clear_model(self, model_name: str) -> bool:
        """Remove a model from memory."""
        if model_name in self._models:
            with self._lock:
                if model_name in self._models:
                    del self._models[model_name]
                    logger.info(f"Cleared embedding model: {model_name}")
                    return True
        return False
    
    def get_loaded_models(self) -> list:
        """Get list of currently loaded model names."""
        return list(self._models.keys())
    
    def clear_all_models(self):
        """Clear all loaded models from memory."""
        with self._lock:
            model_count = len(self._models)
            self._models.clear()
            logger.info(f"Cleared all {model_count} embedding models")

# Global instance
embedding_manager = EmbeddingModelManager()