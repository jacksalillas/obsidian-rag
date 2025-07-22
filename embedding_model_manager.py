import threading
import os
from typing import Dict, Optional
from sentence_transformers import SentenceTransformer
import logging
import torch

logger = logging.getLogger(__name__)

class EmbeddingModelManager:
    """Singleton manager for embedding models to avoid duplicate loading."""
    _instance = None
    _lock = threading.Lock()
    _models: Dict[str, SentenceTransformer] = {}
    
    def __init__(self):
        logger.debug("EmbeddingModelManager initialized.")

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, model_name: str) -> SentenceTransformer:
        """Get or create an embedding model instance."""
        logger.debug(f"EmbeddingModelManager.get_model received request for model: {model_name}")
        
        # Clear all models to ensure a fresh load, especially during debugging
        self.clear_all_models()

        if model_name not in self._models:
            with self._lock:
                if model_name not in self._models:
                    logger.info(f"Loading embedding model: {model_name}")
                    try:
                        # Force CPU for debugging embedding quality
                        device = 'cpu'
                        
                        # Load model with explicit device configuration
                        model = SentenceTransformer(model_name, device=device)
                        
                        # Set model to evaluation mode for inference
                        model.eval()
                        
                        self._models[model_name] = model
                        logger.info(f"Embedding model loaded: {model_name} on device: {device}")
                        
                    except Exception as e:
                        logger.error(f"Failed to load embedding model {model_name}: {e}")
                        # Fallback to CPU-only loading
                        try:
                            logger.info(f"Retrying {model_name} with CPU-only mode...")
                            model = SentenceTransformer(model_name, device='cpu')
                            model.eval()
                            self._models[model_name] = model
                            logger.info(f"Embedding model loaded: {model_name} on CPU (fallback)")
                        except Exception as e2:
                            logger.error(f"Failed to load embedding model {model_name} even with CPU: {e2}")
                            raise RuntimeError(f"Could not load embedding model {model_name}: {e2}")
        
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

