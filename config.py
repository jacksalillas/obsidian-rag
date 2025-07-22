import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

@dataclass
class Config:
    vaults: List[str] = field(default_factory=list)
    text_sources: List[str] = field(default_factory=list)
    model_name: str = "mistral:latest"
    embedding_model_name: str = "all-mpnet-base-v2"
    obsidian_root: Optional[str] = None
    mnemosyne_vault_guidance: List[Dict[str, str]] = field(default_factory=lambda: [
        {"category": "Personal", "vault": "Jack-Rich"},
        {"category": "Work and office-related inquiries", "vault": "Macquarie"},
        {"category": "DevOps Engineering", "vault": "DevOps Engineer"},
        {"category": "Quick notes and general information", "vault": "Heynote"}
    ])
    
    # Performance settings
    use_cache: bool = True
    cache_similarity_threshold: float = 0.95
    max_cache_entries: int = 10000
    
    # Connection pool settings
    max_ollama_connections: int = 10
    connection_timeout: int = 300
    
    # Processing settings
    file_batch_size: int = 100
    max_concurrent_files: int = 10
    
    # Context settings
    dynamic_context: bool = True
    base_context_results: int = 5
    max_context_results: int = 12
    
def load_config() -> Config:
    """Load configuration from config.json or environment variables."""
    config_path = Path("config.json")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return Config(**config_data)
    
    # Default configuration if no config file exists
    return Config(
        vaults=[],
        text_sources=[],
        model_name=os.getenv("MNEMOSYNE_MODEL", "mistral:latest"),
        embedding_model_name=os.getenv("MNEMOSYNE_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        obsidian_root=os.getenv("OBSIDIAN_ROOT"),
        mnemosyne_vault_guidance=[],
        
        # Performance defaults
        use_cache=bool(os.getenv("MNEMOSYNE_USE_CACHE", "true").lower() == "true"),
        cache_similarity_threshold=float(os.getenv("MNEMOSYNE_CACHE_THRESHOLD", "0.95")),
        max_cache_entries=int(os.getenv("MNEMOSYNE_MAX_CACHE_ENTRIES", "10000")),
        
        # Connection pool defaults
        max_ollama_connections=int(os.getenv("MNEMOSYNE_MAX_CONNECTIONS", "10")),
        connection_timeout=int(os.getenv("MNEMOSYNE_CONNECTION_TIMEOUT", "300")),
        
        # Processing defaults
        file_batch_size=int(os.getenv("MNEMOSYNE_FILE_BATCH_SIZE", "100")),
        max_concurrent_files=int(os.getenv("MNEMOSYNE_MAX_CONCURRENT_FILES", "10")),
        
        # Context defaults
        dynamic_context=bool(os.getenv("MNEMOSYNE_DYNAMIC_CONTEXT", "true").lower() == "true"),
        base_context_results=int(os.getenv("MNEMOSYNE_BASE_CONTEXT", "5")),
        max_context_results=int(os.getenv("MNEMOSYNE_MAX_CONTEXT", "12"))
    )

def save_config(config: Config):
    """Save configuration to config.json."""
    config_path = Path("config.json")
    
    config_dict = {
        "vaults": config.vaults,
        "text_sources": config.text_sources,
        "model_name": config.model_name,
        "embedding_model_name": config.embedding_model_name,
        "obsidian_root": config.obsidian_root,
        "mnemosyne_vault_guidance": config.mnemosyne_vault_guidance,
        
        # Performance settings
        "use_cache": config.use_cache,
        "cache_similarity_threshold": config.cache_similarity_threshold,
        "max_cache_entries": config.max_cache_entries,
        
        # Connection settings
        "max_ollama_connections": config.max_ollama_connections,
        "connection_timeout": config.connection_timeout,
        
        # Processing settings
        "file_batch_size": config.file_batch_size,
        "max_concurrent_files": config.max_concurrent_files,
        
        # Context settings
        "dynamic_context": config.dynamic_context,
        "base_context_results": config.base_context_results,
        "max_context_results": config.max_context_results
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)