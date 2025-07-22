import os
# Set multiprocessing method before importing any multiprocessing-related modules
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent tokenizer parallelism issues
os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads
os.environ["MKL_NUM_THREADS"] = "1"  # Limit MKL threads

import multiprocessing
# Set spawn method to avoid fork-related issues on macOS
if hasattr(multiprocessing, 'set_start_method'):
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Method already set

import argparse
from pathlib import Path
from rich.console import Console
import ollama
import logging

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "True"

# Suppress ChromaDB telemetry logs
logging.getLogger('chromadb').setLevel(logging.CRITICAL)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)


from config import load_config, Config
from vault_manager import VaultManager
from vector_store_manager import VectorStoreManager
from memory_manager import MemoryManager
from chat_model import ChatModel
from chat_interface import ChatInterface

class ObsidianRAG:
    def __init__(self, config: Config):
        self.config = config
        self.console = Console()

    async def run(self):
        # Check if Ollama is running
        try:
            ollama.list()
        except Exception:
            self.console.print("[red]❌ Ollama is not running. Please start Ollama first.[/red]")
            self.console.print("[dim]Run: ollama serve[/dim]")
            return

        # Ensure ChromaDB persistence directory exists
        chroma_data_path = Path("./chroma_db/data")
        chroma_data_path.mkdir(parents=True, exist_ok=True)

        # Initialize managers with configuration
        vector_store_manager = VectorStoreManager(
            embedding_model_name=self.config.embedding_model_name
        )
        
        memory_manager = MemoryManager()
        await memory_manager.async_init()
        
        chat_model = ChatModel(
            model_name=self.config.model_name,
            embedding_model_name=self.config.embedding_model_name,
            mnemosyne_vault_guidance=self.config.mnemosyne_vault_guidance,
            use_cache=self.config.use_cache,
            cache_similarity_threshold=self.config.cache_similarity_threshold,
            dynamic_context=self.config.dynamic_context
        )
        await chat_model.async_init()
        
        vault_manager = VaultManager(
            self.config.vaults,
            self.config.text_sources,
            vector_store_manager,
            self.config.obsidian_root,
            self.config.mnemosyne_vault_guidance,
            chat_model
        )

        # Initial indexing with configured batch size
        await vault_manager.index_sources(batch_size=self.config.file_batch_size)

        # Start background services
        vault_manager.start_watching()
        if self.config.obsidian_root:
            await vault_manager.start_vault_scanning()

        # Start chat interface with configuration
        chat_interface = ChatInterface(vault_manager, vector_store_manager, memory_manager, chat_model)
        chat_interface.base_context_results = self.config.base_context_results
        chat_interface.max_context_results = self.config.max_context_results

        try:
            await chat_interface.start()
        finally:
            # Cleanup
            vault_manager.stop_watching()
            if self.config.obsidian_root:
                vault_manager.stop_vault_scanning()
            await memory_manager.backup_memories_to_icloud()

def main():
    parser = argparse.ArgumentParser(description='Obsidian RAG Multi-Vault Support')
    parser.add_argument('--vaults', nargs='+', help='Paths to Obsidian vaults', required=False)
    parser.add_argument('--model', type=str, help='Model name for Ollama', default=None)
    parser.add_argument('--obsidian-root', type=str, help='Root directory to scan for new Obsidian vaults', default=None)
    args = parser.parse_args()

    try:
        config = load_config()
        # Override config with command line arguments if provided
        if args.vaults:
            config.vaults = args.vaults
        if args.model:
            config.model_name = args.model
        if args.obsidian_root:
            config.obsidian_root = args.obsidian_root

        app = ObsidianRAG(config)
        import asyncio
        asyncio.run(app.run())
    except (FileNotFoundError, ValueError) as e:
        Console().print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    main()
