
import os
import glob
import frontmatter
import re
import json
import time
import threading
import logging
from pathlib import Path
from typing import List, Dict, Any, Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from langchain.text_splitter import RecursiveCharacterTextSplitter
from vector_store_manager import VectorStoreManager
import aiofiles
import asyncio
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.style import Style
from rich.text import Text
from rich.console import Console

import logging
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SourceFileHandler(FileSystemEventHandler):
    def __init__(self, vault_manager, vector_store_manager):
        self.vault_manager = vault_manager
        self.vector_store_manager = vector_store_manager
        self.last_modified = {}
        self.debounce_delay = 2.0

    def should_process(self, file_path: str) -> bool:
        return (file_path.endswith(('.md', '.txt'))) and os.path.exists(file_path)

    def debounce_reindex(self, event_type: str, file_path: str):
        current_time = time.time()
        if file_path in self.last_modified and current_time - self.last_modified[file_path] < self.debounce_delay:
            return
        self.last_modified[file_path] = current_time
        time.sleep(0.5)

        if event_type in ('created', 'modified'):
            asyncio.run(self.vault_manager.reindex_file(file_path))
        elif event_type == 'deleted':
            removed_count = self.vector_store_manager.remove_documents_by_file_path(file_path)
            if removed_count > 0:
                logging.info(f"Removed {removed_count} chunks for deleted file: {file_path}")

    def on_modified(self, event):
        if not event.is_directory and self.should_process(event.src_path):
            threading.Thread(target=self.debounce_reindex, args=('modified', event.src_path)).start()

    def on_created(self, event):
        if not event.is_directory and self.should_process(event.src_path):
            threading.Thread(target=self.debounce_reindex, args=('created', event.src_path)).start()

    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith(('.md', '.txt')):
            threading.Thread(target=self.debounce_reindex, args=('deleted', event.src_path)).start()

class VaultManager:
    def __init__(self, vault_paths: List[str], text_sources: List[str], vector_store_manager: VectorStoreManager, obsidian_root: str = None, mnemosyne_vault_guidance: List[Dict[str, str]] = None, chat_model = None):
        self.source_paths = [Path(p).resolve() for p in vault_paths + text_sources]
        self.vector_store_manager = vector_store_manager
        self.obsidian_root = Path(obsidian_root).resolve() if obsidian_root else None
        self.mnemosyne_vault_guidance = mnemosyne_vault_guidance if mnemosyne_vault_guidance is not None else []
        self.chat_model = chat_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        # Semaphore to limit concurrent file processing
        self.file_processing_semaphore = asyncio.Semaphore(10)  # Process max 10 files concurrently
        self.observers = []
        self.is_watching = False
        self.scan_interval = 6 * 3600  # 6 hours
        self.stop_scanning = threading.Event()
        self.vault_scan_thread = None
        self.discovered_vaults = {str(p) for p in self.source_paths}
        self.console = Console()

    async def determine_vault_for_content(self, content: str) -> str:
        if not self.mnemosyne_vault_guidance or not self.chat_model:
            return "Unknown"

        guidance_str = "\n".join([f"- {g['category']}: {g['vault']}" for g in self.mnemosyne_vault_guidance])
        prompt = (
            f"Given the following content, which of the following vaults is most appropriate?\n\n"
            f"Vault Guidance:\n{guidance_str}\n\n"
            f"Content:\n{content[:1000]}...\n\n"  # Limit content to avoid exceeding token limits
            f"Please respond with only the name of the most appropriate vault from the guidance above. "
            f"If none are suitable, respond with 'Unknown'."
        )
        
        try:
            response = await self.chat_model.generate_response(prompt)
            # Attempt to find a vault name from the guidance in the response
            for guidance_item in self.mnemosyne_vault_guidance:
                vault_name = guidance_item['vault']
                if vault_name.lower() in response.lower():
                    return vault_name
            return "Unknown" # Default if no vault name is found in the response
        except Exception as e:
            logging.error(f"Error determining vault for content: {e}")
            return "Unknown"

    def start_watching(self):
        if self.is_watching:
            return
        handler = SourceFileHandler(self, self.vector_store_manager)
        for source_path in self.source_paths:
            if source_path.exists():
                observer = Observer()
                observer.schedule(handler, str(source_path), recursive=True)
                observer.start()
                self.observers.append(observer)
                logging.info(f"Started watching source: {source_path}")
        self.is_watching = True

    def stop_watching(self):
        for observer in self.observers:
            observer.stop()
            observer.join()
        self.observers = []
        self.is_watching = False
        logging.info("Stopped all file watchers.")


    async def index_sources(self, batch_size: int = 100):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("({task.completed}/{task.total})"),
            console=self.console
        ) as progress:
            tasks = [progress.add_task(f"[dim]Indexing {source_path.name}[/dim]", total=len(list(source_path.rglob("*.md")) + list(source_path.rglob("*.txt")))) for source_path in self.source_paths]
            await asyncio.gather(*[self.index_single_source(source_path, batch_size, tasks[i], progress) for i, source_path in enumerate(self.source_paths)])

    async def index_single_source(self, source_path: Path, batch_size: int = 100, task_id=None, progress_bar=None):
        logger.info(f"Indexing source: {source_path.name}")
        files_to_process = list(source_path.rglob("*.md")) + list(source_path.rglob("*.txt"))
        files_to_process = [f for f in files_to_process if '.obsidian' not in str(f)]
        
        documents, metadatas, ids = [], [], []
        
        # Filter out files that haven't changed
        files_to_index = []
        for file_path in files_to_process:
            current_mtime = os.path.getmtime(file_path)
            stored_mtime = self.vector_store_manager.get_file_last_modified(str(file_path))
            
            if stored_mtime is None or current_mtime != stored_mtime:
                files_to_index.append(file_path)
            else:
                if progress_bar:
                    progress_bar.update(task_id, advance=1) # Still advance progress for skipped files
        
        # Process files in parallel batches
        async def process_file_batch(file_batch):
            batch_results = []
            tasks = []
            for file_path in file_batch:
                # Remove old entries before processing new ones
                self.vector_store_manager.remove_documents_by_file_path(str(file_path))
                tasks.append(self._process_file_with_semaphore(file_path))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing file {file_batch[i]}: {result}")
                else:
                    batch_results.append(result)
                    
                if progress_bar:
                    progress_bar.update(task_id, advance=1)
            
            return batch_results
        
        # Process files in concurrent batches of 20
        concurrent_batch_size = 20
        for i in range(0, len(files_to_index), concurrent_batch_size):
            file_batch = files_to_index[i:i + concurrent_batch_size]
            batch_results = await process_file_batch(file_batch)
            
            # Accumulate results
            for docs, metas, doc_ids in batch_results:
                documents.extend(docs)
                metadatas.extend(metas)
                ids.extend(doc_ids)
            
            # Add to vector store when we have enough documents
            if len(documents) >= batch_size:
                self.vector_store_manager.add_documents(documents, metadatas, ids, batch_size=batch_size)
                logger.info(f"Indexed batch of {len(documents)} chunks from {source_path.name}")
                documents, metadatas, ids = [], [], []
        
        # Add remaining documents
        if documents:
            self.vector_store_manager.add_documents(documents, metadatas, ids, batch_size=batch_size)
            logger.info(f"Indexed final batch of {len(documents)} chunks from {source_path.name}")

    async def reindex_file(self, file_path: str):
        logging.info(f"Re-indexing file: {file_path}")
        self.vector_store_manager.remove_documents_by_file_path(file_path)
        documents, metadatas, ids = await self._process_file(Path(file_path))
        if documents:
            self.vector_store_manager.add_documents(documents, metadatas, ids)
            logging.info(f"Indexed {len(documents)} chunks for file: {file_path}")

    async def _process_file_with_semaphore(self, file_path: Path):
        """Process file with semaphore limiting for concurrency control."""
        async with self.file_processing_semaphore:
            return await self._process_file(file_path)
    
    async def _process_file(self, file_path: Path) -> tuple[list[str], list[dict], list[str]]:
        if file_path.suffix == '.md':
            content_data = await self._extract_markdown_content(file_path)
        elif file_path.suffix == '.txt':
            content_data = await self._extract_text_content(file_path)
        else:
            return [], [], []

        if not content_data:
            return [], [], []

        # Determine the appropriate vault for the content
        determined_vault = await self.determine_vault_for_content(content_data['content'])

        # Optimized text splitting - filter during splitting
        chunks = self._split_and_filter_text(content_data['content'])
        documents, metadatas, ids = [], [], []
        for i, chunk in enumerate(chunks):
            doc_id = f"{content_data['title']}_{i}_{hash(str(file_path))}"
            documents.append(chunk)
            metadatas.append({
                'title': content_data['title'],
                'file_path': str(file_path),
                'chunk_id': i,
                'wiki_links': json.dumps(content_data.get('wiki_links', [])),
                'tags': json.dumps(content_data.get('tags', [])),
                'metadata': json.dumps(content_data.get('metadata', {})),
                'determined_vault': determined_vault, # Add the determined vault to metadata
                'file_last_modified': os.path.getmtime(file_path) # Add last modified timestamp
            })
            ids.append(doc_id)
        return documents, metadatas, ids

    async def _extract_markdown_content(self, file_path: Path) -> Dict[str, Any]:
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            post = frontmatter.loads(content)
            return {
                'content': post.content,
                'metadata': post.metadata,
                'title': file_path.stem,
                'wiki_links': re.findall(r'\[\[([^\]]+)\]\]', post.content),
                'tags': re.findall(r'#(\w+)', post.content)
            }
        except Exception as e:
            logging.error(f"Error reading Markdown file {file_path}: {e}")
            return None

    async def _extract_text_content(self, file_path: Path) -> Dict[str, Any]:
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            return {
                'content': content,
                'metadata': {},
                'title': file_path.stem,
            }
        except Exception as e:
            logging.error(f"Error reading text file {file_path}: {e}")
            return None
    
    def _split_and_filter_text(self, text: str, min_chunk_size: int = 10) -> List[str]:
        """Split text and filter out small chunks in one pass for efficiency."""
        chunks = self.text_splitter.split_text(text)
        return [chunk for chunk in chunks if len(chunk.strip()) >= min_chunk_size]

    async def start_vault_scanning(self):
        if self.obsidian_root and not self.vault_scan_thread:
            self.stop_scanning.clear()
            self.vault_scan_thread = threading.Thread(target=lambda: asyncio.run(self._vault_scan_worker()), daemon=True)
            self.vault_scan_thread.start()
            logging.info(f"Started vault scanning in {self.obsidian_root}")

    def stop_vault_scanning(self):
        if self.vault_scan_thread:
            self.stop_scanning.set()
            self.vault_scan_thread.join()
            self.vault_scan_thread = None
            logging.info("Stopped vault scanning.")

    async def _vault_scan_worker(self):
        while not self.stop_scanning.wait(self.scan_interval):
            await self.scan_for_new_vaults()

    async def scan_for_new_vaults(self):
        if not self.obsidian_root or not self.obsidian_root.exists():
            return
        new_vaults = []
        for item in self.obsidian_root.iterdir():
            if item.is_dir() and (item / ".obsidian").is_dir():
                if str(item.resolve()) not in self.discovered_vaults:
                    new_vaults.append(item)
                    self.discovered_vaults.add(str(item.resolve()))
        if new_vaults:
            logging.info(f"Discovered {len(new_vaults)} new vaults.")
            for vault_path in new_vaults:
                self.source_paths.append(vault_path)
                await self.index_single_source(vault_path)
                if self.is_watching:
                    handler = SourceFileHandler(self, self.vector_store_manager)
                    observer = Observer()
                    observer.schedule(handler, str(vault_path), recursive=True)
                    observer.start()
                    self.observers.append(observer)
                    logging.info(f"Started watching new vault: {vault_path}")

