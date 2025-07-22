import chromadb
from chromadb.config import Settings
from typing import Dict, Any, List
from datetime import datetime, timedelta
import zoneinfo
import urllib.request
import json as pyjson
from pathlib import Path
import uuid
import logging
import asyncio
from functools import partial
from embedding_model_manager import embedding_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMCache:
    async def get_atomic_time_manila(self):
        """Fetch atomic (UTC) time from a public API and convert to Manila local time."""
        try:
            # Use worldtimeapi.org for atomic/UTC time
            # Use run_in_executor for blocking urllib.request.urlopen
            data = await self._run_in_executor(self._fetch_atomic_time_data)
            utc_datetime_str = data['utc_datetime']
            utc_dt = datetime.fromisoformat(utc_datetime_str.replace('Z', '+00:00'))
            # Convert to Manila time
            manila_tz = zoneinfo.ZoneInfo("Asia/Manila")
            manila_dt = utc_dt.astimezone(manila_tz)
            return manila_dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            logging.error(f"Error fetching atomic time: {e}")
            return f"Error fetching atomic time: {e}"

    def _fetch_atomic_time_data(self):
        with urllib.request.urlopen('http://worldtimeapi.org/api/timezone/Etc/UTC') as response:
            return pyjson.loads(response.read().decode())

    async def get_current_datetime(self):
        """Return the current date and time in Manila, Philippines."""
        # This is a non-blocking operation, so direct call is fine, but for consistency
        # with other async methods, we can wrap it if needed, or just keep it direct.
        # For now, keeping it direct as it's not I/O bound.
        tz = zoneinfo.ZoneInfo("Asia/Manila")
        now = datetime.now(tz)
        return now.strftime("%Y-%m-%d %H:%M:%S")

    def __init__(self, collection_name: str = "llm_cache_collection", embedding_model_name: str = "all-MiniLM-L6-v2", chroma_path: str = "./chroma_db/llm_cache_data", max_entries: int = 10000):
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.embedding_model = embedding_manager.get_model(embedding_model_name)
        self.max_entries = max_entries
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            logging.info(f"Connected to existing ChromaDB collection: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(collection_name)
            logging.info(f"Created new ChromaDB collection: {collection_name}")

    async def _run_in_executor(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(func, *args))

    async def get(self, prompt: str, similarity_threshold: float = 0.9) -> Any:
        # Fast path: exact string match in cache using metadata
        all_entries = await self._run_in_executor(partial(self.collection.get, include=['metadatas']))
        for i, metadata in enumerate(all_entries['metadatas']):
            if metadata.get('prompt') == prompt:
                cached_response = metadata.get('response')
                # Update access tracking for LRU
                await self._update_access_tracking(all_entries['ids'][i])
                logging.info(f"Cache EXACT MATCH for prompt: '{prompt[:50]}...'")
                return cached_response

        # Usual vector similarity path
        query_embedding = await self._run_in_executor(
            partial(self.embedding_model.encode, [prompt], show_progress_bar=False)
        )
        results = await self._run_in_executor(
            partial(self.collection.query,
                query_embeddings=query_embedding.tolist(),
                n_results=1,
                include=['documents', 'metadatas', 'distances']
            )
        )

        if results['distances'] and results['distances'][0]:
            distance = results['distances'][0][0]
            similarity = 1 - distance 
            if similarity >= similarity_threshold:
                cached_response = results['metadatas'][0][0].get('response')
                # Update access tracking for LRU
                await self._update_access_tracking(results['ids'][0][0])
                logging.info(f"Cache HIT for prompt: '{prompt[:50]}...' (Similarity: {similarity:.4f})")
                return cached_response

        logging.info(f"Cache MISS for prompt: '{prompt[:50]}...'")
        return None

    async def set(self, prompt: str, response: Any):
        # Check if cache is at capacity and perform LRU eviction
        current_size = await self.get_cache_size()
        if current_size >= self.max_entries:
            await self._evict_oldest_entries(int(self.max_entries * 0.1))  # Remove 10% of oldest
        
        prompt_embedding = await self._run_in_executor(
            partial(self.embedding_model.encode, [prompt], show_progress_bar=False)
        )
        entry_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        await self._run_in_executor(
            partial(self.collection.add,
                metadatas=[{"prompt": prompt, "response": response, "timestamp": timestamp, "access_count": 0}],
                ids=[entry_id],
                embeddings=prompt_embedding.tolist()
            )
        )
        logging.info(f"Cache SET for prompt: '{prompt[:50]}...' (ID: {entry_id})")

    async def clear_cache(self):
        """Clears all entries from the cache collection."""
        await self._run_in_executor(partial(self.chroma_client.delete_collection, self.collection.name))
        self.collection = await self._run_in_executor(partial(self.chroma_client.create_collection, self.collection.name))
        logging.info(f"Cache collection '{self.collection.name}' cleared.")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        size = await self.get_cache_size()
        return {
            "size": size,
            "max_entries": self.max_entries,
            "utilization": size / self.max_entries if self.max_entries > 0 else 0
        }

    async def evict_old_entries(self, days_old: int):
        """Evicts cache entries older than a specified number of days."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        all_entries = await self._run_in_executor(partial(self.collection.get, include=['metadatas']))
        ids_to_delete = []
        for i, metadata in enumerate(all_entries['metadatas']):
            entry_timestamp_str = metadata.get('timestamp')
            if entry_timestamp_str:
                entry_datetime = datetime.fromisoformat(entry_timestamp_str)
                if entry_datetime < cutoff_date:
                    ids_to_delete.append(all_entries['ids'][i])
        if ids_to_delete:
            await self._run_in_executor(partial(self.collection.delete, ids=ids_to_delete))
            logging.info(f"Evicted {len(ids_to_delete)} old cache entries.")
        else:
            logging.info("No old cache entries to evict.")

    async def get_cache_size(self) -> int:
        """Returns the number of entries in the cache."""
        return await self._run_in_executor(self.collection.count)
    
    async def _evict_oldest_entries(self, num_to_evict: int):
        """Remove the oldest cache entries for LRU management."""
        try:
            all_entries = await self._run_in_executor(
                partial(self.collection.get, include=['metadatas'])
            )
            
            if not all_entries['metadatas']:
                return
            
            # Sort by timestamp (oldest first)
            entries_with_timestamps = [
                (all_entries['ids'][i], metadata.get('timestamp', '1970-01-01T00:00:00'))
                for i, metadata in enumerate(all_entries['metadatas'])
            ]
            
            entries_with_timestamps.sort(key=lambda x: x[1])
            
            # Get IDs of oldest entries to remove
            ids_to_remove = [entry[0] for entry in entries_with_timestamps[:num_to_evict]]
            
            if ids_to_remove:
                await self._run_in_executor(
                    partial(self.collection.delete, ids=ids_to_remove)
                )
                logging.info(f"LRU evicted {len(ids_to_remove)} oldest cache entries")
                
        except Exception as e:
            logging.error(f"Error during LRU eviction: {e}")
    
    async def _update_access_tracking(self, entry_id: str):
        """Update access count and timestamp for LRU tracking."""
        try:
            # Note: ChromaDB doesn't support in-place updates, so we skip this for now
            # In a production system, you might use a separate metadata store
            pass
        except Exception as e:
            logging.debug(f"Error updating access tracking: {e}")