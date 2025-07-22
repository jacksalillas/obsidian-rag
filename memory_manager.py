import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import shutil

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, vector_store_manager, memory_dir: str = "./memory", 
                 icloud_backup_dir: str = "~/Library/Mobile Documents/com~apple~CloudDocs/Saved Memory Backup"):
        """Initialize the memory manager for conversation history and context."""
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        self.icloud_backup_dir = Path(icloud_backup_dir).expanduser()
        self.icloud_backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.user_preferences_file = self.memory_dir / "user_preferences.json"
        self.saved_memories_file = self.memory_dir / "saved_memories.json"
        
        # In-memory storage
        self.user_preferences: Dict[str, Any] = {}
        self.saved_memories: List[Dict[str, Any]] = []
        self.is_dirty = False  # Track if memory has changed
        
        # Conversation limits (only for context, not history)
        self.max_context_age_days = 30
    
    async def async_init(self):
        """Async initialization to load existing data."""
        await self.load_user_preferences()
        await self.load_saved_memories()
        logger.info("Memory manager initialized")
    
    async def load_user_preferences(self):
        """Load user preferences from disk."""
        try:
            if self.user_preferences_file.exists():
                content = await asyncio.to_thread(self._read_file, self.user_preferences_file)
                self.user_preferences = json.loads(content)
                logger.info("Loaded user preferences")
        except Exception as e:
            logger.error(f"Error loading user preferences: {e}")
            self.user_preferences = {}
    
    async def save_user_preferences(self):
        """Save user preferences to disk."""
        try:
            content = json.dumps(self.user_preferences, indent=2)
            await asyncio.to_thread(self._write_file, self.user_preferences_file, content)
            logger.debug("Saved user preferences to disk")
        except Exception as e:
            logger.error(f"Error saving user preferences: {e}")
    
    async def load_saved_memories(self):
        """Load saved memories from disk."""
        try:
            if self.saved_memories_file.exists():
                content = await asyncio.to_thread(self._read_file, self.saved_memories_file)
                self.saved_memories = json.loads(content)
                logger.info(f"Loaded {len(self.saved_memories)} saved memories")
        except Exception as e:
            logger.error(f"Error loading saved memories: {e}")
            self.saved_memories = []
    
    async def save_saved_memories(self):
        """Save saved memories to disk."""
        try:
            content = json.dumps(self.saved_memories, indent=2)
            await asyncio.to_thread(self._write_file, self.saved_memories_file, content)
            logger.debug("Saved memories to disk")
        except Exception as e:
            logger.error(f"Error saving saved memories: {e}")
    
    async def add_saved_memory(self, synthesized_memory: str, category: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Add a synthesized memory to saved memories."""
        entry_metadata = metadata or {}
        if category:
            entry_metadata["category"] = category

        entry = {
            "timestamp": datetime.now().isoformat(),
            "content": synthesized_memory,
            "metadata": entry_metadata
        }
        self.saved_memories.append(entry)
        self.is_dirty = True
        await self.save_saved_memories()
        
        # Add to vector store for retrieval
        memory_id = f"saved_memory_{len(self.saved_memories) - 1}_{datetime.now().timestamp()}"
        self.vector_store_manager.add_saved_memory_document(synthesized_memory, memory_id, category=category)
        
        logger.debug("Added saved memory")
    
    async def get_all_saved_memories(self) -> List[Dict[str, Any]]:
        """Get all saved memories."""
        return self.saved_memories
    
    async def set_user_preference(self, key: str, value: Any):
        """Set user preference."""
        self.user_preferences[key] = value
        self.is_dirty = True
        await self.save_user_preferences()
        logger.debug(f"Set user preference: {key}")
    
    async def get_user_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference."""
        return self.user_preferences.get(key, default)
    
    async def backup_memories_to_icloud(self):
        """Backup only user preferences and saved memories to iCloud if changes have been made."""
        if not self.is_dirty:
            logger.info("No changes to memories, skipping backup.")
            return
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_subdir = self.icloud_backup_dir / f"backup_{timestamp}"
            backup_subdir.mkdir(exist_ok=True)
            
            # Copy only user preferences and saved memories files
            files_to_backup = [self.user_preferences_file, self.saved_memories_file]
            for file_path in files_to_backup:
                if file_path.exists():
                    backup_file = backup_subdir / file_path.name
                    await asyncio.to_thread(shutil.copy2, file_path, backup_file)
            
            logger.info(f"Backed up memories to: {backup_subdir}")
            self.is_dirty = False  # Reset dirty flag after successful backup
            
            # Keep only the last 10 backups
            await self._cleanup_old_backups()
            
        except Exception as e:
            logger.error(f"Error backing up memories to iCloud: {e}")
    
    async def _cleanup_old_backups(self):
        """Keep only the most recent backups."""
        try:
            backup_dirs = sorted([d for d in self.icloud_backup_dir.iterdir() if d.is_dir() and d.name.startswith("backup_")])
            
            if len(backup_dirs) > 10:
                for old_backup in backup_dirs[:-10]:
                    await asyncio.to_thread(shutil.rmtree, old_backup)
                logger.info(f"Cleaned up {len(backup_dirs) - 10} old backups")
                
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
    
    def _read_file(self, file_path):
        """Synchronous file read helper."""
        with open(file_path, 'r') as f:
            return f.read()
    
    def _write_file(self, file_path, content):
        """Synchronous file write helper."""
        with open(file_path, 'w') as f:
            f.write(content)