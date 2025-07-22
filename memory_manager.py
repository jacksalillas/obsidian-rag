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
    def __init__(self, memory_dir: str = "./memory", 
                 icloud_backup_dir: str = "~/Documents/mnemosyne_memory_backup"):
        """Initialize the memory manager for conversation history and context."""
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        self.icloud_backup_dir = Path(icloud_backup_dir).expanduser()
        self.icloud_backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.conversation_file = self.memory_dir / "conversation_history.json"
        self.context_file = self.memory_dir / "conversation_context.json"
        self.user_preferences_file = self.memory_dir / "user_preferences.json"
        
        # In-memory storage
        self.conversation_history: List[Dict[str, Any]] = []
        self.conversation_context: Dict[str, Any] = {}
        self.user_preferences: Dict[str, Any] = {}
        self.is_dirty = False  # Track if memory has changed
        
        # Conversation limits
        self.max_history_entries = 1000
        self.max_context_age_days = 30
    
    async def async_init(self):
        """Async initialization to load existing data."""
        await self.load_conversation_history()
        await self.load_conversation_context()
        await self.load_user_preferences()
        logger.info("Memory manager initialized")
    
    async def load_conversation_history(self):
        """Load conversation history from disk."""
        try:
            if self.conversation_file.exists():
                content = await asyncio.to_thread(self._read_file, self.conversation_file)
                self.conversation_history = json.loads(content)
                logger.info(f"Loaded {len(self.conversation_history)} conversation entries")
        except Exception as e:
            logger.error(f"Error loading conversation history: {e}")
            self.conversation_history = []
    
    async def save_conversation_history(self):
        """Save conversation history to disk."""
        try:
            content = json.dumps(self.conversation_history, indent=2)
            await asyncio.to_thread(self._write_file, self.conversation_file, content)
            logger.debug("Saved conversation history to disk")
        except Exception as e:
            logger.error(f"Error saving conversation history: {e}")
    
    async def load_conversation_context(self):
        """Load conversation context from disk."""
        try:
            if self.context_file.exists():
                content = await asyncio.to_thread(self._read_file, self.context_file)
                self.conversation_context = json.loads(content)
                logger.info("Loaded conversation context")
        except Exception as e:
            logger.error(f"Error loading conversation context: {e}")
            self.conversation_context = {}
    
    async def save_conversation_context(self):
        """Save conversation context to disk."""
        try:
            content = json.dumps(self.conversation_context, indent=2)
            await asyncio.to_thread(self._write_file, self.context_file, content)
            logger.debug("Saved conversation context to disk")
        except Exception as e:
            logger.error(f"Error saving conversation context: {e}")
    
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
    
    async def add_conversation_entry(self, user_message: str, assistant_response: str, 
                                   metadata: Optional[Dict[str, Any]] = None):
        """Add a conversation entry to history."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "assistant_response": assistant_response,
            "metadata": metadata or {}
        }
        
        self.conversation_history.append(entry)
        self.is_dirty = True
        
        # Maintain history size limit
        if len(self.conversation_history) > self.max_history_entries:
            self.conversation_history = self.conversation_history[-self.max_history_entries:]
        
        await self.save_conversation_history()
        logger.debug("Added conversation entry to history")
    
    async def get_recent_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history."""
        return self.conversation_history[-limit:] if self.conversation_history else []
    
    async def search_conversation_history(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search conversation history for relevant entries."""
        query_lower = query.lower()
        matching_entries = []
        
        for entry in reversed(self.conversation_history):
            user_msg = entry.get("user_message", "").lower()
            assistant_msg = entry.get("assistant_response", "").lower()
            
            if query_lower in user_msg or query_lower in assistant_msg:
                matching_entries.append(entry)
                if len(matching_entries) >= limit:
                    break
        
        return matching_entries
    
    async def update_conversation_context(self, key: str, value: Any):
        """Update conversation context."""
        self.conversation_context[key] = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        self.is_dirty = True
        await self.save_conversation_context()
        logger.debug(f"Updated conversation context: {key}")
    
    async def get_conversation_context(self, key: str) -> Optional[Any]:
        """Get conversation context value."""
        if key in self.conversation_context:
            context_entry = self.conversation_context[key]
            # Check if context is still valid (not too old)
            timestamp = datetime.fromisoformat(context_entry["timestamp"])
            if datetime.now() - timestamp < timedelta(days=self.max_context_age_days):
                return context_entry["value"]
            else:
                # Remove old context
                del self.conversation_context[key]
                await self.save_conversation_context()
        
        return None
    
    async def set_user_preference(self, key: str, value: Any):
        """Set user preference."""
        self.user_preferences[key] = value
        self.is_dirty = True
        await self.save_user_preferences()
        logger.debug(f"Set user preference: {key}")
    
    async def get_user_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference."""
        return self.user_preferences.get(key, default)
    
    async def cleanup_old_entries(self):
        """Clean up old conversation entries and context."""
        cutoff_date = datetime.now() - timedelta(days=self.max_context_age_days)
        
        # Clean conversation history
        original_count = len(self.conversation_history)
        self.conversation_history = [
            entry for entry in self.conversation_history
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_date
        ]
        removed_history = original_count - len(self.conversation_history)
        
        # Clean conversation context
        original_context_count = len(self.conversation_context)
        keys_to_remove = []
        for key, context_entry in self.conversation_context.items():
            timestamp = datetime.fromisoformat(context_entry["timestamp"])
            if timestamp < cutoff_date:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.conversation_context[key]
        
        removed_context = len(keys_to_remove)
        
        if removed_history > 0 or removed_context > 0:
            await self.save_conversation_history()
            await self.save_conversation_context()
            logger.info(f"Cleaned up {removed_history} old history entries and {removed_context} old context entries")
    
    async def backup_memories_to_icloud(self):
        """Backup all memory files to iCloud if changes have been made."""
        if not self.is_dirty:
            logger.info("No changes to memories, skipping backup.")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_subdir = self.icloud_backup_dir / f"backup_{timestamp}"
            backup_subdir.mkdir(exist_ok=True)
            
            # Copy all memory files
            for file_path in self.memory_dir.glob("*.json"):
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