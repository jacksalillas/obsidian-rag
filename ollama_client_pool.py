import asyncio
import threading
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import ollama

logger = logging.getLogger(__name__)

@dataclass
class ConnectionStats:
    """Track connection usage statistics."""
    created_at: datetime
    last_used: datetime
    use_count: int = 0
    is_healthy: bool = True

class OllamaConnectionPool:
    """Connection pool manager for Ollama clients."""
    
    def __init__(self, max_connections: int = 10, connection_timeout: int = 300):
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout  # seconds
        self._pool = asyncio.Queue(maxsize=max_connections)
        self._stats: Dict[str, ConnectionStats] = {}
        self._pool_lock = asyncio.Lock()
        self._initialized = False
        
    async def _initialize_pool(self):
        """Initialize the connection pool."""
        if self._initialized:
            return
            
        async with self._pool_lock:
            if self._initialized:
                return
                
            # Pre-fill pool with initial connections
            initial_connections = min(3, self.max_connections)  # Start with 3 connections
            for i in range(initial_connections):
                connection_id = f"conn_{i}"
                await self._pool.put(connection_id)
                self._stats[connection_id] = ConnectionStats(
                    created_at=datetime.now(),
                    last_used=datetime.now()
                )
                
            self._initialized = True
            logger.info(f"Ollama connection pool initialized with {initial_connections} connections")
    
    async def get_connection(self) -> str:
        """Get a connection from the pool."""
        await self._initialize_pool()
        
        try:
            # Try to get a connection with a short timeout
            connection_id = await asyncio.wait_for(self._pool.get(), timeout=5.0)
            
            # Update usage stats
            if connection_id in self._stats:
                stats = self._stats[connection_id]
                stats.last_used = datetime.now()
                stats.use_count += 1
            
            return connection_id
            
        except asyncio.TimeoutError:
            # If no connection available, create a new one if under limit
            if len(self._stats) < self.max_connections:
                connection_id = f"conn_{len(self._stats)}"
                self._stats[connection_id] = ConnectionStats(
                    created_at=datetime.now(),
                    last_used=datetime.now(),
                    use_count=1
                )
                logger.info(f"Created new Ollama connection: {connection_id}")
                return connection_id
            else:
                # Pool is at capacity and no connections available
                raise RuntimeError("Ollama connection pool exhausted")
    
    async def return_connection(self, connection_id: str):
        """Return a connection to the pool."""
        if connection_id not in self._stats:
            logger.warning(f"Attempting to return unknown connection: {connection_id}")
            return
            
        # Check if connection is still healthy and not too old
        stats = self._stats[connection_id]
        connection_age = datetime.now() - stats.created_at
        
        if connection_age.total_seconds() > self.connection_timeout:
            # Connection is too old, don't return it to pool
            del self._stats[connection_id]
            logger.debug(f"Discarded aged Ollama connection: {connection_id}")
            return
            
        try:
            await self._pool.put(connection_id)
        except asyncio.QueueFull:
            # Pool is full, discard this connection
            del self._stats[connection_id]
            logger.debug(f"Pool full, discarded connection: {connection_id}")
    
    async def generate_with_pool(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response using a pooled connection."""
        connection_id = None
        try:
            connection_id = await self.get_connection()
            logger.debug(f"Using Ollama connection {connection_id} for generation")
            
            # Use asyncio.to_thread for the blocking ollama call
            response = await asyncio.to_thread(
                ollama.generate,
                model=model,
                prompt=prompt,
                **kwargs
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in Ollama generation with {connection_id}: {e}")
            # Mark connection as unhealthy if it was the cause
            if connection_id and connection_id in self._stats:
                self._stats[connection_id].is_healthy = False
            raise
            
        finally:
            if connection_id:
                await self.return_connection(connection_id)
    
    async def list_models_with_pool(self) -> Dict[str, Any]:
        """List models using a pooled connection."""
        connection_id = None
        try:
            connection_id = await self.get_connection()
            logger.debug(f"Using Ollama connection {connection_id} for model listing")
            
            # Use asyncio.to_thread for the blocking ollama call
            models = await asyncio.to_thread(ollama.list)
            return models
            
        except Exception as e:
            logger.error(f"Error listing Ollama models with {connection_id}: {e}")
            if connection_id and connection_id in self._stats:
                self._stats[connection_id].is_healthy = False
            raise
            
        finally:
            if connection_id:
                await self.return_connection(connection_id)
    
    async def cleanup_stale_connections(self):
        """Clean up old/stale connections."""
        cutoff_time = datetime.now() - timedelta(seconds=self.connection_timeout)
        stale_connections = []
        
        for conn_id, stats in self._stats.items():
            if stats.last_used < cutoff_time or not stats.is_healthy:
                stale_connections.append(conn_id)
        
        for conn_id in stale_connections:
            del self._stats[conn_id]
            logger.debug(f"Cleaned up stale Ollama connection: {conn_id}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        healthy_connections = sum(1 for stats in self._stats.values() if stats.is_healthy)
        total_uses = sum(stats.use_count for stats in self._stats.values())
        
        return {
            "total_connections": len(self._stats),
            "healthy_connections": healthy_connections,
            "max_connections": self.max_connections,
            "pool_size": self._pool.qsize(),
            "total_uses": total_uses,
            "connection_timeout": self.connection_timeout
        }

# Global connection pool instance
_ollama_pool = None
_pool_lock = threading.Lock()

def get_ollama_pool(max_connections: int = 10, connection_timeout: int = 300) -> OllamaConnectionPool:
    """Get or create the global Ollama connection pool."""
    global _ollama_pool
    
    if _ollama_pool is None:
        with _pool_lock:
            if _ollama_pool is None:
                _ollama_pool = OllamaConnectionPool(max_connections, connection_timeout)
                
    return _ollama_pool