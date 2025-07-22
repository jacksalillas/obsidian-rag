import asyncio
import logging
from typing import Dict, Any, List, Optional
from llm_cache import LLMCache
from ollama_client_pool import get_ollama_pool

logger = logging.getLogger(__name__)

class ChatModel:
    def __init__(self, model_name: str = "mistral:latest", 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 mnemosyne_vault_guidance: List[Dict[str, str]] = None,
                 use_cache: bool = False,
                 cache_similarity_threshold: float = 0.95,
                 dynamic_context: bool = True,
                 embedding_manager = None):
        """Initialize the chat model with Ollama connection pooling and optional caching."""
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.mnemosyne_vault_guidance = mnemosyne_vault_guidance or []
        self.use_cache = use_cache
        self.cache_similarity_threshold = cache_similarity_threshold
        self.dynamic_context = dynamic_context
        
        # Get connection pool instance
        self.ollama_pool = get_ollama_pool()
        
        # Initialize cache if enabled
        if self.use_cache:
            self.cache = LLMCache(
                collection_name="mnemosyne_chat_cache",
                embedding_model_name=embedding_model_name,
                embedding_manager=embedding_manager
            )
        else:
            self.cache = None
            
        self.system_prompt = self._build_system_prompt()
        
    async def async_init(self):
        """Async initialization."""
        logger.info(f"Chat model initialized with {self.model_name}")
        
    def _build_system_prompt(self) -> str:
        """Build the system prompt for Mnemosyne."""
        base_prompt = """You are Mnemosyne, a sophisticated AI assistant designed to help users explore and understand their personal knowledge base in Obsidian. Your primary goal is to provide clear, comprehensive, and well-structured answers based on the information found in the user's notes.

**Core Directives:**
1.  **Synthesize, Don't Just List:** Do not simply list the source files or raw text. Synthesize the information from the provided context into a coherent and complete answer.
2.  **Answer the Question Directly:** Address the user's query directly. Use the context from their notes as evidence to build your response.
3.  **Cite Sources Clearly:** After providing the synthesized answer, cite the relevant source files in a clear and organized manner (e.g., "Source: `filename.md`"). This should be a supplement to the answer, not the answer itself.
4.  **Be Context-Aware:** Pay attention to conversation history to understand follow-up questions and maintain context.
5.  **Proactive Connections:** If you identify connections between different notes that are relevant to the user's query, highlight them.

**Example Interaction:**
User: "What are the key principles of Zettelkasten?"
Mnemosyne: "The key principles of Zettelkasten, based on your notes, are atomicity, linking, and using a unique identifier for each note. Atomicity means each note should contain a single, discrete idea. This allows for more precise linking between concepts. Your notes in `Zettelkasten Principles.md` and `Note Taking Methods.md` both emphasize this core idea.

Sources:
- `Zettelkasten Principles.md`
- `Note Taking Methods.md`"

When referencing information from the user's notes, always mention the source file when possible."""
        if self.mnemosyne_vault_guidance:
            vault_guidance = "\n\n**Vault Organization Guidance:**\n"
            for guidance in self.mnemosyne_vault_guidance:
                vault_guidance += f"- **{guidance['category']}**: Use vault '{guidance['vault']}' for queries related to this category.\n"
            base_prompt += vault_guidance
        return base_prompt
    
    async def generate_response(self, prompt: str, context: Optional[str] = None, 
                              conversation_history: Optional[List[Dict[str, str]]] = None,
                              max_tokens: Optional[int] = None,
                              temperature: float = 0.7,
                              context_k: Optional[int] = None) -> str:
        """Generate a response using the Ollama model with connection pooling and optional caching."""
        
        # Dynamic context adjustment based on query complexity
        if self.dynamic_context and context_k is None:
            # Simple heuristic: longer queries or questions get more context
            query_words = len(prompt.split())
            has_question = any(word in prompt.lower() for word in ['what', 'how', 'why', 'when', 'where', 'which', 'explain'])
            
            if query_words > 20 or has_question:
                context_k = 10  # More context for complex queries
            elif query_words > 10:
                context_k = 7   # Standard context
            else:
                context_k = 5   # Less context for simple queries
        
        # Build the full prompt with context
        full_prompt = self._build_full_prompt(prompt, context, conversation_history)
        
        # Check cache first if enabled
        if self.cache:
            cached_response = await self.cache.get(full_prompt, self.cache_similarity_threshold)
            if cached_response:
                return cached_response
        
        try:
            # Generate response with Ollama using connection pool
            response = await self.ollama_pool.generate_with_pool(
                model=self.model_name,
                prompt=full_prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens or -1
                }
            )
            
            generated_text = response["response"]
            logger.info(f"Raw LLM response: {generated_text}")
            
            # Cache the response if caching is enabled
            if self.cache and generated_text:
                await self.cache.set(full_prompt, generated_text)
            
            logger.debug(f"Generated response for prompt: {prompt[:50]}...")
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
    def _build_full_prompt(self, user_prompt: str, context: Optional[str] = None,
                          conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Build the full prompt including system prompt, context, and history with token awareness."""
        parts = [self.system_prompt]
        
        # Add conversation history if provided
        if conversation_history:
            parts.append("\n## Recent Conversation History")
            # Limit history based on estimated token count (rough: 4 chars per token)
            max_history_tokens = 2000
            history_text = ""
            for entry in reversed(conversation_history[-10:]):  # Check last 10, use what fits
                entry_text = f"User: {entry.get('user_message', '')}\nMnemosyne: {entry.get('assistant_response', '')}\n"
                if len(history_text + entry_text) * 0.25 < max_history_tokens:  # Rough token estimation
                    history_text = entry_text + history_text
                else:
                    break
            parts.append(history_text.strip())
        
        # Add relevant context from knowledge base
        if context:
            # Truncate context if too long (max ~4000 tokens)
            max_context_chars = 16000
            if len(context) > max_context_chars:
                context = context[:max_context_chars] + "\n... [truncated for length]"
            
            parts.append(f"\n## Relevant Context from Your Notes\nHere is some information from your notes that might be relevant to your query. Use this to construct your answer.")
            parts.append(context)
        
        # Add the current user prompt
        parts.append(f"\n## Your Query\nUser: {user_prompt}")
        parts.append("\n## Your Answer\nAnswer the user's query directly and comprehensively based on the provided context.")
        
        full_prompt = "\n".join(parts)
        
        # Log token estimate for monitoring
        estimated_tokens = len(full_prompt) * 0.25
        logger.debug(f"Prompt estimated tokens: {estimated_tokens:.0f}")
        
        return full_prompt
    
    async def summarize_text(self, text: str, max_length: int = 200) -> str:
        """Summarize a piece of text."""
        prompt = f"Please provide a concise summary (max {max_length} words) of the following text:\n\n{text}"
        
        try:
            summary = await self.generate_response(prompt, temperature=0.3)
            return summary.strip()
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return "Unable to generate summary."
    
    async def extract_key_points(self, text: str) -> List[str]:
        """Extract key points from a piece of text."""
        prompt = f"Please extract the main key points from the following text as a bulleted list:\n\n{text}"
        
        try:
            response = await self.generate_response(prompt, temperature=0.3)
            # Parse the response to extract bullet points
            lines = response.split('\n')
            key_points = []
            for line in lines:
                line = line.strip()
                if line.startswith(('•', '-', '*', '+')):
                    key_points.append(line[1:].strip())
                elif line and not line.startswith(('Here are', 'Key points', 'Main points')):
                    # Handle numbered lists
                    import re
                    if re.match(r'^\d+\.?\s+', line):
                        key_points.append(re.sub(r'^\d+\.?\s+', '', line))
            
            return key_points[:10]  # Limit to top 10 points
        except Exception as e:
            logger.error(f"Error extracting key points: {e}")
            return []
    
    async def suggest_tags(self, text: str) -> List[str]:
        """Suggest relevant tags for a piece of text."""
        prompt = f"Based on the following text, suggest 3-5 relevant tags (single words or short phrases) that would be useful for organizing this content:\n\n{text}"
        
        try:
            response = await self.generate_response(prompt, temperature=0.5)
            # Extract tags from the response
            import re
            # Look for tags in various formats
            tags = re.findall(r'#(\w+)', response)  # hashtags
            if not tags:
                # Look for comma-separated or bulleted lists
                lines = response.replace(',', '\n').split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith(('Here are', 'Tags:', 'Suggested', 'Based on')):
                        # Clean up the tag
                        tag = re.sub(r'[^\w\s-]', '', line).strip()
                        if tag and len(tag) <= 20:
                            tags.append(tag)
            
            return tags[:5]  # Limit to 5 tags
        except Exception as e:
            logger.error(f"Error suggesting tags: {e}")
            return []
    
    async def check_model_availability(self) -> bool:
        """Check if the specified model is available in Ollama."""
        try:
            models = await self.ollama_pool.list_models_with_pool()
            available_models = [model['name'] for model in models['models']]
            return self.model_name in available_models
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False
    
    async def clear_cache(self):
        """Clear the LLM cache."""
        if self.cache:
            await self.cache.clear_cache()
            logger.info("LLM cache cleared")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache and connection pool statistics."""
        stats = {"cache_enabled": False}
        
        if self.cache:
            cache_stats = await self.cache.get_cache_stats()
            stats.update({
                "cache_enabled": True,
                "cache_size": cache_stats["size"],
                "cache_max_entries": cache_stats["max_entries"],
                "cache_utilization": cache_stats["utilization"],
                "similarity_threshold": self.cache_similarity_threshold
            })
        
        # Add connection pool stats
        pool_stats = self.ollama_pool.get_pool_stats()
        stats["connection_pool"] = pool_stats
        
        return stats