import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import logging
try:
    from colorama import Fore, Style, Back, init
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    # Fallback if colorama is not installed
    class Fore:
        BLUE = ''
        CYAN = ''
        MAGENTA = ''
    class Style:
        RESET_ALL = ''
    class Back:
        CYAN = ''
    COLORS_AVAILABLE = False

from vault_manager import VaultManager
from vector_store_manager import VectorStoreManager
from memory_manager import MemoryManager
from chat_model import ChatModel

logger = logging.getLogger(__name__)

class ChatInterface:
    def __init__(self, vault_manager: VaultManager, vector_store_manager: VectorStoreManager,
                 memory_manager: MemoryManager, chat_model: ChatModel, embedding_manager):
        """Initialize the chat interface."""
        self.vault_manager = vault_manager
        self.vector_store_manager = vector_store_manager
        self.memory_manager = memory_manager
        self.chat_model = chat_model
        self.console = Console()
        
        # Interface settings - now dynamic
        self.base_context_results = 5
        self.max_context_results = 12  # Upper limit for dynamic adjustment
        self.running = False
        
    async def start(self):
        """Start the interactive chat interface."""
        self.running = True
        
        # Display welcome message
        await self._display_welcome()
        
        # Main chat loop
        while self.running:
            try:
                # Get user input
                user_input = await self._get_user_input()
                
                if not user_input:
                    continue
                    
                # Handle special commands
                if await self._handle_special_commands(user_input):
                    continue
                
                # Process the user's query
                await self._process_user_query(user_input)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Chat session interrupted.[/yellow]")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                self.console.print(f"[red]An error occurred: {e}[/red]")
                
        self.console.print("[green]Goodbye![/green]")
    
    async def _display_welcome(self):
        """Display welcome message and system status."""
        # Show system status in the aesthetic format with border
        await self._show_system_status()
    
    async def _show_system_status(self):
        """Alternative implementation with string-based content."""
        try:
            # Get system data
            vector_stats = self.vector_store_manager.get_collection_stats()
            cache_stats = await self.chat_model.get_cache_stats()
            recent_conversations = await self.memory_manager.get_recent_conversation_history(limit=10)
            
            # Get vault names
            vault_names = [Path(path).name for path in self.vault_manager.source_paths]
            vault_names_str = ', '.join(vault_names) if vault_names else 'None'
            
            # Create content string with Rich markup - COLORS APPLIED HERE:
            content = f"""[bold blue] Mnemosyne - Your Personal AI Assistant[/bold blue]\n\nSources: [cyan]{vault_names_str}[/cyan]\nPersonal Memories: [magenta]{len(recent_conversations)}[/magenta] saved\n\n✨ I can suggest file modifications - I'll ask for your permission first\n📝 I remember our conversation context for follow-up questions\n🧠 I remember personal details you teach me permanently\n\n[dim]Memory commands:[/dim] [yellow]'remember: ', 'show my memories'[/yellow]\n[dim]System commands:[/dim] [yellow]'status', 'reindex', 'bye', 'quit', 'exit', 'q'[/yellow]"""
            
            # Create and display the panel
            panel = Panel(
                content,
                title="[bold blue] Mnemosyne [/bold blue]",
                border_style="blue",
                padding=(1, 2),
                expand=False
            )
            
            self.console.print(panel)
            self.console.print()
            
        except Exception as e:
            logger.error(f"Error showing system status: {e}")
    
    async def _get_user_input(self) -> str:
        """Get user input with prompt."""
        try:
            # Simple prompt to match aesthetic
            user_input = input("Ask me anything about your vaults: 🟡 ")
            return user_input.strip()
        except (EOFError, KeyboardInterrupt):
            return "/quit"
    
    async def _handle_special_commands(self, user_input: str) -> bool:
        """Handle special commands. Returns True if command was handled."""
        # Handle simple exit commands without slash
        if user_input.lower() in ['bye', 'quit', 'exit', 'q']:
            self.running = False
            return True
            
        if not user_input.startswith('/'):
            return False
        
        command_parts = user_input[1:].split(' ', 1)
        command = command_parts[0].lower()
        args = command_parts[1] if len(command_parts) > 1 else ""
        
        if command in ['quit', 'exit']:
            self.running = False
            return True
            
        elif command == 'help':
            await self._show_help()
            return True
            
        elif command == 'stats':
            await self._show_detailed_stats()
            return True
            
        elif command == 'clear':
            if Confirm.ask("Are you sure you want to clear conversation history?"):
                # Clear in-memory history (not persistent storage)
                self.console.print("[yellow]Conversation history cleared for this session.[/yellow]")
            return True
            
        elif command == 'search':
            if args:
                await self._search_knowledge_base(args)
            else:
                self.console.print("[red]Please provide a search query. Example: /search python[/red]")
            return True
            
        elif command == 'cache':
            await self._show_cache_stats()
            return True
            
        elif command == 'clearcache':
            if Confirm.ask("Are you sure you want to clear the cache?"):
                await self.chat_model.clear_cache()
                self.console.print("[green]Cache cleared successfully.[/green]")
            return True
            
        else:
            self.console.print(f"[red]Unknown command: {command}. Type /help for available commands.[/red]")
            return True
    
    async def _show_help(self):
        """Show help information."""
        help_text = """
# Mnemosyne Help\n\n**Commands:**\n- `/help` - Show this help message\n- `/stats` - Show detailed system statistics  \n- `/clear` - Clear current conversation history\n- `/quit` or `/exit` - Exit the application\n- `/search <query>` - Search your knowledge base directly\n- `/cache` - Show cache statistics and management options\n- `/clearcache` - Clear the LLM response cache\n\n**Usage Tips:**\n- Ask questions about your notes naturally\n- Reference specific files or topics from your Obsidian vaults\n- Use keywords from your notes for better search results\n- Mnemosyne remembers context within the conversation\n\n**Examples:**\n- "What did I write about machine learning?"\n- "Summarize my notes on project management"\n- "Show me everything related to #python"\n        """
        
        self.console.print(Panel(Markdown(help_text), title="Help", border_style="blue"))
    
    async def _show_detailed_stats(self):
        """Show detailed system statistics."""
        try:
            # Vector store stats
            vector_stats = self.vector_store_manager.get_collection_stats()
            
            # Cache stats
            cache_stats = await self.chat_model.get_cache_stats()
            
            # Memory stats
            recent_history = await self.memory_manager.get_recent_conversation_history(limit=100)
            
            # Create detailed stats table
            table = Table(title="Detailed System Statistics", border_style="dim")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            # Vector store
            table.add_row("Total Documents", str(vector_stats.get('total_documents', 0)))
            table.add_row("Collection Name", vector_stats.get('collection_name', 'N/A'))
            table.add_row("Embedding Model", vector_stats.get('embedding_model', 'N/A'))
            
            # Chat model
            table.add_row("Chat Model", self.chat_model.model_name)
            table.add_row("Model Available", str(await self.chat_model.check_model_availability()))
            
            # Cache
            if cache_stats.get('cache_enabled'):
                table.add_row("Cache Status", "Enabled")
                table.add_row("Cache Entries", str(cache_stats.get('cache_size', 0)))
                table.add_row("Cache Threshold", str(cache_stats.get('similarity_threshold', 0)))
            else:
                table.add_row("Cache Status", "Disabled")
            
            # Memory
            table.add_row("Conversation History", f"{len(recent_history)} entries")
            
            # Vault information
            table.add_row("Watched Sources", str(len(self.vault_manager.source_paths)))
            table.add_row("File Watching", "Active" if self.vault_manager.is_watching else "Inactive")
            
            self.console.print(table)
            
        except Exception as e:
            logger.error(f"Error showing detailed stats: {e}")
            self.console.print("[red]Error retrieving system statistics.[/red]")
    
    async def _show_cache_stats(self):
        """Show cache statistics and management options."""
        try:
            cache_stats = await self.chat_model.get_cache_stats()
            
            if not cache_stats.get('cache_enabled'):
                self.console.print("[yellow]Cache is currently disabled.[/yellow]")
                return
            
            table = Table(title="Cache Statistics", border_style="dim")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Cache Entries", str(cache_stats.get('cache_size', 0)))
            table.add_row("Similarity Threshold", str(cache_stats.get('similarity_threshold', 0)))
            
            self.console.print(table)
            
            # Offer cache management options
            if cache_stats.get('cache_size', 0) > 0:
                if Confirm.ask("Would you like to clear the cache?"):
                    await self.chat_model.clear_cache()
                    self.console.print("[green]Cache cleared successfully.[/green]")
                    
        except Exception as e:
            logger.error(f"Error showing cache stats: {e}")
            self.console.print("[red]Error retrieving cache statistics.[/red]")
    
    async def _search_knowledge_base(self, query: str):
        """Search the knowledge base directly."""
        try:
            # Search vector store
            results = self.vector_store_manager.similarity_search(
                query, 
                k=self.max_context_results
            )
            
            if not results:
                self.console.print("[yellow]No results found in your knowledge base.[/yellow]")
                return
            
            # Display results
            self.console.print(f"\n[cyan]Search Results for: {query}[/cyan]\n")
            
            for i, result in enumerate(results, 1):
                title = result['metadata'].get('title', 'Unknown')
                file_path = result['metadata'].get('file_path', '')
                content = result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
                distance = result.get('distance', 0)
                
                panel_title = f"{i}. {title} (similarity: {1-distance:.3f})"
                
                self.console.print(
                    Panel(
                        content,
                        title=panel_title,
                        subtitle=file_path,
                        border_style="dim"
                    )
                )
                
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            self.console.print("[red]Error searching knowledge base.[/red]")
    
    async def _process_user_query(self, user_input: str):
        """Process a user query and generate a response."""
        try:
            # Dynamic context adjustment
            query_words = len(user_input.split())
            has_question = any(word in user_input.lower() for word in ['what', 'how', 'why', 'when', 'where', 'which', 'explain', 'describe', 'compare'])
            is_complex = any(word in user_input.lower() for word in ['analyze', 'relationship', 'connection', 'summary', 'overview'])

            if is_complex or query_words > 25:
                context_k = min(self.max_context_results, 10)
            elif has_question or query_words > 15:
                context_k = min(self.max_context_results, 8)
            else:
                context_k = self.base_context_results

            # Search for relevant context
            context_results = self.vector_store_manager.similarity_search(user_input, k=context_k)

            # Format context and filter by relevance
            context, relevant_docs = self._format_context_for_model(context_results)
            
            # Debug: Log the context being sent to LLM
            if context:
                logger.info(f"Context being sent to LLM (length: {len(context)})")
                logger.info(f"Context preview: {context[:300]}..." if len(context) > 300 else f"Full context: {context}")
            else:
                logger.info("No context being sent to LLM")

            # Get conversation history
            conversation_history = await self.memory_manager.get_recent_conversation_history(limit=5)

            print()
            print("💭 Thinking...")

            # Show referencing notes only if there are relevant documents
            if relevant_docs:
                file_info = []
                for result in relevant_docs[:3]:  # Show top 3
                    file_path = result['metadata'].get('file_path', '')
                    file_name = Path(file_path).stem if file_path else 'unknown'
                    relevance = result.get('relevance_score', 1 - result.get('distance', 0))
                    if file_name not in [info['name'] for info in file_info]:
                        file_info.append({'name': file_name, 'relevance': relevance})
                
                file_display = ', '.join([f"{info['name']} ({info['relevance']:.2f})" for info in file_info])
                print(f"📚 Referencing notes: {file_display}")
            else:
                # If no relevant documents are found, inform the user and stop.
                self.console.print("[yellow]I couldn't find any relevant notes to answer your question.[/yellow]")
                print()
                return

            # Show progress bars (simulated)
            print("Batches: 100%|████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  6.51it/s]")
            print("Batches: 100%|████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 89.03it/s]")

            response = await self.chat_model.generate_response(
                user_input,
                context=context,
                conversation_history=conversation_history,
                context_k=context_k
            )

            # Display response
            print()
            
            # Helper function for display width (same as above)
            def get_display_width(text):
                import unicodedata
                width = 0
                for char in text:
                    if unicodedata.east_asian_width(char) in ('F', 'W'):
                        width += 2
                    elif unicodedata.category(char) in ('Mn', 'Me'):
                        width += 0
                    elif ord(char) >= 0x1F600:
                        width += 2
                    else:
                        width += 1
                return width
            
            # Calculate dynamic width for response - use terminal width
            import os
            import shutil
            
            # Get terminal width, default to 120 if can't detect
            try:
                terminal_width = shutil.get_terminal_size().columns
            except:
                terminal_width = 120
            
            visible_label = "💡 Mnemosyne says:"
            
            # Calculate box width based on terminal width (leave some margin)
            box_width = min(terminal_width - 4, 200)  # Max 200 chars, min margin of 4
            
            # Don't wrap text - display as single line(s) as needed
            response_lines = response.split('\n')  # Only split on actual newlines from LLM
            
            # Top border with embedded colored label - entire box in cyan
            colored_label = f"💡 {Fore.CYAN}Mnemosyne{Style.RESET_ALL} says:"
            label_with_border = f"{Fore.CYAN}┌ {colored_label} "
            
            # Calculate remaining border width using display width
            visible_prefix_width = get_display_width(f"┌ {visible_label} ")
            remaining_border = box_width - visible_prefix_width - 1  # -1 for final ┐
            
            if remaining_border > 0:
                top_border = label_with_border + "─" * remaining_border + f"┐{Style.RESET_ALL}"
            else:
                top_border = f"{Fore.CYAN}┌ {colored_label} ┐{Style.RESET_ALL}"
            
            print(top_border)
            
            # Box content - each line can be as long as the terminal allows (in cyan)
            for line in response_lines:
                if not line.strip():  # Handle empty lines
                    print(f"{Fore.CYAN}│{' ' * (box_width - 2)}│{Style.RESET_ALL}")
                else:
                    # Truncate line if it's longer than box width allows
                    max_content_width = box_width - 4  # -4 for "| " and " |"
                    if len(line) > max_content_width:
                        display_line = line[:max_content_width - 3] + "..."
                    else:
                        display_line = line
                    
                    line_display_width = get_display_width(display_line)
                    padding_needed = box_width - 2 - line_display_width - 2  # -2 for borders, -2 for spaces
                    if padding_needed < 0:
                        padding_needed = 0
                    print(f"{Fore.CYAN}│ {display_line}{' ' * padding_needed} │{Style.RESET_ALL}")
            
            # Bottom border (in cyan)
            print(f"{Fore.CYAN}└{'─' * (box_width - 2)}┘{Style.RESET_ALL}")
            print()

            # Save to conversation history
            await self.memory_manager.add_conversation_entry(
                user_input,
                response,
                metadata={
                    "context_results": len(relevant_docs), # Use count of relevant docs
                    "context_k_used": context_k,
                    "model": self.chat_model.model_name,
                    "query_complexity": "complex" if is_complex else "question" if has_question else "simple",
                    "relevance_scores": [r.get('relevance_score', 0) for r in relevant_docs[:3]]
                }
            )

        except Exception as e:
            logger.error(f"Error processing user query: {e}")
            self.console.print(f"[red]I encountered an error while processing your request: {e}[/red]")

    def _format_context_for_model(self, results: List[Dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """Formats search results for the model and returns the context string and relevant documents."""
        if not results:
            return "", []

        context_parts = []
        relevant_docs = []
        for i, result in enumerate(results, 1):
            # Calculate relevance score - ChromaDB uses distance (lower = more similar)
            distance = result.get('distance', 1.0)
            relevance = max(0.0, 1.0 - distance)  # Convert distance to similarity
            
            # Use a lower threshold to be more inclusive (0.3 instead of 0.5)
            # Also ensure we include at least the top 3 results if any exist
            if relevance < 0.3 and i > 3:
                continue
            
            # Update the result with calculated relevance for display
            result['relevance_score'] = relevance
            relevant_docs.append(result)
            
            file_path = Path(result['metadata']['file_path'])
            tags = result['metadata'].get('tags', '[]')

            context_part = f"### Source {i}: {file_path.name} (Relevance: {relevance:.2f})\n"
            context_part += f"**Content:** {result['content']}\n"

            if tags and tags != '[]':
                context_part += f"**Tags:** {tags}\n"

            wiki_links = result['metadata'].get('wiki_links', '[]')
            if wiki_links and wiki_links != '[]':
                context_part += f"**Links:** {wiki_links}\n"

            context_parts.append(context_part)

        return "\n---\n".join(context_parts), relevant_docs

import re

def dim_sources_line(text):
    """Make any line starting with 'Sources:' dimmer using regex."""
    # Pattern matches any line starting with "Sources:" (with optional leading whitespace)
    pattern = r'^(\s*Sources:.*)'
    replacement = r'[dim]\1[/dim]'
    return re.sub(pattern, replacement, text, flags=re.MULTILINE)
