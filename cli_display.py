
try:
    from colorama import Fore, Style, init
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
    COLORS_AVAILABLE = False

def display_cli_aesthetic(vault_names=None, memory_count=5):
    """
    Display the Mnemosyne CLI banner with dynamic content.
    
    Args:
        vault_names: List of vault names to display. If None, shows example vaults.
        memory_count: Number of saved memories to display.
    """
    title = "Mnemosyne"

    # Use provided vault names or default examples
    if vault_names is None:
        vault_names = ["PersonalNotes", "WorkProjects", "Research"]
    
    # Prepare all content lines
    sources_text = f"Sources: {', '.join(vault_names) if vault_names else 'None'}"
    memories_text = f"Personal Memories: {memory_count} saved"
    
    content_lines = [
        sources_text,
        memories_text,
        "✨ I can suggest file modifications - I'll ask for your permission first",
        "📝 I remember our conversation context for follow-up questions",
        "🧠 I remember personal details you teach me permanently",
        "",  # Empty line
        "Memory commands: 'remember: ', 'show my memories'",
        "System commands: 'status', 'reindex', 'bye', 'quit', 'exit', 'q'"
    ]
    
    # Calculate the maximum width needed, accounting for colors
    # We need to calculate length without color codes for proper alignment
    max_content_width = 0
    for line in content_lines:
        if line:  # Skip empty lines
            max_content_width = max(max_content_width, len(line))
    
    box_width = max(max_content_width + 4, len(title) + 6)  # +6 for padding and spaces

    # Top blue line
    print(Fore.BLUE + "🧠 Mnemosyne - Your Personal AI Assistant")

    # Top border with centered title (with spaces)
    title_with_spaces = f" {title} "
    print(f"┌{title_with_spaces.center(box_width - 2, '─')}┐")

    # Content lines
    for i, line in enumerate(content_lines):
        if i == 0:  # Sources line - color the vault names in cyan
            vault_names_str = ', '.join(vault_names) if vault_names else 'None'
            colored_line = f"Sources: {Fore.CYAN}{vault_names_str}{Style.RESET_ALL}"
            # Calculate padding based on visible text length
            visible_length = len(f"Sources: {vault_names_str}")
            padding = " " * (box_width - visible_length - 3)
            print(f"│ {colored_line}{padding}│")
        elif i == 1:  # Personal Memories line - color the number in magenta
            colored_line = f"Personal Memories: {Fore.MAGENTA}{memory_count}{Style.RESET_ALL} saved"
            # Calculate padding based on visible text length
            visible_length = len(f"Personal Memories: {memory_count} saved")
            padding = " " * (box_width - visible_length - 3)
            print(f"│ {colored_line}{padding}│")
        elif line == "":  # Empty line
            print("│" + " " * (box_width - 2) + "│")
        elif i in [2, 3, 4]:  # Feature lines that might overflow
            # Truncate if too long
            if len(line) > box_width - 4:
                truncated_line = line[:box_width - 7] + "..."
                padding = " " * (box_width - len(truncated_line) - 3)
                print(f"│ {truncated_line}{padding}│")
            else:
                padding = " " * (box_width - len(line) - 3)
                print(f"│ {line}{padding}│")
        else:
            padding = " " * (box_width - len(line) - 3)
            print(f"│ {line}{padding}│")
    
    # Bottom border
    print("└" + "─" * (box_width - 2) + "┘")

def display_chat_flow_example():
    print("\nAsk me anything about your vaults: ")
    print(" Thinking...")
    print(" Referencing notes: Acronyms")
    print(" . . . ") # Simple visual batch progress example

def display_banner_with_real_vaults():
    """Display banner with real vault names from config"""
    try:
        from config import load_config
        from pathlib import Path
        
        config = load_config()
        
        # Extract vault names from paths
        vault_names = []
        for vault_path in config.vaults:
            vault_names.append(Path(vault_path).name)
        
        # Add text sources
        for text_source in config.text_sources:
            vault_names.append(Path(text_source).name)
        
        # Mock memory count - in real app this would come from MemoryManager
        memory_count = 7  # This would be dynamic in the real app
        
        display_cli_aesthetic(vault_names, memory_count)
        
    except Exception as e:
        print(f"Error loading config: {e}")
        # Fallback to example display
        display_cli_aesthetic()

if __name__ == "__main__":
    display_banner_with_real_vaults()
    display_chat_flow_example()
