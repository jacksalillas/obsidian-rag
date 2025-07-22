#!/usr/bin/env python3

try:
    from colorama import Back, Style, init
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    # Fallback if colorama is not installed
    class Back:
        CYAN = ''
    class Style:
        RESET_ALL = ''
    COLORS_AVAILABLE = False

def test_response_box():
    """Test the new cyan response box design with dynamic sizing"""
    
    # Sample response text
    response = "Based on your DevOps Engineer vault, I can see you've been working on container orchestration with Kubernetes. Your notes mention setting up monitoring with Prometheus and Grafana, which is a solid observability stack. For your current project, I'd recommend implementing proper resource limits and requests to ensure stable performance across your cluster environments."
    
    import textwrap
    label = "💡 Mnemosyne says:"
    
    # First, wrap text to a reasonable max width (say 80 chars)
    wrapped_lines = textwrap.wrap(response, width=76)
    
    # Calculate the actual width needed
    if wrapped_lines:
        max_content_width = max(len(line) for line in wrapped_lines)
    else:
        max_content_width = 0
        
    # Box width should fit the content + label + padding
    box_width = max(max_content_width + 4, len(label) + 6)  # +6 for "┌ " and " ┐"
    
    # Re-wrap text to fit the calculated box width
    wrapped_lines = textwrap.wrap(response, width=box_width - 4)

    # Top border with embedded label
    label_prefix = f"┌{label} "
    remaining_width = box_width - len(label_prefix) - 1  # -1 for final ┐
    top_border = label_prefix + "─" * remaining_width + "┐"
    print(Back.CYAN + top_border)

    # Box content
    for line in wrapped_lines:
        print(Back.CYAN + f"│ {line.ljust(box_width - 2)} │")

    # Bottom border
    bottom_border = "└" + "─" * (box_width - 2) + "┘"
    print(Back.CYAN + bottom_border)
    print()

if __name__ == "__main__":
    test_response_box()
