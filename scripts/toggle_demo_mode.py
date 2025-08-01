#!/usr/bin/env python3
"""
Demo Mode Toggle Script
======================

Easily enable/disable demo mode in chat_server.py for presentations.
Demo mode provides extended timeouts to ensure successful completion.
"""

import re
import sys
from pathlib import Path

def toggle_demo_mode(enable=None):
    """Toggle demo mode in chat_server.py"""
    
    chat_server_path = Path("chat_server.py")
    
    if not chat_server_path.exists():
        print("❌ chat_server.py not found!")
        return False
    
    # Read current content
    with open(chat_server_path, 'r') as f:
        content = f.read()
    
    # Find current demo mode setting
    demo_pattern = r'DEMO_MODE = (True|False)'
    match = re.search(demo_pattern, content)
    
    if not match:
        print("❌ DEMO_MODE setting not found in chat_server.py")
        return False
    
    current_mode = match.group(1) == 'True'
    
    if enable is None:
        # Toggle mode
        new_mode = not current_mode
    else:
        # Set specific mode
        new_mode = enable
    
    # Update content
    new_content = re.sub(
        demo_pattern, 
        f'DEMO_MODE = {new_mode}', 
        content
    )
    
    # Write back
    with open(chat_server_path, 'w') as f:
        f.write(new_content)
    
    print(f"🔄 Demo mode changed: {current_mode} → {new_mode}")
    
    if new_mode:
        print("🎭 DEMO MODE ENABLED")
        print("   ✅ Extended timeouts (2+ minutes)")
        print("   ✅ Prioritizes completion over speed")
        print("   ✅ Ideal for presentations")
        print("   💡 Restart chat_server.py to apply changes")
    else:
        print("⚡ DEMO MODE DISABLED")  
        print("   ✅ Standard timeouts (~45-50s)")
        print("   ✅ Balanced speed/reliability")
        print("   ✅ Ideal for development")
        print("   💡 Restart chat_server.py to apply changes")
    
    return True

def main():
    """Main function"""
    
    print("🎭 Demo Mode Toggle Utility")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['on', 'enable', 'true', '1']:
            toggle_demo_mode(True)
        elif arg in ['off', 'disable', 'false', '0']:
            toggle_demo_mode(False)
        else:
            print("❌ Invalid argument. Use: on/off, enable/disable, true/false")
    else:
        # Interactive toggle
        toggle_demo_mode()

if __name__ == "__main__":
    main()