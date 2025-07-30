# Chat Server Development Files

This directory contains all development, testing, and alternative implementations for the chat server system.

## 📁 **File Organization**

### **Alternative Server Implementations:**
- `simple_chat_server.py` - Minimal HTTP-only server
- `working_chat_server.py` - WebSocket-enabled server with fixes
- `start_server.py` - Development server launcher

### **Test Scripts:**
- `test_chat_server_flow.py` - Tests conversation flow (selection → continuation)
- `test_conversation_flow.py` - Backend API testing
- `test_chat_system.py` - Complete system integration test
- `test_dynamic_selector.py` - Model selection testing
- `test_session_management.py` - Session persistence testing
- `test_websocket_flow.py` - WebSocket communication testing
- `test_working_server.py` - Server validation script

### **Simple Chat Interface:**
- `simple_chat.html` - Basic HTTP-only chat interface
- `simple_chat.js` - Pure JavaScript (no WebSocket) chat logic
- `test_chat_interface.html` - Test interface for debugging

### **System Verification:**
- `verify_system.py` - Complete system health check
- `verify_system.sh` - Shell script for system verification

## 🚀 **Usage**

### **Main Production Server:**
Use the main `chat_server.py` in the root directory:
```bash
# From project root
python chat_server.py
```

### **Development/Alternative Servers:**
```bash
# From project root
python scripts/chat_server_development/simple_chat_server.py  # HTTP-only
python scripts/chat_server_development/working_chat_server.py  # WebSocket
python scripts/chat_server_development/start_server.py        # Dev launcher
```

### **Testing:**
```bash
# From project root
python scripts/chat_server_development/test_chat_server_flow.py
python scripts/chat_server_development/verify_system.py
```

## 📋 **Development Notes**

- All files moved here from root directory for better organization
- Main production server (`chat_server.py`) remains in root
- These files are for development, testing, and alternative implementations
- Use main `chat_server.py` for actual deployment

## 🔧 **File History**

These files were moved from the main directory to improve organization:
- Keeps main directory clean and focused
- Groups all development/testing files together
- Maintains backup implementations for reference
- Preserves all test scripts for future development