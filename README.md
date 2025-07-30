# Mental Health LLM Evaluation

**Comparing local vs cloud-based LLMs for mental health telemedicine applications with intelligent model selection and conversation flow.**

## 🎯 **Project Overview**

This system evaluates 4 AI models (OpenAI, Claude, DeepSeek, Gemma) for mental health support through dynamic model selection and comprehensive conversation interfaces.

**Key Features:**
- **Dynamic Model Selection:** First message intelligently selects best AI model
- **Conversation Continuity:** Follow-up messages use selected model consistently
- **Professional Chat Interface:** Dark theme, session management, conversation history
- **Research Framework:** Complete evaluation system for model comparison

---

## 🚀 **Quick Start**

### **1. Start the Chat Server**
```bash
python chat_server.py
```

### **2. Access the Interface**
```bash
# Open browser to:
http://localhost:8000/chat
```

### **3. Test Conversation Flow**
1. First message: "I'm feeling anxious" → Triggers model selection (3-5 seconds)
2. Follow-up: "What can help?" → Uses selected model (1-2 seconds)
3. New conversation: Click "✨ New Chat" → Fresh model selection

### **4. Verify System**
```bash
# Check server status
curl http://localhost:8000/api/status

# Run system verification
python scripts/chat_server_development/verify_system.py
```

---

## 📚 **Documentation**

**Complete documentation available in [`docs/`](docs/README.md)**

### **Quick Links:**
- **[Quick Start Guide](docs/guides/QUICK_START.md)** - Detailed setup instructions
- **[UI Development Phase](docs/phases/UI_Development_Phase_Summary.md)** - Complete development history
- **[Troubleshooting](docs/debugging/CHAT_INTERFACE_DEBUG_GUIDE.md)** - Debug procedures and common issues
- **[Chat Server Fixes](docs/CHAT_SERVER_FIXES_SUMMARY.md)** - Technical implementation details

---

## 🏗️ **System Architecture**

```
mental-health-llm-evaluation/
├── chat_server.py              # Main production server
├── src/                        # Source code modules
│   ├── chat/                   # Conversation logic & session management
│   ├── models/                 # AI model clients (OpenAI, Claude, DeepSeek, Gemma)
│   └── ui/                     # Frontend templates and assets
├── scripts/chat_server_development/ # Development & testing files
├── config/                     # Configuration files
├── data/                       # Input scenarios
├── results/                    # Evaluation outputs
└── docs/                       # Comprehensive documentation
```

---

## 🔧 **Development**

### **Alternative Servers:**
```bash
# HTTP-only version
python scripts/chat_server_development/simple_chat_server.py

# WebSocket-enabled version
python scripts/chat_server_development/working_chat_server.py
```

### **Testing:**
```bash
# Test conversation flow
python scripts/chat_server_development/test_chat_server_flow.py

# Run research evaluation
python scripts/run_research.py
```

---

## 📊 **Research Results**

**Winner: DeepSeek R1 (Local Model)**
- Superior therapeutic performance
- Zero per-request costs
- Complete privacy (no external data transmission)
- Equivalent safety performance

**Statistical Validation:** All results significant (p < 0.05) with large effect sizes.

*Full research findings and methodology available in [`results/`](results/) directory.*

---

## 📞 **Support**

1. **Check [docs/README.md](docs/README.md)** for comprehensive documentation
2. **Run debug tools:** `python scripts/chat_server_development/verify_system.py`
3. **API status:** http://localhost:8000/api/status (when server running)

**Ready to use! Start with the Quick Start steps above.**