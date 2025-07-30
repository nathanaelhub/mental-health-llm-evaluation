# 🚀 Quick Start Guide

## Prerequisites

- Python 3.8+ installed
- 2GB free disk space
- API keys for OpenAI and/or Anthropic (optional)

## 1. Installation (2 minutes)

```bash
# Clone the repository (if not already done)
git clone https://github.com/your-repo/mental-health-llm-evaluation.git
cd mental-health-llm-evaluation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_chat.txt
pip install -r requirements_api.txt
```

## 2. Configuration (1 minute)

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add API keys (optional)
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
```

## 3. Start the Chat Server (30 seconds)

```bash
# Option A: Full chat server with UI (recommended)
python chat_server.py

# Option B: Simple test server
python simple_server.py
```

## 4. Access the Interface

Open your browser and navigate to:
- **Full UI**: http://localhost:8000/chat
- **Simple UI**: http://localhost:8000

## 5. Test the System

### First Message (Tests Model Selection)
Type: `I'm feeling anxious about work`

**Expected behavior:**
- Shows "Selecting best AI model..." (3-5 seconds)
- Displays selected model with confidence score
- Shows AI response

### Follow-up Message (Tests Continuation)
Type: `What techniques can help me?`

**Expected behavior:**
- Uses same model (no selection delay)
- Shows model name while responding
- Maintains conversation context

### New Chat
Click the "New Chat" button to start fresh.

## Common Issues & Solutions

### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use different port
python chat_server.py --port 8001
```

### Missing Dependencies
```bash
# Reinstall all dependencies
pip install --force-reinstall -r requirements.txt
```

### No Models Available
If all models timeout, check:
1. API keys in `.env` file
2. Internet connection for cloud APIs
3. Local model servers if using DeepSeek/Gemma

### Chat Not Responding
The first message may take 15-20 seconds due to model selection. This is normal.

## What's Next?

- **Configure Models**: Add API keys for more model options
- **Run Tests**: `python scripts/test_chat_api.py`
- **Check Health**: `python scripts/test_health_check.py`
- **Read Docs**: See `docs/TECHNICAL_REFERENCE.md` for details

## Quick Commands Reference

```bash
# Start chat server
python chat_server.py

# Test API
python scripts/test_chat_api.py

# Check model connectivity
python scripts/chat_server_development/test_local_models.py

# View logs
tail -f server_debug.log
```

---

🎉 **That's it!** Your Mental Health Chat System is ready. Type a message and see intelligent model selection in action.