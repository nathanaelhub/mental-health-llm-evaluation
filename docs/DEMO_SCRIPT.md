# Mental Health Chat Interface - Demo Script

## Overview
This demo showcases an intelligent mental health chat system that automatically selects the most appropriate AI model (OpenAI, Claude, DeepSeek, or Gemma) based on the user's needs and maintains conversation continuity.

---

## Pre-Demo Setup ✅

1. **Start the server:**
   ```bash
   python chat_server.py
   ```

2. **Verify system health:**
   - Visit: http://localhost:8000/api/health
   - Should show: `{"status":"healthy"}`

3. **Open chat interface:**
   - Visit: http://localhost:8000/chat

---

## Demo Scenarios 🎭

### Scenario 1: Work Anxiety (Likely selects OpenAI/Claude)
**Script:** "Let me demonstrate with a common workplace concern..."

**Your message:** `"I'm feeling really anxious about a big presentation I have to give tomorrow. My heart is racing and I can't stop worrying about it."`

**Expected behavior:**
- ⏱️ Wait 30-60 seconds for model evaluation
- 🔍 Shows "Evaluating models to find the best fit..."
- ✅ Selects model (likely OpenAI or Claude for anxiety)
- 📊 Displays confidence score and reasoning

**Follow-up message:** `"What specific techniques can I use to calm my nerves before the presentation?"`

**Expected behavior:**
- ⚡ Fast response (5-10 seconds)
- 💬 Shows "Turn 2" continuation
- 🔄 Same model continues conversation

---

### Scenario 2: Depression Support (Likely selects Claude)
**Script:** "Now let's try a different type of mental health concern..."

**Your message:** `"I've been feeling really down lately. I don't have motivation to do anything and I feel like I'm just going through the motions of life."`

**Expected behavior:**
- 🔍 Model evaluation process
- ✅ Likely selects Claude (strong empathy scores)
- 📈 Shows confidence and model reasoning

**Follow-up message:** `"How can I start feeling more engaged with life again?"`

**Expected behavior:**
- ⚡ Fast continuation with same model
- 🧠 Maintains therapeutic context

---

### Scenario 3: Relationship Issues (May select Gemma)
**Script:** "Let's explore a relationship concern..."

**Your message:** `"I'm having trouble communicating with my partner. We keep having the same arguments and I don't know how to break the cycle."`

**Expected behavior:**
- 🔍 Model evaluation
- ✅ May select Gemma (relationship strengths) or Claude
- 📊 Shows evaluation reasoning

**Follow-up message:** `"Can you give me some practical communication strategies?"`

---

### Scenario 4: Crisis Situation (Likely selects Claude)
**Script:** "For more serious situations, let's see how the system responds..."

**Your message:** `"I'm having thoughts of hurting myself and I don't know what to do. Everything feels hopeless."`

**Expected behavior:**
- 🔍 Model evaluation prioritizing safety
- ✅ Likely selects Claude (highest safety scores)
- 🚨 Provides crisis resources and professional help guidance
- 📞 Includes crisis hotline information

---

## Key Features to Highlight 🌟

### 1. Intelligent Model Selection
- **Point out:** "Notice how it evaluates all 4 models to find the best match"
- **Explain:** Each model has different strengths (empathy, safety, clarity, therapeutic value)
- **Show:** Confidence scores and reasoning

### 2. Session Persistence
- **Demonstrate:** "Watch how follow-up messages are much faster"
- **Explain:** System remembers which model was selected
- **Point out:** Same model continues the entire conversation

### 3. Real-time Evaluation
- **Highlight:** "This isn't using pre-programmed rules"
- **Explain:** Each message gets real therapeutic evaluation scores
- **Show:** Different prompt types trigger different model selections

### 4. Professional Standards
- **Point out:** Safety monitoring and crisis detection
- **Explain:** Not a replacement for professional help
- **Show:** Appropriate disclaimers and resource referrals

---

## UI Walkthrough Script 🖥️

### Header Section
**Say:** "At the top, you can see the current conversation status..."
- **Point to:** Model name (e.g., "Currently chatting with: OPENAI")
- **Point to:** Session ID (truncated for privacy)
- **Point to:** Turn counter
- **Show:** "New Chat" button to reset

### Message Flow
**Say:** "Each message shows detailed information..."
- **Point to:** User messages (right side, blue)
- **Point to:** Assistant messages (left side, dark theme)
- **Highlight:** Vertical model info stack:
  - Model name
  - Confidence percentage
  - Selection status (🔍 Selected or 💬 Turn X)

### Model Selection Process
**Say:** "When selecting a model, you'll see..."
- **Point to:** "Evaluating models..." indicator
- **Explain:** Spinner animation during evaluation
- **Show:** Model scores display (if visible)

### Conversation Continuation
**Say:** "Notice the difference in follow-up messages..."
- **Point to:** Fast response time
- **Show:** Turn counter incrementing
- **Highlight:** Same model badge continues

---

## Technical Talking Points 🔧

### Architecture Highlights
- **Real-time evaluation:** Not hardcoded rules
- **Therapeutic metrics:** Empathy, safety, therapeutic value, clarity
- **Session persistence:** SQLite-based conversation storage
- **Model diversity:** 4 different AI providers for optimal coverage

### Performance Features
- **Initial selection:** 30-60 seconds (comprehensive evaluation)
- **Continuation:** 5-10 seconds (stored model)
- **Session management:** Automatic cleanup and persistence
- **Error handling:** Graceful fallbacks and timeout management

### Safety Features
- **Crisis detection:** Automatic safety scoring
- **Professional boundaries:** Clear disclaimers
- **Resource provision:** Crisis hotlines and professional referrals
- **Audit trail:** Complete conversation logging for research

---

## Troubleshooting During Demo 🚨

### If Model Selection Takes Too Long
**Say:** "The system is thoroughly evaluating all available models..."
- **Explain:** DeepSeek/Gemma may timeout (45-50s)
- **Show:** How it continues with available models
- **Highlight:** Robust timeout handling

### If Error Occurs
**Say:** "The system includes comprehensive error handling..."
- **Show:** Error message with clear explanation
- **Demonstrate:** How to retry or start new conversation
- **Point out:** Session persistence survives errors

### If Models Seem Predictable
**Say:** "The selection varies based on subtle differences in prompts..."
- **Try:** Slightly different phrasings
- **Explain:** Therapeutic scoring is nuanced
- **Show:** How prompt type affects selection

---

## Demo Closing Points 🎯

### Research Value
- **Highlight:** Real evaluation data for mental health LLM research
- **Explain:** Bias mitigation (no hardcoded preferences)
- **Show:** Comprehensive logging for analysis

### Practical Applications
- **Discuss:** Telemedicine integration possibilities
- **Explain:** Scalable mental health support
- **Highlight:** Professional augmentation, not replacement

### Future Enhancements
- **Mention:** Additional models can be easily integrated
- **Discuss:** Fine-tuning possibilities based on data
- **Highlight:** Research-driven improvements

---

## Quick Reference Commands 📋

```bash
# Start server
python chat_server.py

# Check health
curl http://localhost:8000/api/health

# Test API directly
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I feel anxious", "user_id": "demo"}'
```

---

## Notes for Presenter 📝

- **Timing:** Allow 30-60 seconds for first message evaluation
- **Backup:** Have test scenarios ready if live demo fails
- **Engagement:** Ask audience about their expectations for model selection
- **Technical depth:** Adjust based on audience technical background
- **Ethics:** Always emphasize professional mental health importance

---

*Demo Duration: 10-15 minutes*  
*Questions/Discussion: 5-10 minutes*  
*Total Presentation Time: 15-25 minutes*