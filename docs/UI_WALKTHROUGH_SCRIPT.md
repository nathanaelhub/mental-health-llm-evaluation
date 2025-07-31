# Chat Interface UI Walkthrough Script

## Opening the Interface 🌐

**Action:** Navigate to `http://localhost:8000/chat`

**Say:** "Welcome to our Mental Health Chat Interface. This is a research-grade system that intelligently selects the most appropriate AI model for mental health conversations."

---

## Interface Layout Overview 📱

### Header Section (Top)
**Point to each element and explain:**

1. **Title Area (Left)**
   - "Mental Health Support" with heart icon
   - Subtitle: "Intelligent LLM-powered mental health assistance"

2. **Model Status Area (Right)**
   - **Current Model Display:** "Currently chatting with: Selecting best model..."
   - **Confidence Display:** (Hidden initially, shows after selection)
   - **Session ID:** Shows truncated session identifier
   - **Turn Counter:** "Turn: 0" (increments with each exchange)

3. **Control Buttons**
   - **"New Chat" button:** Resets conversation and triggers new model selection
   - **Model switch button:** (Advanced feature for demonstration)

**Say:** "The header gives you real-time information about which AI model you're talking to and the conversation status."

---

## Welcome Message Section 💬

**Point to the center welcome message**

**Read key points:**
- "Our intelligent system will automatically select the best AI model"
- "All 4 models (OpenAI, Claude, DeepSeek, Gemma) are evaluated"
- "Subsequent messages continue with the selected model"
- "Important disclaimer about professional mental health services"

**Say:** "This explains how the system works and sets appropriate expectations about professional mental health care."

---

## Message Input Area (Bottom) ✍️

**Point to the input section:**

1. **Text Input Box**
   - Auto-resizing based on content
   - Placeholder: "Type your message here..."

2. **Send Button**
   - Blue button with send icon
   - Disabled during message processing

3. **Options (if visible)**
   - Streaming toggle
   - Cache usage toggle

**Say:** "The interface is clean and focused on the conversation, just like professional chat applications."

---

## Demonstration Flow 🎬

### Step 1: First Message Entry
**Action:** Click in the text box

**Say:** "Let me demonstrate with a typical mental health concern..."

**Type:** `"I'm feeling really anxious about my upcoming job interview tomorrow. I keep imagining all the things that could go wrong."`

**Before sending, point out:**
- Text auto-resizing in the input box
- Send button is active and ready

### Step 2: Model Selection Process
**Action:** Click Send

**Immediately point to:**
1. **Welcome message disappears** (interface focuses on conversation)
2. **"Evaluating models..." indicator appears** with spinner animation
3. **Header updates:** "Currently chatting with: Selecting best model..."
4. **Input is disabled** during processing

**Say:** "Now the system is evaluating all 4 AI models against this specific anxiety-related prompt. This takes 30-60 seconds because we're doing real-time evaluation, not using pre-programmed rules."

**While waiting, explain:**
- "Each model gets a therapeutic score based on empathy, safety, therapeutic value, and clarity"
- "The system analyzes the prompt type (anxiety, depression, crisis, etc.)"
- "Some models may timeout (DeepSeek/Gemma), but evaluation continues"

### Step 3: Model Selection Result
**When response appears, point to:**

1. **Message appears in conversation**
   - Left side (assistant message)
   - Robot avatar
   - Dark theme bubble with response

2. **Model information (vertical stack):**
   - **Model Badge:** "OPENAI" (or whichever was selected)
   - **Confidence Score:** "85.5% confidence" (example)
   - **Selection Status:** "🔍 Selected"
   - **Timestamp:** Current time

3. **Header updates:**
   - "Currently chatting with: OPENAI"
   - Confidence display becomes visible
   - Turn counter: "Turn: 1"

**Say:** "Great! The system selected OpenAI with 85.5% confidence for this anxiety-related prompt. Notice how all the information is clearly displayed in a vertical stack that's easy to read."

### Step 4: Conversation Continuation
**Type follow-up message:** `"What specific breathing techniques work best for interview anxiety?"`

**Before sending, say:** "Now watch how fast the continuation is..."

**Action:** Click Send

**Point out immediately:**
1. **Much faster response** (5-10 seconds vs 30-60)
2. **No model evaluation indicator** (goes straight to typing indicator)
3. **Same model continues** the conversation

**When response appears:**
1. **Same model badge:** "OPENAI"
2. **Same confidence:** "85.5% confidence"
3. **Status change:** "💬 Turn 2" (instead of "🔍 Selected")
4. **Turn counter updates:** "Turn: 2"

**Say:** "Notice how much faster that was! The system remembered OpenAI was selected and continued the conversation without re-evaluation."

---

## Advanced Features Demonstration 🔧

### New Conversation Reset
**Action:** Click "New Chat" button

**Point out:**
1. **Conversation history clears**
2. **Welcome message returns**
3. **Header resets:** "Selecting best model..."
4. **Turn counter:** "Turn: 0"
5. **Session ID changes**

**Say:** "The 'New Chat' button starts fresh, which will trigger new model selection for different types of concerns."

### Different Prompt Types
**Try a different scenario:** `"I've been feeling really depressed and unmotivated lately. Nothing seems to bring me joy anymore."`

**Say:** "Let's see if a depression-focused prompt selects a different model..."

**Point out during evaluation:**
- Same evaluation process
- May select Claude (known for empathy)
- Different confidence score reflects prompt type

---

## UI Design Philosophy 🎨

### Dark Theme Choice
**Say:** "The dark theme was chosen for mental health applications because:"
- Less eye strain during emotional conversations
- Calming, professional appearance
- Better focus on conversation content

### Vertical Information Stack
**Point to model info:** "The vertical layout for model information:"
- Cleaner than horizontal badges
- Better information hierarchy
- More space-efficient
- Easier to scan

### Conversation Bubbles
**Point to message layout:** "Professional chat interface:"
- Clear speaker identification (user vs assistant)
- Consistent with modern messaging apps
- Appropriate spacing and typography
- Timestamps for reference

---

## Error Handling Demonstration 🚨

### Timeout Scenario
**If models timeout during demo:**

**Say:** "You might notice some timeout warnings - this is normal:"
- DeepSeek and Gemma are local models that may timeout
- System continues with available models (OpenAI, Claude)
- Robust error handling ensures conversation continues

### Network Issues
**If API errors occur:**

**Say:** "The system includes comprehensive error handling:"
- Clear error messages to user
- Option to retry
- Session persistence survives errors
- Graceful degradation

---

## Behind-the-Scenes Information 🔍

### What's Happening During Model Selection
**Technical explanation:**
1. **Prompt Analysis:** System classifies the prompt type (anxiety, depression, crisis, etc.)
2. **Parallel Evaluation:** All 4 models generate responses simultaneously
3. **Therapeutic Scoring:** Each response gets scored on 4 dimensions
4. **Selection Logic:** Highest composite score wins
5. **Session Creation:** Winner is stored for conversation continuity

### Why Different Models Matter
**Explain model strengths:**
- **OpenAI:** Balanced, good for general anxiety and information
- **Claude:** Exceptional empathy and crisis handling
- **DeepSeek:** Analytical approach, good for problem-solving
- **Gemma:** Warm communication style, good for relationships

### Session Management
**Technical details:**
- **SQLite Storage:** Conversations persist across server restarts
- **Session IDs:** Unique identifiers for each conversation
- **Metadata Tracking:** Model selection reasoning and scores stored
- **Privacy:** Sessions are user-controlled and can be reset

---

## Closing the Walkthrough 🎯

### Key Takeaways
**Summarize for audience:**
1. **Intelligent Selection:** Real-time evaluation, not hardcoded rules
2. **Conversation Continuity:** Fast follow-ups with selected model
3. **Professional Interface:** Clean, focused, appropriate for sensitive conversations
4. **Research Grade:** Comprehensive logging and evaluation for research purposes

### Research Applications
**Discuss potential uses:**
- **Bias Research:** No hardcoded preferences enable fair comparison
- **Therapeutic Effectiveness:** Real scoring metrics for analysis
- **Model Improvement:** Data-driven insights for better mental health AI
- **Clinical Integration:** Framework for professional mental health tools

### Ethical Considerations
**Always emphasize:**
- **Not a replacement** for professional mental health care
- **Crisis situations** should involve human professionals
- **Privacy and security** considerations for real deployments
- **Research purposes** of current implementation

---

## Q&A Preparation 💭

### Common Questions and Answers

**Q: "How accurate is the model selection?"**  
**A:** "The system uses research-based therapeutic evaluation metrics. We've removed all hardcoded biases that were present in earlier versions, so selection is based purely on response quality."

**Q: "What happens in a crisis situation?"**  
**A:** "The system prioritizes safety scoring and typically selects Claude, which has the highest safety scores. It also provides crisis resources and emphasizes professional help."

**Q: "Can users choose their preferred model?"**  
**A:** "For research purposes, we focus on automatic selection, but the architecture supports user choice. The goal is to find the objectively best model for each situation."

**Q: "How do you prevent bias toward certain models?"**  
**A:** "This was a major focus of our research. We removed hardcoded preferences and use transparent scoring metrics. All models compete fairly based on therapeutic effectiveness."

---

*Walkthrough Duration: 15-20 minutes*  
*Includes live demonstration and explanation*  
*Adaptable to audience technical level*