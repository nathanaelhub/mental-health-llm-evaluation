/**
 * Simple Mental Health Chat Interface
 * 
 * Features:
 * - First message triggers model selection
 * - All subsequent messages go to selected model
 * - Clear visual indication of selected model
 * - Working "New Chat" button
 * - No page-breaking buttons - pure JavaScript
 */

class SimpleMentalHealthChat {
    constructor() {
        // Chat state
        this.currentSessionId = null;
        this.selectedModel = null;
        this.conversationMode = 'initial'; // 'initial' or 'continued'
        this.turnCount = 0;
        this.isProcessing = false;
        
        // API configuration
        this.apiBaseUrl = 'http://localhost:8000';
        
        // Initialize
        this.initializeElements();
        this.bindEvents();
        
        console.log('Simple Mental Health Chat initialized');
    }
    
    initializeElements() {
        // Core elements
        this.messagesContainer = document.getElementById('messages-container');
        this.messageInput = document.getElementById('message-input');
        this.sendButton = document.getElementById('send-button');
        this.newChatBtn = document.getElementById('new-chat-btn');
        this.typingIndicator = document.getElementById('typing-indicator');
        this.typingText = document.getElementById('typing-text');
        this.welcomeMessage = document.getElementById('welcome-message');
        
        // Header elements
        this.modelLabel = document.getElementById('model-label');
        this.modelName = document.getElementById('model-name');
    }
    
    bindEvents() {
        // Send button
        this.sendButton.addEventListener('click', (e) => {
            e.preventDefault();
            this.sendMessage();
        });
        
        // Enter key to send (Shift+Enter for new line)
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => {
            this.autoResizeTextarea();
        });
        
        // New chat button
        this.newChatBtn.addEventListener('click', (e) => {
            e.preventDefault();
            this.startNewChat();
        });
    }
    
    autoResizeTextarea() {
        const textarea = this.messageInput;
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isProcessing) return;
        
        // Set processing state
        this.isProcessing = true;
        this.setInputState(false);
        
        // Clear input
        this.messageInput.value = '';
        this.autoResizeTextarea();
        
        // Hide welcome message on first user message
        if (this.conversationMode === 'initial') {
            this.welcomeMessage.style.display = 'none';
        }
        
        // Add user message to chat
        this.addMessage(message, 'user');
        
        try {
            // Show typing indicator with appropriate text
            if (this.conversationMode === 'initial' || !this.selectedModel) {
                this.typingText.textContent = 'Selecting best AI model for your needs...';
            } else {
                this.typingText.textContent = `${this.selectedModel.toUpperCase()} is responding...`;
            }
            this.showTypingIndicator();
            
            // Send to API
            const response = await this.callChatAPI(message);
            
            // Hide typing indicator
            this.hideTypingIndicator();
            
            // Process response
            if (response) {
                this.handleChatResponse(response);
            } else {
                this.addStatusMessage('Failed to get response from server. Please try again.', 'error');
            }
            
        } catch (error) {
            this.hideTypingIndicator();
            console.error('Chat error:', error);
            this.addStatusMessage('Error connecting to chat service. Please check if the server is running.', 'error');
        } finally {
            this.isProcessing = false;
            this.setInputState(true);
            this.messageInput.focus();
        }
    }
    
    async callChatAPI(message) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    session_id: this.currentSessionId,
                    user_id: 'simple-chat-user',
                    force_reselection: false
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
            
        } catch (error) {
            console.error('API call failed:', error);
            return null;
        }
    }
    
    handleChatResponse(data) {
        // Update session state
        this.currentSessionId = data.session_id;
        this.selectedModel = data.selected_model;
        this.conversationMode = data.conversation_mode;
        this.turnCount = data.turn_number || data.turn_count; // Handle both field names
        
        // Update UI header
        this.updateModelStatus(data);
        
        // Add assistant response
        this.addMessage(data.response, 'assistant', {
            model: data.selected_model,
            conversationMode: data.conversation_mode,
            confidence: data.confidence_score,
            turnNumber: data.turn_number
        });
        
        // Show status message for model selection
        if (data.conversation_mode === 'selection') {
            const confidence = Math.round(data.confidence_score * 100);
            this.addStatusMessage(`🎯 Selected ${data.selected_model.toUpperCase()} (${confidence}% confidence)`, 'success');
        }
    }
    
    updateModelStatus(data) {
        if (data.conversation_mode === 'selection') {
            this.modelLabel.textContent = 'Chatting with';
            this.modelName.textContent = `${data.selected_model.toUpperCase()} (Selected)`;
        } else if (data.conversation_mode === 'continuation') {
            this.modelLabel.textContent = 'Chatting with';
            this.modelName.textContent = `${data.selected_model.toUpperCase()} • Turn ${data.turn_number || data.turn_count}`;
        } else {
            this.modelLabel.textContent = 'Chatting with';
            this.modelName.textContent = `${data.selected_model.toUpperCase()}`;
        }
    }
    
    addMessage(content, sender, metadata = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'message-bubble';
        bubbleDiv.innerHTML = this.formatMessageContent(content);
        
        messageDiv.appendChild(bubbleDiv);
        
        // Add metadata for assistant messages
        if (sender === 'assistant' && metadata.model) {
            const metaDiv = document.createElement('div');
            metaDiv.className = 'message-meta';
            
            const timestamp = new Date().toLocaleTimeString([], { 
                hour: '2-digit', 
                minute: '2-digit' 
            });
            
            let metaHTML = `<span>${timestamp}</span>`;
            metaHTML += `<span class="model-badge">${metadata.model.toUpperCase()}</span>`;
            
            if (metadata.conversationMode === 'selection') {
                metaHTML += `<span class="model-badge">🎯 Selected</span>`;
            } else if (metadata.conversationMode === 'continuation') {
                metaHTML += `<span class="model-badge">💬 Turn ${metadata.turnNumber || ''}</span>`;
            }
            
            metaDiv.innerHTML = metaHTML;
            messageDiv.appendChild(metaDiv);
        }
        
        // Add timestamp for user messages
        if (sender === 'user') {
            const metaDiv = document.createElement('div');
            metaDiv.className = 'message-meta';
            const timestamp = new Date().toLocaleTimeString([], { 
                hour: '2-digit', 
                minute: '2-digit' 
            });
            metaDiv.innerHTML = `<span>${timestamp}</span>`;
            messageDiv.appendChild(metaDiv);
        }
        
        // Insert before typing indicator
        this.messagesContainer.insertBefore(messageDiv, this.typingIndicator);
        this.scrollToBottom();
    }
    
    addStatusMessage(message, type = 'info') {
        const statusDiv = document.createElement('div');
        statusDiv.className = `status-message ${type}`;
        statusDiv.textContent = message;
        
        // Insert before typing indicator
        this.messagesContainer.insertBefore(statusDiv, this.typingIndicator);
        this.scrollToBottom();
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (statusDiv.parentNode) {
                statusDiv.remove();
            }
        }, 5000);
    }
    
    formatMessageContent(content) {
        // Simple formatting
        return content
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/(https?:\\/\\/[^\\s]+)/g, '<a href=\"$1\" target=\"_blank\" style=\"color: #06b6d4;\">$1</a>');
    }
    
    showTypingIndicator() {
        this.typingIndicator.style.display = 'flex';
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        this.typingIndicator.style.display = 'none';
    }
    
    setInputState(enabled) {
        this.messageInput.disabled = !enabled;
        this.sendButton.disabled = !enabled;
        
        if (enabled) {
            this.messageInput.focus();
        }
    }
    
    scrollToBottom() {
        // Smooth scroll to bottom
        setTimeout(() => {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        }, 100);
    }
    
    startNewChat() {
        // Reset chat state
        this.currentSessionId = null;
        this.selectedModel = null;
        this.conversationMode = 'initial';
        this.turnCount = 0;
        
        // Clear messages (keep welcome message and typing indicator)
        const messages = this.messagesContainer.querySelectorAll('.message, .status-message');
        messages.forEach(message => message.remove());
        
        // Show welcome message
        this.welcomeMessage.style.display = 'block';
        
        // Reset header
        this.modelLabel.textContent = 'Ready to chat';
        this.modelName.textContent = 'Select a model by sending a message';
        
        // Focus input
        this.messageInput.focus();
        
        // Show status
        this.addStatusMessage('✨ Started new conversation. Your next message will select the best AI model.', 'info');
        
        console.log('New chat session started');
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.simpleChat = new SimpleMentalHealthChat();
});

// Error handling for missing server
window.addEventListener('error', (e) => {
    console.error('JavaScript error:', e.error);
});

// Handle unhandled promise rejections
window.addEventListener('unhandledrejection', (e) => {
    console.error('Unhandled promise rejection:', e.reason);
});