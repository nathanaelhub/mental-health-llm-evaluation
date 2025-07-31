/**
 * Mental Health Chat Interface JavaScript
 * Handles dynamic model selection, real-time streaming, and session management
 */

class MentalHealthChat {
    constructor() {
        this.currentSessionId = null;
        this.selectedModel = null;
        this.conversationMode = 'initial';  // 'initial' or 'continued'
        this.turnCount = 0;
        this.websocket = null;
        this.isStreaming = false;
        this.messageHistory = [];
        this.isSending = false;  // Prevent double-sending
        
        // Configuration from server
        this.config = window.chatConfig || {
            enableStreaming: true,
            enableCaching: true,
            availableModels: ['openai', 'deepseek', 'claude', 'gemma']
        };
        
        this.initializeElements();
        this.bindEvents();
        this.loadSystemStatus();
        
        console.log('Mental Health Chat initialized', this.config);
    }
    
    initializeElements() {
        // Chat elements
        this.chatMessages = document.getElementById('chat-messages');
        this.messageInput = document.getElementById('message-input');
        this.sendButton = document.getElementById('send-button');
        
        // Status elements
        this.currentModelElement = document.getElementById('current-model');
        this.sessionIdElement = document.getElementById('session-id');
        this.modelScoresElement = document.getElementById('model-scores');
        
        // Indicators
        this.selectionIndicator = document.getElementById('selection-indicator');
        this.typingIndicator = document.getElementById('typing-indicator');
        
        // Validate critical elements
        const requiredElements = {
            'chat-messages': this.chatMessages,
            'message-input': this.messageInput,
            'send-button': this.sendButton,
            'current-model': this.currentModelElement,
            'session-id': this.sessionIdElement,
            'model-scores': this.modelScoresElement
        };
        
        for (const [id, element] of Object.entries(requiredElements)) {
            if (!element) {
                console.error(`❌ Required element missing: ${id}`);
            }
        }
        
        // Options
        this.useCacheCheckbox = document.getElementById('use-cache');
        this.useStreamingCheckbox = document.getElementById('use-streaming');
        
        // Modals
        this.chatHistoryModal = document.getElementById('chat-history-modal');
        this.settingsModal = document.getElementById('settings-modal');
        
        // Toast container
        this.toastContainer = document.getElementById('toast-container');
    }
    
    bindEvents() {
        // Send message events
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
        });
        
        // Modal events
        window.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                this.closeModal();
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeModal();
            }
        });
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isStreaming) return;
        
        // Prevent double-sending
        if (this.isSending) {
            console.log('⚠️ Already sending a message, ignoring duplicate request');
            return;
        }
        this.isSending = true;
        
        // Add user message to chat
        this.addMessage(message, 'user');
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        
        // Disable input while processing
        this.setInputState(false);
        
        try {
            console.log('🔍 Sending message - Streaming checkbox:', this.useStreamingCheckbox.checked, 'Config streaming:', this.config.enableStreaming);
            if (this.useStreamingCheckbox.checked && this.config.enableStreaming) {
                console.log('📡 Using streaming mode');
                await this.sendStreamingMessage(message);
            } else {
                console.log('📬 Using regular mode');
                await this.sendRegularMessage(message);
            }
            console.log('✅ Message sent successfully');
        } catch (error) {
            console.error('❌ Error sending message:', error);
            console.error('Error stack:', error.stack);
            this.showToast('Failed to send message. Please try again.', 'error');
            this.addMessage('Sorry, I encountered an error. Please try again.', 'assistant', { error: true });
        } finally {
            this.setInputState(true);
            this.isSending = false;
        }
    }
    
    async sendRegularMessage(message) {
        // Show appropriate indicator based on conversation state
        if (this.conversationMode === 'initial' || !this.currentSessionId) {
            this.showSelectionIndicator();
        } else {
            this.showTypingIndicator();
        }
        
        // Debug logging
        const requestData = {
            message: message,
            session_id: this.currentSessionId,
            user_id: 'demo-user',
            force_reselection: false
        };
        console.log('🔍 Sending request:', JSON.stringify(requestData, null, 2));
        
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Hide indicators
        this.hideSelectionIndicator();
        this.hideTypingIndicator();
        
        // Update session info
        this.updateSessionInfo(data);
        
        // Add assistant message
        this.addMessage(data.response, 'assistant', {
            model: data.selected_model,
            responseTime: data.response_time_ms || 100,
            turnCount: data.turn_count,
            conversationMode: data.conversation_mode,
            confidenceScore: data.confidence_score,
            modelScores: data.model_scores
        });
        
        // Show model scores if this is a new selection
        if (data.conversation_mode === 'selection' && data.model_scores) {
            this.showModelScores({
                all_scores: data.model_scores,
                selected_model: data.selected_model,
                prompt_type: data.prompt_type
            });
        }
        
        // Update UI state
        this.currentSessionId = data.session_id;
        this.selectedModel = data.selected_model;
        this.conversationMode = data.conversation_mode;
        this.turnCount = data.turn_count;
    }
    
    async sendStreamingMessage(message) {
        // Streaming is not implemented in the mock server
        console.warn('⚠️ Streaming mode not available in mock server, falling back to regular mode');
        this.showToast('Streaming not available, using regular mode', 'warning');
        
        // Fall back to regular message sending
        return this.sendRegularMessage(message);
    }
    
    addMessage(content, sender, metadata = {}) {
        // Get conversation history container
        const conversationHistory = document.getElementById('conversation-history');
        if (!conversationHistory) {
            console.error('Conversation history container not found');
            return null;
        }
        
        // Hide welcome message on first user message
        if (sender === 'user' && this.messageHistory.length === 0) {
            const welcomeMessage = document.getElementById('welcome-message');
            if (welcomeMessage) {
                welcomeMessage.style.display = 'none';
            }
        }
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Add avatar
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        
        if (sender === 'user') {
            avatar.textContent = 'U';
        } else {
            avatar.innerHTML = '<i class="fas fa-robot"></i>';
        }
        
        // Create message bubble
        const messageBubble = document.createElement('div');
        messageBubble.className = 'message-bubble';
        
        if (metadata.streaming) {
            messageBubble.innerHTML = '<span class="streaming-cursor">|</span>';
        } else {
            messageBubble.innerHTML = this.formatMessage(content);
        }
        
        // Add content to message
        messageContent.appendChild(avatar);
        messageContent.appendChild(messageBubble);
        
        // Add metadata for assistant messages
        if (sender === 'assistant' && !metadata.streaming) {
            const messageMeta = document.createElement('div');
            messageMeta.className = 'message-meta';
            
            let metaHTML = '';
            
            // Add timestamp
            const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            metaHTML += `<span class="message-time">${timestamp}</span>`;
            
            // Create vertical stack for model info
            if (metadata.model) {
                metaHTML += `
                    <div class="model-selection-stack">
                        <div class="model-badge">${metadata.model.toUpperCase()}</div>`;
                
                if (metadata.confidenceScore !== undefined) {
                    metaHTML += `<div class="confidence-score">${(metadata.confidenceScore * 100).toFixed(1)}% confidence</div>`;
                }
                
                if (metadata.conversationMode === 'selection' || metadata.conversationMode === 'initial') {
                    metaHTML += `<div class="selection-status">🔍 Selected</div>`;
                } else if (metadata.conversationMode === 'continuation' || metadata.conversationMode === 'continued') {
                    metaHTML += `<div class="selection-status">💬 Turn ${metadata.turnCount || ''}</div>`;
                }
                
                if (metadata.cached) {
                    metaHTML += `<div class="selection-status">📚 Cached</div>`;
                }
                
                if (metadata.error) {
                    metaHTML += `<div class="selection-status">⚠️ Error</div>`;
                }
                
                metaHTML += `</div>`;
            }
            
            messageMeta.innerHTML = metaHTML;
            messageContent.appendChild(messageMeta);
        }
        
        // Add timestamp for user messages
        if (sender === 'user') {
            const messageMeta = document.createElement('div');
            messageMeta.className = 'message-meta';
            const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            messageMeta.innerHTML = `<span class="message-time">${timestamp}</span>`;
            messageContent.appendChild(messageMeta);
        }
        
        messageDiv.appendChild(messageContent);
        conversationHistory.appendChild(messageDiv);
        
        // Store in message history
        this.messageHistory.push({
            content,
            sender,
            metadata,
            timestamp: new Date()
        });
        
        this.scrollToBottom();
        
        return messageDiv;
    }
    
    updateStreamingMessage(messageElement, content) {
        const messageBubble = messageElement.querySelector('.message-bubble');
        messageBubble.innerHTML = this.formatMessage(content) + '<span class="streaming-cursor">|</span>';
        this.scrollToBottom();
    }
    
    finalizeStreamingMessage(messageElement) {
        const messageBubble = messageElement.querySelector('.message-bubble');
        const cursor = messageBubble.querySelector('.streaming-cursor');
        if (cursor) {
            cursor.remove();
        }
        
        // Add metadata
        const messageContent = messageElement.querySelector('.message-content');
        const messageMeta = document.createElement('div');
        messageMeta.className = 'message-meta';
        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        messageMeta.innerHTML = `<span class="message-time">${timestamp}</span><span class="model-badge">📡 Streamed</span>`;
        messageContent.appendChild(messageMeta);
    }
    
    formatMessage(content) {
        // Simple formatting for mental health content
        return content
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank">$1</a>');
    }
    
    updateSessionInfo(data) {
        // Handle both old and new API response formats
        let sessionId, modelName, selectionInfo, conversationMode, turnCount, confidenceScore;
        
        if (data.session_id) {
            // New API format
            sessionId = data.session_id;
            modelName = data.selected_model;
            conversationMode = data.conversation_mode;
            turnCount = data.turn_count;
            confidenceScore = data.confidence_score;
            selectionInfo = {
                selected_model: data.selected_model,
                selection_score: data.confidence_score,
                selection_reasoning: data.reasoning
            };
        } else {
            // Legacy format fallback
            sessionId = data;
            modelName = arguments[1];
            selectionInfo = arguments[2];
        }
        
        // Update UI elements
        this.currentSessionId = sessionId;
        this.selectedModel = modelName;
        this.conversationMode = conversationMode;
        this.turnCount = turnCount;
        
        // Update session display
        if (this.sessionIdElement) {
            this.sessionIdElement.textContent = sessionId.substring(0, 8);
        }
        
        // Update model display
        if (conversationMode === 'initial') {
            if (this.currentModelElement) {
                this.currentModelElement.textContent = `${modelName.toUpperCase()} (Selected)`;
                this.currentModelElement.classList.add('model-selecting');
            }
            
            // Show confidence score
            const confidenceDisplay = document.getElementById('confidence-display');
            const confidenceScore = document.getElementById('confidence-score');
            if (confidenceDisplay && confidenceScore) {
                confidenceScore.textContent = `${(selectionInfo.selection_score * 100).toFixed(1)}%`;
                confidenceDisplay.style.display = 'block';
            }
            
            // Remove selecting state after animation
            setTimeout(() => {
                if (this.currentModelElement) {
                    this.currentModelElement.classList.remove('model-selecting');
                }
            }, 2000);
            
            // Create detailed toast message
            const confidencePercent = (confidenceScore * 100).toFixed(1);
            const promptTypeDisplay = data.prompt_type ? data.prompt_type.replace('_', ' ') : 'general';
            this.showToast(`🔍 Selected ${modelName.toUpperCase()} (${confidencePercent}% confidence) for ${promptTypeDisplay} prompt`, 'success');
        } else {
            if (this.currentModelElement) {
                this.currentModelElement.textContent = modelName.toUpperCase();
            }
        }
        
        // Update turn counter
        const turnCountElement = document.getElementById('turn-count');
        if (turnCountElement) {
            turnCountElement.textContent = turnCount || '0';
        }
        
        // Add conversation mode indicator
        this.addConversationModeIndicator(conversationMode, turnCount);
        
        if (selectionInfo && conversationMode === 'initial') {
            this.showModelScores(selectionInfo);
        }
    }
    
    showModelScores(selectionInfo) {
        console.log('📊 Showing model scores:', selectionInfo);
        
        const scores = selectionInfo.all_scores || {};
        const promptType = selectionInfo.prompt_type || 'general';
        
        let scoresHTML = `<div class="scores-header">
            <strong>Model Evaluation (${promptType.replace('_', ' ')} prompt)</strong>
        </div>`;
        
        // Sort models by score (highest first)
        const sortedScores = Object.entries(scores).sort((a, b) => b[1] - a[1]);
        
        for (const [model, score] of sortedScores) {
            const isSelected = model === selectionInfo.selected_model;
            const percentage = ((score / 10) * 100).toFixed(0);
            scoresHTML += `<div class="score-item ${isSelected ? 'selected' : ''}">
                <span class="model-name">${model.toUpperCase()}</span>
                <span class="score-value">${score.toFixed(2)}/10 (${percentage}%)</span>
                ${isSelected ? '<span class="selected-indicator">✓ Selected</span>' : ''}
            </div>`;
        }
        
        if (this.modelScoresElement) {
            this.modelScoresElement.innerHTML = scoresHTML;
            this.modelScoresElement.style.display = 'block';
        }
    }
    
    showSelectionIndicator() {
        this.selectionIndicator.style.display = 'flex';
    }
    
    hideSelectionIndicator() {
        this.selectionIndicator.style.display = 'none';
    }
    
    showTypingIndicator() {
        this.typingIndicator.style.display = 'flex';
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
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        toast.innerHTML = `
            <div>${message}</div>
            <div class="toast-progress"></div>
        `;
        
        this.toastContainer.appendChild(toast);
        
        setTimeout(() => {
            toast.remove();
        }, 5000);
        
        // Click to dismiss
        toast.addEventListener('click', () => {
            toast.remove();
        });
    }
    
    async loadSystemStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();
            
            // Update model availability in UI
            this.updateModelHealth(status.model_health);
            
        } catch (error) {
            console.error('Failed to load system status:', error);
        }
    }
    
    updateModelHealth(modelHealth) {
        // You can add visual indicators for model health here
        console.log('Model health status:', modelHealth);
    }
    
    // Modal Management
    showChatHistory() {
        this.loadChatHistory();
        this.chatHistoryModal.style.display = 'flex';
    }
    
    showSettings() {
        this.loadSystemSettings();
        this.settingsModal.style.display = 'flex';
    }
    
    showSystemStatus() {
        this.loadSystemStatus();
        document.getElementById('system-status-modal').style.display = 'flex';
    }
    
    showModelStatus() {
        this.loadModelStatus();
        document.getElementById('model-status-modal').style.display = 'flex';
    }
    
    closeModal() {
        this.chatHistoryModal.style.display = 'none';
        this.settingsModal.style.display = 'none';
        document.getElementById('system-status-modal').style.display = 'none';
        document.getElementById('model-status-modal').style.display = 'none';
        document.getElementById('model-switch-modal').style.display = 'none';
    }
    
    showModelSwitchModal() {
        // Update current model display
        const currentModelDisplay = document.getElementById('current-model-display');
        if (currentModelDisplay) {
            currentModelDisplay.textContent = this.selectedModel ? this.selectedModel.toUpperCase() : 'Unknown';
        }
        
        document.getElementById('model-switch-modal').style.display = 'flex';
    }
    
    loadChatHistory() {
        // Update chat history info section
        const historySessionId = document.getElementById('history-session-id');
        const historyModel = document.getElementById('history-model');
        const historyTurns = document.getElementById('history-turns');
        const historyMode = document.getElementById('history-mode');
        
        if (!this.currentSessionId) {
            historySessionId.textContent = 'No active session';
            historyModel.textContent = 'None';
            historyTurns.textContent = '0';
            historyMode.textContent = 'New conversation';
        } else {
            historySessionId.textContent = this.currentSessionId.substring(0, 8) + '...';
            historyModel.textContent = this.selectedModel ? this.selectedModel.toUpperCase() : 'Unknown';
            historyTurns.textContent = this.turnCount || 0;
            historyMode.textContent = this.conversationMode === 'selection' ? 'Model Selection' : 'Continuing Conversation';
        }
        
        // Load conversation history
        const historyContent = document.getElementById('chat-history-content');
        
        if (!this.messageHistory || this.messageHistory.length === 0) {
            historyContent.innerHTML = `
                <div class="empty-history">
                    <i class="fas fa-comment-slash" style="font-size: 3rem; color: var(--text-secondary); margin-bottom: 1rem;"></i>
                    <p>No conversation history yet.</p>
                    <p>Start chatting to see your conversation history here!</p>
                </div>
            `;
            return;
        }
        
        // Generate history HTML
        let historyHTML = '<div class="history-timeline">';
        
        this.messageHistory.forEach((messageData, index) => {
            const timestamp = messageData.timestamp.toLocaleString();
            const isUser = messageData.sender === 'user';
            const metadata = messageData.metadata || {};
            
            historyHTML += `
                <div class="history-item ${isUser ? 'user-item' : 'assistant-item'}">
                    <div class="history-meta">
                        <span class="history-timestamp">${timestamp}</span>
                        <span class="history-sender">${isUser ? 'You' : 'Assistant'}</span>
                        ${!isUser && metadata.model ? `<span class="history-model">${metadata.model.toUpperCase()}</span>` : ''}
                        ${!isUser && metadata.conversationMode === 'selection' ? '<span class="history-badge">🔍 Selected</span>' : ''}
                        ${!isUser && metadata.conversationMode === 'continued' ? `<span class="history-badge">💬 Turn ${metadata.turnCount || ''}</span>` : ''}
                    </div>
                    <div class="history-content">
                        ${this.formatMessage(messageData.content)}
                    </div>
                </div>
            `;
        });
        
        historyHTML += '</div>';
        
        // Add summary statistics
        const userMessages = this.messageHistory.filter(m => m.sender === 'user').length;
        const assistantMessages = this.messageHistory.filter(m => m.sender === 'assistant').length;
        const firstMessage = this.messageHistory[0];
        const lastMessage = this.messageHistory[this.messageHistory.length - 1];
        
        historyHTML += `
            <div class="history-summary">
                <h4>Conversation Summary</h4>
                <div class="summary-stats">
                    <div class="summary-stat">
                        <span class="stat-label">Your Messages:</span>
                        <span class="stat-value">${userMessages}</span>
                    </div>
                    <div class="summary-stat">
                        <span class="stat-label">Assistant Responses:</span>
                        <span class="stat-value">${assistantMessages}</span>
                    </div>
                    <div class="summary-stat">
                        <span class="stat-label">Started:</span>
                        <span class="stat-value">${firstMessage ? firstMessage.timestamp.toLocaleString() : 'Unknown'}</span>
                    </div>
                    <div class="summary-stat">
                        <span class="stat-label">Last Activity:</span>
                        <span class="stat-value">${lastMessage ? lastMessage.timestamp.toLocaleString() : 'Unknown'}</span>
                    </div>
                </div>
            </div>
        `;
        
        historyContent.innerHTML = historyHTML;
    }
    
    async loadSystemSettings() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();
            
            // Update model health
            const modelHealthGrid = document.getElementById('model-health');
            modelHealthGrid.innerHTML = '';
            
            for (const [model, isHealthy] of Object.entries(status.model_health)) {
                const healthItem = document.createElement('div');
                healthItem.className = 'health-item';
                healthItem.innerHTML = `
                    <span>${model.toUpperCase()}</span>
                    <span class="health-status ${isHealthy ? 'healthy' : 'unhealthy'}">
                        ${isHealthy ? '✓ Healthy' : '✗ Unhealthy'}
                    </span>
                `;
                modelHealthGrid.appendChild(healthItem);
            }
            
            // Update cache stats
            const cacheStatsGrid = document.getElementById('cache-stats');
            cacheStatsGrid.innerHTML = '';
            
            if (status.cache_stats && Object.keys(status.cache_stats).length > 0) {
                const cacheStats = [
                    { label: 'Total Entries', value: status.cache_stats.total_entries },
                    { label: 'Hit Rate', value: `${(status.cache_stats.hit_rate * 100).toFixed(1)}%` },
                    { label: 'Cache Hits', value: status.cache_stats.cache_hits },
                    { label: 'Cache Misses', value: status.cache_stats.cache_misses }
                ];
                
                cacheStats.forEach(stat => {
                    const statItem = document.createElement('div');
                    statItem.className = 'stat-item';
                    statItem.innerHTML = `<strong>${stat.label}:</strong> ${stat.value}`;
                    cacheStatsGrid.appendChild(statItem);
                });
            }
            
            // Update session analytics
            const sessionAnalyticsGrid = document.getElementById('session-analytics');
            sessionAnalyticsGrid.innerHTML = '';
            
            if (status.session_analytics) {
                const sessionStats = [
                    { label: 'Active Sessions', value: status.session_analytics.total_sessions },
                    { label: 'Total Turns', value: status.session_analytics.total_turns },
                    { label: 'Avg Conversation Length', value: status.session_analytics.avg_conversation_length?.toFixed(1) || '0' },
                    { label: 'Active Users', value: status.session_analytics.active_users }
                ];
                
                sessionStats.forEach(stat => {
                    const statItem = document.createElement('div');
                    statItem.className = 'stat-item';
                    statItem.innerHTML = `<strong>${stat.label}:</strong> ${stat.value}`;
                    sessionAnalyticsGrid.appendChild(statItem);
                });
            }
            
        } catch (error) {
            console.error('Failed to load system settings:', error);
        }
    }
    
    async loadSystemStatus() {
        const contentDiv = document.getElementById('system-status-content');
        
        try {
            // Show loading spinner
            contentDiv.innerHTML = `
                <div class="loading-spinner">
                    <div class="spinner"></div>
                    <span>Loading system status...</span>
                </div>
            `;
            
            const response = await fetch('/api/status');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const status = await response.json();
            
            // Display system status
            contentDiv.innerHTML = `
                <div class="status-grid">
                    <div class="status-card">
                        <h4><i class="fas fa-heartbeat"></i> System Health</h4>
                        <div class="status-item">
                            <span class="status-label">Status:</span>
                            <span class="status-badge ${status.status === 'healthy' ? 'healthy' : status.status}">${status.status || 'Unknown'}</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">Version:</span>
                            <span class="status-value">${status.version || 'Unknown'}</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">Uptime:</span>
                            <span class="status-value">${this.formatUptime(status.uptime_seconds || 0)}</span>
                        </div>
                    </div>
                    
                    <div class="status-card">
                        <h4><i class="fas fa-database"></i> Models</h4>
                        <div class="status-item">
                            <span class="status-label">Available Models:</span>
                            <span class="status-value">${status.available_models ? status.available_models.length : 0}</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">Models:</span>
                            <span class="status-value">${status.available_models ? status.available_models.join(', ') : 'None'}</span>
                        </div>
                    </div>
                </div>
            `;
            
        } catch (error) {
            console.error('Failed to load system status:', error);
            contentDiv.innerHTML = `
                <div class="status-card">
                    <h4><i class="fas fa-exclamation-triangle"></i> Error</h4>
                    <p style="color: var(--danger-color);">Failed to load system status: ${error.message}</p>
                    <button onclick="chat.loadSystemStatus()" style="margin-top: 10px; padding: 8px 16px; background: var(--primary-color); color: white; border: none; border-radius: 4px; cursor: pointer;">Retry</button>
                </div>
            `;
        }
    }
    
    async loadModelStatus() {
        const contentDiv = document.getElementById('model-status-content');
        
        try {
            // Show loading spinner
            contentDiv.innerHTML = `
                <div class="loading-spinner">
                    <div class="spinner"></div>
                    <span>Loading model status...</span>
                </div>
            `;
            
            const response = await fetch('/api/models/status');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const modelData = await response.json();
            
            // Display model status
            const modelsHtml = Object.entries(modelData.models).map(([modelId, modelInfo]) => `
                <div class="model-card">
                    <div class="model-header">
                        <span class="model-name">${modelId}</span>
                        <span class="status-badge ${modelInfo.enabled ? 'healthy' : 'critical'}">
                            ${modelInfo.enabled ? '✓ Enabled' : '✗ Disabled'}
                        </span>
                    </div>
                    <div class="model-details">
                        <div class="model-detail">
                            <span class="status-label">Model Name:</span>
                            <span class="status-value">${modelInfo.model_name || 'Unknown'}</span>
                        </div>
                        <div class="model-detail">
                            <span class="status-label">Cost per Token:</span>
                            <span class="status-value">$${(modelInfo.cost_per_token || 0).toFixed(6)}</span>
                        </div>
                        <div class="model-detail">
                            <span class="status-label">Status:</span>
                            <span class="status-value">${modelInfo.status || 'Unknown'}</span>
                        </div>
                    </div>
                    ${modelInfo.specialties ? `
                        <div class="specialties">
                            ${modelInfo.specialties.map(specialty => 
                                `<span class="specialty-tag">${specialty}</span>`
                            ).join('')}
                        </div>
                    ` : ''}
                </div>
            `).join('');
            
            contentDiv.innerHTML = `
                <div style="margin-bottom: 20px;">
                    <div class="status-card">
                        <h4><i class="fas fa-info-circle"></i> Overview</h4>
                        <div class="status-item">
                            <span class="status-label">Total Available:</span>
                            <span class="status-value">${modelData.total_available || 0} models</span>
                        </div>
                    </div>
                </div>
                <div class="model-grid">
                    ${modelsHtml}
                </div>
            `;
            
        } catch (error) {
            console.error('Failed to load model status:', error);
            contentDiv.innerHTML = `
                <div class="status-card">
                    <h4><i class="fas fa-exclamation-triangle"></i> Error</h4>
                    <p style="color: var(--danger-color);">Failed to load model status: ${error.message}</p>
                    <button onclick="chat.loadModelStatus()" style="margin-top: 10px; padding: 8px 16px; background: var(--primary-color); color: white; border: none; border-radius: 4px; cursor: pointer;">Retry</button>
                </div>
            `;
        }
    }
    
    formatUptime(seconds) {
        if (seconds < 60) return `${Math.floor(seconds)}s`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
        if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
        return `${Math.floor(seconds / 86400)}d ${Math.floor((seconds % 86400) / 3600)}h`;
    }
    
    
    clearChat(resetSession = true) {
        // Clear messages except welcome message
        const welcomeMessage = this.chatMessages.querySelector('.welcome-message');
        this.chatMessages.innerHTML = '';
        if (welcomeMessage) {
            this.chatMessages.appendChild(welcomeMessage);
        }
        
        if (resetSession) {
            // Reset session
            this.currentSessionId = null;
            this.selectedModel = null;
            this.conversationMode = 'initial';
            this.turnCount = 0;
            if (this.sessionIdElement) {
                this.sessionIdElement.textContent = 'New';
            }
            if (this.currentModelElement) {
                this.currentModelElement.textContent = 'Selecting...';
            }
            if (this.modelScoresElement) {
                this.modelScoresElement.style.display = 'none';
            }
            
            // Remove turn counter if it exists
            const turnCounter = document.querySelector('.model-info .turn-counter');
            if (turnCounter) {
                turnCounter.remove();
            }
            
            this.showToast('Chat cleared. New session will start with model selection.', 'info');
        }
    }
    
    async forceReselection() {
        if (!this.currentSessionId) {
            this.showToast('No active session to force re-selection', 'warning');
            return;
        }
        
        const message = this.messageInput.value.trim() || "Please re-evaluate and select the best model for this conversation.";
        
        try {
            this.showSelectionIndicator();
            this.setInputState(false);
            
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    session_id: this.currentSessionId,
                    user_id: 'demo-user',
                    force_reselection: true
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            this.hideSelectionIndicator();
            this.setInputState(true);
            
            // Update session info
            this.updateSessionInfo(data);
            
            // Update UI state
            this.selectedModel = data.selected_model;
            this.conversationMode = data.conversation_mode;
            
            this.showToast(`🔄 Re-selected ${data.selected_model.toUpperCase()} (confidence: ${(data.confidence_score * 100).toFixed(0)}%)`, 'success');
            
            if (message !== "Please re-evaluate and select the best model for this conversation.") {
                // Add user message and response if custom message was provided
                this.addMessage(message, 'user');
                this.addMessage(data.response, 'assistant', {
                    model: data.selected_model,
                    responseTime: data.response_time_ms || 100,
                    turnCount: data.turn_count,
                    conversationMode: data.conversation_mode,
                    confidenceScore: data.confidence_score,
                    modelScores: data.model_scores
                });
                this.messageInput.value = '';
            }
            
        } catch (error) {
            this.hideSelectionIndicator();
            this.setInputState(true);
            console.error('Force re-selection failed:', error);
            this.showToast('Failed to force re-selection. Please try again.', 'error');
        }
    }
    
    async switchToModel(modelName) {
        if (!this.currentSessionId) {
            this.showToast('No active session to switch models', 'warning');
            return;
        }
        
        try {
            const response = await fetch(`/api/sessions/${this.currentSessionId}/switch-model?new_model=${modelName}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // Update UI
            this.selectedModel = data.new_model;
            if (this.currentModelElement) {
                this.currentModelElement.textContent = data.new_model.toUpperCase();
            }
            
            // Close modal and show success
            this.closeModal();
            this.showToast(`Switched to ${data.new_model.toUpperCase()}`, 'success');
            
        } catch (error) {
            console.error('Model switch failed:', error);
            this.showToast('Failed to switch model. Please try again.', 'error');
        }
    }
    
    startNewConversation() {
        // Clear conversation state
        this.currentSessionId = null;
        this.selectedModel = null;
        this.conversationMode = 'initial';
        this.turnCount = 0;
        this.messageHistory = [];
        
        // Clear UI
        const conversationHistory = document.getElementById('conversation-history');
        if (conversationHistory) {
            conversationHistory.innerHTML = '';
        }
        
        // Show welcome message
        const welcomeMessage = document.getElementById('welcome-message');
        if (welcomeMessage) {
            welcomeMessage.style.display = 'block';
        }
        
        // Reset UI elements
        if (this.currentModelElement) {
            this.currentModelElement.textContent = 'Selecting best model...';
        }
        if (this.sessionIdElement) {
            this.sessionIdElement.textContent = 'New';
        }
        
        const confidenceDisplay = document.getElementById('confidence-display');
        if (confidenceDisplay) {
            confidenceDisplay.style.display = 'none';
        }
        
        const turnCountElement = document.getElementById('turn-count');
        if (turnCountElement) {
            turnCountElement.textContent = '0';
        }
        
        // Focus on input
        this.messageInput.focus();
        
        this.showToast('🆕 Started new conversation', 'info');
    }
    
    addConversationModeIndicator(mode, turnCount) {
        // Remove existing indicator
        const existingIndicator = document.querySelector('.conversation-mode-indicator');
        if (existingIndicator) {
            existingIndicator.remove();
        }
        
        if (mode === 'initial') {
            const indicator = document.createElement('div');
            indicator.className = 'conversation-mode-indicator mode-initial';
            indicator.innerHTML = '🔍 <strong>Model Selection:</strong> Analyzing your message to select the best AI model for this conversation...';
            
            const conversationHistory = document.getElementById('conversation-history');
            if (conversationHistory && conversationHistory.children.length === 0) {
                conversationHistory.appendChild(indicator);
            }
        }
    }
    
    showModelSwitchModal() {
        const modal = document.getElementById('model-switch-modal');
        if (modal) {
            // Update current model display
            const currentModelDisplay = document.getElementById('current-model-display');
            if (currentModelDisplay && this.selectedModel) {
                currentModelDisplay.textContent = this.selectedModel.toUpperCase();
            }
            
            modal.style.display = 'flex';
        }
    }
}

// Global functions for HTML onclick handlers
window.showChatHistory = () => chat.showChatHistory();
window.showSettings = () => chat.showSettings();
window.showSystemStatus = () => chat.showSystemStatus();
window.showModelStatus = () => chat.showModelStatus();
window.closeModal = () => chat.closeModal();
window.clearChat = () => chat.clearChat();
window.forceReselection = () => chat.forceReselection();
window.switchToModel = (modelName) => chat.switchToModel(modelName);
window.startNewConversation = () => chat.startNewConversation();
window.showModelSwitchModal = () => chat.showModelSwitchModal();

// Initialize chat when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chat = new MentalHealthChat();
});

// Add some CSS for streaming cursor
const style = document.createElement('style');
style.textContent = `
    .streaming-cursor {
        animation: blink 1s infinite;
        color: var(--primary-color);
        font-weight: bold;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    .scores-header {
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-bottom: 8px;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 4px;
    }
    
    .score-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 3px 0;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid transparent;
    }
    
    .score-item.selected {
        background: rgba(74, 144, 226, 0.2);
        border-color: var(--primary-color);
        font-weight: bold;
    }
    
    .model-name {
        font-weight: 600;
    }
    
    .score-value {
        color: var(--text-secondary);
        font-family: monospace;
    }
    
    .selected-indicator {
        color: var(--success-color);
        font-size: 0.7rem;
        font-weight: normal;
    }
    
    /* Chat History Modal Styles */
    .chat-history-info {
        margin-bottom: 20px;
        padding: 15px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }
    
    .history-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 10px;
    }
    
    .stat-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0;
    }
    
    .stat-label {
        font-weight: 600;
        color: var(--text-secondary);
    }
    
    .stat-value {
        font-family: monospace;
        color: var(--primary-color);
        font-weight: bold;
    }
    
    .chat-history-content {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 8px;
    }
    
    .empty-history {
        text-align: center;
        padding: 40px 20px;
        color: var(--text-secondary);
    }
    
    .history-timeline {
        space-y: 15px;
    }
    
    .history-item {
        margin-bottom: 15px;
        padding: 12px;
        border-radius: 8px;
        border-left: 3px solid transparent;
    }
    
    .history-item.user-item {
        background: rgba(74, 144, 226, 0.1);
        border-left-color: var(--primary-color);
    }
    
    .history-item.assistant-item {
        background: rgba(34, 197, 94, 0.1);
        border-left-color: var(--success-color);
    }
    
    .history-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 8px;
        font-size: 0.75rem;
        color: var(--text-secondary);
    }
    
    .history-timestamp {
        font-family: monospace;
    }
    
    .history-sender {
        font-weight: 600;
    }
    
    .history-model {
        background: var(--primary-color);
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: bold;
    }
    
    .history-badge {
        background: var(--secondary-color);
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.7rem;
    }
    
    .history-content {
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    .history-summary {
        margin-top: 20px;
        padding: 15px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        border-top: 2px solid var(--primary-color);
    }
    
    .history-summary h4 {
        margin: 0 0 10px 0;
        color: var(--primary-color);
    }
    
    .summary-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 8px;
    }
    
    .summary-stat {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
`;
document.head.appendChild(style);