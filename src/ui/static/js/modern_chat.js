/**
 * Modern Mental Health Chat Interface JavaScript
 * 
 * Comprehensive chat system with dynamic model selection, real-time streaming,
 * accessibility features, and professional UX patterns.
 */

class MentalHealthChat {
    constructor(config = {}) {
        this.config = {
            apiEndpoint: '/api/chat',
            wsEndpoint: '/api/chat/stream',
            maxMessageLength: 2000,
            enableStreaming: true,
            enableCaching: true,
            availableModels: ['openai', 'claude', 'deepseek', 'gemma'],
            features: {
                modelSelection: true,
                quickActions: true,
                exportConversation: true,
                darkMode: true,
                accessibility: true
            },
            ...config
        };

        // State management
        this.state = {
            sessionId: null,
            currentModel: 'intelligent-selection',
            isConnected: false,
            isStreaming: false,
            messageHistory: [],
            settings: {
                theme: localStorage.getItem('chat-theme') || 'auto',
                fontSize: localStorage.getItem('chat-font-size') || 'medium',
                highContrast: localStorage.getItem('chat-high-contrast') === 'true',
                reduceMotion: localStorage.getItem('chat-reduce-motion') === 'true',
                enableQuickActions: localStorage.getItem('chat-quick-actions') !== 'false',
                showModelDetails: localStorage.getItem('chat-model-details') === 'true',
                enableStreaming: localStorage.getItem('chat-streaming') !== 'false'
            }
        };

        // WebSocket connection
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;

        // DOM elements
        this.elements = {};

        // Event handlers
        this.handlers = new Map();

        // Initialize the chat system
        this.init();
    }

    /**
     * Initialize the chat system
     */
    async init() {
        try {
            console.log('🚀 Initializing Mental Health Chat System...');
            
            // Get DOM elements
            this.getDOMElements();
            
            // Apply saved settings
            this.applySettings();
            
            // Set up event listeners
            this.setupEventListeners();
            
            // Initialize session
            await this.initializeSession();
            
            // Connect WebSocket if streaming enabled
            if (this.config.enableStreaming && this.state.settings.enableStreaming) {
                this.connectWebSocket();
            }
            
            // Enable input
            this.enableInput();
            
            // Show quick actions if enabled
            if (this.state.settings.enableQuickActions) {
                this.showQuickActions();
            }
            
            // Update connection status
            this.updateConnectionStatus(true);
            
            console.log('✅ Mental Health Chat System initialized successfully');
            
            // Show welcome toast
            this.showToast('Welcome! Your secure mental health support session is ready.', 'success');
            
        } catch (error) {
            console.error('❌ Failed to initialize chat system:', error);
            this.showToast('Failed to initialize chat system. Please refresh the page.', 'error');
        }
    }

    /**
     * Get all required DOM elements
     */
    getDOMElements() {
        const selectors = {
            // Main containers
            chatApp: '#chat-app',
            messagesContainer: '#messages-container',
            chatMessages: '#chat-messages',
            
            // Input elements
            messageForm: '#message-form',
            messageInput: '#message-input',
            sendButton: '#send-button',
            sendText: '#send-text',
            sendIcon: '#send-icon',
            charCount: '#char-count',
            
            // Header elements
            currentModel: '#current-model',
            modelIndicator: '#model-indicator',
            themeToggle: '#theme-toggle',
            settingsBtn: '#settings-btn',
            
            // Settings modal
            settingsModal: '#settings-modal',
            closeSettings: '#close-settings',
            saveSettings: '#save-settings',
            resetSettings: '#reset-settings',
            
            // Settings controls
            highContrast: '#high-contrast',
            reduceMotion: '#reduce-motion',
            fontSize: '#font-size',
            showModelDetails: '#show-model-details',
            enableQuickActions: '#enable-quick-actions',
            enableStreaming: '#enable-streaming',
            
            // Quick actions
            quickActions: '#quick-actions',
            
            // Model selection
            modelSelectionPanel: '#model-selection-panel',
            closeSelectionPanel: '#close-selection-panel',
            modelComparison: '#model-comparison',
            selectionReasoning: '#selection-reasoning',
            advancedMode: '#advanced-mode',
            
            // Status elements
            typingIndicator: '#typing-indicator',
            typingText: '#typing-text',
            connectionStatus: '#connection-status',
            sessionDisplay: '#session-display',
            modelsCount: '#models-count',
            
            // Utility elements
            toastContainer: '#toast-container',
            loadingOverlay: '#loading-overlay',
            exportBtn: '#export-btn',
            clearConversation: '#clear-conversation',
            downloadData: '#download-data'
        };

        this.elements = {};
        for (const [key, selector] of Object.entries(selectors)) {
            this.elements[key] = document.querySelector(selector);
            if (!this.elements[key]) {
                console.warn(`⚠️ Element not found: ${selector}`);
            }
        }
    }

    /**
     * Set up all event listeners
     */
    setupEventListeners() {
        // Message form submission
        if (this.elements.messageForm) {
            this.elements.messageForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.sendMessage();
            });
        }

        // Message input handling
        if (this.elements.messageInput) {
            this.elements.messageInput.addEventListener('input', (e) => {
                this.updateCharCount();
                this.updateSendButton();
            });

            this.elements.messageInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                } else if (e.key === 'Enter' && e.shiftKey) {
                    // Allow new line
                }
            });

            // Auto-resize textarea
            this.elements.messageInput.addEventListener('input', () => {
                this.autoResizeTextarea(this.elements.messageInput);
            });
        }

        // Theme toggle
        if (this.elements.themeToggle) {
            this.elements.themeToggle.addEventListener('click', () => {
                this.toggleTheme();
            });
        }

        // Settings modal
        if (this.elements.settingsBtn) {
            this.elements.settingsBtn.addEventListener('click', () => {
                this.openSettings();
            });
        }

        if (this.elements.closeSettings) {
            this.elements.closeSettings.addEventListener('click', () => {
                this.closeSettings();
            });
        }

        if (this.elements.saveSettings) {
            this.elements.saveSettings.addEventListener('click', () => {
                this.saveSettings();
            });
        }

        if (this.elements.resetSettings) {
            this.elements.resetSettings.addEventListener('click', () => {
                this.resetSettings();
            });
        }

        // Advanced mode toggle
        if (this.elements.advancedMode) {
            this.elements.advancedMode.addEventListener('change', (e) => {
                this.toggleAdvancedMode(e.target.checked);
            });
        }

        // Model selection panel
        if (this.elements.closeSelectionPanel) {
            this.elements.closeSelectionPanel.addEventListener('click', () => {
                this.hideModelSelection();
            });
        }

        // Export and data management
        if (this.elements.exportBtn) {
            this.elements.exportBtn.addEventListener('click', () => {
                this.exportConversation();
            });
        }

        if (this.elements.clearConversation) {
            this.elements.clearConversation.addEventListener('click', () => {
                this.clearConversation();
            });
        }

        if (this.elements.downloadData) {
            this.elements.downloadData.addEventListener('click', () => {
                this.downloadUserData();
            });
        }

        // Quick action buttons
        this.setupQuickActionListeners();

        // Accessibility: Close modals with Escape
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeModals();
            }
        });

        // Handle window focus for reconnection
        window.addEventListener('focus', () => {
            if (!this.state.isConnected) {
                this.reconnectWebSocket();
            }
        });

        // Handle connection status changes
        window.addEventListener('online', () => {
            this.updateConnectionStatus(true);
            this.reconnectWebSocket();
        });

        window.addEventListener('offline', () => {
            this.updateConnectionStatus(false);
        });
    }

    /**
     * Set up quick action button listeners
     */
    setupQuickActionListeners() {
        const quickActionButtons = document.querySelectorAll('.quick-action-btn');
        quickActionButtons.forEach(button => {
            button.addEventListener('click', () => {
                const message = button.getAttribute('data-message');
                if (message) {
                    this.sendQuickMessage(message);
                }
            });
        });
    }

    /**
     * Apply saved settings to the interface
     */
    applySettings() {
        // Apply theme
        this.applyTheme(this.state.settings.theme);
        
        // Apply font size
        this.applyFontSize(this.state.settings.fontSize);
        
        // Apply accessibility settings
        if (this.state.settings.highContrast) {
            document.body.classList.add('high-contrast');
        }
        
        if (this.state.settings.reduceMotion) {
            document.body.classList.add('reduce-motion');
        }
        
        // Update settings form
        this.updateSettingsForm();
    }

    /**
     * Update settings form with current values
     */
    updateSettingsForm() {
        if (this.elements.highContrast) {
            this.elements.highContrast.checked = this.state.settings.highContrast;
        }
        
        if (this.elements.reduceMotion) {
            this.elements.reduceMotion.checked = this.state.settings.reduceMotion;
        }
        
        if (this.elements.fontSize) {
            this.elements.fontSize.value = this.state.settings.fontSize;
        }
        
        if (this.elements.showModelDetails) {
            this.elements.showModelDetails.checked = this.state.settings.showModelDetails;
        }
        
        if (this.elements.enableQuickActions) {
            this.elements.enableQuickActions.checked = this.state.settings.enableQuickActions;
        }
        
        if (this.elements.enableStreaming) {
            this.elements.enableStreaming.checked = this.state.settings.enableStreaming;
        }
    }

    /**
     * Initialize chat session
     */
    async initializeSession() {
        try {
            const response = await fetch(`${this.config.apiEndpoint}/session`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    user_id: this.getUserId(),
                    metadata: {
                        user_agent: navigator.userAgent,
                        timestamp: new Date().toISOString()
                    }
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            this.state.sessionId = data.session_id;
            
            // Update session display
            if (this.elements.sessionDisplay) {
                this.elements.sessionDisplay.textContent = this.state.sessionId.substring(0, 8);
            }
            
            // Update models count
            if (this.elements.modelsCount) {
                this.elements.modelsCount.textContent = this.config.availableModels.length;
            }

            console.log('✅ Session initialized:', this.state.sessionId);
            
        } catch (error) {
            console.error('❌ Failed to initialize session:', error);
            throw error;
        }
    }

    /**
     * Connect WebSocket for real-time updates
     */
    connectWebSocket() {
        if (!this.state.sessionId) {
            console.warn('⚠️ Cannot connect WebSocket: No session ID');
            return;
        }

        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}${this.config.wsEndpoint}?session_id=${this.state.sessionId}`;
            
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('✅ WebSocket connected');
                this.state.isConnected = true;
                this.reconnectAttempts = 0;
                this.updateConnectionStatus(true);
            };
            
            this.ws.onmessage = (event) => {
                this.handleWebSocketMessage(event);
            };
            
            this.ws.onclose = (event) => {
                console.log('🔌 WebSocket disconnected:', event.code);
                this.state.isConnected = false;
                this.updateConnectionStatus(false);
                
                // Attempt reconnection if not a clean close
                if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
                    setTimeout(() => {
                        this.reconnectWebSocket();
                    }, Math.pow(2, this.reconnectAttempts) * 1000); // Exponential backoff
                }
            };
            
            this.ws.onerror = (error) => {
                console.error('❌ WebSocket error:', error);
            };
            
        } catch (error) {
            console.error('❌ Failed to connect WebSocket:', error);
        }
    }

    /**
     * Reconnect WebSocket
     */
    reconnectWebSocket() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.warn('⚠️ Max reconnection attempts reached');
            return;
        }
        
        this.reconnectAttempts++;
        console.log(`🔄 Reconnecting WebSocket (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.connectWebSocket();
    }

    /**
     * Handle incoming WebSocket messages
     */
    handleWebSocketMessage(event) {
        try {
            const data = JSON.parse(event.data);
            
            switch (data.type) {
                case 'message_chunk':
                    this.handleStreamingChunk(data);
                    break;
                case 'message_complete':
                    this.handleStreamingComplete(data);
                    break;
                case 'model_selected':
                    this.handleModelSelection(data);
                    break;
                case 'crisis_alert':
                    this.handleCrisisAlert(data);
                    break;
                case 'session_update':
                    this.handleSessionUpdate(data);
                    break;
                default:
                    console.log('📨 Received WebSocket message:', data);
            }
        } catch (error) {
            console.error('❌ Error handling WebSocket message:', error);
        }
    }

    /**
     * Send a message
     */
    async sendMessage() {
        const content = this.elements.messageInput?.value.trim();
        if (!content || this.state.isStreaming) {
            return;
        }

        try {
            // Add user message to UI immediately
            this.addMessage({
                role: 'user',
                content: content,
                timestamp: new Date().toISOString()
            });

            // Clear input
            this.elements.messageInput.value = '';
            this.updateCharCount();
            this.updateSendButton();

            // Show typing indicator
            this.showTypingIndicator();

            // Send message to API
            const response = await fetch(`${this.config.apiEndpoint}/message`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.state.sessionId,
                    message: content,
                    streaming: this.state.settings.enableStreaming
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            // Handle non-streaming response
            if (!this.state.settings.enableStreaming) {
                const data = await response.json();
                this.hideTypingIndicator();
                this.addMessage(data.message);
                
                if (data.model_selection) {
                    this.showModelSelection(data.model_selection);
                }
            }

        } catch (error) {
            console.error('❌ Error sending message:', error);
            this.hideTypingIndicator();
            this.showToast('Failed to send message. Please try again.', 'error');
        }
    }

    /**
     * Send a quick action message
     */
    sendQuickMessage(message) {
        if (this.elements.messageInput) {
            this.elements.messageInput.value = message;
            this.updateCharCount();
            this.updateSendButton();
            this.sendMessage();
        }
    }

    /**
     * Add a message to the chat display
     */
    addMessage(message) {
        if (!this.elements.chatMessages) return;

        const messageElement = this.createMessageElement(message);
        this.elements.chatMessages.appendChild(messageElement);
        
        // Store in history
        this.state.messageHistory.push(message);
        
        // Scroll to bottom
        this.scrollToBottom();
        
        // Announce to screen readers
        this.announceMessage(message);
    }

    /**
     * Create a message DOM element
     */
    createMessageElement(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'flex items-start space-x-3';
        
        // Avatar
        const avatar = document.createElement('div');
        avatar.className = `w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
            message.role === 'user' 
                ? 'bg-indigo-600' 
                : message.role === 'assistant'
                ? 'bg-indigo-100 dark:bg-indigo-900'
                : 'bg-amber-100 dark:bg-amber-900'
        }`;
        
        const icon = document.createElement('i');
        icon.className = `fas ${
            message.role === 'user' 
                ? 'fa-user text-white' 
                : message.role === 'assistant'
                ? 'fa-robot text-indigo-600 dark:text-indigo-400'
                : 'fa-info-circle text-amber-600 dark:text-amber-400'
        } text-sm`;
        icon.setAttribute('aria-hidden', 'true');
        
        avatar.appendChild(icon);
        
        // Message bubble
        const bubble = document.createElement('div');
        bubble.className = `message-bubble ${message.role} max-w-3xl`;
        
        // Message content
        const content = document.createElement('div');
        content.className = 'message-content';
        content.innerHTML = this.formatMessageContent(message.content);
        
        // Message metadata
        const metadata = document.createElement('div');
        metadata.className = 'flex items-center justify-between mt-2 text-xs opacity-75';
        
        const timestamp = document.createElement('span');
        timestamp.textContent = this.formatTimestamp(message.timestamp);
        
        const modelInfo = document.createElement('span');
        if (message.model_used && message.role === 'assistant') {
            modelInfo.textContent = `via ${message.model_used}`;
        }
        
        metadata.appendChild(timestamp);
        if (modelInfo.textContent) {
            metadata.appendChild(modelInfo);
        }
        
        bubble.appendChild(content);
        bubble.appendChild(metadata);
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(bubble);
        
        return messageDiv;
    }

    /**
     * Format message content (basic markdown support)
     */
    formatMessageContent(content) {
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }

    /**
     * Format timestamp for display
     */
    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    /**
     * Handle streaming message chunks
     */
    handleStreamingChunk(data) {
        if (!this.currentStreamingMessage) {
            // Start new streaming message
            this.currentStreamingMessage = {
                role: 'assistant',
                content: '',
                timestamp: new Date().toISOString(),
                model_used: data.model
            };
            
            this.currentStreamingElement = this.createMessageElement(this.currentStreamingMessage);
            this.elements.chatMessages.appendChild(this.currentStreamingElement);
            this.hideTypingIndicator();
        }
        
        // Append chunk to content
        this.currentStreamingMessage.content += data.chunk;
        
        // Update the message element
        const contentElement = this.currentStreamingElement.querySelector('.message-content');
        if (contentElement) {
            contentElement.innerHTML = this.formatMessageContent(this.currentStreamingMessage.content);
        }
        
        // Scroll to bottom
        this.scrollToBottom();
    }

    /**
     * Handle streaming message completion
     */
    handleStreamingComplete(data) {
        if (this.currentStreamingMessage) {
            // Store final message in history
            this.state.messageHistory.push(this.currentStreamingMessage);
            
            // Clean up streaming state
            this.currentStreamingMessage = null;
            this.currentStreamingElement = null;
            this.state.isStreaming = false;
            
            // Show model selection if provided
            if (data.model_selection) {
                this.showModelSelection(data.model_selection);
            }
        }
    }

    /**
     * Handle model selection updates
     */
    handleModelSelection(data) {
        this.state.currentModel = data.selected_model;
        
        // Update model indicator
        if (this.elements.currentModel) {
            this.elements.currentModel.textContent = data.selected_model;
        }
        
        // Show detailed selection if advanced mode is enabled
        if (this.state.settings.showModelDetails && data.selection_details) {
            this.showModelSelection(data.selection_details);
        }
    }

    /**
     * Handle crisis alerts
     */
    handleCrisisAlert(data) {
        console.warn('🚨 Crisis alert:', data);
        
        // Show urgent toast
        this.showToast(
            'Crisis content detected. Professional support resources are available.',
            'warning',
            10000 // Show for 10 seconds
        );
        
        // Additional crisis handling could go here
        // (e.g., show crisis resources, contact emergency services prompt)
    }

    /**
     * Handle session updates
     */
    handleSessionUpdate(data) {
        console.log('📋 Session updated:', data);
        // Handle session state changes
    }

    /**
     * Show model selection panel
     */
    showModelSelection(selectionData) {
        if (!this.elements.modelSelectionPanel || !this.state.settings.showModelDetails) {
            return;
        }
        
        // Populate model comparison
        if (this.elements.modelComparison && selectionData.models) {
            this.elements.modelComparison.innerHTML = '';
            
            selectionData.models.forEach(model => {
                const card = this.createModelCard(model);
                this.elements.modelComparison.appendChild(card);
            });
        }
        
        // Show selection reasoning
        if (this.elements.selectionReasoning && selectionData.reasoning) {
            this.elements.selectionReasoning.textContent = selectionData.reasoning;
        }
        
        // Show panel
        this.elements.modelSelectionPanel.classList.remove('hidden');
    }

    /**
     * Hide model selection panel
     */
    hideModelSelection() {
        if (this.elements.modelSelectionPanel) {
            this.elements.modelSelectionPanel.classList.add('hidden');
        }
    }

    /**
     * Create model comparison card
     */
    createModelCard(modelData) {
        const card = document.createElement('div');
        card.className = `model-card ${modelData.selected ? 'selected' : ''}`;
        
        card.innerHTML = `
            <div class="flex items-center justify-between mb-2">
                <h4 class="font-medium text-gray-900 dark:text-white">${modelData.name}</h4>
                <span class="text-sm font-medium ${modelData.selected ? 'text-indigo-600' : 'text-gray-500'}">
                    ${Math.round(modelData.score * 100)}%
                </span>
            </div>
            <div class="space-y-1">
                <div class="flex justify-between text-xs">
                    <span class="text-gray-500">Empathy</span>
                    <span class="text-gray-700 dark:text-gray-300">${Math.round(modelData.metrics.empathy * 100)}%</span>
                </div>
                <div class="flex justify-between text-xs">
                    <span class="text-gray-500">Safety</span>
                    <span class="text-gray-700 dark:text-gray-300">${Math.round(modelData.metrics.safety * 100)}%</span>
                </div>
                <div class="flex justify-between text-xs">
                    <span class="text-gray-500">Clarity</span>
                    <span class="text-gray-700 dark:text-gray-300">${Math.round(modelData.metrics.clarity * 100)}%</span>
                </div>
            </div>
        `;
        
        return card;
    }

    /**
     * Show/hide typing indicator
     */
    showTypingIndicator(message = 'AI is thinking...') {
        if (this.elements.typingIndicator) {
            if (this.elements.typingText) {
                this.elements.typingText.textContent = message;
            }
            this.elements.typingIndicator.classList.remove('hidden');
            this.state.isStreaming = true;
            this.scrollToBottom();
        }
    }

    hideTypingIndicator() {
        if (this.elements.typingIndicator) {
            this.elements.typingIndicator.classList.add('hidden');
            this.state.isStreaming = false;
        }
    }

    /**
     * Show/hide quick actions
     */
    showQuickActions() {
        if (this.elements.quickActions && this.state.settings.enableQuickActions) {
            this.elements.quickActions.classList.remove('hidden');
        }
    }

    hideQuickActions() {
        if (this.elements.quickActions) {
            this.elements.quickActions.classList.add('hidden');
        }
    }

    /**
     * Toggle advanced mode
     */
    toggleAdvancedMode(enabled) {
        this.state.settings.showModelDetails = enabled;
        localStorage.setItem('chat-model-details', enabled.toString());
        
        if (!enabled) {
            this.hideModelSelection();
        }
    }

    /**
     * Update character count
     */
    updateCharCount() {
        if (this.elements.messageInput && this.elements.charCount) {
            const length = this.elements.messageInput.value.length;
            this.elements.charCount.textContent = length;
            
            // Change color based on limit
            if (length > this.config.maxMessageLength * 0.9) {
                this.elements.charCount.className = 'text-red-500';
            } else if (length > this.config.maxMessageLength * 0.7) {
                this.elements.charCount.className = 'text-amber-500';
            } else {
                this.elements.charCount.className = 'text-gray-400';
            }
        }
    }

    /**
     * Update send button state
     */
    updateSendButton() {
        if (this.elements.sendButton && this.elements.messageInput) {
            const hasContent = this.elements.messageInput.value.trim().length > 0;
            const withinLimit = this.elements.messageInput.value.length <= this.config.maxMessageLength;
            
            this.elements.sendButton.disabled = !hasContent || !withinLimit || this.state.isStreaming;
        }
    }

    /**
     * Auto-resize textarea
     */
    autoResizeTextarea(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
    }

    /**
     * Enable input
     */
    enableInput() {
        if (this.elements.messageInput && this.elements.sendButton) {
            this.elements.messageInput.disabled = false;
            this.elements.messageInput.placeholder = 'Share what\'s on your mind...';
            this.updateSendButton();
        }
    }

    /**
     * Scroll to bottom of messages
     */
    scrollToBottom() {
        if (this.elements.messagesContainer) {
            setTimeout(() => {
                this.elements.messagesContainer.scrollTop = this.elements.messagesContainer.scrollHeight;
            }, 100);
        }
    }

    /**
     * Announce message to screen readers
     */
    announceMessage(message) {
        if (message.role === 'assistant') {
            // Create live region announcement
            const announcement = document.createElement('div');
            announcement.setAttribute('aria-live', 'polite');
            announcement.setAttribute('aria-atomic', 'true');
            announcement.className = 'sr-only';
            announcement.textContent = `Assistant: ${message.content}`;
            
            document.body.appendChild(announcement);
            
            // Remove after announcement
            setTimeout(() => {
                document.body.removeChild(announcement);
            }, 1000);
        }
    }

    /**
     * Theme management
     */
    toggleTheme() {
        const currentTheme = this.state.settings.theme;
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        this.setTheme(newTheme);
    }

    setTheme(theme) {
        this.state.settings.theme = theme;
        localStorage.setItem('chat-theme', theme);
        this.applyTheme(theme);
    }

    applyTheme(theme) {
        if (theme === 'dark') {
            document.documentElement.classList.add('dark');
        } else if (theme === 'light') {
            document.documentElement.classList.remove('dark');
        } else {
            // Auto theme - follow system preference
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            if (prefersDark) {
                document.documentElement.classList.add('dark');
            } else {
                document.documentElement.classList.remove('dark');
            }
        }
    }

    /**
     * Font size management
     */
    applyFontSize(size) {
        document.body.classList.remove('font-large', 'font-extra-large');
        
        if (size === 'large') {
            document.body.classList.add('font-large');
        } else if (size === 'extra-large') {
            document.body.classList.add('font-extra-large');
        }
    }

    /**
     * Settings management
     */
    openSettings() {
        if (this.elements.settingsModal) {
            this.elements.settingsModal.classList.remove('hidden');
            
            // Focus first input for accessibility
            const firstInput = this.elements.settingsModal.querySelector('input, select');
            if (firstInput) {
                firstInput.focus();
            }
        }
    }

    closeSettings() {
        if (this.elements.settingsModal) {
            this.elements.settingsModal.classList.add('hidden');
        }
    }

    saveSettings() {
        // Collect settings from form
        const newSettings = {
            highContrast: this.elements.highContrast?.checked || false,
            reduceMotion: this.elements.reduceMotion?.checked || false,
            fontSize: this.elements.fontSize?.value || 'medium',
            showModelDetails: this.elements.showModelDetails?.checked || false,
            enableQuickActions: this.elements.enableQuickActions?.checked || true,
            enableStreaming: this.elements.enableStreaming?.checked || true
        };

        // Apply settings
        Object.assign(this.state.settings, newSettings);

        // Save to localStorage
        Object.entries(newSettings).forEach(([key, value]) => {
            localStorage.setItem(`chat-${key.replace(/([A-Z])/g, '-$1').toLowerCase()}`, value.toString());
        });

        // Apply changes
        this.applySettings();

        // Show success message
        this.showToast('Settings saved successfully', 'success');

        // Close modal
        this.closeSettings();
    }

    resetSettings() {
        // Reset to defaults
        const defaultSettings = {
            theme: 'auto',
            fontSize: 'medium',
            highContrast: false,
            reduceMotion: false,
            enableQuickActions: true,
            showModelDetails: false,
            enableStreaming: true
        };

        Object.assign(this.state.settings, defaultSettings);

        // Clear localStorage
        Object.keys(defaultSettings).forEach(key => {
            localStorage.removeItem(`chat-${key.replace(/([A-Z])/g, '-$1').toLowerCase()}`);
        });

        // Apply changes
        this.applySettings();

        // Show success message
        this.showToast('Settings reset to defaults', 'success');
    }

    /**
     * Close all modals
     */
    closeModals() {
        this.closeSettings();
        this.hideModelSelection();
    }

    /**
     * Update connection status
     */
    updateConnectionStatus(connected) {
        this.state.isConnected = connected;
        
        if (this.elements.connectionStatus) {
            const statusText = this.elements.connectionStatus.querySelector('span:last-child');
            const statusDot = this.elements.connectionStatus.querySelector('div');
            
            if (statusText) {
                statusText.textContent = connected ? 'Connected' : 'Disconnected';
            }
            
            if (statusDot) {
                statusDot.className = `w-2 h-2 rounded-full mr-1 ${
                    connected ? 'bg-green-500' : 'bg-red-500'
                }`;
            }
        }
    }

    /**
     * Export conversation
     */
    async exportConversation() {
        try {
            const conversation = {
                session_id: this.state.sessionId,
                timestamp: new Date().toISOString(),
                messages: this.state.messageHistory,
                settings: this.state.settings
            };

            const blob = new Blob([JSON.stringify(conversation, null, 2)], {
                type: 'application/json'
            });

            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `mental-health-chat-${new Date().toISOString().slice(0, 10)}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            this.showToast('Conversation exported successfully', 'success');
        } catch (error) {
            console.error('❌ Export failed:', error);
            this.showToast('Failed to export conversation', 'error');
        }
    }

    /**
     * Clear conversation
     */
    async clearConversation() {
        if (!confirm('Are you sure you want to clear this conversation? This action cannot be undone.')) {
            return;
        }

        try {
            // Clear UI
            if (this.elements.chatMessages) {
                this.elements.chatMessages.innerHTML = '';
            }

            // Clear state
            this.state.messageHistory = [];

            // Clear session on server
            if (this.state.sessionId) {
                await fetch(`${this.config.apiEndpoint}/session/${this.state.sessionId}`, {
                    method: 'DELETE'
                });
            }

            // Initialize new session
            await this.initializeSession();

            this.showToast('Conversation cleared', 'success');
        } catch (error) {
            console.error('❌ Failed to clear conversation:', error);
            this.showToast('Failed to clear conversation', 'error');
        }
    }

    /**
     * Download user data
     */
    async downloadUserData() {
        try {
            const response = await fetch(`${this.config.apiEndpoint}/user/data?user_id=${this.getUserId()}`);
            
            if (!response.ok) {
                throw new Error('Failed to download user data');
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `user-data-${new Date().toISOString().slice(0, 10)}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            this.showToast('User data downloaded', 'success');
        } catch (error) {
            console.error('❌ Download failed:', error);
            this.showToast('Failed to download user data', 'error');
        }
    }

    /**
     * Show toast notification
     */
    showToast(message, type = 'info', duration = 5000) {
        if (!this.elements.toastContainer) return;

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <div class="flex items-start">
                <div class="flex-shrink-0">
                    <i class="fas ${this.getToastIcon(type)} w-5 h-5" aria-hidden="true"></i>
                </div>
                <div class="ml-3 flex-1">
                    <p class="text-sm font-medium text-gray-900 dark:text-white">${message}</p>
                </div>
                <div class="ml-4 flex-shrink-0 flex">
                    <button class="inline-flex text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 focus:outline-none">
                        <i class="fas fa-times w-4 h-4" aria-hidden="true"></i>
                    </button>
                </div>
            </div>
        `;

        // Add close functionality
        const closeBtn = toast.querySelector('button');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                this.removeToast(toast);
            });
        }

        this.elements.toastContainer.appendChild(toast);

        // Auto-remove after duration
        setTimeout(() => {
            this.removeToast(toast);
        }, duration);
    }

    /**
     * Remove toast notification
     */
    removeToast(toast) {
        if (toast && toast.parentNode) {
            toast.style.transform = 'translateX(100%)';
            toast.style.opacity = '0';
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }
    }

    /**
     * Get toast icon for type
     */
    getToastIcon(type) {
        const icons = {
            success: 'fa-check-circle text-green-500',
            error: 'fa-exclamation-circle text-red-500',
            warning: 'fa-exclamation-triangle text-amber-500',
            info: 'fa-info-circle text-indigo-500'
        };
        return icons[type] || icons.info;
    }

    /**
     * Get user ID (in production, this would be from authentication)
     */
    getUserId() {
        // For demo purposes, generate or retrieve from localStorage
        let userId = localStorage.getItem('chat-user-id');
        if (!userId) {
            userId = 'user_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('chat-user-id', userId);
        }
        return userId;
    }

    /**
     * Cleanup resources
     */
    destroy() {
        // Close WebSocket
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        // Clear handlers
        this.handlers.clear();

        // Clear state
        this.state = null;

        console.log('🧹 Mental Health Chat System cleaned up');
    }
}

// Initialize chat system when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Get configuration from global variable set by the server
    const config = window.chatConfig || {};
    
    // Initialize chat system
    window.mentalHealthChat = new MentalHealthChat(config);
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (window.mentalHealthChat) {
        window.mentalHealthChat.destroy();
    }
});