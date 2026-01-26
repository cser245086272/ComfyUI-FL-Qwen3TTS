export const TRAINING_WIDGET_STYLES = `
  .qwen3-training-widget {
    background: #1a1a2e;
    border-radius: 8px;
    padding: 16px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #e0e0e0;
    box-sizing: border-box;
  }

  .qwen3-training-widget * {
    box-sizing: border-box;
  }

  .training-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 16px;
  }

  .status-panel {
    display: flex;
    flex-direction: column;
    gap: 4px;
    flex: 1;
  }

  .epoch-display, .loss-display {
    font-size: 14px;
    color: #a0a0a0;
  }

  .epoch-display span, .loss-display span {
    color: #E93D82;
    font-weight: bold;
    font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
  }

  .status-text {
    font-size: 12px;
    color: #808080;
    margin-top: 4px;
  }

  .stop-btn {
    padding: 8px 16px;
    background: transparent;
    border: 1px solid #E93D82;
    color: #E93D82;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 12px;
    font-weight: 500;
    flex-shrink: 0;
  }

  .stop-btn:hover:not(:disabled) {
    background: rgba(233, 61, 130, 0.2);
  }

  .stop-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    border-color: #666;
    color: #666;
  }

  .progress-container {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
  }

  .progress-bar {
    flex: 1;
    height: 8px;
    background: #2a2a4e;
    border-radius: 4px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #E93D82, #ff6b9d);
    transition: width 0.3s ease;
    border-radius: 4px;
  }

  .progress-text {
    font-size: 12px;
    color: #a0a0a0;
    min-width: 40px;
    text-align: right;
    font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
  }

  .validation-section {
    border-top: 1px solid #2a2a4e;
    padding-top: 12px;
  }

  .section-title {
    font-size: 11px;
    color: #808080;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .audio-timeline {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 12px;
    min-height: 32px;
  }

  .audio-timeline:empty::before {
    content: 'Validation samples will appear here...';
    color: #555;
    font-size: 12px;
    font-style: italic;
  }

  .timeline-chip {
    padding: 6px 12px;
    background: #2a2a4e;
    border: 1px solid #3a3a5e;
    color: #e0e0e0;
    border-radius: 16px;
    cursor: pointer;
    font-size: 12px;
    transition: all 0.2s;
    font-family: inherit;
  }

  .timeline-chip:hover {
    background: #3a3a5e;
    border-color: #E93D82;
  }

  .timeline-chip.active {
    background: #E93D82;
    border-color: #E93D82;
    color: white;
  }

  .audio-player-container {
    margin-top: 8px;
  }

  .audio-player {
    width: 100%;
    height: 36px;
    border-radius: 4px;
    outline: none;
  }

  .audio-player::-webkit-media-controls-panel {
    background: #2a2a4e;
  }

  .no-audio-message {
    color: #555;
    font-size: 12px;
    font-style: italic;
    text-align: center;
    padding: 8px;
  }

  /* Animations */
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .training-active .status-text {
    animation: pulse 2s ease-in-out infinite;
  }
`

export function injectStyles(): void {
  // Prevent duplicate injection
  if (document.getElementById('qwen3-training-widget-styles')) {
    return
  }

  const style = document.createElement('style')
  style.id = 'qwen3-training-widget-styles'
  style.textContent = TRAINING_WIDGET_STYLES
  document.head.appendChild(style)
}
