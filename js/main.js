var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
const TRAINING_WIDGET_STYLES = `
  .qwen3-training-widget {
    --primary: #8b5cf6;
    --primary-glow: rgba(139, 92, 246, 0.4);
    --secondary: #06b6d4;
    --success: #22c55e;
    --danger: #ef4444;
    --warning: #f59e0b;
    --bg-dark: #0f0f12;
    --bg-card: #18181b;
    --bg-elevated: #1f1f23;
    --border: #27272a;
    --border-hover: #3f3f46;
    --text-primary: #fafafa;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;

    background: var(--bg-card);
    border-radius: 16px;
    border: 1px solid var(--border);
    overflow: hidden;
    position: relative;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
    box-sizing: border-box;
    height: 100%;
    min-height: 420px;
    display: flex;
    flex-direction: column;
  }

  .qwen3-training-widget * {
    box-sizing: border-box;
  }

  /* Ambient glow effect */
  .qwen3-training-widget::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(
      circle at 30% 20%,
      rgba(139, 92, 246, 0.08) 0%,
      transparent 50%
    );
    pointer-events: none;
    animation: ambientShift 10s ease-in-out infinite;
  }

  @keyframes ambientShift {
    0%, 100% { transform: translate(0, 0); }
    50% { transform: translate(5%, 5%); }
  }

  /* Particles Background */
  .particles {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    overflow: hidden;
    pointer-events: none;
    z-index: 0;
  }

  .particle {
    position: absolute;
    width: 2px;
    height: 2px;
    background: var(--primary);
    border-radius: 50%;
    opacity: 0.3;
    animation: float 15s linear infinite;
  }

  @keyframes float {
    0% {
      transform: translateY(100%) translateX(0);
      opacity: 0;
    }
    10% { opacity: 0.3; }
    90% { opacity: 0.3; }
    100% {
      transform: translateY(-100%) translateX(50px);
      opacity: 0;
    }
  }

  .particle:nth-child(1) { left: 10%; animation-delay: 0s; animation-duration: 12s; }
  .particle:nth-child(2) { left: 20%; animation-delay: 2s; animation-duration: 14s; }
  .particle:nth-child(3) { left: 35%; animation-delay: 4s; animation-duration: 11s; }
  .particle:nth-child(4) { left: 50%; animation-delay: 1s; animation-duration: 13s; }
  .particle:nth-child(5) { left: 65%; animation-delay: 3s; animation-duration: 15s; }
  .particle:nth-child(6) { left: 80%; animation-delay: 5s; animation-duration: 12s; }
  .particle:nth-child(7) { left: 90%; animation-delay: 2.5s; animation-duration: 14s; }

  /* Top Bar */
  .top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    background: var(--bg-elevated);
    border-bottom: 1px solid var(--border);
    position: relative;
    z-index: 1;
    gap: 10px;
  }

  .breadcrumb {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: var(--text-muted);
    flex: 1;
  }

  .breadcrumb strong {
    color: var(--text-primary);
    font-weight: 500;
  }

  .top-bar-controls {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .stop-btn-top {
    padding: 4px 10px;
    background: transparent;
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 6px;
    color: var(--danger);
    font-size: 10px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
  }

  .stop-btn-top:hover:not(:disabled) {
    border-color: var(--danger);
    background: rgba(239, 68, 68, 0.1);
  }

  .stop-btn-top:disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }

  .live-indicator {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 3px 8px;
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.2);
    border-radius: 12px;
    animation: glowPulse 2s ease-in-out infinite;
  }

  .live-indicator.idle {
    background: rgba(113, 113, 122, 0.1);
    border-color: rgba(113, 113, 122, 0.2);
    animation: none;
  }

  @keyframes glowPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.2); }
    50% { box-shadow: 0 0 12px 2px rgba(34, 197, 94, 0.15); }
  }

  .live-dot {
    width: 6px;
    height: 6px;
    background: var(--success);
    border-radius: 50%;
    position: relative;
  }

  .live-indicator.idle .live-dot {
    background: var(--text-muted);
  }

  .live-dot::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 100%;
    height: 100%;
    background: var(--success);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    animation: ping 1.5s cubic-bezier(0, 0, 0.2, 1) infinite;
  }

  .live-indicator.idle .live-dot::after {
    animation: none;
    opacity: 0;
  }

  @keyframes ping {
    0% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
    75%, 100% { transform: translate(-50%, -50%) scale(2.5); opacity: 0; }
  }

  .live-text {
    font-size: 9px;
    font-weight: 600;
    color: var(--success);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .live-indicator.idle .live-text {
    color: var(--text-muted);
  }

  /* Main Content */
  .main-content {
    padding: 10px 12px;
    position: relative;
    z-index: 1;
    flex: 1;
    min-height: 0;
    overflow-y: auto;
  }

  /* KPI Row */
  .kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 6px;
    margin-bottom: 10px;
  }

  .kpi-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 8px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  }

  .kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--accent, var(--primary));
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.3s;
  }

  .kpi-card:hover {
    border-color: var(--border-hover);
    transform: translateY(-2px);
  }

  .kpi-card:hover::before {
    transform: scaleX(1);
  }

  .kpi-card:nth-child(1) { --accent: var(--primary); }
  .kpi-card:nth-child(2) { --accent: var(--secondary); }
  .kpi-card:nth-child(3) { --accent: var(--warning); }
  .kpi-card:nth-child(4) { --accent: var(--success); }

  .kpi-label {
    font-size: 8px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 2px;
  }

  .kpi-value {
    font-size: 13px;
    font-weight: 700;
    color: var(--text-primary);
    font-variant-numeric: tabular-nums;
    line-height: 1;
  }

  .kpi-trend {
    display: inline-flex;
    align-items: center;
    gap: 2px;
    margin-top: 2px;
    font-size: 8px;
    font-weight: 500;
    padding: 1px 4px;
    border-radius: 3px;
  }

  .kpi-trend.down {
    color: var(--success);
    background: rgba(34, 197, 94, 0.1);
  }

  .kpi-trend.up {
    color: var(--danger);
    background: rgba(239, 68, 68, 0.1);
  }

  .kpi-trend svg {
    width: 8px;
    height: 8px;
  }

  /* Progress Section */
  .progress-section {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
    margin-bottom: 10px;
  }

  .progress-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }

  .progress-title {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .progress-meta {
    font-size: 10px;
    color: var(--primary);
    font-weight: 500;
  }

  .progress-bars {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .progress-item {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .progress-label {
    font-size: 10px;
    color: var(--text-muted);
    min-width: 40px;
  }

  .progress-track {
    flex: 1;
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
    position: relative;
  }

  .progress-fill {
    height: 100%;
    border-radius: 3px;
    position: relative;
    transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
  }

  .progress-fill.epoch {
    background: linear-gradient(90deg, var(--primary), #a78bfa);
  }

  .progress-fill.step {
    background: linear-gradient(90deg, var(--secondary), #22d3ee);
  }

  /* Animated shimmer effect */
  .training-active .progress-fill::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(
      90deg,
      transparent 0%,
      rgba(255, 255, 255, 0.3) 50%,
      transparent 100%
    );
    animation: shimmer 2s infinite;
  }

  @keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
  }

  .progress-value {
    font-size: 10px;
    color: var(--text-secondary);
    min-width: 50px;
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-weight: 500;
  }

  /* Loss Chart */
  .chart-section {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
    margin-bottom: 10px;
  }

  .chart-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }

  .chart-title {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .chart-legend {
    display: flex;
    gap: 10px;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 9px;
    color: var(--text-muted);
  }

  .legend-dot {
    width: 6px;
    height: 6px;
    border-radius: 2px;
  }

  .legend-dot.main { background: var(--primary); }

  .chart-container {
    position: relative;
    height: 70px;
    margin-bottom: 4px;
  }

  .y-axis {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 16px;
    width: 26px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    font-size: 8px;
    color: var(--text-muted);
    text-align: right;
    padding-right: 6px;
  }

  .chart-area {
    position: absolute;
    left: 30px;
    right: 0;
    top: 0;
    bottom: 16px;
    display: flex;
    align-items: flex-end;
    gap: 3px;
    border-left: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    padding: 0 3px 0 6px;
  }

  .chart-bar {
    flex: 1;
    border-radius: 3px 3px 0 0;
    position: relative;
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
    background: linear-gradient(to top, var(--primary), rgba(139, 92, 246, 0.6));
    min-width: 8px;
    max-width: 24px;
  }

  .chart-bar:hover {
    filter: brightness(1.2);
    transform: scaleY(1.02);
    transform-origin: bottom;
  }

  .chart-bar::after {
    content: attr(data-value);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%) translateY(-4px);
    background: var(--bg-dark);
    color: var(--text-primary);
    font-size: 10px;
    padding: 4px 8px;
    border-radius: 4px;
    white-space: nowrap;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s;
    border: 1px solid var(--border);
  }

  .chart-bar:hover::after {
    opacity: 1;
  }

  .chart-labels {
    position: absolute;
    left: 30px;
    right: 0;
    bottom: 0;
    display: flex;
    justify-content: space-around;
    padding: 2px 3px 0 6px;
  }

  .chart-label {
    font-size: 8px;
    color: var(--text-muted);
    min-width: 6px;
    max-width: 20px;
    text-align: center;
    flex: 1;
  }

  .chart-empty {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 60px;
    color: var(--text-muted);
    font-size: 11px;
    font-style: italic;
  }

  /* Validation Section */
  .validation-section {
    margin-bottom: 10px;
  }

  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
  }

  .section-title {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .validation-row {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    min-height: 24px;
  }

  .validation-row:empty::before {
    content: 'Samples appear here...';
    color: var(--text-muted);
    font-size: 10px;
    font-style: italic;
  }

  .epoch-btn {
    padding: 4px 10px;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }

  .epoch-btn:hover {
    border-color: var(--border-hover);
    color: var(--text-primary);
  }

  .epoch-btn.active {
    background: var(--primary);
    border-color: var(--primary);
    color: #fff;
  }

  .epoch-btn.new {
    animation: pulse 2s ease-in-out;
  }

  @keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(139, 92, 246, 0); }
    50% { box-shadow: 0 0 8px 2px rgba(139, 92, 246, 0.4); }
  }

  /* Player Bar */
  .player-bar {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 10px;
    display: flex;
    align-items: center;
    gap: 10px;
    position: relative;
    overflow: hidden;
  }

  .player-bar.hidden {
    display: none;
  }

  /* Animated gradient border on player */
  .player-bar::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--primary), var(--secondary), var(--primary));
    background-size: 200% 100%;
    animation: gradientMove 3s linear infinite;
  }

  @keyframes gradientMove {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
  }

  .player-btn {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, var(--primary), #a78bfa);
    border: none;
    border-radius: 50%;
    color: #fff;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    flex-shrink: 0;
  }

  .player-btn::before {
    content: '';
    position: absolute;
    inset: -3px;
    background: linear-gradient(135deg, var(--primary), #a78bfa);
    border-radius: 50%;
    opacity: 0;
    z-index: -1;
    transition: opacity 0.3s;
    filter: blur(8px);
  }

  .player-btn:hover {
    transform: scale(1.05);
  }

  .player-btn:hover::before {
    opacity: 0.5;
  }

  .player-btn:active {
    transform: scale(0.95);
  }

  .player-btn svg {
    width: 14px;
    height: 14px;
  }

  .player-info {
    flex: 1;
    min-width: 0;
  }

  .player-title {
    font-size: 11px;
    font-weight: 500;
    color: var(--text-primary);
    margin-bottom: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .player-progress {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .player-track {
    flex: 1;
    height: 5px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
    cursor: pointer;
    position: relative;
  }

  .player-fill {
    width: 0%;
    height: 100%;
    background: linear-gradient(90deg, var(--primary), #a78bfa);
    border-radius: 2px;
    position: relative;
    transition: width 0.1s;
  }

  .player-fill::after {
    content: '';
    position: absolute;
    right: 0;
    top: 50%;
    transform: translateY(-50%);
    width: 10px;
    height: 10px;
    background: #fff;
    border-radius: 50%;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
    opacity: 0;
    transition: opacity 0.2s;
  }

  .player-track:hover .player-fill::after {
    opacity: 1;
  }

  .player-time {
    font-size: 10px;
    color: var(--text-muted);
    font-variant-numeric: tabular-nums;
    min-width: 60px;
    text-align: right;
  }

  /* Hidden audio element */
  .audio-element {
    display: none;
  }

  /* Action Section - Hidden, stop button moved to top bar */
  .action-section {
    display: none;
  }

  /* Status Bar */
  .status-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: var(--bg-elevated);
    border-top: 1px solid var(--border);
    font-size: 10px;
    color: var(--text-muted);
  }

  .status-spinner {
    width: 14px;
    height: 14px;
    border: 2px solid var(--border);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  .status-spinner.hidden {
    display: none;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .status-text {
    flex: 1;
  }

  .status-time {
    font-variant-numeric: tabular-nums;
    color: var(--text-secondary);
  }
`;
function injectStyles() {
  if (document.getElementById("qwen3-training-widget-styles")) {
    return;
  }
  const style = document.createElement("style");
  style.id = "qwen3-training-widget-styles";
  style.textContent = TRAINING_WIDGET_STYLES;
  document.head.appendChild(style);
}
class TrainingWidget {
  constructor(options) {
    __publicField(this, "container");
    __publicField(this, "widgetRoot");
    // KPI elements
    __publicField(this, "kpiEpoch");
    __publicField(this, "kpiLoss");
    __publicField(this, "kpiSteps");
    __publicField(this, "kpiEta");
    __publicField(this, "lossTrend");
    // Progress elements
    __publicField(this, "progressMeta");
    __publicField(this, "epochFill");
    __publicField(this, "epochValue");
    __publicField(this, "stepFill");
    __publicField(this, "stepValue");
    // Chart elements
    __publicField(this, "chartArea");
    __publicField(this, "chartLabels");
    __publicField(this, "yAxisMax");
    __publicField(this, "yAxisMid");
    // Validation elements
    __publicField(this, "validationRow");
    __publicField(this, "playerBar");
    __publicField(this, "playerBtn");
    __publicField(this, "playerTitle");
    __publicField(this, "playerFill");
    __publicField(this, "playerTime");
    __publicField(this, "audioElement");
    // Status elements
    __publicField(this, "liveIndicator");
    __publicField(this, "liveText");
    __publicField(this, "stopButton");
    __publicField(this, "statusSpinner");
    __publicField(this, "statusText");
    __publicField(this, "statusTime");
    // State
    __publicField(this, "samples", []);
    __publicField(this, "lossHistory", []);
    __publicField(this, "activeCard", null);
    __publicField(this, "currentSample", null);
    __publicField(this, "isPlaying", false);
    __publicField(this, "startTime", null);
    __publicField(this, "timerInterval", null);
    __publicField(this, "previousLoss", null);
    injectStyles();
    this.container = options.container;
    this.createDOM();
    this.bindEvents();
  }
  createDOM() {
    this.container.innerHTML = `
      <div class="qwen3-training-widget">
        <div class="particles">
          <div class="particle"></div>
          <div class="particle"></div>
          <div class="particle"></div>
          <div class="particle"></div>
          <div class="particle"></div>
          <div class="particle"></div>
          <div class="particle"></div>
        </div>

        <div class="top-bar">
          <div class="breadcrumb">
            Training / <strong>Voice Model</strong>
          </div>
          <div class="top-bar-controls">
            <button class="stop-btn-top" disabled>Stop</button>
            <div class="live-indicator idle">
              <div class="live-dot"></div>
              <span class="live-text">Idle</span>
            </div>
          </div>
        </div>

        <div class="main-content">
          <div class="kpi-row">
            <div class="kpi-card">
              <div class="kpi-label">Epoch</div>
              <div class="kpi-value kpi-epoch">0/0</div>
            </div>
            <div class="kpi-card">
              <div class="kpi-label">Loss</div>
              <div class="kpi-value kpi-loss">--</div>
              <div class="kpi-trend down" style="display: none;">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                  <path d="M6 9l6 6 6-6"/>
                </svg>
                <span class="trend-value">0%</span>
              </div>
            </div>
            <div class="kpi-card">
              <div class="kpi-label">Steps</div>
              <div class="kpi-value kpi-steps">0</div>
            </div>
            <div class="kpi-card">
              <div class="kpi-label">Elapsed</div>
              <div class="kpi-value kpi-eta">--</div>
            </div>
          </div>

          <div class="progress-section">
            <div class="progress-header">
              <span class="progress-title">Training Progress</span>
              <span class="progress-meta">0% complete</span>
            </div>
            <div class="progress-bars">
              <div class="progress-item">
                <span class="progress-label">Epoch</span>
                <div class="progress-track">
                  <div class="progress-fill epoch" style="width: 0%"></div>
                </div>
                <span class="progress-value epoch-value">0 / 0</span>
              </div>
              <div class="progress-item">
                <span class="progress-label">Step</span>
                <div class="progress-track">
                  <div class="progress-fill step" style="width: 0%"></div>
                </div>
                <span class="progress-value step-value">0 / 0</span>
              </div>
            </div>
          </div>

          <div class="chart-section">
            <div class="chart-header">
              <span class="chart-title">Loss Over Epochs</span>
              <div class="chart-legend">
                <div class="legend-item">
                  <div class="legend-dot main"></div>
                  Loss
                </div>
              </div>
            </div>
            <div class="chart-container">
              <div class="y-axis">
                <span class="y-max">0.10</span>
                <span class="y-mid">0.05</span>
                <span>0.00</span>
              </div>
              <div class="chart-area"></div>
              <div class="chart-labels"></div>
            </div>
          </div>

          <div class="validation-section">
            <div class="section-header">
              <span class="section-title">Validation Samples</span>
            </div>
            <div class="validation-row"></div>
          </div>

          <div class="player-bar hidden">
            <button class="player-btn">
              <svg class="play-icon" viewBox="0 0 24 24" fill="currentColor">
                <path d="M8 5v14l11-7z"/>
              </svg>
              <svg class="pause-icon" viewBox="0 0 24 24" fill="currentColor" style="display: none;">
                <rect x="6" y="4" width="4" height="16"/>
                <rect x="14" y="4" width="4" height="16"/>
              </svg>
            </button>
            <div class="player-info">
              <div class="player-title">No sample selected</div>
              <div class="player-progress">
                <div class="player-track">
                  <div class="player-fill"></div>
                </div>
                <span class="player-time">0:00 / 0:00</span>
              </div>
            </div>
          </div>

          <audio class="audio-element"></audio>

          <div class="action-section">
            <button class="stop-btn" disabled>Stop Training</button>
          </div>
        </div>

        <div class="status-bar">
          <div class="status-spinner hidden"></div>
          <span class="status-text">Ready to train</span>
          <span class="status-time"></span>
        </div>
      </div>
    `;
    // Cache element references
    this.widgetRoot = this.container.querySelector(".qwen3-training-widget");
    // KPI elements
    this.kpiEpoch = this.container.querySelector(".kpi-epoch");
    this.kpiLoss = this.container.querySelector(".kpi-loss");
    this.kpiSteps = this.container.querySelector(".kpi-steps");
    this.kpiEta = this.container.querySelector(".kpi-eta");
    this.lossTrend = this.container.querySelector(".kpi-trend");
    // Progress elements
    this.progressMeta = this.container.querySelector(".progress-meta");
    this.epochFill = this.container.querySelector(".progress-fill.epoch");
    this.epochValue = this.container.querySelector(".epoch-value");
    this.stepFill = this.container.querySelector(".progress-fill.step");
    this.stepValue = this.container.querySelector(".step-value");
    // Chart elements
    this.chartArea = this.container.querySelector(".chart-area");
    this.chartLabels = this.container.querySelector(".chart-labels");
    this.yAxisMax = this.container.querySelector(".y-max");
    this.yAxisMid = this.container.querySelector(".y-mid");
    // Validation elements
    this.validationRow = this.container.querySelector(".validation-row");
    this.playerBar = this.container.querySelector(".player-bar");
    this.playerBtn = this.container.querySelector(".player-btn");
    this.playerTitle = this.container.querySelector(".player-title");
    this.playerFill = this.container.querySelector(".player-fill");
    this.playerTime = this.container.querySelector(".player-time");
    this.audioElement = this.container.querySelector(".audio-element");
    // Status elements
    this.liveIndicator = this.container.querySelector(".live-indicator");
    this.liveText = this.container.querySelector(".live-text");
    this.stopButton = this.container.querySelector(".stop-btn-top");
    this.statusSpinner = this.container.querySelector(".status-spinner");
    this.statusText = this.container.querySelector(".status-text");
    this.statusTime = this.container.querySelector(".status-time");
  }
  bindEvents() {
    this.stopButton.addEventListener("click", () => {
      this.requestStop();
    });
    this.playerBtn.addEventListener("click", () => {
      this.togglePlayPause();
    });
    this.audioElement.addEventListener("timeupdate", () => {
      this.updatePlayerProgress();
    });
    this.audioElement.addEventListener("ended", () => {
      this.onAudioEnded();
    });
    this.audioElement.addEventListener("loadedmetadata", () => {
      this.updatePlayerTime();
    });
    // Click on progress track to seek
    const playerTrack = this.container.querySelector(".player-track");
    playerTrack.addEventListener("click", (e) => {
      this.seekAudio(e);
    });
  }
  requestStop() {
    this.statusText.textContent = "Stopping...";
    this.stopButton.disabled = true;

    // Call ComfyUI's interrupt endpoint to stop training
    api.fetchApi("/interrupt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    }).catch(err => {
      console.warn("Failed to send interrupt:", err);
    });
  }
  formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }
  formatElapsed(ms) {
    const totalSeconds = Math.floor(ms / 1000);
    const mins = Math.floor(totalSeconds / 60);
    const secs = totalSeconds % 60;
    if (mins >= 60) {
      const hours = Math.floor(mins / 60);
      const remainingMins = mins % 60;
      return `${hours}h${remainingMins}m`;
    }
    return `${mins}m${secs}s`;
  }
  startTimer() {
    if (!this.startTime) {
      this.startTime = Date.now();
    }
    if (this.timerInterval) {
      clearInterval(this.timerInterval);
    }
    this.timerInterval = setInterval(() => {
      const elapsed = Date.now() - this.startTime;
      this.kpiEta.textContent = this.formatElapsed(elapsed);
      this.statusTime.textContent = this.formatElapsed(elapsed) + " elapsed";
    }, 1000);
  }
  stopTimer() {
    if (this.timerInterval) {
      clearInterval(this.timerInterval);
      this.timerInterval = null;
    }
  }
  updateProgress(epoch, totalEpochs, loss, step, totalSteps) {
    // Start timer on first update
    if (!this.startTime) {
      this.startTimer();
    }
    const overallProgress = (epoch - 1 + step / totalSteps) / totalEpochs * 100;
    const epochProgress = (epoch / totalEpochs) * 100;
    const stepProgress = (step / totalSteps) * 100;
    // Update KPIs
    this.kpiEpoch.textContent = `${epoch}/${totalEpochs}`;
    this.kpiLoss.textContent = loss.toFixed(4);
    this.kpiSteps.textContent = ((epoch - 1) * totalSteps + step).toString();
    // Update loss trend
    if (this.previousLoss !== null && this.previousLoss > 0) {
      const change = ((loss - this.previousLoss) / this.previousLoss) * 100;
      if (Math.abs(change) > 0.1) {
        this.lossTrend.style.display = "inline-flex";
        const trendValue = this.lossTrend.querySelector(".trend-value");
        const isDown = change < 0;
        this.lossTrend.className = `kpi-trend ${isDown ? 'down' : 'up'}`;
        trendValue.textContent = `${Math.abs(change).toFixed(1)}%`;
        // Update arrow direction
        const svg = this.lossTrend.querySelector("svg");
        svg.innerHTML = isDown
          ? '<path d="M6 9l6 6 6-6"/>'
          : '<path d="M6 15l6-6 6 6"/>';
      }
    }
    this.previousLoss = loss;
    // Update progress section
    this.progressMeta.textContent = `${Math.round(overallProgress)}% complete`;
    this.epochFill.style.width = `${epochProgress}%`;
    this.epochValue.textContent = `${epoch} / ${totalEpochs}`;
    this.stepFill.style.width = `${stepProgress}%`;
    this.stepValue.textContent = `${step} / ${totalSteps}`;
    // Update status
    this.statusText.textContent = `Training epoch ${epoch}, step ${step}/${totalSteps}...`;
    this.statusSpinner.classList.remove("hidden");
    this.stopButton.disabled = false;
    // Update live indicator
    this.liveIndicator.classList.remove("idle");
    this.liveText.textContent = "Live";
    this.widgetRoot.classList.add("training-active");
  }
  updateChart(epoch, loss) {
    // Add to loss history
    this.lossHistory.push({ epoch, loss });
    // Keep last 10 epochs
    if (this.lossHistory.length > 10) {
      this.lossHistory.shift();
    }
    // Calculate max for y-axis
    const maxLoss = Math.max(...this.lossHistory.map(h => h.loss), 0.1);
    const yMax = Math.ceil(maxLoss * 100) / 100;
    this.yAxisMax.textContent = yMax.toFixed(2);
    this.yAxisMid.textContent = (yMax / 2).toFixed(2);
    // Clear and rebuild chart
    this.chartArea.innerHTML = '';
    this.chartLabels.innerHTML = '';
    this.lossHistory.forEach(h => {
      const heightPercent = (h.loss / yMax) * 100;
      const bar = document.createElement("div");
      bar.className = "chart-bar";
      bar.style.height = `${Math.max(heightPercent, 5)}%`;
      bar.setAttribute("data-value", h.loss.toFixed(4));
      this.chartArea.appendChild(bar);
      const label = document.createElement("span");
      label.className = "chart-label";
      label.textContent = `E${h.epoch}`;
      this.chartLabels.appendChild(label);
    });
  }
  addValidationSample(epoch, audioBase64, checkpointPath) {
    const sample = { epoch, audioBase64, checkpointPath };
    this.samples.push(sample);
    // Update chart with current loss if available
    if (this.previousLoss !== null) {
      this.updateChart(epoch, this.previousLoss);
    }
    // Create simple epoch button
    const btn = document.createElement("button");
    btn.className = "epoch-btn new";
    btn.textContent = `E${epoch}`;
    btn.title = `Epoch ${epoch} - ${checkpointPath}`;
    btn.addEventListener("click", () => this.selectSample(sample, btn));
    this.validationRow.appendChild(btn);
    // Remove 'new' animation class after it plays
    setTimeout(() => btn.classList.remove("new"), 2000);
    // Auto-select the new sample
    this.selectSample(sample, btn);
  }
  selectSample(sample, card) {
    // Deselect previous card
    if (this.activeCard) {
      this.activeCard.classList.remove("active");
    }
    // Select new card
    card.classList.add("active");
    this.activeCard = card;
    this.currentSample = sample;
    // Show player bar
    this.playerBar.classList.remove("hidden");
    this.playerTitle.textContent = `Epoch ${sample.epoch} - Validation Sample`;
    // Remember if we were playing before switching
    const wasPlaying = this.isPlaying;
    // Stop current playback
    this.audioElement.pause();
    this.isPlaying = false;
    // Load new audio
    this.audioElement.src = sample.audioBase64;
    this.playerFill.style.width = "0%";
    this.updatePlayerTime();
    // If user was already playing, continue playing the new sample
    if (wasPlaying) {
      this.playAudio();
    } else {
      this.updatePlayPauseIcon();
    }
  }
  playAudio() {
    this.audioElement.play().then(() => {
      this.isPlaying = true;
      this.updatePlayPauseIcon();
    }).catch((e) => {
      console.warn("Audio autoplay prevented:", e);
    });
  }
  togglePlayPause() {
    if (this.isPlaying) {
      this.audioElement.pause();
      this.isPlaying = false;
    } else {
      this.audioElement.play().catch(console.warn);
      this.isPlaying = true;
    }
    this.updatePlayPauseIcon();
  }
  updatePlayPauseIcon() {
    const playIcon = this.playerBtn.querySelector(".play-icon");
    const pauseIcon = this.playerBtn.querySelector(".pause-icon");
    if (this.isPlaying) {
      playIcon.style.display = "none";
      pauseIcon.style.display = "block";
    } else {
      playIcon.style.display = "block";
      pauseIcon.style.display = "none";
    }
  }
  updatePlayerProgress() {
    if (this.audioElement.duration) {
      const progress = (this.audioElement.currentTime / this.audioElement.duration) * 100;
      this.playerFill.style.width = `${progress}%`;
      this.updatePlayerTime();
    }
  }
  updatePlayerTime() {
    const current = this.formatTime(this.audioElement.currentTime || 0);
    const duration = this.formatTime(this.audioElement.duration || 0);
    this.playerTime.textContent = `${current} / ${duration}`;
  }
  onAudioEnded() {
    this.isPlaying = false;
    this.updatePlayPauseIcon();
    this.playerFill.style.width = "100%";
  }
  seekAudio(e) {
    const track = e.currentTarget;
    const rect = track.getBoundingClientRect();
    const percent = (e.clientX - rect.left) / rect.width;
    if (this.audioElement.duration) {
      this.audioElement.currentTime = percent * this.audioElement.duration;
    }
  }
  updateStatus(message) {
    this.statusText.textContent = message;
    if (message.toLowerCase().includes("complete") || message.toLowerCase().includes("stopped")) {
      this.widgetRoot.classList.remove("training-active");
      this.stopButton.disabled = true;
      this.liveIndicator.classList.add("idle");
      this.liveText.textContent = "Done";
      this.statusSpinner.classList.add("hidden");
      this.stopTimer();
      // Set progress to 100%
      this.progressMeta.textContent = "100% complete";
      this.epochFill.style.width = "100%";
      this.stepFill.style.width = "100%";
    }
  }
  onTrainingComplete(checkpointPath) {
    if (checkpointPath) {
      this.statusText.textContent = `Complete! Saved to: ${checkpointPath}`;
    } else {
      this.statusText.textContent = "Training complete!";
    }
    this.widgetRoot.classList.remove("training-active");
    this.stopButton.disabled = true;
    this.liveIndicator.classList.add("idle");
    this.liveText.textContent = "Done";
    this.statusSpinner.classList.add("hidden");
    this.stopTimer();
  }
  reset() {
    // Reset state
    this.samples = [];
    this.lossHistory = [];
    this.activeCard = null;
    this.currentSample = null;
    this.isPlaying = false;
    this.startTime = null;
    this.previousLoss = null;
    this.stopTimer();
    // Reset KPIs
    this.kpiEpoch.textContent = "0/0";
    this.kpiLoss.textContent = "--";
    this.kpiSteps.textContent = "0";
    this.kpiEta.textContent = "--";
    this.lossTrend.style.display = "none";
    // Reset progress
    this.progressMeta.textContent = "0% complete";
    this.epochFill.style.width = "0%";
    this.epochValue.textContent = "0 / 0";
    this.stepFill.style.width = "0%";
    this.stepValue.textContent = "0 / 0";
    // Reset chart
    this.chartArea.innerHTML = '';
    this.chartLabels.innerHTML = '';
    this.yAxisMax.textContent = "0.10";
    this.yAxisMid.textContent = "0.05";
    // Reset validation
    this.validationRow.innerHTML = "";
    this.playerBar.classList.add("hidden");
    this.audioElement.pause();
    this.audioElement.src = "";
    this.playerFill.style.width = "0%";
    this.playerTime.textContent = "0:00 / 0:00";
    this.updatePlayPauseIcon();
    // Reset status
    this.liveIndicator.classList.add("idle");
    this.liveText.textContent = "Idle";
    this.statusText.textContent = "Ready to train";
    this.statusSpinner.classList.add("hidden");
    this.statusTime.textContent = "";
    this.stopButton.disabled = true;
    this.widgetRoot.classList.remove("training-active");
  }
  dispose() {
    this.stopTimer();
    this.audioElement.pause();
    this.audioElement.src = "";
    this.container.innerHTML = "";
  }
}
const widgetInstances = /* @__PURE__ */ new Map();
function createTrainingWidget(node) {
  const container = document.createElement("div");
  container.id = `qwen3-training-widget-${node.id}`;
  container.style.width = "100%";
  container.style.height = "100%";
  const widget = node.addDOMWidget(
    "training_ui",
    "training-widget",
    container,
    {
      getMinHeight: () => 420,
      hideOnZoom: false,
      serialize: false,
      computeSize: function(width) {
        // Calculate height based on node size, allowing widget to fill available space
        const nodeHeight = node.size[1];
        // Account for node header and other widgets (approximately 100px for title bar)
        const availableHeight = Math.max(nodeHeight - 100, 420);
        return [width, availableHeight];
      }
    }
  );
  setTimeout(() => {
    const trainingWidget = new TrainingWidget({
      node,
      container
    });
    widgetInstances.set(node.id, trainingWidget);
  }, 100);
  widget.onRemove = () => {
    const instance = widgetInstances.get(node.id);
    if (instance) {
      instance.dispose();
      widgetInstances.delete(node.id);
    }
  };
  return { widget };
}
app.registerExtension({
  name: "ComfyUI.FL_Qwen3TTS_TrainingUI",
  // Called when any node is created
  nodeCreated(node) {
    var _a;
    if (((_a = node.constructor) == null ? void 0 : _a.comfyClass) !== "FL_Qwen3TTS_TrainingUI") {
      return;
    }
    const [oldWidth, oldHeight] = node.size;
    node.setSize([Math.max(oldWidth, 380), Math.max(oldHeight, 540)]);
    createTrainingWidget(node);
  }
});
api.addEventListener("qwen3tts_training_update", ((event) => {
  const detail = event.detail;
  if (!(detail == null ? void 0 : detail.node)) return;
  const nodeId = parseInt(detail.node, 10);
  const widget = widgetInstances.get(nodeId);
  if (!widget) return;
  switch (detail.type) {
    case "progress":
      widget.updateProgress(
        detail.epoch,
        detail.total_epochs,
        detail.loss,
        detail.step,
        detail.total_steps
      );
      break;
    case "validation":
      widget.addValidationSample(
        detail.epoch,
        detail.audio_base64,
        detail.checkpoint_path
      );
      break;
    case "status":
      widget.updateStatus(detail.message);
      break;
  }
}));
api.addEventListener("executed", ((event) => {
  var _a;
  const detail = event.detail;
  if (!(detail == null ? void 0 : detail.node) || !(detail == null ? void 0 : detail.output)) return;
  const nodeId = parseInt(detail.node, 10);
  const widget = widgetInstances.get(nodeId);
  if (!widget) return;
  const checkpointPath = (_a = detail.output) == null ? void 0 : _a.checkpoint_path;
  if (checkpointPath && checkpointPath.length > 0) {
    widget.onTrainingComplete(checkpointPath[0]);
  }
}));
api.addEventListener("executing", ((event) => {
  const detail = event.detail;
  if (!(detail == null ? void 0 : detail.node)) return;
  const nodeId = parseInt(detail.node, 10);
  const widget = widgetInstances.get(nodeId);
  if (widget) {
    widget.reset();
  }
}));
export {
  TrainingWidget
};
//# sourceMappingURL=main.js.map
