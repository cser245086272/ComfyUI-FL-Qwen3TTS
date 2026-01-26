import type { TrainingWidgetOptions, ValidationSample } from './types'
import { injectStyles } from './styles'

export class TrainingWidget {
  private container: HTMLElement
  private progressFill!: HTMLElement
  private progressText!: HTMLElement
  private statusText!: HTMLElement
  private epochSpan!: HTMLElement
  private lossSpan!: HTMLElement
  private stopButton!: HTMLButtonElement
  private timeline!: HTMLElement
  private audioPlayer!: HTMLAudioElement
  private audioPlayerContainer!: HTMLElement
  private widgetRoot!: HTMLElement
  private samples: ValidationSample[] = []
  private activeChip: HTMLElement | null = null

  constructor(options: TrainingWidgetOptions) {
    injectStyles()
    this.container = options.container
    this.createDOM()
    this.bindEvents()
  }

  private createDOM(): void {
    this.container.innerHTML = `
      <div class="qwen3-training-widget">
        <div class="training-header">
          <div class="status-panel">
            <div class="epoch-display">Epoch: <span>0/0</span></div>
            <div class="loss-display">Loss: <span>--</span></div>
            <div class="status-text">Ready to train</div>
          </div>
          <button class="stop-btn" disabled>Stop</button>
        </div>

        <div class="progress-container">
          <div class="progress-bar">
            <div class="progress-fill" style="width: 0%"></div>
          </div>
          <div class="progress-text">0%</div>
        </div>

        <div class="validation-section">
          <div class="section-title">Validation Samples</div>
          <div class="audio-timeline"></div>
          <div class="audio-player-container">
            <audio class="audio-player" controls style="display: none;"></audio>
            <div class="no-audio-message">Click a sample to play</div>
          </div>
        </div>
      </div>
    `

    // Cache element references
    this.widgetRoot = this.container.querySelector('.qwen3-training-widget') as HTMLElement
    this.progressFill = this.container.querySelector('.progress-fill') as HTMLElement
    this.progressText = this.container.querySelector('.progress-text') as HTMLElement
    this.statusText = this.container.querySelector('.status-text') as HTMLElement
    this.epochSpan = this.container.querySelector('.epoch-display span') as HTMLElement
    this.lossSpan = this.container.querySelector('.loss-display span') as HTMLElement
    this.stopButton = this.container.querySelector('.stop-btn') as HTMLButtonElement
    this.timeline = this.container.querySelector('.audio-timeline') as HTMLElement
    this.audioPlayer = this.container.querySelector('.audio-player') as HTMLAudioElement
    this.audioPlayerContainer = this.container.querySelector('.audio-player-container') as HTMLElement
  }

  private bindEvents(): void {
    this.stopButton.addEventListener('click', () => {
      this.requestStop()
    })

    this.audioPlayer.addEventListener('ended', () => {
      // Optionally deselect chip when audio ends
    })
  }

  private requestStop(): void {
    // The stop functionality is handled by ComfyUI's interrupt mechanism
    // This button is mainly visual - the actual stop happens via ComfyUI's cancel
    this.statusText.textContent = 'Stopping...'
    this.stopButton.disabled = true
  }

  public updateProgress(
    epoch: number,
    totalEpochs: number,
    loss: number,
    step: number,
    totalSteps: number
  ): void {
    const overallProgress = ((epoch - 1 + step / totalSteps) / totalEpochs) * 100
    this.progressFill.style.width = `${Math.min(overallProgress, 100)}%`
    this.progressText.textContent = `${Math.round(overallProgress)}%`
    this.epochSpan.textContent = `${epoch}/${totalEpochs}`
    this.lossSpan.textContent = loss.toFixed(4)
    this.statusText.textContent = `Training epoch ${epoch}, step ${step}/${totalSteps}...`
    this.stopButton.disabled = false
    this.widgetRoot.classList.add('training-active')
  }

  public addValidationSample(epoch: number, audioBase64: string, checkpointPath: string): void {
    const sample: ValidationSample = { epoch, audioBase64, checkpointPath }
    this.samples.push(sample)

    // Create timeline chip
    const chip = document.createElement('button')
    chip.className = 'timeline-chip'
    chip.textContent = `Ep ${epoch}`
    chip.title = `Checkpoint: ${checkpointPath}`
    chip.addEventListener('click', () => this.playSample(sample, chip))
    this.timeline.appendChild(chip)

    // Auto-play the new sample
    this.playSample(sample, chip)
  }

  private playSample(sample: ValidationSample, chip: HTMLElement): void {
    // Update active state
    if (this.activeChip) {
      this.activeChip.classList.remove('active')
    }
    chip.classList.add('active')
    this.activeChip = chip

    // Show and play audio
    const noAudioMessage = this.audioPlayerContainer.querySelector('.no-audio-message') as HTMLElement
    if (noAudioMessage) {
      noAudioMessage.style.display = 'none'
    }
    this.audioPlayer.style.display = 'block'
    this.audioPlayer.src = sample.audioBase64
    this.audioPlayer.play().catch(e => {
      console.warn('Audio autoplay prevented:', e)
    })
  }

  public updateStatus(message: string): void {
    this.statusText.textContent = message

    // Check if training is complete
    if (message.toLowerCase().includes('complete') || message.toLowerCase().includes('stopped')) {
      this.widgetRoot.classList.remove('training-active')
      this.stopButton.disabled = true
      this.progressFill.style.width = '100%'
      this.progressText.textContent = '100%'
    }
  }

  public onTrainingComplete(checkpointPath: string | undefined): void {
    if (checkpointPath) {
      this.statusText.textContent = `Complete! Saved to: ${checkpointPath}`
    } else {
      this.statusText.textContent = 'Training complete!'
    }
    this.widgetRoot.classList.remove('training-active')
    this.stopButton.disabled = true
  }

  public reset(): void {
    this.samples = []
    this.activeChip = null
    this.timeline.innerHTML = ''
    this.audioPlayer.style.display = 'none'
    this.audioPlayer.src = ''
    const noAudioMessage = this.audioPlayerContainer.querySelector('.no-audio-message') as HTMLElement
    if (noAudioMessage) {
      noAudioMessage.style.display = 'block'
    }
    this.progressFill.style.width = '0%'
    this.progressText.textContent = '0%'
    this.epochSpan.textContent = '0/0'
    this.lossSpan.textContent = '--'
    this.statusText.textContent = 'Ready to train'
    this.stopButton.disabled = true
    this.widgetRoot.classList.remove('training-active')
  }

  public dispose(): void {
    this.audioPlayer.pause()
    this.audioPlayer.src = ''
    this.container.innerHTML = ''
  }
}
