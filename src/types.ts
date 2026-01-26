import type { LGraphNode } from '@comfyorg/comfyui-frontend-types'

export interface TrainingNode extends LGraphNode {
  widgets?: Array<{
    name: string
    value: unknown
    callback?: (value: unknown) => void
  }>
}

export interface DOMWidgetOptions {
  getMinHeight?: () => number
  hideOnZoom?: boolean
  serialize?: boolean
}

export interface DOMWidget {
  name: string
  type: string
  element: HTMLElement
  options: DOMWidgetOptions
  onRemove?: () => void
  serializeValue?: () => Promise<string> | string
}

export interface TrainingWidgetOptions {
  node: LGraphNode
  container: HTMLElement
}

export interface ValidationSample {
  epoch: number
  audioBase64: string
  checkpointPath: string
}

export interface ProgressData {
  epoch: number
  total_epochs: number
  loss: number
  step: number
  total_steps: number
}

export interface ValidationData {
  epoch: number
  audio_base64: string
  checkpoint_path: string
}

export interface StatusData {
  message: string
}

export interface TrainingUpdateEvent {
  node: string
  type: 'progress' | 'validation' | 'status'
  [key: string]: unknown
}
