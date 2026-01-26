import { app } from '../../../scripts/app.js'
import { api } from '../../../scripts/api.js'
import { TrainingWidget } from './TrainingWidget'
import type { TrainingNode, DOMWidget, TrainingUpdateEvent } from './types'

// Store widget instances for cleanup and updates
const widgetInstances = new Map<number, TrainingWidget>()

function createTrainingWidget(node: TrainingNode): { widget: DOMWidget } {
  // Create container element
  const container = document.createElement('div')
  container.id = `qwen3-training-widget-${node.id}`
  container.style.width = '100%'
  container.style.height = '100%'
  container.style.minHeight = '280px'

  // Create DOM widget using ComfyUI's API
  const widget = node.addDOMWidget(
    'training_ui',
    'training-widget',
    container,
    {
      getMinHeight: () => 300,
      hideOnZoom: false,
      serialize: false
    }
  ) as DOMWidget

  // Create the actual widget after container is mounted
  setTimeout(() => {
    const trainingWidget = new TrainingWidget({
      node,
      container
    })
    widgetInstances.set(node.id, trainingWidget)
  }, 100)

  // Cleanup when node is removed
  widget.onRemove = () => {
    const instance = widgetInstances.get(node.id)
    if (instance) {
      instance.dispose()
      widgetInstances.delete(node.id)
    }
  }

  return { widget }
}

// Register the extension with ComfyUI
app.registerExtension({
  name: 'ComfyUI.FL_Qwen3TTS_TrainingUI',

  // Called when any node is created
  nodeCreated(node: TrainingNode) {
    // Only handle our specific node type
    if (node.constructor?.comfyClass !== 'FL_Qwen3TTS_TrainingUI') {
      return
    }

    // Adjust default node size to accommodate the widget
    const [oldWidth, oldHeight] = node.size
    node.setSize([Math.max(oldWidth, 380), Math.max(oldHeight, 550)])

    // Create the training widget
    createTrainingWidget(node)
  }
})

// Listen for custom training updates from Python
api.addEventListener('qwen3tts_training_update', ((event: CustomEvent<TrainingUpdateEvent>) => {
  const detail = event.detail
  if (!detail?.node) return

  const nodeId = parseInt(detail.node, 10)
  const widget = widgetInstances.get(nodeId)
  if (!widget) return

  switch (detail.type) {
    case 'progress':
      widget.updateProgress(
        detail.epoch as number,
        detail.total_epochs as number,
        detail.loss as number,
        detail.step as number,
        detail.total_steps as number
      )
      break

    case 'validation':
      widget.addValidationSample(
        detail.epoch as number,
        detail.audio_base64 as string,
        detail.checkpoint_path as string
      )
      break

    case 'status':
      widget.updateStatus(detail.message as string)
      break
  }
}) as EventListener)

// Listen for node execution results (data from Python's "ui" dict)
api.addEventListener('executed', ((event: CustomEvent) => {
  const detail = event.detail
  if (!detail?.node || !detail?.output) return

  const nodeId = parseInt(detail.node, 10)
  const widget = widgetInstances.get(nodeId)
  if (!widget) return

  // Access the ui data sent from Python
  const checkpointPath = detail.output?.checkpoint_path as string[] | undefined

  if (checkpointPath && checkpointPath.length > 0) {
    widget.onTrainingComplete(checkpointPath[0])
  }
}) as EventListener)

// Listen for execution start to reset the widget
api.addEventListener('executing', ((event: CustomEvent) => {
  const detail = event.detail
  if (!detail?.node) return

  const nodeId = parseInt(detail.node, 10)
  const widget = widgetInstances.get(nodeId)
  if (widget) {
    widget.reset()
  }
}) as EventListener)

export { TrainingWidget }
