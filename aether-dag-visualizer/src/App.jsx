import { useState, useCallback, useRef } from 'react'
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Panel,
} from '@xyflow/react'
import dagre from '@dagrejs/dagre'
import '@xyflow/react/dist/style.css'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Textarea } from '@/components/ui/textarea.jsx'
import { Switch } from '@/components/ui/switch.jsx'
import { Label } from '@/components/ui/label.jsx'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip.jsx'
import { Play, Upload, FileUp, Eye, Code, Clock, DollarSign, AlertCircle, CheckCircle, XCircle, SkipForward, Loader2 } from 'lucide-react'
import './App.css'

// Layout configuration
const LAYOUT_CONFIG = {
  rankdir: 'LR', // Left to right (alternatives: TB, BT, RL)
  nodesep: 80,   // Horizontal separation between nodes
  ranksep: 120,  // Vertical separation between ranks (levels)
  marginx: 40,
  marginy: 40,
}

// Node dimensions for dagre layout
const NODE_WIDTH = 220
const NODE_HEIGHT = 80

// Sample DAG data
const sampleDag = {
  "nodes": [
    {
      "id": "summarize",
      "node_type": "llm_fn",
      "prompt": "Summarize the following text: This is a test document about AI and machine learning.",
      "model": "gpt-4o",
      "dependencies": []
    },
    {
      "id": "extract_entities",
      "node_type": "llm_fn", 
      "prompt": "Extract entities from: This is a test document about AI and machine learning.",
      "model": "claude-3-opus",
      "dependencies": []
    },
    {
      "id": "process_results",
      "node_type": "fn",
      "dependencies": ["summarize", "extract_entities"]
    }
  ]
}

// Execution status enum matching runtime response
const ExecutionStatus = {
  PENDING: 'pending',
  RUNNING: 'running',
  SUCCEEDED: 'succeeded',
  FAILED: 'failed',
  SKIPPED: 'skipped',
}

// Get status color for node styling
const getStatusColor = (status, cacheHit) => {
  if (cacheHit) return { border: 'border-green-400', bg: 'bg-green-50', text: 'text-green-700' }
  
  switch (status) {
    case ExecutionStatus.SUCCEEDED:
      return { border: 'border-blue-400', bg: 'bg-blue-50', text: 'text-blue-700' }
    case ExecutionStatus.FAILED:
      return { border: 'border-red-400', bg: 'bg-red-50', text: 'text-red-700' }
    case ExecutionStatus.SKIPPED:
      return { border: 'border-gray-400', bg: 'bg-gray-50', text: 'text-gray-500' }
    case ExecutionStatus.RUNNING:
      return { border: 'border-yellow-400', bg: 'bg-yellow-50', text: 'text-yellow-700' }
    case ExecutionStatus.PENDING:
    default:
      return { border: 'border-gray-300', bg: 'bg-white', text: 'text-gray-600' }
  }
}

// Get status icon component
const StatusIcon = ({ status, cacheHit, className = "w-4 h-4" }) => {
  if (cacheHit) return <CheckCircle className={`${className} text-green-500`} />
  
  switch (status) {
    case ExecutionStatus.SUCCEEDED:
      return <CheckCircle className={`${className} text-blue-500`} />
    case ExecutionStatus.FAILED:
      return <XCircle className={`${className} text-red-500`} />
    case ExecutionStatus.SKIPPED:
      return <SkipForward className={`${className} text-gray-400`} />
    case ExecutionStatus.RUNNING:
      return <Loader2 className={`${className} text-yellow-500 animate-spin`} />
    default:
      return null
  }
}

// Layout nodes using dagre algorithm
const layoutWithDagre = (nodes, edges, direction = LAYOUT_CONFIG.rankdir) => {
  const dagreGraph = new dagre.graphlib.Graph()
  dagreGraph.setDefaultEdgeLabel(() => ({}))
  dagreGraph.setGraph({
    rankdir: direction,
    nodesep: LAYOUT_CONFIG.nodesep,
    ranksep: LAYOUT_CONFIG.ranksep,
    marginx: LAYOUT_CONFIG.marginx,
    marginy: LAYOUT_CONFIG.marginy,
  })

  // Add nodes to dagre graph
  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT })
  })

  // Add edges to dagre graph
  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target)
  })

  // Run the layout algorithm
  dagre.layout(dagreGraph)

  // Apply computed positions to nodes
  const layoutedNodes = nodes.map((node) => {
    const dagreNode = dagreGraph.node(node.id)
    return {
      ...node,
      position: {
        x: dagreNode.x - NODE_WIDTH / 2,
        y: dagreNode.y - NODE_HEIGHT / 2,
      },
    }
  })

  return layoutedNodes
}

// Map execution results to node statuses
const mapExecutionResults = (executionResults) => {
  if (!executionResults) return {}
  
  const statusMap = {}
  
  // Map from node_status field (runtime response format)
  if (executionResults.node_status) {
    Object.entries(executionResults.node_status).forEach(([nodeId, statusInfo]) => {
      statusMap[nodeId] = {
        state: statusInfo.state || ExecutionStatus.PENDING,
        attempts: statusInfo.attempts || 0,
        error: statusInfo.error || null,
      }
    })
  }
  
  // Enhance with data from results array
  if (executionResults.results) {
    executionResults.results.forEach((result) => {
      if (!statusMap[result.node_id]) {
        statusMap[result.node_id] = { state: ExecutionStatus.SUCCEEDED }
      }
      statusMap[result.node_id].output = result.output
      statusMap[result.node_id].cacheHit = result.cache_hit
      statusMap[result.node_id].executionTimeMs = result.execution_time_ms
      statusMap[result.node_id].tokenCost = result.token_cost
    })
  }
  
  // Map skipped nodes
  if (executionResults.skipped_nodes) {
    executionResults.skipped_nodes.forEach((nodeId) => {
      if (!statusMap[nodeId]) {
        statusMap[nodeId] = {}
      }
      statusMap[nodeId].state = ExecutionStatus.SKIPPED
    })
  }
  
  // Map errors
  if (executionResults.errors) {
    executionResults.errors.forEach((error) => {
      // Try to extract node_id from error string if possible
      const match = error.match(/node\s+['"]?(\w+)['"]?/i)
      if (match && statusMap[match[1]]) {
        statusMap[match[1]].error = error
        statusMap[match[1]].state = ExecutionStatus.FAILED
      }
    })
  }
  
  return statusMap
}

// Convert DAG to React Flow format with dagre layout
const convertDagToFlow = (dag, executionResults = null) => {
  const nodeStatusMap = mapExecutionResults(executionResults)
  
  const nodes = dag.nodes.map((node) => ({
    id: node.id,
    type: node.node_type === 'llm_fn' ? 'llmNode' : 'functionNode',
    position: { x: 0, y: 0 }, // Will be set by dagre
    data: {
      label: node.id,
      nodeType: node.node_type,
      prompt: node.prompt,
      model: node.model,
      dependencies: node.dependencies,
      // Execution status data
      status: nodeStatusMap[node.id]?.state || null,
      cacheHit: nodeStatusMap[node.id]?.cacheHit || false,
      output: nodeStatusMap[node.id]?.output || null,
      error: nodeStatusMap[node.id]?.error || null,
      executionTimeMs: nodeStatusMap[node.id]?.executionTimeMs || null,
      tokenCost: nodeStatusMap[node.id]?.tokenCost || null,
    }
  }))

  const edges = []
  dag.nodes.forEach(node => {
    node.dependencies.forEach(dep => {
      edges.push({
        id: `${dep}-${node.id}`,
        source: dep,
        target: node.id,
        animated: true,
        style: { stroke: '#6366f1' }
      })
    })
  })

  // Apply dagre layout
  const layoutedNodes = layoutWithDagre(nodes, edges)

  return { nodes: layoutedNodes, edges }
}

// Custom LLM node component with status support
const LLMNode = ({ data, selected }) => {
  const hasStatus = data.status !== null
  const statusColors = hasStatus ? getStatusColor(data.status, data.cacheHit) : null
  
  const baseClasses = `px-4 py-3 shadow-lg rounded-lg min-w-[200px] transition-all duration-200`
  const colorClasses = hasStatus 
    ? `bg-white ${statusColors.border} border-2`
    : `bg-gradient-to-r from-blue-500 to-purple-600 text-white border-2 ${selected ? 'border-yellow-400' : 'border-transparent'}`
  
  const textColorClasses = hasStatus ? statusColors.text : 'text-white'
  
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className={`${baseClasses} ${colorClasses}`}>
            <div className="flex items-center gap-2 mb-2">
              <Code className={`w-4 h-4 ${textColorClasses}`} />
              <div className={`font-bold text-sm ${hasStatus ? 'text-gray-900' : ''}`}>{data.label}</div>
              {hasStatus && <StatusIcon status={data.status} cacheHit={data.cacheHit} />}
            </div>
            <div className={`text-xs ${hasStatus ? 'text-gray-600' : 'opacity-90'}`}>
              <div className="mb-1">Type: LLM Function</div>
              {data.model && <div className="mb-1">Model: {data.model}</div>}
              {data.cacheHit && <Badge variant="outline" className="text-green-600 border-green-600 text-[10px]">CACHED</Badge>}
            </div>
          </div>
        </TooltipTrigger>
        {data.output && (
          <TooltipContent side="bottom" className="max-w-md">
            <p className="text-sm font-medium mb-1">Output:</p>
            <p className="text-xs text-gray-600 whitespace-pre-wrap">{data.output.slice(0, 300)}{data.output.length > 300 ? '...' : ''}</p>
          </TooltipContent>
        )}
      </Tooltip>
    </TooltipProvider>
  )
}

// Custom function node component with status support
const FunctionNode = ({ data, selected }) => {
  const hasStatus = data.status !== null
  const statusColors = hasStatus ? getStatusColor(data.status, data.cacheHit) : null
  
  const baseClasses = `px-4 py-3 shadow-lg rounded-lg min-w-[200px] transition-all duration-200`
  const colorClasses = hasStatus 
    ? `bg-white ${statusColors.border} border-2`
    : `bg-gradient-to-r from-green-500 to-teal-600 text-white border-2 ${selected ? 'border-yellow-400' : 'border-transparent'}`
  
  const textColorClasses = hasStatus ? statusColors.text : 'text-white'
  
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className={`${baseClasses} ${colorClasses}`}>
            <div className="flex items-center gap-2 mb-2">
              <Play className={`w-4 h-4 ${textColorClasses}`} />
              <div className={`font-bold text-sm ${hasStatus ? 'text-gray-900' : ''}`}>{data.label}</div>
              {hasStatus && <StatusIcon status={data.status} cacheHit={data.cacheHit} />}
            </div>
            <div className={`text-xs ${hasStatus ? 'text-gray-600' : 'opacity-90'}`}>
              <div>Type: Function</div>
            </div>
          </div>
        </TooltipTrigger>
        {data.output && (
          <TooltipContent side="bottom" className="max-w-md">
            <p className="text-sm font-medium mb-1">Output:</p>
            <p className="text-xs text-gray-600 whitespace-pre-wrap">{data.output.slice(0, 300)}{data.output.length > 300 ? '...' : ''}</p>
          </TooltipContent>
        )}
      </Tooltip>
    </TooltipProvider>
  )
}

const nodeTypes = {
  llmNode: LLMNode,
  functionNode: FunctionNode,
}

function App() {
  const [dagInput, setDagInput] = useState(JSON.stringify(sampleDag, null, 2))
  const [selectedNode, setSelectedNode] = useState(null)
  const [executionResults, setExecutionResults] = useState(null)
  const [isExecuting, setIsExecuting] = useState(false)
  const [showStatus, setShowStatus] = useState(true)
  const [isDragOver, setIsDragOver] = useState(false)
  
  const dagFileInputRef = useRef(null)
  const resultFileInputRef = useRef(null)

  const { nodes: initialNodes, edges: initialEdges } = convertDagToFlow(sampleDag)
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  )

  const onNodeClick = useCallback((event, node) => {
    setSelectedNode(node)
  }, [])

  // Load DAG from JSON string
  const loadDag = useCallback((dagJson = dagInput, results = executionResults) => {
    try {
      const dag = JSON.parse(dagJson)
      const effectiveResults = showStatus ? results : null
      const { nodes: newNodes, edges: newEdges } = convertDagToFlow(dag, effectiveResults)
      setNodes(newNodes)
      setEdges(newEdges)
      setSelectedNode(null)
    } catch (error) {
      alert('Invalid JSON format: ' + error.message)
    }
  }, [dagInput, executionResults, showStatus, setNodes, setEdges])

  // Handle DAG file upload
  const handleDagFileUpload = useCallback((file) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      const content = e.target.result
      setDagInput(content)
      loadDag(content, executionResults)
    }
    reader.readAsText(file)
  }, [loadDag, executionResults])

  // Handle execution result file upload
  const handleResultFileUpload = useCallback((file) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const results = JSON.parse(e.target.result)
        setExecutionResults(results)
        loadDag(dagInput, results)
      } catch (error) {
        alert('Invalid execution result JSON: ' + error.message)
      }
    }
    reader.readAsText(file)
  }, [dagInput, loadDag])

  // Handle drag and drop
  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(false)
    
    const files = Array.from(e.dataTransfer.files)
    const jsonFile = files.find(f => f.name.endsWith('.json'))
    
    if (jsonFile) {
      // Check if it is a DAG or result file by naming convention
      if (jsonFile.name.includes('result') || jsonFile.name.includes('execution')) {
        handleResultFileUpload(jsonFile)
      } else {
        handleDagFileUpload(jsonFile)
      }
    }
  }, [handleDagFileUpload, handleResultFileUpload])

  // Re-apply layout when showStatus changes
  const handleShowStatusChange = useCallback((checked) => {
    setShowStatus(checked)
    try {
      const dag = JSON.parse(dagInput)
      const effectiveResults = checked ? executionResults : null
      const { nodes: newNodes, edges: newEdges } = convertDagToFlow(dag, effectiveResults)
      setNodes(newNodes)
      setEdges(newEdges)
    } catch {
      // Ignore parse errors during toggle
    }
  }, [dagInput, executionResults, setNodes, setEdges])

  const executeDag = async () => {
    setIsExecuting(true)
    try {
      const dag = JSON.parse(dagInput)
      const response = await fetch('http://localhost:3000/execute', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ dag, context: {} }),
      })
      
      if (response.ok) {
        const results = await response.json()
        setExecutionResults(results)
        loadDag(dagInput, results)
      } else {
        const errorText = await response.text()
        alert('Failed to execute DAG: ' + errorText)
      }
    } catch (error) {
      alert('Error executing DAG: ' + error.message)
    } finally {
      setIsExecuting(false)
    }
  }

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b p-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Aether DAG Visualizer</h1>
            <p className="text-gray-600">Interactive visualization of Aether execution graphs</p>
          </div>
          <div className="flex items-center gap-4">
            {/* Status toggle */}
            {executionResults && (
              <div className="flex items-center gap-2">
                <Switch
                  id="show-status"
                  checked={showStatus}
                  onCheckedChange={handleShowStatusChange}
                />
                <Label htmlFor="show-status" className="text-sm text-gray-600">Show Status</Label>
              </div>
            )}
            <div className="flex gap-2">
              {/* File upload buttons */}
              <input
                ref={dagFileInputRef}
                type="file"
                accept=".json"
                className="hidden"
                onChange={(e) => e.target.files[0] && handleDagFileUpload(e.target.files[0])}
              />
              <input
                ref={resultFileInputRef}
                type="file"
                accept=".json"
                className="hidden"
                onChange={(e) => e.target.files[0] && handleResultFileUpload(e.target.files[0])}
              />
              <Button onClick={() => dagFileInputRef.current?.click()} variant="outline" size="sm">
                <FileUp className="w-4 h-4 mr-2" />
                Load DAG File
              </Button>
              <Button onClick={() => resultFileInputRef.current?.click()} variant="outline" size="sm">
                <Upload className="w-4 h-4 mr-2" />
                Load Result
              </Button>
              <Button onClick={() => loadDag()} variant="outline" size="sm">
                <Upload className="w-4 h-4 mr-2" />
                Apply JSON
              </Button>
              <Button onClick={executeDag} disabled={isExecuting} size="sm">
                <Play className="w-4 h-4 mr-2" />
                {isExecuting ? 'Executing...' : 'Execute'}
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 flex">
        {/* Left Panel - DAG Input */}
        <div className="w-1/3 p-4 bg-white border-r overflow-y-auto">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">DAG Definition</CardTitle>
              <CardDescription>
                Edit the JSON DAG definition below or drag and drop a .json file
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div
                className={`relative ${isDragOver ? 'ring-2 ring-blue-400 ring-offset-2' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <Textarea
                  value={dagInput}
                  onChange={(e) => setDagInput(e.target.value)}
                  className="font-mono text-sm h-64 resize-none"
                  placeholder="Enter DAG JSON or drag and drop a file..."
                />
                {isDragOver && (
                  <div className="absolute inset-0 bg-blue-50 bg-opacity-90 flex items-center justify-center rounded-md">
                    <div className="text-blue-600 font-medium">Drop JSON file here</div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Status Legend */}
          {executionResults && showStatus && (
            <Card className="mt-4">
              <CardHeader className="py-3">
                <CardTitle className="text-sm">Status Legend</CardTitle>
              </CardHeader>
              <CardContent className="py-2">
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    <span>Cached</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-blue-500" />
                    <span>Succeeded</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <XCircle className="w-4 h-4 text-red-500" />
                    <span>Failed</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <SkipForward className="w-4 h-4 text-gray-400" />
                    <span>Skipped</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Node Details */}
          {selectedNode && (
            <Card className="mt-4">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Eye className="w-5 h-5" />
                  Node Details
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <div className="font-semibold text-sm text-gray-700">ID</div>
                  <div className="text-sm">{selectedNode.data.label}</div>
                </div>
                <div>
                  <div className="font-semibold text-sm text-gray-700">Type</div>
                  <Badge variant={selectedNode.data.nodeType === 'llm_fn' ? 'default' : 'secondary'}>
                    {selectedNode.data.nodeType}
                  </Badge>
                </div>
                {selectedNode.data.status && (
                  <div>
                    <div className="font-semibold text-sm text-gray-700">Status</div>
                    <div className="flex items-center gap-2">
                      <StatusIcon status={selectedNode.data.status} cacheHit={selectedNode.data.cacheHit} />
                      <span className="text-sm capitalize">{selectedNode.data.cacheHit ? 'Cached' : selectedNode.data.status}</span>
                    </div>
                  </div>
                )}
                {selectedNode.data.model && (
                  <div>
                    <div className="font-semibold text-sm text-gray-700">Model</div>
                    <div className="text-sm">{selectedNode.data.model}</div>
                  </div>
                )}
                {selectedNode.data.prompt && (
                  <div>
                    <div className="font-semibold text-sm text-gray-700">Prompt</div>
                    <div className="text-sm bg-gray-50 p-2 rounded border text-gray-700 max-h-32 overflow-y-auto">
                      {selectedNode.data.prompt}
                    </div>
                  </div>
                )}
                {selectedNode.data.output && (
                  <div>
                    <div className="font-semibold text-sm text-gray-700">Output</div>
                    <div className="text-sm bg-blue-50 p-2 rounded border border-blue-200 text-gray-700 max-h-32 overflow-y-auto">
                      {selectedNode.data.output}
                    </div>
                  </div>
                )}
                {selectedNode.data.error && (
                  <div>
                    <div className="font-semibold text-sm text-red-700 flex items-center gap-1">
                      <AlertCircle className="w-4 h-4" />
                      Error
                    </div>
                    <div className="text-sm bg-red-50 p-2 rounded border border-red-200 text-red-700">
                      {selectedNode.data.error}
                    </div>
                  </div>
                )}
                {selectedNode.data.executionTimeMs !== null && (
                  <div>
                    <div className="font-semibold text-sm text-gray-700">Execution Time</div>
                    <div className="text-sm">{selectedNode.data.executionTimeMs}ms</div>
                  </div>
                )}
                {selectedNode.data.dependencies && selectedNode.data.dependencies.length > 0 && (
                  <div>
                    <div className="font-semibold text-sm text-gray-700">Dependencies</div>
                    <div className="flex flex-wrap gap-1">
                      {selectedNode.data.dependencies.map(dep => (
                        <Badge key={dep} variant="outline" className="text-xs">
                          {dep}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Execution Results */}
          {executionResults && (
            <Card className="mt-4">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Play className="w-5 h-5" />
                  Execution Results
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="flex items-center gap-2">
                    <Clock className="w-4 h-4 text-blue-500" />
                    <span className="font-semibold">Total Time:</span>
                    <span>{executionResults.total_execution_time_ms}ms</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <DollarSign className="w-4 h-4 text-green-500" />
                    <span className="font-semibold">Token Cost:</span>
                    <span>{executionResults.total_token_cost}</span>
                  </div>
                </div>

                {executionResults.cache_hit_rate !== undefined && (
                  <div className="text-sm">
                    <span className="font-semibold">Cache Hit Rate:</span>
                    <span className="ml-2">{(executionResults.cache_hit_rate * 100).toFixed(1)}%</span>
                  </div>
                )}
                
                <div>
                  <div className="font-semibold text-sm text-gray-700 mb-2">Node Results</div>
                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {executionResults.results && executionResults.results.map(result => (
                      <div key={result.node_id} className={`p-2 rounded border ${result.cache_hit ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'}`}>
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-sm">{result.node_id}</span>
                          {result.cache_hit && <Badge variant="outline" className="text-green-600 border-green-600 text-[10px]">CACHED</Badge>}
                        </div>
                        <div className="text-xs text-gray-600">
                          Time: {result.execution_time_ms}ms | Tokens: {result.token_cost}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {executionResults.errors && executionResults.errors.length > 0 && (
                  <div>
                    <div className="font-semibold text-sm text-red-700 mb-2">Errors</div>
                    <div className="space-y-1">
                      {executionResults.errors.map((error, index) => (
                        <div key={index} className="bg-red-50 text-red-700 p-2 rounded border border-red-200 text-sm">
                          {error}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right Panel - Flow Visualization */}
        <div className="flex-1 relative">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            nodeTypes={nodeTypes}
            fitView
            className="bg-gray-50"
          >
            <Controls />
            <MiniMap />
            <Background variant="dots" gap={12} size={1} />
            <Panel position="top-right" className="bg-white p-2 rounded shadow">
              <div className="text-xs text-gray-600">
                Click nodes to view details. Hover for output preview.
              </div>
            </Panel>
          </ReactFlow>
        </div>
      </div>
    </div>
  )
}

export default App

