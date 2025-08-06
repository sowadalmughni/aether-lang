import { useState, useCallback, useEffect } from 'react'
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
import '@xyflow/react/dist/style.css'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Textarea } from '@/components/ui/textarea.jsx'
import { Play, Upload, Download, Eye, Code, Clock, DollarSign } from 'lucide-react'
import './App.css'

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

// Convert DAG to React Flow format
const convertDagToFlow = (dag) => {
  const nodes = dag.nodes.map((node, index) => ({
    id: node.id,
    type: node.node_type === 'llm_fn' ? 'llmNode' : 'functionNode',
    position: { x: index * 250, y: node.dependencies.length * 150 },
    data: {
      label: node.id,
      nodeType: node.node_type,
      prompt: node.prompt,
      model: node.model,
      dependencies: node.dependencies
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

  return { nodes, edges }
}

// Custom node components
const LLMNode = ({ data, selected }) => {
  return (
    <div className={`px-4 py-3 shadow-lg rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white border-2 ${selected ? 'border-yellow-400' : 'border-transparent'} min-w-[200px]`}>
      <div className="flex items-center gap-2 mb-2">
        <Code className="w-4 h-4" />
        <div className="font-bold text-sm">{data.label}</div>
      </div>
      <div className="text-xs opacity-90">
        <div className="mb-1">Type: LLM Function</div>
        {data.model && <div className="mb-1">Model: {data.model}</div>}
      </div>
    </div>
  )
}

const FunctionNode = ({ data, selected }) => {
  return (
    <div className={`px-4 py-3 shadow-lg rounded-lg bg-gradient-to-r from-green-500 to-teal-600 text-white border-2 ${selected ? 'border-yellow-400' : 'border-transparent'} min-w-[200px]`}>
      <div className="flex items-center gap-2 mb-2">
        <Play className="w-4 h-4" />
        <div className="font-bold text-sm">{data.label}</div>
      </div>
      <div className="text-xs opacity-90">
        <div>Type: Function</div>
      </div>
    </div>
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

  const loadDag = () => {
    try {
      const dag = JSON.parse(dagInput)
      const { nodes: newNodes, edges: newEdges } = convertDagToFlow(dag)
      setNodes(newNodes)
      setEdges(newEdges)
      setSelectedNode(null)
    } catch (error) {
      alert('Invalid JSON format')
    }
  }

  const executeDag = async () => {
    setIsExecuting(true)
    try {
      const dag = JSON.parse(dagInput)
      const response = await fetch('http://localhost:3000/execute', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(dag),
      })
      
      if (response.ok) {
        const results = await response.json()
        setExecutionResults(results)
      } else {
        alert('Failed to execute DAG')
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
          <div className="flex gap-2">
            <Button onClick={loadDag} variant="outline" size="sm">
              <Upload className="w-4 h-4 mr-2" />
              Load DAG
            </Button>
            <Button onClick={executeDag} disabled={isExecuting} size="sm">
              <Play className="w-4 h-4 mr-2" />
              {isExecuting ? 'Executing...' : 'Execute'}
            </Button>
          </div>
        </div>
      </div>

      <div className="flex-1 flex">
        {/* Left Panel - DAG Input */}
        <div className="w-1/3 p-4 bg-white border-r">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">DAG Definition</CardTitle>
              <CardDescription>
                Edit the JSON DAG definition below
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Textarea
                value={dagInput}
                onChange={(e) => setDagInput(e.target.value)}
                className="font-mono text-sm h-64 resize-none"
                placeholder="Enter DAG JSON..."
              />
            </CardContent>
          </Card>

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
                {selectedNode.data.model && (
                  <div>
                    <div className="font-semibold text-sm text-gray-700">Model</div>
                    <div className="text-sm">{selectedNode.data.model}</div>
                  </div>
                )}
                {selectedNode.data.prompt && (
                  <div>
                    <div className="font-semibold text-sm text-gray-700">Prompt</div>
                    <div className="text-sm bg-gray-50 p-2 rounded border text-gray-700">
                      {selectedNode.data.prompt}
                    </div>
                  </div>
                )}
                {selectedNode.data.dependencies.length > 0 && (
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
                
                <div>
                  <div className="font-semibold text-sm text-gray-700 mb-2">Node Results</div>
                  <div className="space-y-2">
                    {executionResults.results.map(result => (
                      <div key={result.node_id} className="bg-gray-50 p-2 rounded border">
                        <div className="font-medium text-sm">{result.node_id}</div>
                        <div className="text-xs text-gray-600">
                          Output: {result.output} | Time: {result.execution_time_ms}ms | Cost: {result.token_cost}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {executionResults.errors.length > 0 && (
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
                Click nodes to view details
              </div>
            </Panel>
          </ReactFlow>
        </div>
      </div>
    </div>
  )
}

export default App

