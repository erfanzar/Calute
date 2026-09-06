// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { MCPManager } from '../src/mcp/manager.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { addMcpToolsToBuiltinAgents, mcpRuntimeToolName, registerMcpTools } from '../src/tools/mcpTools.js'
import { BUILTIN_AGENTS } from '../src/agents/definitions.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

const context = { sessionId: 'owner', metadata: {} }
const call = (name: string, args: JsonObject = {}): ToolCall => ({ id: 'mcp', type: 'function', function: { name, arguments: args } })
test('MCP runtime schemas validate input, retain server identity and reject stale connections', async () => {
  const calls: Array<{ server: string; args: JsonObject; signal?: AbortSignal }> = []
  const manager = new MCPManager({ clientFactory: config => ({
    config, connected: true, tools: [{ name: 'same tool', inputSchema: { type: 'object', properties: { value: { type: 'string' } }, required: ['value'], additionalProperties: false } }], resources: [], prompts: [],
    async connect() {}, async disconnect() {},
    async callTool(_name, args = {}, options = {}) { calls.push({ server: config.name, args, ...(options.signal ? { signal: options.signal } : {}) }); return { content: [{ type: 'text', text: config.name }] } },
    async readResource() { return { contents: [] } }, async getPrompt() { return { messages: [] } },
  }) })
  await manager.addServer({ name: 'a/b' }); await manager.addServer({ name: 'a_b' })
  const registry = new ToolRegistry()
  registerMcpTools(registry, manager)
  const first = mcpRuntimeToolName('a/b', 'same tool'), second = mcpRuntimeToolName('a_b', 'same tool')
  expect(first).not.toBe(second)
  expect(first).toMatch(/^[a-zA-Z0-9_-]{1,64}$/)
  expect(registry.definitions()).toHaveLength(2)
  expect(registry.capabilities(first)).toMatchObject({ readOnly: false, destructive: true, openWorld: true })
  await expect(registry.execute(call(first), context)).rejects.toThrow('value')
  expect(calls).toHaveLength(0)
  const signal = new AbortController().signal
  await registry.execute(call(second, { value: 'test' }), context, signal)
  expect(calls).toEqual([{ server: 'a_b', args: { value: 'test' }, signal }])
  await manager.reconnect('a_b')
  await expect(registry.execute(call(second, { value: 'test' }), context)).rejects.toThrow('changed or disconnected')
  const refreshed = new ToolRegistry(); registerMcpTools(refreshed, manager)
  await refreshed.execute(call(second, { value: 'fresh' }), context)
  await manager.disconnectAll()
  await expect(refreshed.execute(call(second, { value: 'test' }), context)).rejects.toThrow('changed or disconnected')
})

test('MCP runtime surfaces server errors and cancellation rather than successful tool results', async () => {
  let invoked = 0
  const manager = new MCPManager({ clientFactory: config => ({
    config, tools: [{ name: 'error', inputSchema: { type: 'object' } }], resources: [], prompts: [],
    async connect() {}, async disconnect() {}, async callTool() { invoked++; return { isError: true, content: [{ type: 'text', text: 'fixture failure' }] } },
    async readResource() { return { contents: [] } }, async getPrompt() { return { messages: [] } },
  }) })
  await manager.addServer({ name: 'fixture' })
  const registry = new ToolRegistry(); registerMcpTools(registry, manager)
  const invocation = call(mcpRuntimeToolName('fixture', 'error'))
  await expect(registry.execute(invocation, { metadata: {} })).rejects.toThrow('authenticated')
  await expect(registry.execute(invocation, context, AbortSignal.abort())).rejects.toThrow('cancelled')
  expect(invoked).toBe(0)
  await expect(registry.execute(invocation, context)).rejects.toThrow('fixture failure')
  await manager.disconnectAll()
})

test('MCP profile wiring preserves user definitions and restricted mode allowlists', () => {
  const definitions = new Map(BUILTIN_AGENTS)
  const name = mcpRuntimeToolName('fixture', 'echo')
  const before = new Map(definitions)
  addMcpToolsToBuiltinAgents(definitions, [name])
  expect(definitions.get('default')?.tools).toContain(name)
  expect(definitions.get('creator')?.tools).toContain(name)
  for (const [key, value] of before) {
    if (key !== 'default' && key !== 'creator') expect(definitions.get(key)).toBe(value)
  }
  const userProfile = { ...before.get('default')!, source: '/workspace/custom.yaml', tools: ['ReadFile'] }
  definitions.set('default', userProfile)
  addMcpToolsToBuiltinAgents(definitions, [name])
  expect(definitions.get('default')).toBe(userProfile)
})
