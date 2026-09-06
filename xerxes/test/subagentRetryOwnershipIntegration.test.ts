// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { BUILTIN_AGENTS } from '../src/agents/definitions.js'
import { DaemonSubagentEventBus } from '../src/daemon/subagentEvents.js'
import { createNativeSubagentHost } from '../src/daemon/subagentHost.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { LlmClient } from '../src/llms/client.js'
import type { SpawnedAgentSnapshot } from '../src/operators/subagents.js'

class DeterministicClient implements LlmClient {
  readonly requests: string[] = []
  async *stream(request: { messages?: readonly { content?: unknown }[] }): AsyncGenerator<{ content: string }> {
    this.requests.push(String(request.messages?.at(-1)?.content ?? ''))
    yield { content: 'retry complete' }
  }
}

function host(client: LlmClient) {
  const toolExecutor = new ToolRegistry()
  return createNativeSubagentHost({ agentDefinitions: BUILTIN_AGENTS, cwd: process.cwd(), eventBus: new DaemonSubagentEventBus(),
    llm: client, model: 'test-model', permissionMode: 'accept-all', toolExecutor, tools: toolExecutor.definitions() })
}

function recovered(id: string, sourceAgentId: string): SpawnedAgentSnapshot {
  const now = new Date().toISOString()
  return { id, name: 'shared-review', title: `Review ${sourceAgentId}`, agentId: 'default', promptProfile: 'default', sourceAgentId,
    status: 'completed', closed: true, createdAt: now, updatedAt: now, queueSize: 0, lastInput: `finish ${sourceAgentId}` }
}

test('restored same-name retries resolve within the requesting session and preserve stable ids', async () => {
  const client = new DeterministicClient()
  const native = host(client)
  const taskA = recovered('task-a', 'source-a')
  const taskB = recovered('task-b', 'source-b')
  try {
    native.turnCoordinator.restore?.('source-a', [taskA])
    native.turnCoordinator.restore?.('source-b', [taskB])
    native.managerPort.resume(taskA.id)
    native.managerPort.resume(taskB.id)
    const retryA = await native.retry('shared-review', { sourceAgentId: 'source-a', message: 'retry A' })
    const retryB = await native.retry('shared-review', { sourceAgentId: 'source-b', message: 'retry B' })
    await native.managerPort.wait([retryA.id, retryB.id], 5_000)
    expect(retryA.id).toBe(taskA.id)
    expect(retryB.id).toBe(taskB.id)
    expect(client.requests).toEqual(expect.arrayContaining(['retry A', 'retry B']))
  } finally { await native.manager.shutdown() }
})

test('foreign id and unknown owner fail before provider execution', async () => {
  const client = new DeterministicClient()
  const native = host(client)
  const taskA = recovered('task-a', 'source-a')
  const taskB = recovered('task-b', 'source-b')
  try {
    native.turnCoordinator.restore?.('source-a', [taskA])
    native.turnCoordinator.restore?.('source-b', [taskB])
    native.managerPort.resume(taskA.id)
    native.managerPort.resume(taskB.id)
    await expect(native.retry(taskB.id, { sourceAgentId: 'source-a', message: 'foreign id' })).rejects.toThrow('another session')
    await expect(native.retry('shared-review', { sourceAgentId: 'unknown-source', message: 'unknown owner' })).rejects.toThrow('not found')
    expect(client.requests).toHaveLength(0)
  } finally { await native.manager.shutdown() }
})

test('live terminal task rejects a foreign owner before retry execution', async () => {
  const client = new DeterministicClient()
  const native = host(client)
  try {
    const live = await native.managerPort.spawn({ message: 'live task', promptProfile: 'default', sourceAgentId: 'source-b', title: 'Live' })
    await native.managerPort.wait([live.id], 5_000)
    await expect(native.retry(live.id, { sourceAgentId: 'source-a', message: 'foreign live retry' })).rejects.toThrow('another session')
    expect(client.requests).toHaveLength(1)
  } finally { await native.manager.shutdown() }
})
