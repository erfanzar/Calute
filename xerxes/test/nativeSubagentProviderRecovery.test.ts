// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { BUILTIN_AGENTS } from '../src/agents/definitions.js'
import { DaemonSubagentEventBus } from '../src/daemon/subagentEvents.js'
import { createNativeSubagentHost, type NativeSubagentHostOptions } from '../src/daemon/subagentHost.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { LlmClient, CompletionRequest } from '../src/llms/client.js'
import type { SpawnedAgentSnapshot } from '../src/operators/subagents.js'

const ROUTE_A = 'a'.repeat(64)
const ROUTE_B = 'b'.repeat(64)
const PROFILE = 'saved-profile'

class RouteClient implements LlmClient {
  calls = 0
  readonly models: string[] = []
  readonly efforts: Array<string | undefined> = []
  async *stream(request: CompletionRequest): AsyncGenerator<{ content: string }> {
    this.calls++
    this.models.push(request.model)
    this.efforts.push(request.thinking?.effort)
    yield { content: 'provider route accepted' }
  }
}

function hostOptions(client: LlmClient, route: () => string, inheritedRoute?: string,
  validateInheritedSelection?: (model: string, effort?: string, signal?: AbortSignal) => Promise<void>,
  validateProviderSelection?: (profile: string, model: string, effort?: string, signal?: AbortSignal) => Promise<void>): NativeSubagentHostOptions {
  const toolExecutor = new ToolRegistry()
  return {
    agentDefinitions: BUILTIN_AGENTS,
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: client,
    model: 'child-model',
    permissionMode: 'accept-all',
    toolExecutor,
    tools: toolExecutor.definitions(),
    ...(inheritedRoute === undefined ? {} : { inheritedProviderRoute: inheritedRoute }),
    resolveProviderRoute: () => route(),
    resolveProviderProfile: (_providerProfile, _model, expectedRoute) => {
      if (expectedRoute !== route()) throw new Error(`provider route changed: expected ${expectedRoute}, current ${route()}`)
      return { llm: client }
    },
    ...(validateInheritedSelection ? { validateInheritedSelection } : {}),
    ...(validateProviderSelection ? { validateProviderSelection } : {}),
  }
}

function host(...args: Parameters<typeof hostOptions>) {
  return createNativeSubagentHost(hostOptions(...args))
}

function recovered(providerRoute?: string, explicitProfile = false): SpawnedAgentSnapshot {
  const now = new Date().toISOString()
  return {
    id: 'provider-recovery-child', name: 'provider-recovery-child', title: 'Provider recovery', agentId: 'default', promptProfile: 'default',
    sourceAgentId: 'provider-session', status: 'completed', closed: true, createdAt: now, updatedAt: now, queueSize: 0,
    lastInput: 'retry provider recovery', model: 'child-model', ...(explicitProfile ? { providerProfile: PROFILE } : {}), ...(providerRoute === undefined ? {} : { providerRoute }),
  }
}

test('matching provider route survives persisted recovery with model and effort identity', async () => {
  const firstClient = new RouteClient()
  const first = host(firstClient, () => ROUTE_A, ROUTE_A)
  try {
    const task = await first.managerPort.spawn({ agent: { id: 'default', model: 'child-model', providerProfile: PROFILE, reasoningEffort: 'high' }, promptProfile: 'default', sourceAgentId: 'provider-session', message: 'initial provider call' })
    await first.managerPort.wait([task.id], 5_000)
    const snapshot = first.managerPort.listHandles().find(item => item.id === task.id)!
    expect(snapshot.providerRoute).toBe(ROUTE_A)
    await first.manager.shutdown()
    const retryClient = new RouteClient()
    const restarted = host(retryClient, () => ROUTE_A, ROUTE_A)
    try {
      restarted.turnCoordinator.restore?.('provider-session', [snapshot])
      restarted.managerPort.resume(snapshot.id)
      const retried = await restarted.retry(snapshot.id, { message: 'retry provider recovery', sourceAgentId: 'provider-session' })
      await restarted.managerPort.wait([retried.id], 5_000)
      expect(retryClient.calls).toBe(1)
      expect(retried.model).toBe('child-model')
      expect(retried.reasoningEffort).toBe('high')
      expect(retryClient.models).toEqual(['child-model'])
      expect(retryClient.efforts).toEqual(['high'])
    } finally { await restarted.manager.shutdown() }
  } finally { if (first.manager.listTasks().length) await first.manager.shutdown() }
})

test('changed inherited route rejects recovered retry before provider execution', async () => {
  const client = new RouteClient()
  const restarted = host(client, () => ROUTE_B, ROUTE_B)
  try {
    const snapshot = recovered(ROUTE_A)
    restarted.turnCoordinator.restore?.('provider-session', [snapshot]); restarted.managerPort.resume(snapshot.id)
    await expect(restarted.retry(snapshot.id, { message: 'retry changed route', sourceAgentId: 'provider-session' })).rejects.toThrow(/route|provider/)
    expect(client.calls).toBe(0)
  } finally { await restarted.manager.shutdown() }
})

test('route-aware recovery rejects legacy snapshots without provider identity', async () => {
  const client = new RouteClient()
  const restarted = host(client, () => ROUTE_A, ROUTE_A)
  try {
    const snapshot = recovered()
    restarted.turnCoordinator.restore?.('provider-session', [snapshot]); restarted.managerPort.resume(snapshot.id)
    await expect(restarted.retry(snapshot.id, { message: 'retry legacy route', sourceAgentId: 'provider-session' })).rejects.toThrow(/route|provider|identity/)
    expect(client.calls).toBe(0)
  } finally { await restarted.manager.shutdown() }
})

test('inherited route changes during async validation are rejected before provider execution', async () => {
  let currentRoute = ROUTE_A
  const client = new RouteClient()
  const options = hostOptions(client, () => currentRoute, ROUTE_A, async () => {
    currentRoute = ROUTE_B
    restarted.reconfigure({ ...options, inheritedProviderRoute: ROUTE_B })
  })
  const restarted = createNativeSubagentHost(options)
  try {
    const snapshot = recovered(ROUTE_A)
    restarted.turnCoordinator.restore?.('provider-session', [snapshot]); restarted.managerPort.resume(snapshot.id)
    await expect(restarted.retry(snapshot.id, { message: 'retry after route mutation', sourceAgentId: 'provider-session' })).rejects.toThrow(/provider route changed|host changed/)
    expect(client.calls).toBe(0)
  } finally { await restarted.manager.shutdown() }
})

test('explicit profile route changes during async validation are rejected before provider execution', async () => {
  let currentRoute = ROUTE_A
  const client = new RouteClient()
  const restarted = host(client, () => currentRoute, ROUTE_A, undefined, async () => { currentRoute = ROUTE_B })
  try {
    const snapshot = recovered(ROUTE_A, true)
    restarted.turnCoordinator.restore?.('provider-session', [snapshot]); restarted.managerPort.resume(snapshot.id)
    await expect(restarted.retry(snapshot.id, { message: 'retry explicit mutation', sourceAgentId: 'provider-session' })).rejects.toThrow(/provider route changed/)
    expect(client.calls).toBe(0)
  } finally { await restarted.manager.shutdown() }
})

test('inherited recovery reset retains its route and refuses a changed connection', async () => {
  const client = new RouteClient()
  const restarted = host(client, () => ROUTE_A, ROUTE_A)
  try {
    const snapshot = recovered(ROUTE_A)
    restarted.turnCoordinator.restore?.('provider-session', [snapshot])
    restarted.managerPort.resume(snapshot.id)
    const reset = await restarted.managerPort.sendInput(snapshot.id, { message: 'reset under original route' })
    await restarted.managerPort.wait([reset.id], 5_000)
    expect(reset.id).not.toBe(snapshot.id)
    expect(reset.providerRoute).toBe(ROUTE_A)
    expect(client.calls).toBe(1)
    expect(client.models).toEqual(['child-model'])
    const changedClient = new RouteClient()
    const changed = host(changedClient, () => ROUTE_B, ROUTE_B)
    try {
      changed.turnCoordinator.restore?.('provider-session', restarted.managerPort.listHandles())
      changed.managerPort.resume(reset.id)
      await expect(changed.managerPort.sendInput(reset.id, { message: 'reset after configuration change' })).rejects.toThrow('provider route changed')
      expect(changedClient.calls).toBe(0)
    } finally { await changed.manager.shutdown() }
  } finally { await restarted.manager.shutdown() }
})

test('profile mutation at execution validation is rejected before creating the selected client', async () => {
  let route = ROUTE_A
  let validations = 0
  let creations = 0
  const client = new RouteClient()
  const options = hostOptions(client, () => route, ROUTE_A, undefined, async () => {
    if (++validations === 2) route = ROUTE_B
  })
  const selected = options.resolveProviderProfile!
  const runtime = createNativeSubagentHost({ ...options, resolveProviderProfile: (...args) => {
    creations++
    return selected(...args)
  } })
  try {
    const task = await runtime.managerPort.spawn({ agent: { id: 'default', providerProfile: PROFILE }, message: 'validate twice' })
    await runtime.managerPort.wait([task.id], 5_000)
    const finished = runtime.managerPort.listHandles().find(item => item.id === task.id)
    expect(validations).toBe(2)
    expect(finished?.error).toContain('provider route changed')
    expect(creations).toBe(0)
    expect(client.calls).toBe(0)
  } finally { await runtime.manager.shutdown() }
})
