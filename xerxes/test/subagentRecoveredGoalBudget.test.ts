// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { BUILTIN_AGENTS } from '../src/agents/definitions.js'
import { replacePersistedSubagentSnapshots, persistedSubagentSnapshotValues } from '../src/agents/subagentPersistence.js'
import { recoverSubagentSnapshots } from '../src/daemon/subagentCoordinator.js'
import { createNativeSubagentHost } from '../src/daemon/subagentHost.js'
import { DaemonSubagentEventBus } from '../src/daemon/subagentEvents.js'
import { withModelCallBudget, type ModelCallScope } from '../src/llms/callBudget.js'
import type { LlmClient } from '../src/llms/client.js'
import { createGoal, completeGoal, getGoal } from '../src/runtime/goalDomain.js'
import { restoreRecoveredModelCallScopes } from '../src/runtime/recoveredModelCallScopes.js'
import { GoalTokenBudget } from '../src/runtime/goalTokenBudget.js'
import { GoalTokenLedger } from '../src/runtime/goalTokenLedger.js'
import type { SpawnedAgentSnapshot } from '../src/operators/subagents.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'

function session(id: string, metadata: Record<string, unknown>): { id: string; metadata: Record<string, unknown> } {
  return { id, metadata }
}

class DeterministicClient implements LlmClient {
  calls = 0
  async *stream(): AsyncGenerator<{ content: string; usage: { inputTokens: number; outputTokens: number } }> {
    this.calls++
    yield { content: 'recovered result', usage: { inputTokens: 3, outputTokens: 2 } }
  }
}

function host(client: LlmClient, restoreModelCallScopes?: (snapshot: SpawnedAgentSnapshot) => readonly ModelCallScope[], validateInheritedSelection?: (model: string, effort?: string, signal?: AbortSignal) => Promise<void>) {
  const toolExecutor = new ToolRegistry()
  return createNativeSubagentHost({ agentDefinitions: BUILTIN_AGENTS, cwd: process.cwd(), eventBus: new DaemonSubagentEventBus(),
    llm: client, model: 'test-model', permissionMode: 'accept-all', toolExecutor, tools: toolExecutor.definitions(),
    ...(validateInheritedSelection ? { validateInheritedSelection } : {}),
    ...(restoreModelCallScopes ? { restoreModelCallScopes } : {}) })
}

test('recovered native subagent restores goal ownership, charges a retry, then denies the next call at the durable cap', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-recovered-goal-budget-'))
  const ledgerPath = join(directory, 'ledger.sqlite')
  let ledger = new GoalTokenLedger(ledgerPath)
  const metadata: Record<string, unknown> = {}
  const sourceId = 'recovered-budget-session'
  const live = session(sourceId, metadata)
  const sessions = new Map([[sourceId, live]])
  const goal = createGoal(metadata, sourceId, { objective: 'bounded work', maxTotalTokens: 10 }, Date.now())
  ledger.initialize(sourceId, goal.id, true)
  const client = new DeterministicClient()
  const firstHost = host(client)
  try {
    const budget = new GoalTokenBudget(() => live, ledger, 'owner-before-restart', sourceId)
    const task = await withModelCallBudget(budget, () => firstHost.managerPort.spawn({ message: 'do bounded work', promptProfile: 'default', sourceAgentId: sourceId }))
    await firstHost.managerPort.wait([task.id], 5_000)
    expect(ledger.inspect(sourceId, goal.id)).toMatchObject({ inputTokens: 3, outputTokens: 2, settledCalls: 1, complete: true })
    const persisted: Record<string, unknown> = {}
    replacePersistedSubagentSnapshots(persisted, firstHost.managerPort.listHandles())
    const reparsed = recoverSubagentSnapshots([], sourceId, persistedSubagentSnapshotValues(persisted))
    expect(reparsed[0]?.modelCallBindings).toEqual([{ kind: 'goal', sessionId: sourceId, goalId: goal.id }])
    await firstHost.manager.shutdown()
    const restartedClient = new DeterministicClient()
    ledger.close()
    ledger = new GoalTokenLedger(ledgerPath)
    const restarted = host(restartedClient, snapshot => restoreRecoveredModelCallScopes(snapshot.modelCallBindings, snapshot.sourceAgentId, {
      readSession: id => sessions.get(id), ledger, ownerId: 'owner-after-restart',
    }))
    try {
      restarted.turnCoordinator.restore?.(sourceId, reparsed)
      restarted.managerPort.resume(task.id)
      const retried = await restarted.retry(task.id, { message: 'retry bounded work' })
      await restarted.managerPort.wait([retried.id], 5_000)
      expect(restartedClient.calls).toBe(1)
      expect(ledger.inspect(sourceId, goal.id)).toMatchObject({ settledCalls: 2, inputTokens: 6, outputTokens: 4 })
      const retrySnapshot = restarted.managerPort.listHandles().find(item => item.id === retried.id)!
      await restarted.manager.shutdown()
      const finalClient = new DeterministicClient()
      const finalHost = host(finalClient, snapshot => restoreRecoveredModelCallScopes(snapshot.modelCallBindings, snapshot.sourceAgentId, {
        readSession: id => sessions.get(id), ledger, ownerId: 'owner-third-retry',
      }))
      try {
        finalHost.turnCoordinator.restore?.(sourceId, [retrySnapshot])
        finalHost.managerPort.resume(retrySnapshot.id)
        await expect(finalHost.retry(retrySnapshot.id, { message: 'third bounded retry' })).rejects.toThrow()
        expect(finalClient.calls).toBe(0)
      } finally { await finalHost.manager.shutdown() }
    } finally { if (restarted.manager.listTasks().length) await restarted.manager.shutdown() }
  } finally { if (firstHost.manager.listTasks().length) await firstHost.manager.shutdown(); ledger.close(); await rm(directory, { recursive: true, force: true }) }
})

test('recovered resume then sendInput reset keeps goal ownership and workspace', async () => {
  const ledger = new GoalTokenLedger(':memory:')
  const metadata: Record<string, unknown> = {}
  const sourceId = 'recovered-reset-session'
  const live = session(sourceId, metadata)
  const goal = createGoal(metadata, sourceId, { objective: 'reset work', maxTotalTokens: 2 }, Date.now())
  ledger.initialize(sourceId, goal.id, true)
  const client = new DeterministicClient()
  const restarted = host(client, snapshot => restoreRecoveredModelCallScopes(snapshot.modelCallBindings, snapshot.sourceAgentId, {
    readSession: id => id === sourceId ? live : undefined, ledger, ownerId: 'reset-owner',
  }))
  const snapshot: SpawnedAgentSnapshot = {
    id: 'recovered-reset-child', name: 'recovered-reset-child', title: 'recovered-reset-child', agentId: 'default',
    promptProfile: 'default', sourceAgentId: sourceId, workspace: '/tmp/recovered-reset-workspace',
    status: 'interrupted', closed: false, createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(),
    queueSize: 0, lastInput: 'old work',
    modelCallBindings: [{ kind: 'goal', sessionId: sourceId, goalId: goal.id }],
  }
  try {
    restarted.turnCoordinator.restore?.(sourceId, [snapshot])
    restarted.managerPort.resume(snapshot.id)
    const replacement = await restarted.managerPort.sendInput(snapshot.id, { message: 'reset work' })
    await restarted.managerPort.wait([replacement.id], 5_000)
    expect(replacement.id).not.toBe(snapshot.id)
    expect(replacement.workspace).toBe(snapshot.workspace)
    expect(replacement.modelCallBindings).toEqual([{ kind: 'goal', sessionId: sourceId, goalId: goal.id }])
    expect(client.calls).toBe(1)
    expect(ledger.inspect(sourceId, goal.id)).toMatchObject({ settledCalls: 1, inputTokens: 3, outputTokens: 2 })
    const cappedClient = new DeterministicClient()
    const cappedHost = host(cappedClient, child => restoreRecoveredModelCallScopes(child.modelCallBindings, child.sourceAgentId, {
      readSession: id => id === sourceId ? live : undefined, ledger, ownerId: 'second-reset-owner',
    }))
    try {
      cappedHost.turnCoordinator.restore?.(sourceId, restarted.managerPort.listHandles())
      cappedHost.managerPort.resume(replacement.id)
      await expect(cappedHost.managerPort.sendInput(replacement.id, { message: 'reset after cap' })).rejects.toThrow('exhausted')
      expect(cappedClient.calls).toBe(0)
    } finally { await cappedHost.manager.shutdown() }
  } finally {
    await restarted.manager.shutdown()
    ledger.close()
  }
})

test('recovered goal binding refuses a replaced goal without calling the provider', async () => {
  const ledger = new GoalTokenLedger(':memory:')
  const metadata: Record<string, unknown> = {}
  const sourceId = 'replaced-recovery-session'
  const live = session(sourceId, metadata)
  const sessions = new Map([[sourceId, live]])
  const oldGoal = createGoal(metadata, sourceId, { objective: 'old', maxTotalTokens: 20 }, Date.now())
  ledger.initialize(sourceId, oldGoal.id, true)
  completeGoal(metadata, sourceId, oldGoal, Date.now() + 1)
  const replacement = createGoal(metadata, sourceId, { objective: 'replacement', maxTotalTokens: 20 }, Date.now() + 2)
  ledger.initialize(sourceId, replacement.id, true)
  const client = new DeterministicClient()
  const restored = host(client, snapshot => restoreRecoveredModelCallScopes(snapshot.modelCallBindings, snapshot.sourceAgentId, {
    readSession: id => sessions.get(id), ledger, ownerId: 'fresh-owner',
  }))
  try {
    const snapshot = { id: 'replaced-child', name: 'replaced-child', title: 'replaced-child', agentId: 'default', promptProfile: 'default', sourceAgentId: sourceId,
      status: 'completed' as const, closed: true, createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), queueSize: 0,
      lastInput: 'old work', modelCallBindings: [{ kind: 'goal' as const, sessionId: sourceId, goalId: oldGoal.id }] }
    restored.turnCoordinator.restore?.(sourceId, [snapshot])
    restored.managerPort.resume(snapshot.id)
    await expect(restored.retry(snapshot.id, { message: 'retry old work' })).rejects.toThrow(/no longer active|resume the current goal/)
    expect(client.calls).toBe(0)
    expect(getGoal(metadata, sourceId)?.id).toBe(replacement.id)
    expect(ledger.inspect(sourceId, replacement.id)).toMatchObject({ settledCalls: 0, inputTokens: 0, outputTokens: 0 })
  } finally { await restored.manager.shutdown(); ledger.close() }
})

test('recovered goal bindings are rejected when the restarted host lacks restoration authority', async () => {
  const ledger = new GoalTokenLedger(':memory:')
  const metadata: Record<string, unknown> = {}
  const sourceId = 'unbound-recovery-session'
  const goal = createGoal(metadata, sourceId, { objective: 'restore me', maxTotalTokens: 20 }, Date.now())
  ledger.initialize(sourceId, goal.id, true)
  const client = new DeterministicClient()
  const restarted = host(client)
  try {
    const snapshot = { id: 'unbound-child', name: 'unbound-child', title: 'unbound-child', agentId: 'default', promptProfile: 'default', sourceAgentId: sourceId,
      status: 'completed' as const, closed: true, createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), queueSize: 0,
      lastInput: 'old work', modelCallBindings: [{ kind: 'goal' as const, sessionId: sourceId, goalId: goal.id }] }
    restarted.turnCoordinator.restore?.(sourceId, [snapshot])
    restarted.managerPort.resume(snapshot.id)
    await expect(restarted.retry(snapshot.id, { message: 'retry without restore' })).rejects.toThrow('Cannot restore delegated budget ownership')
    expect(client.calls).toBe(0)
  } finally { await restarted.manager.shutdown(); ledger.close() }
})

test('recovered ownership is rechecked after async provider validation changes the goal', async () => {
  const ledger = new GoalTokenLedger(':memory:')
  const metadata: Record<string, unknown> = {}
  const sourceId = 'stale-recovery-session'
  const live = session(sourceId, metadata)
  const sessions = new Map([[sourceId, live]])
  const goal = createGoal(metadata, sourceId, { objective: 'stale work', maxTotalTokens: 20 }, Date.now())
  ledger.initialize(sourceId, goal.id, true)
  const client = new DeterministicClient()
  const restored = host(client, snapshot => restoreRecoveredModelCallScopes(snapshot.modelCallBindings, snapshot.sourceAgentId, {
    readSession: id => sessions.get(id), ledger, ownerId: 'stale-owner',
  }), async () => { completeGoal(metadata, sourceId, goal, Date.now() + 1) })
  try {
    const snapshot = { id: 'stale-child', name: 'stale-child', title: 'stale-child', agentId: 'default', promptProfile: 'default', sourceAgentId: sourceId,
      status: 'completed' as const, closed: true, createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), queueSize: 0,
      lastInput: 'old work', modelCallBindings: [{ kind: 'goal' as const, sessionId: sourceId, goalId: goal.id }] }
    restored.turnCoordinator.restore?.(sourceId, [snapshot])
    restored.managerPort.resume(snapshot.id)
    await expect(restored.retry(snapshot.id, { message: 'retry stale work' })).rejects.toThrow()
    expect(client.calls).toBe(0)
  } finally { await restored.manager.shutdown(); ledger.close() }
})
