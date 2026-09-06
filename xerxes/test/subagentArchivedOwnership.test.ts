// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { SubAgentManager } from '../src/agents/subagentManager.js'
import { chargeModelCall, withModelCallBudget } from '../src/llms/callBudget.js'
import { createGoal } from '../src/runtime/goalDomain.js'
import { GoalTokenBudget } from '../src/runtime/goalTokenBudget.js'
import { GoalTokenLedger } from '../src/runtime/goalTokenLedger.js'
import { resolveOwnedSubagentRetry } from '../src/daemon/subagentRetryOwnership.js'

test('archived subagent rebuild retains workspace and goal ownership for retry', async () => {
  const sessionId = 'archived-owner-session'
  const workspace = '/tmp/xerxes-archived-owner-workspace'
  const metadata: Record<string, unknown> = {}
  const session = { id: sessionId, metadata }
  const ledger = new GoalTokenLedger(':memory:')
  const goal = createGoal(metadata, sessionId, { objective: 'archive and retry', maxTotalTokens: 20 }, Date.now())
  ledger.initialize(sessionId, goal.id, true)
  const budget = new GoalTokenBudget(() => session, ledger, 'owner', sessionId)
  const manager = new SubAgentManager({
    maxRetainedTerminalTasks: 1,
    runner: async () => {
      const receipt = chargeModelCall()
      receipt?.({ inputTokens: 2, outputTokens: 1 })
      return 'done'
    },
  })
  try {
    const first = await withModelCallBudget(budget, () => manager.spawn({
      prompt: 'first',
      sourceId: sessionId,
      config: { _nativeSubagentWorkspace: workspace, _nativeSubagentProviderRoute: 'a'.repeat(64), providerProfile: 'original-profile', reasoningEffort: 'high' },
    }))
    await manager.waitAll([first.id], 1_000)
    const second = await manager.spawn({ prompt: 'second' })
    await manager.waitAll([second.id], 1_000)

    expect(manager.findTask(first.id)).toMatchObject({ archived: true, id: first.id })
    expect(resolveOwnedSubagentRetry(manager.listRetryTasks(), first.id, sessionId).id).toBe(first.id)
    expect(() => resolveOwnedSubagentRetry(manager.listRetryTasks(), first.id, 'other-session')).toThrow('another session')
    const retried = await manager.retry(first.id, 'retry')
    if (!retried) throw new Error('Archived task could not be retried')
    await manager.waitAll([retried.id], 1_000)

    const restored = manager.listTasks().find(task => task.id === first.id)
    expect(restored?.workspace).toBe(workspace)
    expect(restored?.providerRoute).toBe('a'.repeat(64))
    expect(restored?.providerProfile).toBe('original-profile')
    expect(restored?.reasoningEffort).toBe('high')
    expect(restored?.modelCallBindings).toEqual([{ kind: 'goal', sessionId, goalId: goal.id }])
    expect(ledger.inspect(sessionId, goal.id)).toMatchObject({ settledCalls: 2, inputTokens: 4, outputTokens: 2 })
  } finally {
    await manager.close()
    ledger.close()
  }
})

test('same-process reset preserves the captured workspace and goal scope outside ambient ownership', async () => {
  const sessionId = 'same-process-reset-session'
  const workspace = '/tmp/xerxes-same-process-reset-workspace'
  const metadata: Record<string, unknown> = {}
  const session = { id: sessionId, metadata }
  const ledger = new GoalTokenLedger(':memory:')
  const goal = createGoal(metadata, sessionId, { objective: 'reset safely', maxTotalTokens: 20 }, Date.now())
  ledger.initialize(sessionId, goal.id, true)
  const budget = new GoalTokenBudget(() => session, ledger, 'owner', sessionId)
  const manager = new SubAgentManager({
    runner: async () => {
      const receipt = chargeModelCall()
      receipt?.({ inputTokens: 1, outputTokens: 1 })
      return 'done'
    },
  })
  try {
    const first = await withModelCallBudget(budget, () => manager.spawn({
      prompt: 'first',
      sourceId: sessionId,
      config: { _nativeSubagentWorkspace: workspace },
    }))
    await manager.waitAll([first.id], 1_000)
    const reset = await manager.reset(first.id, 'reset outside ambient scope')
    expect(reset).toBeDefined()
    await manager.waitAll([reset!.id], 1_000)
    expect(reset!.id).not.toBe(first.id)
    expect(reset!.workspace).toBe(workspace)
    expect(reset!.modelCallBindings).toEqual([{ kind: 'goal', sessionId, goalId: goal.id }])
    expect(ledger.inspect(sessionId, goal.id)).toMatchObject({ settledCalls: 2, inputTokens: 2, outputTokens: 2 })
  } finally {
    await manager.close()
    ledger.close()
  }
})
