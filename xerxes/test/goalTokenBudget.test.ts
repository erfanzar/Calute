// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { captureModelCallScopes, chargeModelCall, withCapturedModelCallScopes, withModelCallBudget } from '../src/llms/callBudget.js'
import { createGoal, completeGoal, getGoal } from '../src/runtime/goalDomain.js'
import { GoalTokenLedger } from '../src/runtime/goalTokenLedger.js'
import { GoalTokenBudget } from '../src/runtime/goalTokenBudget.js'
import { SubAgentManager } from '../src/agents/subagentManager.js'
import { completeLlm, type LlmClient } from '../src/llms/client.js'

function fixture(work: (ledger: GoalTokenLedger) => void) {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-goal-token-scope-'))
  const ledger = new GoalTokenLedger(join(directory, 'ledger.sqlite'))
  try { work(ledger) } finally { ledger.close(); rmSync(directory, { recursive: true, force: true }) }
}

test('detached scopes retain goal ownership and nested scopes charge once', () => fixture(ledger => {
  const session = { id: 'scope-parent', metadata: {} }
  const goal = createGoal(session.metadata, session.id, { objective: 'work', maxTotalTokens: 20 }, 1000)
  ledger.initialize(session.id, goal.id, true)
  const budget = new GoalTokenBudget(() => session, ledger, 'owner', session.id)
  const child = withModelCallBudget(budget, () => {
    chargeModelCall()!({ inputTokens: 5, outputTokens: 5 })
    return captureModelCallScopes()
  })
  withModelCallBudget(budget, () => withCapturedModelCallScopes(child, () => chargeModelCall()!({ inputTokens: 5, outputTokens: 5 })))
  expect(ledger.inspect(session.id, goal.id)).toMatchObject({ inputTokens: 10, outputTokens: 10, settledCalls: 2 })
  expect(() => withCapturedModelCallScopes(child, () => chargeModelCall())).toThrow()
  expect(getGoal(session.metadata, session.id)?.blockedReason?.code).toBe('token-budget')
  expect(() => withModelCallBudget(budget, () => chargeModelCall())).toThrow()
}))

test('a scope activates for a model-created goal before the next provider call', () => fixture(ledger => {
  const session = { id: 'scope-new', metadata: {} }
  const budget = new GoalTokenBudget(() => session, ledger, 'owner', session.id)
  withModelCallBudget(budget, () => {
    const beforeGoal = chargeModelCall()!
    const goal = createGoal(session.metadata, session.id, { objective: 'new goal', maxTotalTokens: 10 }, 1000)
    ledger.initialize(session.id, goal.id, true)
    beforeGoal({ inputTokens: 100, outputTokens: 100 })
    chargeModelCall()!({ inputTokens: 8, outputTokens: 2 })
    expect(ledger.inspect(session.id, goal.id)).toMatchObject({ inputTokens: 8, outputTokens: 2, settledCalls: 1 })
    expect(() => chargeModelCall()).toThrow()
  })
}))

test('a captured child cannot silently charge a replacement goal', () => fixture(ledger => {
  const session = { id: 'scope-replace', metadata: {} }
  const goal = createGoal(session.metadata, session.id, { objective: 'old', maxTotalTokens: 10 }, 1000)
  ledger.initialize(session.id, goal.id, true)
  const budget = new GoalTokenBudget(() => session, ledger, 'owner', session.id)
  const captured = withModelCallBudget(budget, captureModelCallScopes)
  completeGoal(session.metadata, session.id, goal, 2000)
  const replacement = createGoal(session.metadata, session.id, { objective: 'new', maxTotalTokens: 10 }, 3000)
  ledger.initialize(session.id, replacement.id, true)
  expect(() => withCapturedModelCallScopes(captured, () => chargeModelCall())).toThrow()
  expect(getGoal(session.metadata, session.id)?.phase).toBe('active')
  expect(ledger.inspect(session.id, replacement.id)?.settledCalls).toBe(0)
}))

test('an existing capped goal without a ledger baseline fails closed', () => fixture(ledger => {
  const session = { id: 'scope-legacy', metadata: {} }
  createGoal(session.metadata, session.id, { objective: 'legacy', maxTotalTokens: 10 }, 1000)
  const budget = new GoalTokenBudget(() => session, ledger, 'owner', session.id)
  expect(() => withModelCallBudget(budget, () => chargeModelCall())).toThrow()
  expect(getGoal(session.metadata, session.id)?.phase).toBe('blocked')
}))

test('subagent retries outside the parent async context retain the goal cap', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-goal-child-budget-'))
  const ledger = new GoalTokenLedger(join(directory, 'ledger.sqlite'))
  const session = { id: 'child-owner', metadata: {} }
  const goal = createGoal(session.metadata, session.id, { objective: 'child work', maxTotalTokens: 10 }, 1000)
  ledger.initialize(session.id, goal.id, true)
  const budget = new GoalTokenBudget(() => session, ledger, 'owner', session.id)
  const manager = new SubAgentManager({ runner: async () => {
    chargeModelCall()!({ inputTokens: 4, outputTokens: 2 })
    return 'done'
  } })
  try {
    const task = await withModelCallBudget(budget, () => manager.spawn({ prompt: 'work', name: 'worker' }))
    await manager.wait(task.id, 2000)
    await manager.retry(task.id)
    await manager.wait(task.id, 2000)
    expect(ledger.inspect(session.id, goal.id)).toMatchObject({ inputTokens: 8, outputTokens: 4, settledCalls: 2 })
    await manager.retry(task.id)
    await manager.wait(task.id, 2000)
    expect(ledger.inspect(session.id, goal.id)?.settledCalls).toBe(2)
    expect(getGoal(session.metadata, session.id)?.blockedReason?.code).toBe('token-budget')
  } finally { ledger.close(); rmSync(directory, { recursive: true, force: true }) }
})

test('provider completions account for reported cache tokens before admitting another call', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-goal-provider-budget-'))
  const ledger = new GoalTokenLedger(join(directory, 'ledger.sqlite'))
  const session = { id: 'provider-owner', metadata: {} }
  const goal = createGoal(session.metadata, session.id, { objective: 'provider work', maxTotalTokens: 10 }, 1000)
  ledger.initialize(session.id, goal.id, true)
  let calls = 0
  const llm: LlmClient = { async *stream() {
    calls++
    yield { content: 'done', usage: { inputTokens: 2, outputTokens: 2, cacheReadTokens: 3, cacheCreationTokens: 3 } }
  } }
  const budget = new GoalTokenBudget(() => session, ledger, 'owner', session.id)
  try {
    await withModelCallBudget(budget, async () => {
      await completeLlm(llm, { model: 'test', messages: [{ role: 'user', content: 'auxiliary' }] })
      await expect(completeLlm(llm, { model: 'test', messages: [{ role: 'user', content: 'next' }] })).rejects.toThrow()
    })
    expect(calls).toBe(1)
    expect(ledger.inspect(session.id, goal.id)).toMatchObject({ inputTokens: 8, outputTokens: 2, measuredCalls: 1, complete: true })
  } finally { ledger.close(); rmSync(directory, { recursive: true, force: true }) }
})
