// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { chargeModelCall, ModelCallBudget, serializeModelCallScopes, withModelCallBudget } from '../src/llms/callBudget.js'
import { completeGoal, createGoal, pauseGoal, resetGoalActivations } from '../src/runtime/goalDomain.js'
import { GoalTokenBudget } from '../src/runtime/goalTokenBudget.js'
import { GoalTokenLedger } from '../src/runtime/goalTokenLedger.js'
import { restoreRecoveredModelCallScopes } from '../src/runtime/recoveredModelCallScopes.js'

function fixture(fn: (ledger: GoalTokenLedger, session: { id: string; metadata: Record<string, unknown> }) => void): void {
  const dir = mkdtempSync(join(tmpdir(), 'xerxes-recovered-scopes-'))
  const ledger = new GoalTokenLedger(join(dir, 'tokens.sqlite'))
  const session = { id: 'parent', metadata: {} as Record<string, unknown> }
  try { fn(ledger, session) } finally { ledger.close(); rmSync(dir, { recursive: true, force: true }); resetGoalActivations() }
}

test('goal forks serialize and restore with the current owner', () => fixture((ledger, session) => {
  const goal = createGoal(session.metadata, session.id, { objective: 'work', maxTotalTokens: 10 }, 1_000)
  ledger.initialize(session.id, goal.id, true)
  const parent = new GoalTokenBudget(() => session, ledger, 'old-owner', session.id)
  const scopes = withModelCallBudget(parent, () => {
    const fork = parent.fork()
    return fork === undefined ? [] : [fork]
  })
  expect(serializeModelCallScopes(scopes)).toEqual([{ kind: 'goal', sessionId: session.id, goalId: goal.id }])
  const restored = restoreRecoveredModelCallScopes(serializeModelCallScopes(scopes), session.id, { readSession: id => id === session.id ? session : undefined, ledger, ownerId: 'new-owner' })
  withModelCallBudget(restored[0]!, () => chargeModelCall()!({ inputTokens: 3, outputTokens: 2 }))
  expect(ledger.inspect(session.id, goal.id)).toMatchObject({ inputTokens: 3, outputTokens: 2, settledCalls: 1 })
}))

test('dynamic goal scopes are deliberately unrecoverable until forked', () => fixture((_ledger, session) => {
  const goal = createGoal(session.metadata, session.id, { objective: 'work' }, 1_000)
  const dynamic = new GoalTokenBudget(() => session, undefined, 'owner', session.id)
  expect(serializeModelCallScopes([dynamic])).toEqual([{ kind: 'unrecoverable' }])
  expect(goal.id).toBeDefined()
}))

test('unknown scopes serialize as unrecoverable and explicit empty bindings bypass newer goals', () => fixture((_ledger, session) => {
  const budget = new ModelCallBudget()
  expect(serializeModelCallScopes([budget])).toEqual([{ kind: 'unrecoverable' }])
  createGoal(session.metadata, session.id, { objective: 'new goal' }, 1_000)
  expect(restoreRecoveredModelCallScopes([], session.id, { readSession: id => id === session.id ? session : undefined, ledger: undefined, ownerId: undefined })).toEqual([])
}))

test('legacy undefined bindings are allowed only without goal history', () => fixture((_ledger, session) => {
  expect(restoreRecoveredModelCallScopes(undefined, session.id, { readSession: id => id === session.id ? session : undefined, ledger: undefined, ownerId: undefined })).toEqual([])
  createGoal(session.metadata, session.id, { objective: 'historical' }, 1_000)
  expect(() => restoreRecoveredModelCallScopes(undefined, session.id, { readSession: id => id === session.id ? session : undefined, ledger: undefined, ownerId: undefined })).toThrow('unknown goal ownership')
}))

test('recovery refuses stale, paused, replaced, and malformed bindings', () => fixture((ledger, session) => {
  const goal = createGoal(session.metadata, session.id, { objective: 'old', maxTotalTokens: 10 }, 1_000)
  ledger.initialize(session.id, goal.id, true)
  const options = { readSession: (id: string) => id === session.id ? session : undefined, ledger, ownerId: 'owner' }
  expect(() => restoreRecoveredModelCallScopes([{ kind: 'goal', sessionId: 'other', goalId: goal.id }], session.id, options)).toThrow('different session')
  resetGoalActivations()
  expect(() => restoreRecoveredModelCallScopes([{ kind: 'goal', sessionId: session.id, goalId: goal.id }], session.id, options)).toThrow('no longer active')
  const paused = pauseGoal(session.metadata, session.id, goal, 2_000)
  expect(() => restoreRecoveredModelCallScopes([{ kind: 'goal', sessionId: session.id, goalId: goal.id }], session.id, options)).toThrow('no longer active')
  const completed = completeGoal(session.metadata, session.id, paused, 3_000)
  expect(() => restoreRecoveredModelCallScopes([{ kind: 'goal', sessionId: session.id, goalId: completed.id }], session.id, options)).toThrow('no longer active')
  const replacement = createGoal(session.metadata, session.id, { objective: 'new' }, 4_000)
  expect(() => restoreRecoveredModelCallScopes([{ kind: 'goal', sessionId: session.id, goalId: goal.id }], session.id, options)).toThrow('no longer active')
  expect(() => restoreRecoveredModelCallScopes([{ kind: 'unrecoverable' }], session.id, options)).toThrow('unrecoverable')
  expect(() => restoreRecoveredModelCallScopes([{ kind: 'goal', sessionId: session.id } as never], session.id, options)).toThrow('unrecoverable')
  expect(replacement.id).not.toBe(goal.id)
}))

test('restored capped scopes retain pending-owner and cap fences', () => fixture((ledger, session) => {
  const goal = createGoal(session.metadata, session.id, { objective: 'work', maxTotalTokens: 5 }, 1_000)
  ledger.initialize(session.id, goal.id, true)
  const pending = ledger.admit(session.id, goal.id, 'old-owner', goal.maxTotalTokens)
  const binding = [{ kind: 'goal' as const, sessionId: session.id, goalId: goal.id }]
  const restored = restoreRecoveredModelCallScopes(binding, session.id, { readSession: id => id === session.id ? session : undefined, ledger, ownerId: 'new-owner' })
  expect(() => withModelCallBudget(restored[0]!, () => chargeModelCall())).toThrow('pending call owned by another process')
  pending(undefined, false)
}))

test('restored scopes retain measured cap exhaustion and require ledger ownership', () => fixture((ledger, session) => {
  const goal = createGoal(session.metadata, session.id, { objective: 'work', maxTotalTokens: 5 }, 1_000)
  ledger.initialize(session.id, goal.id, true)
  const receipt = ledger.admit(session.id, goal.id, 'owner', goal.maxTotalTokens)
  receipt({ inputTokens: 5, outputTokens: 0 })
  const binding = [{ kind: 'goal' as const, sessionId: session.id, goalId: goal.id }]
  const restored = restoreRecoveredModelCallScopes(binding, session.id, { readSession: id => id === session.id ? session : undefined, ledger, ownerId: 'owner' })
  expect(() => withModelCallBudget(restored[0]!, () => chargeModelCall())).toThrow('exhausted')
  expect(() => restoreRecoveredModelCallScopes(binding, session.id, { readSession: id => id === session.id ? session : undefined, ledger: undefined, ownerId: 'owner' })).toThrow('durable token ledger')
  expect(() => restoreRecoveredModelCallScopes(binding, session.id, { readSession: id => id === session.id ? session : undefined, ledger, ownerId: undefined })).toThrow('current daemon owner')
}))
