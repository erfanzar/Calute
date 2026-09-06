// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  GoalWakeError,
  cancelGoalWake,
  claimGoalWake,
  finishGoalWake,
  queueGoalWake,
  readGoalWake,
  recoverGoalWake,
} from '../src/runtime/goalWake.js'

const session = 'session-1'
const fresh = () => ({}) as Record<string, unknown>

test('goal wake supports queue, claim, finish, and restart recovery transitions', () => {
  const metadata = fresh()
  const queued = queueGoalWake(metadata, session, 'goal-1', 3, 100)
  expect(readGoalWake(metadata, session)).toEqual(queued)
  expect(queueGoalWake(metadata, session, 'goal-1', 3, 999)).toEqual(queued)

  const running = claimGoalWake(metadata, session, queued.id, 'owner-1', 4, 110)
  expect(running).toMatchObject({ state: 'running', ownerId: 'owner-1', round: 4, startedAt: 110 })
  expect(recoverGoalWake(metadata, session, 'owner-1', 120)).toEqual(running)
  const interrupted = recoverGoalWake(metadata, session, 'owner-2', 130)!
  expect(interrupted).toMatchObject({ state: 'interrupted', settledAt: 130 })

  const replacement = queueGoalWake(metadata, session, 'goal-1', 4, 140)
  const replacementRunning = claimGoalWake(metadata, session, replacement.id, 'owner-3', 5, 150)
  const settled = finishGoalWake(metadata, session, replacementRunning.id, 'owner-3', 'settled', undefined, 160)
  expect(settled).toMatchObject({ state: 'settled', settledAt: 160 })
})

test('wake CAS and state rules reject stale or unauthorized mutations atomically', () => {
  const metadata = fresh()
  const queued = queueGoalWake(metadata, session, 'goal-1', 1, 1)
  const before = structuredClone(metadata)
  expect(() => claimGoalWake(metadata, session, 'wrong-id', 'owner', 1, 2)).toThrow(GoalWakeError)
  expect(metadata).toEqual(before)
  const running = claimGoalWake(metadata, session, queued.id, 'owner', 1, 2)
  const runningBefore = structuredClone(metadata)
  expect(() => finishGoalWake(metadata, session, running.id, 'other-owner', 'settled', undefined, 3)).toThrow(GoalWakeError)
  expect(metadata).toEqual(runningBefore)
  expect(() => queueGoalWake(metadata, session, 'goal-2', 2, 4)).toThrow(GoalWakeError)
  expect(() => cancelGoalWake(metadata, session, running.id, 'late', 5)).toThrow(GoalWakeError)
})

test('queued wakes can be cancelled and terminal wakes can be replaced', () => {
  const metadata = fresh()
  const queued = queueGoalWake(metadata, session, 'goal-1', 1, 1)
  const cancelled = cancelGoalWake(metadata, session, queued.id, 'user cancelled', 2)
  expect(cancelled).toMatchObject({ state: 'cancelled', reason: 'user cancelled', settledAt: 2 })
  const next = queueGoalWake(metadata, session, 'goal-2', 2, 3)
  expect(next.id).not.toBe(cancelled.id)
})

test('read rejects malformed, foreign, and mixed persisted wake records fail closed', () => {
  const cases = [
    { version: 2 },
    { version: 1, id: 'x', sessionId: session, goalId: 'g', revision: 1, state: 'queued', queuedAt: 1, ownerId: 'unexpected' },
    { version: 1, id: 'x', sessionId: 'other', goalId: 'g', revision: 1, state: 'queued', queuedAt: 1 },
    { version: 1, id: 'x', sessionId: session, goalId: 'g', revision: 1, state: 'unknown', queuedAt: 1 },
  ]
  for (const wake of cases) {
    expect(() => readGoalWake({ goal_wake: wake }, session)).toThrow(GoalWakeError)
  }
  expect(readGoalWake({ unrelated: { state: 'running', sessionId: session } }, session)).toBeUndefined()
})
