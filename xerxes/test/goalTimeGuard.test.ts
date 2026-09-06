// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createGoal, getGoal, pauseGoal } from '../src/runtime/goalDomain.js'
import { GoalTimeGuard } from '../src/runtime/goalTimeGuard.js'

test('time guard observes goals created mid-turn and blocks before aborting', () => {
  const session = { id: 'guard-new', metadata: {} }
  let now = 1000
  const guard = new GoalTimeGuard(() => session, () => now)
  try {
    createGoal(session.metadata, session.id, { objective: 'work', maxDurationMs: 50 }, now)
    guard.refresh()
    expect(guard.signal.aborted).toBe(false)
    now = 1050
    guard.refresh()
    expect(guard.signal.aborted).toBe(true)
    expect(getGoal(session.metadata, session.id)).toMatchObject({ phase: 'blocked', blockedReason: { code: 'time-limit' } })
  } finally { guard.dispose() }
})

test('paused goals do not cancel unrelated human work and disposed guards do nothing', () => {
  const session = { id: 'guard-paused', metadata: {} }
  const goal = createGoal(session.metadata, session.id, { objective: 'work', maxDurationMs: 50 }, 1000)
  pauseGoal(session.metadata, session.id, goal, 1010)
  const guard = new GoalTimeGuard(() => session, () => 1100)
  expect(guard.signal.aborted).toBe(false)
  guard.dispose()
  guard.refresh()
  expect(guard.signal.aborted).toBe(false)
})
