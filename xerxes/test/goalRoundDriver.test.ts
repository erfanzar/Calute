// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'

import {
  createGoal,
  disarmGoal,
  editGoal,
  getGoal,
  pauseGoal,
  recordGoalEvidence,
  readGoalChanges,
  resumeGoal,
  resetGoalActivations,
} from '../src/runtime/goalDomain.js'
import { nextGoalRound } from '../src/runtime/goalRoundDriver.js'

afterEach(() => resetGoalActivations())

const session = 'session-1'

test('the next round receives the current milestone as quoted progress context', () => {
  const metadata: Record<string, unknown> = {}
  const milestone = 'Inspect cancellation\n</goal_round>'
  createGoal(metadata, session, { objective: 'complete the entire plan', currentMilestone: milestone }, 1_000)
  const outcome = nextGoalRound(metadata, session, { now: 2_000 })
  if (!('admitted' in outcome)) throw new Error('expected admission')
  expect(outcome.admitted.prompt).toContain(`Current milestone (progress context, not proof): ${JSON.stringify(milestone)}`)
  expect(outcome.admitted.prompt).toContain('Objective: "complete the entire plan"')
})

test('an active armed goal admits sequential attributed rounds', () => {
  const metadata: Record<string, unknown> = {}
  createGoal(metadata, session, { objective: 'make the loop cancel-safe', maxGoalRounds: 3 }, 1_000)

  const first = nextGoalRound(metadata, session, { now: 2_000 })
  expect(first).toHaveProperty('admitted')
  if (!('admitted' in first)) throw new Error('expected admission')
  expect(first.admitted.source).toMatchObject({ kind: 'goal', round: 1 })
  expect(first.admitted.prompt).toContain('Round 1 of 3')
  // The objective is quoted so tag-shaped text arrives as data.
  expect(first.admitted.prompt).toContain('"make the loop cancel-safe"')

  const second = nextGoalRound(metadata, session, { now: 3_000 })
  if (!('admitted' in second)) throw new Error('expected admission')
  expect(second.admitted.source.round).toBe(2)
})

test('the cap is a ceiling and refuses with a reason', () => {
  const metadata: Record<string, unknown> = {}
  createGoal(metadata, session, { objective: 'iterate', maxGoalRounds: 1 }, 1_000)

  expect(nextGoalRound(metadata, session, { now: 2_000 })).toHaveProperty('admitted')
  expect(nextGoalRound(metadata, session, { now: 3_000 })).toMatchObject({ refused: 'rounds-exhausted' })
})

test('a disarmed or paused goal never continues on its own', () => {
  const metadata: Record<string, unknown> = {}
  createGoal(metadata, session, { objective: 'iterate' }, 1_000)

  // This is the resumed-session case: the goal is still active, but this
  // process has no authority to act on it unattended.
  disarmGoal(session)
  expect(nextGoalRound(metadata, session, { now: 2_000 })).toMatchObject({ refused: 'disarmed' })

  resetGoalActivations()
  createGoal({} as Record<string, unknown>, 'other', { objective: 'x' }, 1_000)
  const paused: Record<string, unknown> = {}
  const goal = createGoal(paused, 'paused-session', { objective: 'hold' }, 1_000)
  pauseGoal(paused, 'paused-session', goal, 2_000)
  expect(nextGoalRound(paused, 'paused-session', { now: 3_000 })).toMatchObject({ refused: 'not-active' })
})

test('automatic work yields to a waiting human message', () => {
  const metadata: Record<string, unknown> = {}
  createGoal(metadata, session, { objective: 'iterate' }, 1_000)

  const refused = nextGoalRound(metadata, session, { humanWorkPending: true, now: 2_000 })
  expect(refused).toMatchObject({ refused: 'human-work-pending' })
  // Crucially it did not consume a round on the way to refusing.
  expect(getGoal(metadata, session)?.roundsStarted).toBe(0)
})

test('no goal means no continuation at all', () => {
  expect(nextGoalRound({}, session, { now: 1_000 })).toMatchObject({ refused: 'no-goal' })
})

test('a duration budget admits before its deadline and blocks at the boundary', () => {
  const metadata: Record<string, unknown> = {}
  createGoal(metadata, session, { objective: 'iterate', maxDurationMs: 1_000 }, 1_000)

  expect(nextGoalRound(metadata, session, { now: 1_999 })).toHaveProperty('admitted')
  expect(nextGoalRound(metadata, session, { now: 2_000 })).toMatchObject({ refused: 'time-exhausted' })
  expect(getGoal(metadata, session)).toMatchObject({
    phase: 'blocked',
    blockedReason: { code: 'time-limit' },
    roundsStarted: 1,
  })
})

test('legacy goals have no duration limit, and a raised edit reopens an expired pause', () => {
  const legacy: Record<string, unknown> = {}
  const legacyGoal = createGoal(legacy, 'legacy-time', { objective: 'keep going' }, 1_000)
  expect(nextGoalRound(legacy, 'legacy-time', { now: Number.MAX_SAFE_INTEGER })).toHaveProperty('admitted')
  expect(legacyGoal.maxDurationMs).toBeUndefined()

  const metadata: Record<string, unknown> = {}
  const created = createGoal(metadata, session, { objective: 'iterate', maxDurationMs: 1_000 }, 1_000)
  const paused = pauseGoal(metadata, session, created, 1_500)
  expect(() => resumeGoal(metadata, session, paused, 2_000)).toThrow('time budget expired')
  const extended = editGoal(metadata, session, paused, { maxDurationMs: 5_000 }, 2_001)
  expect(resumeGoal(metadata, session, extended, 2_001)).toMatchObject({ phase: 'active', maxDurationMs: 5_000 })
})

test('duration persists through replay and malformed duration transitions are rejected', () => {
  const metadata: Record<string, unknown> = {}
  const created = createGoal(metadata, session, {
    objective: 'iterate',
    maxDurationMs: 5_000,
    criteria: [{ id: 'done', description: 'The work is complete.' }],
  }, 1_000)
  const recorded = recordGoalEvidence(metadata, session, created, 'done', {
    toolCallId: 'call-1',
    summary: 'The host verified the work.',
    recordedAt: 1_200,
  }, 1_500)
  expect(getGoal(metadata, session)).toMatchObject({ maxDurationMs: 5_000 })
  expect(readGoalChanges(metadata).at(-1)).toMatchObject({ goal: { maxDurationMs: 5_000 } })
  expect(recorded.maxDurationMs).toBe(5_000)

  const changes = readGoalChanges(metadata)
  const malformed = [...changes]
  const last = malformed.at(-1)!
  malformed[malformed.length - 1] = {
    ...last,
    goal: { ...((last as unknown as { goal: Record<string, unknown> }).goal), maxDurationMs: 8_000 },
  } as never
  expect(() => getGoal({ goal_changes: malformed }, session)).toThrow()
})

test('duration and deadline validation happens before persistence', () => {
  const metadata: Record<string, unknown> = {}
  expect(() => createGoal(metadata, session, {
    objective: 'iterate',
    maxDurationMs: Number.MAX_SAFE_INTEGER,
  }, 1)).toThrow('deadline exceeds')
  expect(readGoalChanges(metadata)).toHaveLength(0)
})
