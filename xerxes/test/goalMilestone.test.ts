// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'

import {
  MAX_CURRENT_MILESTONE_CHARS,
  GoalError,
  admitGoalRound,
  blockGoal,
  completeGoal,
  createGoal,
  editGoal,
  foldGoalChanges,
  getGoal,
  pauseGoal,
  readGoalChanges,
  recordGoalEvidence,
  resetGoalActivations,
  resumeGoal,
  setGoalMilestone,
} from '../src/runtime/goalDomain.js'

const session = 'milestone-session'
const fresh = () => ({}) as Record<string, unknown>

afterEach(() => resetGoalActivations())

test('milestones are durable, replayable, and absent from legacy goals', () => {
  const metadata = fresh()
  const created = createGoal(metadata, session, { objective: 'ship', currentMilestone: 'Run verification' }, 1_000)
  expect(created.currentMilestone).toBe('Run verification')
  expect(readGoalChanges(metadata)[0]).toMatchObject({ goal: { currentMilestone: 'Run verification' } })
  expect(foldGoalChanges(readGoalChanges(metadata)).goal?.currentMilestone).toBe('Run verification')

  const legacy = fresh()
  const old = createGoal(legacy, 'legacy', { objective: 'old' }, 1_000)
  expect(old.currentMilestone).toBeUndefined()
  expect(foldGoalChanges(readGoalChanges(legacy)).goal?.currentMilestone).toBeUndefined()
})

test('setGoalMilestone uses CAS, trims, clears, and preserves goal state', () => {
  const metadata = fresh()
  const created = createGoal(metadata, session, { objective: 'ship', criteria: [{ id: 'tests', description: 'tests pass' }] }, 1_000)
  const withEvidence = recordGoalEvidence(metadata, session, created, 'tests', { toolCallId: 'call-1', summary: 'passed', recordedAt: 1_500 }, 1_500)
  const paused = pauseGoal(metadata, session, withEvidence, 2_000)
  const marked = setGoalMilestone(metadata, session, paused, '  Review the report  ', 2_500)
  expect(marked).toMatchObject({ phase: 'paused', currentMilestone: 'Review the report', revision: paused.revision + 1 })
  expect(marked.criteria?.[0]?.evidence).toMatchObject({ toolCallId: 'call-1' })
  expect(() => setGoalMilestone(metadata, session, paused, 'stale', 3_000)).toThrow(GoalError)
  const cleared = setGoalMilestone(metadata, session, marked, null, 3_000)
  expect(cleared.currentMilestone).toBeUndefined()
  expect(cleared.criteria?.[0]?.evidence).toBeDefined()
})

test('objective edits clear a milestone unless explicitly replaced', () => {
  const metadata = fresh()
  const created = createGoal(metadata, session, { objective: 'first', currentMilestone: 'Old step' }, 1_000)
  const changed = editGoal(metadata, session, created, { objective: 'second' }, 2_000)
  expect(changed.currentMilestone).toBeUndefined()
  const replaced = editGoal(metadata, session, changed, { objective: 'third', currentMilestone: 'New step' }, 3_000)
  expect(replaced.currentMilestone).toBe('New step')
  const retained = editGoal(metadata, session, replaced, { maxGoalRounds: 30 }, 4_000)
  expect(retained.currentMilestone).toBe('New step')
  const cleared = editGoal(metadata, session, retained, { currentMilestone: null }, 5_000)
  expect(cleared.currentMilestone).toBeUndefined()
})

test('completed goals reject explicit milestone edits', () => {
  const metadata = fresh()
  const created = createGoal(metadata, session, { objective: 'ship', currentMilestone: 'final check' }, 1_000)
  const completed = completeGoal(metadata, session, created, 2_000)
  expect(() => editGoal(metadata, session, completed, { currentMilestone: 'late change' }, 3_000))
    .toThrow('completed goal')
  expect(() => editGoal(metadata, session, completed, { currentMilestone: null }, 3_000))
    .toThrow('completed goal')
  expect(completed.currentMilestone).toBe('final check')
  const history = readGoalChanges(metadata)
  const last = history.at(-1)!
  if (last.operation === 'clear') throw new Error('expected completed snapshot')
  expect(() => foldGoalChanges([...history, { ...last, operation: 'milestone',
    goal: { ...last.goal, revision: last.goal.revision + 1, currentMilestone: 'tampered' }, updatedAt: 3_000 }]))
    .toThrow('completed goal')
})

test('milestone validation rejects malformed and oversized values', () => {
  const metadata = fresh()
  expect(() => createGoal(metadata, session, { objective: 'x', currentMilestone: '   ' }, 1_000)).toThrow('current milestone')
  expect(() => createGoal(metadata, session, { objective: 'x', currentMilestone: 'x'.repeat(MAX_CURRENT_MILESTONE_CHARS + 1) }, 1_000)).toThrow('current milestone')
  const created = createGoal(metadata, session, { objective: 'x' }, 2_000)
  expect(() => setGoalMilestone(metadata, session, created, '   ', 3_000)).toThrow('current milestone')
  expect(() => editGoal(metadata, session, created, { currentMilestone: 42 as never }, 3_000)).toThrow('current milestone')
})

test('rounds and phase changes preserve the milestone', () => {
  const metadata = fresh()
  const created = createGoal(metadata, session, { objective: 'ship', currentMilestone: 'Run tests', maxGoalRounds: 2 }, 1_000)
  const round = admitGoalRound(metadata, session, 2_000)
  expect(getGoal(metadata, session)?.currentMilestone).toBe('Run tests')
  const paused = pauseGoal(metadata, session, getGoal(metadata, session)!, 3_000)
  expect(paused.currentMilestone).toBe('Run tests')
  const resumed = resumeGoal(metadata, session, paused, 4_000)
  const blocked = blockGoal(metadata, session, resumed, { code: 'x', message: 'stop' }, 5_000)
  expect(blocked.currentMilestone).toBe('Run tests')
  expect(round?.round).toBe(1)
  expect(foldGoalChanges(readGoalChanges(metadata)).goal?.currentMilestone).toBe('Run tests')
})

test('malformed milestone fields and milestone mutations fail strict replay', () => {
  const metadata = fresh()
  const created = createGoal(metadata, session, { objective: 'ship', currentMilestone: 'step' }, 1_000)
  const marked = setGoalMilestone(metadata, session, created, 'next', 2_000)
  const [first, second] = readGoalChanges(metadata)
  if (!first || !second || first.operation === 'clear' || second.operation === 'clear') throw new Error('expected goal snapshots')
  expect(() => foldGoalChanges([
    { ...first, goal: { ...first.goal, currentMilestone: ' ' } },
  ])).toThrow(GoalError)
  expect(() => foldGoalChanges([
    first,
    { ...second, operation: 'milestone', goal: { ...second.goal, objective: 'changed' } },
  ])).toThrow(GoalError)
  expect(marked.revision).toBe(2)
})
