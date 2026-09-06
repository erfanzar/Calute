// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  GoalError,
  completeGoal,
  createGoal,
  editGoal,
  foldGoalChanges,
  getGoal,
  blockGoal,
  pauseGoal,
  recordGoalEvidence,
  readGoalChanges,
} from '../src/runtime/goalDomain.js'

const session = 'criteria-session'

test('explicit tool-result and legacy evidence kinds are equivalent during replay', () => {
  const metadata: Record<string, unknown> = {}
  const goal = createGoal(metadata, 'tool-kind', { objective: 'verify', criteria: [{ id: 'test', description: 'Checks pass' }] }, 1000)
  const recorded = recordGoalEvidence(metadata, 'tool-kind', goal, 'test', {
    toolCallId: 'check', summary: 'Checks passed', recordedAt: 1001,
  }, 1001)
  pauseGoal(metadata, 'tool-kind', recorded, 1002)
  const changes = JSON.parse(JSON.stringify(readGoalChanges(metadata)))
  changes[1].goal.criteria[0].evidence.kind = 'tool-result'
  expect(() => foldGoalChanges(changes)).not.toThrow()
})

test('replacement goals created in the same millisecond retain distinct accounting identities', () => {
  const metadata: Record<string, unknown> = {}
  const first = createGoal(metadata, 'same-timestamp', { objective: 'first' }, 1000)
  completeGoal(metadata, 'same-timestamp', first, 1000)
  const second = createGoal(metadata, 'same-timestamp', { objective: 'second' }, 1000)
  expect(second.id).not.toBe(first.id)
})
const fresh = () => ({}) as Record<string, unknown>
const evidence = (toolCallId = 'call-1') => ({
  toolCallId,
  summary: 'The host verified the required result.',
  recordedAt: 1_234,
})
const decisionEvidence = (decisionId = 'decision-1') => ({
  kind: 'user-decision' as const,
  decisionId,
  summary: 'The user approved the required result.',
  recordedAt: 1_234,
})

test('human decision evidence completes criteria and preserves paused or blocked state', () => {
  const pausedMetadata = fresh()
  const paused = createGoal(pausedMetadata, 'paused-decision', {
    objective: 'wait for review',
    criteria: [{ id: 'review', description: 'The user approved the change.' }],
  }, 1_000)
  const pausedGoal = pauseGoal(pausedMetadata, 'paused-decision', paused, 1_100)
  const recordedPaused = recordGoalEvidence(
    pausedMetadata,
    'paused-decision',
    pausedGoal,
    'review',
    decisionEvidence(),
    1_200,
  )
  expect(recordedPaused.phase).toBe('paused')
  expect(recordedPaused.roundsStarted).toBe(0)
  expect(recordedPaused.criteria?.[0]?.evidence).toEqual(decisionEvidence())
  expect(completeGoal(pausedMetadata, 'paused-decision', recordedPaused, 1_300).phase).toBe('complete')

  const blockedMetadata = fresh()
  const blocked = createGoal(blockedMetadata, 'blocked-decision', {
    objective: 'resolve review',
    criteria: [{ id: 'review', description: 'The user approved the change.' }],
  }, 1_000)
  const blockedGoal = blockGoal(blockedMetadata, 'blocked-decision', blocked, { code: 'needs-review', message: 'Waiting for review.' }, 1_100)
  const recordedBlocked = recordGoalEvidence(
    blockedMetadata,
    'blocked-decision',
    blockedGoal,
    'review',
    decisionEvidence('decision-2'),
    1_200,
  )
  expect(recordedBlocked.phase).toBe('blocked')
  expect(recordedBlocked.blockedReason).toEqual(blockedGoal.blockedReason)
})

test('decision evidence rejects mixed or unknown kinds and invalid identities atomically', () => {
  const metadata = fresh()
  const created = createGoal(metadata, session, {
    objective: 'review',
    criteria: [{ id: 'review', description: 'The user approved the change.' }],
  }, 1_000)
  const before = JSON.stringify(metadata)
  expect(() => recordGoalEvidence(metadata, session, created, 'review', {
    ...decisionEvidence(),
    toolCallId: 'mixed',
  } as never, 1_100)).toThrow(GoalError)
  expect(() => recordGoalEvidence(metadata, session, created, 'review', {
    ...decisionEvidence(),
    kind: 'mystery',
  } as never, 1_100)).toThrow(GoalError)
  expect(() => recordGoalEvidence(metadata, session, created, 'review', decisionEvidence(' '.repeat(201)), 1_100))
    .toThrow(GoalError)
  expect(JSON.stringify(metadata)).toBe(before)
})

test('legacy goals can add pending criteria and replay without losing prior history', () => {
  const metadata = fresh()
  const created = createGoal(metadata, session, { objective: 'legacy goal' }, 1_000)
  const edited = editGoal(metadata, session, created, {
    criteria: [{ id: 'tests', description: 'Focused tests pass.' }],
  }, 2_000)
  expect(edited.criteria).toEqual([{ id: 'tests', description: 'Focused tests pass.' }])
  expect(readGoalChanges(metadata)).toHaveLength(2)
  const reloaded = JSON.parse(JSON.stringify(metadata)) as Record<string, unknown>
  expect(getGoal(reloaded, session)?.criteria).toEqual(edited.criteria)
  expect(() => completeGoal(reloaded, session, edited, 3_000)).toThrow(GoalError)
  const forged = structuredClone(readGoalChanges(metadata))
  const last = forged[1]!
  if (last.operation === 'clear') throw new Error('expected edit')
  const invalid = { ...last, goal: { ...last.goal, criteria: [{ ...edited.criteria![0]!, evidence: evidence() }] } }
  expect(() => foldGoalChanges([forged[0]!, invalid])).toThrow(GoalError)
})

test('a rejected candidate change leaves existing goal history unchanged', () => {
  const metadata = fresh()
  const created = createGoal(metadata, session, { objective: 'keep valid history' }, 1_000)
  const before = JSON.stringify(metadata)
  expect(() => editGoal(metadata, session, created, { maxGoalRounds: 10 }, Number.NaN)).toThrow(GoalError)
  expect(JSON.stringify(metadata)).toBe(before)
  expect(getGoal(metadata, session)?.revision).toBe(created.revision)
})

test('criteria are persisted and evidence is recorded by CAS', () => {
  const metadata = fresh()
  const created = createGoal(metadata, session, {
    objective: 'ship the feature',
    criteria: [
      { id: 'tests', description: 'The focused tests pass.' },
      { id: 'review', description: 'The change has been reviewed.' },
    ],
  }, 1_000)

  expect(created.criteria).toEqual([
    { id: 'tests', description: 'The focused tests pass.' },
    { id: 'review', description: 'The change has been reviewed.' },
  ])
  const recorded = recordGoalEvidence(metadata, session, created, 'tests', evidence(), 2_000)
  expect(recorded.criteria?.[0]).toEqual({
    id: 'tests',
    description: 'The focused tests pass.',
    evidence: evidence(),
  })
  expect(recorded.revision).toBe(2)
  expect(recordGoalEvidence(metadata, session, recorded, 'tests', evidence(), 9_000)).toMatchObject(recorded)
  expect(recordGoalEvidence(metadata, session, recorded, 'tests', evidence(), 9_000).revision).toBe(recorded.revision)
  expect(readGoalChanges(metadata)).toHaveLength(2)
  expect(() => recordGoalEvidence(metadata, session, created, 'review', evidence('stale'), 3_000))
    .toThrow(GoalError)
  expect(getGoal(metadata, session)?.criteria?.[1]?.evidence).toBeUndefined()
})

test('editing criteria preserves only unchanged id and description', () => {
  const metadata = fresh()
  const created = createGoal(metadata, session, {
    objective: 'ship the feature',
    criteria: [
      { id: 'tests', description: 'Tests pass.' },
      { id: 'docs', description: 'Docs are updated.' },
    ],
  }, 1_000)
  const tested = recordGoalEvidence(metadata, session, created, 'tests', evidence(), 2_000)
  const edited = editGoal(metadata, session, tested, {
    criteria: [
      { id: 'tests', description: 'Tests pass.' },
      { id: 'docs', description: 'Docs are now updated.' },
      { id: 'release', description: 'Release notes are present.' },
    ],
  }, 3_000)
  expect(edited.criteria).toEqual([
    { id: 'tests', description: 'Tests pass.', evidence: evidence() },
    { id: 'docs', description: 'Docs are now updated.' },
    { id: 'release', description: 'Release notes are present.' },
  ])

  const objectiveEdited = editGoal(metadata, session, edited, { objective: 'ship it safely' }, 4_000)
  expect(objectiveEdited.criteria?.every(criterion => criterion.evidence === undefined)).toBe(true)
})

test('completion is gated by declared criteria while legacy goals remain completable', () => {
  const metadata = fresh()
  const created = createGoal(metadata, session, {
    objective: 'ship the feature',
    criteria: [{ id: 'tests', description: 'Tests pass.' }],
  }, 1_000)
  expect(() => completeGoal(metadata, session, created, 2_000)).toThrow(GoalError)
  const recorded = recordGoalEvidence(metadata, session, created, 'tests', evidence(), 3_000)
  expect(completeGoal(metadata, session, recorded, 4_000).phase).toBe('complete')

  const legacyMetadata = fresh()
  const legacy = createGoal(legacyMetadata, 'legacy-session', { objective: 'legacy goal' }, 1_000)
  expect(completeGoal(legacyMetadata, 'legacy-session', legacy, 2_000).phase).toBe('complete')
})

test('a completed goal cannot be edited into an unmet criteria state', () => {
  const metadata = fresh()
  const created = createGoal(metadata, session, {
    objective: 'ship the feature',
    criteria: [{ id: 'tests', description: 'Tests pass.' }],
  }, 1_000)
  const certified = recordGoalEvidence(metadata, session, created, 'tests', evidence(), 2_000)
  const completed = completeGoal(metadata, session, certified, 3_000)
  const before = readGoalChanges(metadata)

  expect(() => editGoal(metadata, session, completed, { objective: 'ship it safely' }, 4_000))
    .toThrow(GoalError)
  expect(() => editGoal(metadata, session, completed, {
    criteria: [{ id: 'tests', description: 'A different test requirement.' }],
  }, 4_000)).toThrow(GoalError)
  expect(readGoalChanges(metadata)).toEqual(before)
  expect(getGoal(metadata, session)).toMatchObject({ phase: 'complete', revision: completed.revision })
})

test('replay validates criteria and evidence changes preserve all other snapshot fields', () => {
  const metadata = fresh()
  const created = createGoal(metadata, session, {
    objective: 'ship the feature',
    maxGoalRounds: 3,
    criteria: [{ id: 'tests', description: 'Tests pass.' }],
  }, 1_000)
  const recorded = recordGoalEvidence(metadata, session, created, 'tests', evidence(), 2_000)
  const changes = readGoalChanges(metadata)
  expect(foldGoalChanges(changes).goal).toMatchObject({
    objective: 'ship the feature',
    maxGoalRounds: 3,
    criteria: recorded.criteria,
  })

  const malformed = [...changes]
  const last = malformed.at(-1)!
  malformed[malformed.length - 1] = {
    ...last,
    goal: { ...((last as unknown as { goal: Record<string, unknown> }).goal), criteria: [{ id: 'tests', description: 'Tests pass.' }] },
  } as never
  expect(() => foldGoalChanges(malformed as never)).toThrow(GoalError)
  expect(() => getGoal({ goal_changes: malformed }, session)).toThrow(GoalError)

  const removedEvidence = [...changes]
  const evidenceChange = removedEvidence.at(-1)!
  removedEvidence[removedEvidence.length - 1] = {
    ...evidenceChange,
    goal: {
      ...((evidenceChange as unknown as { goal: Record<string, unknown> }).goal),
      criteria: [{ id: 'tests', description: 'Tests pass.' }],
    },
  } as never
  expect(() => foldGoalChanges(removedEvidence as never)).toThrow(GoalError)

  const completedMetadata = fresh()
  const completedCreated = createGoal(completedMetadata, 'complete-replay', {
    objective: 'finish',
    criteria: [{ id: 'done', description: 'The work is done.' }],
  }, 1_000)
  const completedCertified = recordGoalEvidence(
    completedMetadata,
    'complete-replay',
    completedCreated,
    'done',
    evidence(),
    2_000,
  )
  completeGoal(completedMetadata, 'complete-replay', completedCertified, 3_000)
  const completedChanges = readGoalChanges(completedMetadata)
  const completeChange = completedChanges.at(-1)!
  const unmetComplete = [...completedChanges]
  unmetComplete[unmetComplete.length - 1] = {
    ...completeChange,
    goal: {
      ...((completeChange as unknown as { goal: Record<string, unknown> }).goal),
      criteria: [{ id: 'done', description: 'The work is done.' }],
    },
  } as never
  expect(() => foldGoalChanges(unmetComplete as never)).toThrow(GoalError)
  expect(() => foldGoalChanges([unmetComplete.at(-1)!] as never)).toThrow(GoalError)
})

test('criteria and evidence bounds, ids, and active state are enforced', () => {
  const metadata = fresh()
  expect(() => createGoal(metadata, session, {
    objective: 'x',
    criteria: Array.from({ length: 33 }, (_, index) => ({ id: `c${index}`, description: 'ok' })),
  }, 1_000)).toThrow(GoalError)
  expect(() => createGoal(metadata, session, {
    objective: 'x',
    criteria: [{ id: 'same', description: 'one' }, { id: 'same', description: 'two' }],
  }, 1_000)).toThrow(GoalError)
  expect(() => createGoal(metadata, session, {
    objective: 'x',
    criteria: [{ id: ' '.repeat(81), description: 'ok' }],
  }, 1_000)).toThrow(GoalError)

  const created = createGoal(metadata, session, {
    objective: 'x',
    criteria: [{ id: 'c', description: 'ok' }],
  }, 1_000)
  expect(() => recordGoalEvidence(metadata, session, created, 'c', evidence('x'.repeat(201)), 2_000))
    .toThrow(GoalError)
  expect(() => recordGoalEvidence(metadata, session, created, 'c', { ...evidence(), recordedAt: Number.NaN }, 2_000))
    .toThrow(GoalError)
  expect(() => recordGoalEvidence(metadata, session, created, 'missing', evidence(), 2_000))
    .toThrow(GoalError)
})

test('aggregate criteria bytes are bounded before create and evidence append', () => {
  const oversizedMetadata = fresh()
  expect(() => createGoal(oversizedMetadata, session, {
    objective: 'x',
    criteria: Array.from({ length: 32 }, (_, index) => ({ id: `c${index}`, description: 'd'.repeat(1_000) })),
  }, 1_000)).toThrow(GoalError)
  expect(readGoalChanges(oversizedMetadata)).toHaveLength(0)

  const metadata = fresh()
  const created = createGoal(metadata, session, {
    objective: 'x',
    criteria: Array.from({ length: 32 }, (_, index) => ({ id: `c${index}`, description: 'd'.repeat(980) })),
  }, 1_000)
  const before = readGoalChanges(metadata)
  expect(() => recordGoalEvidence(metadata, session, created, 'c0', {
    toolCallId: 't'.repeat(200),
    summary: 's'.repeat(2_000),
    recordedAt: 1_234,
  }, 2_000)).toThrow(GoalError)
  expect(readGoalChanges(metadata)).toEqual(before)
  expect(getGoal(metadata, session)?.revision).toBe(created.revision)
})

test('total token cap edits preserve criteria, evidence, and duration through replay', () => {
  const metadata = fresh()
  const created = createGoal(metadata, session, {
    objective: 'ship the feature',
    maxDurationMs: 5_000,
    maxTotalTokens: 1_000,
    criteria: [{ id: 'tests', description: 'Tests pass.' }],
  }, 1_000)
  const recorded = recordGoalEvidence(metadata, session, created, 'tests', evidence(), 1_500)
  const edited = editGoal(metadata, session, recorded, { maxTotalTokens: 2_000 }, 1_600)

  expect(edited).toMatchObject({
    maxTotalTokens: 2_000,
    maxDurationMs: 5_000,
    criteria: [{ id: 'tests', description: 'Tests pass.', evidence: evidence() }],
  })
  expect(foldGoalChanges(readGoalChanges(metadata)).goal).toMatchObject({
    maxTotalTokens: 2_000,
    maxDurationMs: 5_000,
    criteria: edited.criteria,
  })

  const malformed = [...structuredClone(readGoalChanges(metadata))]
  const evidenceChange = malformed[1]!
  malformed[1] = {
    ...evidenceChange,
    goal: { ...((evidenceChange as unknown as { goal: Record<string, unknown> }).goal), maxTotalTokens: 3_000 },
  } as never
  expect(() => foldGoalChanges(malformed as never)).toThrow(GoalError)
})

test('total token cap accepts only positive safe integers and failed edits are atomic', () => {
  const invalid = [0, -1, Number.NaN, Number.POSITIVE_INFINITY, 1.5, Number.MAX_SAFE_INTEGER + 1]
  for (const maxTotalTokens of invalid) {
    expect(() => createGoal(fresh(), `${session}-${String(maxTotalTokens)}`, {
      objective: 'x',
      maxTotalTokens,
    }, 1_000)).toThrow(GoalError)
  }

  const metadata = fresh()
  const created = createGoal(metadata, session, { objective: 'x', maxTotalTokens: 100 }, 1_000)
  const before = JSON.stringify(metadata)
  expect(() => editGoal(metadata, session, created, { maxTotalTokens: 0 }, 2_000)).toThrow(GoalError)
  expect(JSON.stringify(metadata)).toBe(before)
  expect(getGoal(metadata, session)).toMatchObject({ maxTotalTokens: 100, revision: created.revision })
})
