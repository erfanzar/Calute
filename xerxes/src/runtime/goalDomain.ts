// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Event-sourced same-session goals.
 *
 * The design — phases, compare-and-set refs, whole-snapshot change events,
 * process-local activation, round attribution — follows the goal subsystem of
 * DeepSeek Harness (github.com/deepseek-ai/deepseek-harness, MIT), which
 * solves this problem better than the marker-matching guard this replaces.
 * The implementation is written against Xerxes's own session metadata rather
 * than copied: their service is a Cordis plugin over a session-log service and
 * shares no runtime with ours. No DeepSeek source is reproduced here.
 *
 * What the old ledger got wrong, and this fixes:
 *
 *   - Completion was inferred by grepping the model's prose for English
 *     phrases ("objective met", "all tests pass"). A model answering in
 *     another language, or phrasing success differently, could never stop.
 *     Lifecycle is now a typed transition the model requests explicitly.
 *   - There was no `paused`, so there was no way to hold a goal without
 *     abandoning it.
 *   - Activation was implied by the durable phase, so a resumed session
 *     silently resumed autonomous work. Activation is now process-local and
 *     starts disarmed on every load; only a human-authorised resume rearms it.
 *
 * Every mutation is compare-and-set on `revision` and appends a whole-value
 * change event. State is a strict fold over those events, so replay after a
 * restart reconstructs exactly what happened rather than trusting a mutable
 * blob.
 */

/** Identifies one goal across its durable revisions. */
export type GoalId = string

/** Compare-and-set identity for one exact goal revision. */
export interface GoalRef {
  readonly id: GoalId
  /** Positive; every durable mutation increments it. */
  readonly revision: number
}

/** Durable continuation phase. Activation is process-local and separate. */
export type GoalPhase = 'active' | 'paused' | 'blocked' | 'complete'

/** Whether this process may automatically continue an active goal. */
export type GoalActivation = 'armed' | 'disarmed'

/** Machine-routable and human-readable explanation for a blocked goal. */
export interface GoalBlockReason {
  /** Stable lower-kebab-case classification chosen by the blocking policy. */
  readonly code: string
  /** Non-empty explanation shown to humans and models. */
  readonly message: string
}

/** A completion requirement declared by the human for a goal. */
export interface GoalCriterionSpec {
  readonly id: string
  readonly description: string
}

/** Evidence recorded after the host has verified a tool result or human decision. */
export type GoalCriterionEvidence = {
  readonly summary: string
  readonly recordedAt: number
} & (
  | {
      /** Legacy persisted tool evidence has no kind field. */
      readonly kind?: 'tool-result'
      readonly toolCallId: string
      readonly decisionId?: never
    }
  | {
      readonly kind: 'user-decision'
      readonly decisionId: string
      readonly toolCallId?: never
    }
)

/** A requirement together with its optional durable proof. */
export interface GoalCriterion extends GoalCriterionSpec {
  readonly evidence?: GoalCriterionEvidence
}

/** Full durable state written by every non-clear mutation. */
export interface GoalSnapshot extends GoalRef {
  readonly objective: string
  readonly currentMilestone?: string
  readonly phase: GoalPhase
  /** Present exactly while `phase` is `blocked`. */
  readonly blockedReason?: GoalBlockReason
  /** Total admitted goal-round cap. */
  readonly maxGoalRounds: number
  /** Optional wall-clock budget from the goal's creation timestamp. */
  readonly maxDurationMs?: number
  /** Optional aggregate input-plus-output token budget for this goal. */
  readonly maxTotalTokens?: number
  /** Explicit completion requirements; absent on legacy goals. */
  readonly criteria?: readonly GoalCriterion[]
}

/** Current projection, including values derived from the change log. */
export interface GoalView extends GoalSnapshot {
  /** Highest admitted round number for this goal. */
  readonly roundsStarted: number
  readonly createdAt: number
  readonly updatedAt: number
  /** Process-local continuation eligibility; never persisted. */
  readonly activation: GoalActivation
}

/** State-changing verbs recorded in the durable change. */
/**
 * `round` is a first-class operation rather than an edit that happens to move a
 * counter: it is the only mutation that spends the budget, and the strict fold
 * cannot check budget accounting if it is indistinguishable from an objective
 * edit in the log.
 */
export type GoalOperation =
  | 'create'
  | 'edit'
  | 'pause'
  | 'resume'
  | 'complete'
  | 'block'
  | 'round'
  | 'evidence'
  | 'milestone'
  | 'clear'

/** Whole-snapshot mutation committed by a durable change event. */
export interface GoalSnapshotChange {
  readonly kind: 'goal/change'
  readonly version: 1
  readonly operation: Exclude<GoalOperation, 'clear'>
  readonly goal: GoalSnapshot
  readonly roundsStarted: number
  readonly createdAt: number
  readonly updatedAt: number
}

/** Tombstone retained when the current goal is cleared. */
export interface GoalClearChange {
  readonly kind: 'goal/change'
  readonly version: 1
  readonly operation: 'clear'
  readonly cleared: GoalRef
  readonly clearedAt: number
}

export type GoalChange = GoalSnapshotChange | GoalClearChange

/** Attribution carried by an admitted continuation round's prompt. */
export interface GoalMessageSource {
  readonly kind: 'goal'
  readonly goalId: GoalId
  readonly revision: number
  /** Positive admitted continuation round. */
  readonly round: number
}

/** Stable codes for rejected reads and mutations. */
export type GoalErrorCode =
  | 'GOAL_NOT_FOUND'
  | 'GOAL_ALREADY_EXISTS'
  | 'GOAL_STALE_REVISION'
  | 'GOAL_INVALID_OBJECTIVE'
  | 'GOAL_INVALID_MAX_ROUNDS'
  | 'GOAL_INVALID_BLOCK_REASON'
  | 'GOAL_INVALID_CRITERIA'
  | 'GOAL_INVALID_EVIDENCE'
  | 'GOAL_INVALID_EDIT'
  | 'GOAL_INVALID_TRANSITION'
  | 'GOAL_ROUNDS_EXHAUSTED'
  | 'GOAL_INVALID_DURATION'
  | 'GOAL_INVALID_TOTAL_TOKENS'
  | 'GOAL_INVALID_MILESTONE'
  | 'GOAL_TIME_EXHAUSTED'

export class GoalError extends Error {
  constructor(message: string, readonly code: GoalErrorCode) {
    super(message)
    this.name = 'GoalError'
  }
}

/** Default cap when a create omits one. Deliberately finite. */
export const DEFAULT_MAX_GOAL_ROUNDS = 24
/** No implicit wall-clock cap: omitted duration preserves legacy behavior. */
export const MAX_GOAL_DURATION_MS = Number.MAX_SAFE_INTEGER
/** No implicit token cap: omitted total tokens preserves legacy behavior. */
export const MAX_GOAL_TOTAL_TOKENS = Number.MAX_SAFE_INTEGER
/** Objectives longer than this are truncated before they reach the log. */
export const MAX_OBJECTIVE_CHARS = 4_000
/** Bounded history: a session cannot grow its metadata without limit. */
export const MAX_GOAL_CHANGES = 256
/** Maximum number of declared completion requirements. */
export const MAX_GOAL_CRITERIA = 32
export const MAX_CRITERION_ID_CHARS = 80
export const MAX_CRITERION_DESCRIPTION_CHARS = 1_000
export const MAX_EVIDENCE_TOOL_CALL_ID_CHARS = 200
export const MAX_EVIDENCE_DECISION_ID_CHARS = 200
export const MAX_EVIDENCE_SUMMARY_CHARS = 2_000
export const MAX_CRITERIA_JSON_BYTES = 32 * 1024
export const MAX_CURRENT_MILESTONE_CHARS = 1_000

export const GOAL_CHANGES_KEY = 'goal_changes'

/**
 * Process-local activation, keyed by session.
 *
 * Never persisted, and never inherited by a fresh process: a resumed or forked
 * session comes back disarmed so it cannot silently continue autonomous work
 * that a human has not re-authorised. This is the single most important
 * difference from the phase, which IS durable.
 */
const activations = new Map<string, GoalActivation>()

/** Read the change log from session metadata. */
export function readGoalChanges(metadata: Readonly<Record<string, unknown>>): readonly GoalChange[] {
  const raw = metadata[GOAL_CHANGES_KEY]
  if (!Array.isArray(raw)) return []
  return raw.filter((entry): entry is GoalChange => {
    if (isGoalChange(entry)) return true
    // Preserve the old tolerance for unrelated metadata values, but never
    // silently drop a malformed goal event. In particular, a damaged
    // criteria array must not make a goal look legacy and completable.
    if (isGoalChangeEnvelope(entry)) rejectLog('malformed goal change')
    return false
  })
}

/** Pure replay fold of durable goal facts. */
export interface FoldedGoal {
  readonly goal?: GoalSnapshot
  readonly roundsStarted: number
  readonly createdAt?: number
  readonly updatedAt?: number
  readonly lastRef?: GoalRef
}

/**
 * Fold the change log into current state, strictly.
 *
 * Last-wins over whole values: every non-clear change carries the complete
 * post-mutation snapshot, so a partial or reordered write cannot produce a
 * half-applied goal the way a field-by-field patch log could.
 *
 * Strict, because this log lives in session metadata — a file on disk that
 * survives crashes, gets copied between machines, and can be hand-edited. A
 * permissive fold would happily accept a log whose revisions skip, whose round
 * counter jumps past the cap, or which admits a round against a completed goal,
 * and the result is autonomous work running on state nobody can account for.
 * Every rejection below describes a log that could not have been produced by
 * this module's own mutations.
 */
export function foldGoalChanges(changes: readonly GoalChange[]): FoldedGoal {
  let folded: FoldedGoal = { roundsStarted: 0 }
  for (const [index, change] of changes.entries()) {
    if (!isGoalChange(change)) rejectLog(`malformed goal change at index ${index}`)
    // Compaction (see `append`) replaces an overlong prefix with the single
    // snapshot it folded to, so the log may legitimately open on a mid-life
    // change rather than a create. Only the first entry may do so.
    if (index === 0 && change.operation !== 'create' && change.operation !== 'clear') {
      if (change.goal.phase === 'complete') assertCompleteCriteria(change.goal)
      validateGoalDuration(change.goal, change.createdAt)
      folded = {
        goal: change.goal,
        roundsStarted: change.roundsStarted,
        createdAt: change.createdAt,
        updatedAt: change.updatedAt,
        lastRef: { id: change.goal.id, revision: change.goal.revision },
      }
      continue
    }
    if (change.operation === 'clear') {
      if (!folded.goal) rejectLog('clear with no current goal')
      if (change.cleared.id !== folded.goal.id) rejectLog('clear of a different goal')
      if (change.cleared.revision !== folded.goal.revision + 1) {
        rejectLog('clear must advance the revision by exactly one')
      }
      folded = { roundsStarted: 0, lastRef: change.cleared }
      continue
    }

    const { goal } = change
    validateGoalDuration(goal, folded.createdAt ?? change.createdAt)
    if (change.operation === 'create') {
      if (folded.goal && folded.goal.phase !== 'complete') {
        rejectLog(`create over a goal in phase "${folded.goal.phase}"`)
      }
      if (goal.revision !== 1) rejectLog('create must start at revision 1')
      if (change.roundsStarted !== 0) rejectLog('create must start with zero rounds')
      assertNoCriteriaEvidence(goal)
    } else {
      if (!folded.goal) rejectLog(`${change.operation} with no current goal`)
      if (goal.id !== folded.goal.id) rejectLog(`${change.operation} of a different goal`)
      if (goal.revision !== folded.goal.revision + 1) {
        rejectLog(`${change.operation} must advance the revision by exactly one`)
      }
      if (change.operation === 'round') {
        // The one operation that is not idempotent under replay: it is what
        // spends the budget, so its accounting is checked hardest.
        if (folded.goal.phase !== 'active') rejectLog('round admitted against a non-active goal')
        if (change.roundsStarted !== folded.roundsStarted + 1) rejectLog('round numbers must be consecutive')
        if (change.roundsStarted > goal.maxGoalRounds) rejectLog('round admitted past the cap')
        if (!sameCriteria(folded.goal.criteria, goal.criteria)) rejectLog('round must preserve criteria')
        if (folded.goal.maxDurationMs !== goal.maxDurationMs) rejectLog('round must preserve duration')
        if (folded.goal.maxTotalTokens !== goal.maxTotalTokens) rejectLog('round must preserve total token cap')
        if (folded.goal.currentMilestone !== goal.currentMilestone) rejectLog('round must preserve current milestone')
      } else if (change.operation === 'evidence') {
        validateEvidenceTransition(folded.goal, goal)
        if (change.roundsStarted !== folded.roundsStarted) rejectLog('evidence must not change the round count')
        if (folded.goal.currentMilestone !== goal.currentMilestone) rejectLog('evidence must preserve current milestone')
      } else if (change.operation === 'edit') {
        validateEditCriteriaTransition(folded.goal, goal)
        if (change.roundsStarted !== folded.roundsStarted) rejectLog('edit must not change the round count')
      } else if (change.operation === 'milestone') {
        if (folded.goal.phase === 'complete') rejectLog('milestone cannot change a completed goal')
        if (change.roundsStarted !== folded.roundsStarted) rejectLog('milestone must not change the round count')
        if (folded.goal.objective !== goal.objective) rejectLog('milestone must preserve objective')
        if (folded.goal.phase !== goal.phase) rejectLog('milestone must preserve phase')
        if (!sameOptionalBlockReason(folded.goal.blockedReason, goal.blockedReason)) rejectLog('milestone must preserve blocker')
        if (folded.goal.maxGoalRounds !== goal.maxGoalRounds) rejectLog('milestone must preserve max rounds')
        if (folded.goal.maxDurationMs !== goal.maxDurationMs) rejectLog('milestone must preserve duration')
        if (folded.goal.maxTotalTokens !== goal.maxTotalTokens) rejectLog('milestone must preserve total token cap')
        if (!sameCriteria(folded.goal.criteria, goal.criteria)) rejectLog('milestone must preserve criteria')
      } else if (change.roundsStarted !== folded.roundsStarted) {
        rejectLog(`${change.operation} must not change the round count`)
      } else if (!sameCriteria(folded.goal.criteria, goal.criteria)) {
        rejectLog(`${change.operation} must preserve criteria`)
      } else if (folded.goal.maxDurationMs !== goal.maxDurationMs) {
        rejectLog(`${change.operation} must preserve duration`)
      } else if (folded.goal.maxTotalTokens !== goal.maxTotalTokens) {
        rejectLog(`${change.operation} must preserve total token cap`)
      } else if (folded.goal.currentMilestone !== goal.currentMilestone) {
        rejectLog(`${change.operation} must preserve current milestone`)
      }
    }
    if (change.roundsStarted < 0) rejectLog('round count must not be negative')
    if (goal.phase === 'complete') assertCompleteCriteria(goal)

    folded = {
      goal,
      roundsStarted: change.roundsStarted,
      createdAt: change.operation === 'create' ? change.createdAt : folded.createdAt ?? change.createdAt,
      updatedAt: change.updatedAt,
      lastRef: { id: goal.id, revision: goal.revision },
    }
  }
  return folded
}

/**
 * Refuse a change log this module could not have written.
 *
 * Thrown rather than repaired: a goal is the authority for unattended work, and
 * silently continuing from a best-guess reconstruction of it is strictly worse
 * than stopping and saying the state is not trustworthy.
 */
function rejectLog(detail: string): never {
  throw new GoalError(`goal change log is inconsistent: ${detail}`, 'GOAL_INVALID_TRANSITION')
}

/** Current goal for a session, or undefined before the first create / after a clear. */
export function getGoal(
  metadata: Readonly<Record<string, unknown>>,
  sessionId: string,
): GoalView | undefined {
  const folded = foldGoalChanges(readGoalChanges(metadata))
  if (!folded.goal) return undefined
  return Object.freeze({
    ...folded.goal,
    roundsStarted: folded.roundsStarted,
    createdAt: folded.createdAt ?? 0,
    updatedAt: folded.updatedAt ?? 0,
    activation: activations.get(sessionId) ?? 'disarmed',
  })
}

export interface CreateGoalRequest {
  readonly objective: string
  readonly currentMilestone?: string
  readonly maxGoalRounds?: number
  readonly maxDurationMs?: number
  readonly maxTotalTokens?: number
  readonly criteria?: readonly GoalCriterionSpec[]
}

export interface EditGoalRequest {
  readonly objective?: string
  readonly currentMilestone?: string | null
  readonly maxGoalRounds?: number
  readonly maxDurationMs?: number
  readonly maxTotalTokens?: number
  readonly criteria?: readonly GoalCriterionSpec[]
}

/**
 * Create and arm a goal.
 *
 * A completed goal may be replaced; every other phase must be cleared or
 * resumed instead, so a second create cannot silently discard work in flight.
 */
export function createGoal(
  metadata: Record<string, unknown>,
  sessionId: string,
  request: CreateGoalRequest,
  now: number,
): GoalView {
  const current = getGoal(metadata, sessionId)
  if (current && current.phase !== 'complete') {
    throw new GoalError(
      `goal "${current.id}" already exists with phase "${current.phase}"`,
      'GOAL_ALREADY_EXISTS',
    )
  }
  const objective = requireObjective(request.objective)
  const currentMilestone = request.currentMilestone === undefined ? undefined : requireCurrentMilestone(request.currentMilestone)
  const maxGoalRounds = requireMaxRounds(request.maxGoalRounds ?? DEFAULT_MAX_GOAL_ROUNDS)
  const maxDurationMs = request.maxDurationMs === undefined ? undefined : requireMaxDuration(request.maxDurationMs)
  if (maxDurationMs !== undefined) requireDeadline(now, maxDurationMs)
  const maxTotalTokens = request.maxTotalTokens === undefined ? undefined : requireMaxTotalTokens(request.maxTotalTokens)
  const criteria = request.criteria === undefined ? undefined : requireCriteria(request.criteria)
  const snapshot: GoalSnapshot = {
    id: `goal_${crypto.randomUUID()}`,
    revision: 1,
    objective,
    ...(currentMilestone === undefined ? {} : { currentMilestone }),
    phase: 'active',
    maxGoalRounds,
    ...(maxDurationMs === undefined ? {} : { maxDurationMs }),
    ...(maxTotalTokens === undefined ? {} : { maxTotalTokens }),
    ...(criteria === undefined ? {} : { criteria }),
  }
  append(metadata, {
    kind: 'goal/change',
    version: 1,
    operation: 'create',
    goal: snapshotOf(snapshot),
    roundsStarted: 0,
    createdAt: now,
    updatedAt: now,
  })
  activations.set(sessionId, 'armed')
  return getGoal(metadata, sessionId)!
}

/** Edit objective and/or cap without changing phase. At least one field is required. */
export function editGoal(
  metadata: Record<string, unknown>,
  sessionId: string,
  ref: GoalRef,
  request: EditGoalRequest,
  now: number,
): GoalView {
  const current = expectCurrent(metadata, sessionId, ref)
  if (request.objective === undefined && request.currentMilestone === undefined && request.maxGoalRounds === undefined && request.maxDurationMs === undefined && request.maxTotalTokens === undefined && request.criteria === undefined) {
    throw new GoalError('edit requires an objective, current_milestone, max_goal_rounds, max_duration_ms, max_total_tokens, or criteria', 'GOAL_INVALID_EDIT')
  }
  if (request.currentMilestone !== undefined && current.phase === 'complete') {
    throw new GoalError(`cannot edit current_milestone on completed goal "${current.id}"`, 'GOAL_INVALID_TRANSITION')
  }
  const objective = request.objective === undefined ? current.objective : requireObjective(request.objective)
  const currentMilestone = request.currentMilestone === undefined
    ? (objective === current.objective ? current.currentMilestone : undefined)
    : request.currentMilestone === null ? undefined : requireCurrentMilestone(request.currentMilestone)
  const maxGoalRounds = request.maxGoalRounds === undefined
    ? current.maxGoalRounds
    : requireMaxRounds(request.maxGoalRounds)
  const maxDurationMs = request.maxDurationMs === undefined
    ? current.maxDurationMs
    : requireMaxDuration(request.maxDurationMs)
  if (maxDurationMs !== undefined) requireDeadline(current.createdAt, maxDurationMs)
  const maxTotalTokens = request.maxTotalTokens === undefined
    ? current.maxTotalTokens
    : requireMaxTotalTokens(request.maxTotalTokens)
  const objectiveChanged = objective !== current.objective
  const criteria = request.criteria === undefined
    ? copyCriteria(current.criteria, !objectiveChanged)
    : mergeCriteria(requireCriteria(request.criteria), current.criteria, !objectiveChanged)
  const { currentMilestone: _oldMilestone, ...withoutMilestone } = current
  const next: GoalSnapshot = {
    ...withoutMilestone,
    objective,
    ...(currentMilestone === undefined ? {} : { currentMilestone }),
    maxGoalRounds,
    ...(maxDurationMs === undefined ? {} : { maxDurationMs }),
    ...(maxTotalTokens === undefined ? {} : { maxTotalTokens }),
    ...(criteria === undefined ? {} : { criteria }),
  }
  if (next.phase === 'complete') assertCompleteCriteria(next)
  return commit(metadata, sessionId, 'edit', next, undefined, now)
}

/** Set or clear the current milestone using the same revision discipline as other goal mutations. */
export function setGoalMilestone(
  metadata: Record<string, unknown>,
  sessionId: string,
  ref: GoalRef,
  value: string | null,
  now: number,
): GoalView {
  const current = expectCurrent(metadata, sessionId, ref)
  assertPhase(current, 'milestone', ['active', 'paused', 'blocked'])
  const currentMilestone = value === null ? undefined : requireCurrentMilestone(value)
  const { currentMilestone: _previousMilestone, ...withoutMilestone } = current
  return commit(metadata, sessionId, 'milestone', {
    ...withoutMilestone,
    ...(currentMilestone === undefined ? {} : { currentMilestone }),
  }, undefined, now)
}

/** Hold an active goal without abandoning it. */
export function pauseGoal(
  metadata: Record<string, unknown>,
  sessionId: string,
  ref: GoalRef,
  now: number,
): GoalView {
  const current = expectCurrent(metadata, sessionId, ref)
  assertPhase(current, 'pause', ['active'])
  return commit(metadata, sessionId, 'pause', withPhase(current, 'paused'), 'disarmed', now)
}

/**
 * Rearm a goal after a pause, a block, or a session resume.
 *
 * The rounds check is here rather than at continuation time so an exhausted
 * goal fails where a human can read the reason and raise the cap, instead of
 * silently never continuing.
 */
export function resumeGoal(
  metadata: Record<string, unknown>,
  sessionId: string,
  ref: GoalRef,
  now: number,
): GoalView {
  const current = expectCurrent(metadata, sessionId, ref)
  assertPhase(current, 'resume', ['active', 'paused', 'blocked'])
  const folded = foldGoalChanges(readGoalChanges(metadata))
  if (current.phase === 'active' && (activations.get(sessionId) ?? 'disarmed') === 'armed') {
    throw new GoalError(`goal "${current.id}" is already active and armed`, 'GOAL_INVALID_TRANSITION')
  }
  if (goalTimeRemainingMs(current, now) === 0) {
    throw new GoalError(
      `goal "${current.id}" time budget expired; raise max_duration_ms before resuming`,
      'GOAL_TIME_EXHAUSTED',
    )
  }
  if (folded.roundsStarted >= current.maxGoalRounds) {
    throw new GoalError(
      `goal "${current.id}" exhausted ${current.maxGoalRounds} goal rounds; raise max_goal_rounds before resuming`,
      'GOAL_ROUNDS_EXHAUSTED',
    )
  }
  return commit(metadata, sessionId, 'resume', withPhase(current, 'active'), 'armed', now)
}

/** Remaining wall-clock budget; undefined means the legacy goal has no cap. */
export function goalTimeRemainingMs(goal: Pick<GoalSnapshot, 'maxDurationMs'> & { readonly createdAt: number }, now: number): number | undefined {
  if (goal.maxDurationMs === undefined) return undefined
  const deadline = requireDeadline(goal.createdAt, goal.maxDurationMs)
  if (!Number.isFinite(now)) return 0
  return Math.max(0, deadline - now)
}

/** Mark a goal complete and disarm it. */
export function completeGoal(
  metadata: Record<string, unknown>,
  sessionId: string,
  ref: GoalRef,
  now: number,
): GoalView {
  const current = expectCurrent(metadata, sessionId, ref)
  assertPhase(current, 'complete', ['active', 'paused', 'blocked'])
  assertCompleteCriteria(current)
  return commit(metadata, sessionId, 'complete', withPhase(current, 'complete'), 'disarmed', now)
}

/** Record host-verified evidence for one declared completion criterion. */
export function recordGoalEvidence(
  metadata: Record<string, unknown>,
  sessionId: string,
  ref: GoalRef,
  criterionId: string,
  evidence: GoalCriterionEvidence,
  now: number,
): GoalView {
  const current = expectCurrent(metadata, sessionId, ref)
  const id = requireCriterionId(criterionId)
  const criteria = current.criteria
  if (criteria === undefined) {
    throw new GoalError('goal has no declared criteria', 'GOAL_INVALID_EVIDENCE')
  }
  const recorded = requireEvidence(evidence)
  assertPhase(current, 'evidence', recorded.kind === 'user-decision'
    ? ['active', 'paused', 'blocked']
    : ['active'])
  let found = false
  let unchanged = false
  const nextCriteria = criteria.map(criterion => {
    if (criterion.id !== id) return criterion
    found = true
    unchanged = sameEvidence(criterion.evidence, recorded)
    return { ...criterion, evidence: recorded }
  })
  if (!found) throw new GoalError(`unknown goal criterion "${id}"`, 'GOAL_INVALID_EVIDENCE')
  if (unchanged) return current
  return commit(metadata, sessionId, 'evidence', { ...current, criteria: nextCriteria }, undefined, now)
}

/** Mark an active goal blocked, with a durable reason, and disarm it. */
export function blockGoal(
  metadata: Record<string, unknown>,
  sessionId: string,
  ref: GoalRef,
  reason: GoalBlockReason,
  now: number,
): GoalView {
  const current = expectCurrent(metadata, sessionId, ref)
  assertPhase(current, 'block', ['active'])
  const blockedReason = requireBlockReason(reason)
  return commit(
    metadata,
    sessionId,
    'block',
    { ...withPhase(current, 'blocked'), blockedReason },
    'disarmed',
    now,
  )
}

/** Clear the current goal, retaining a tombstone so history stays readable. */
export function clearGoal(
  metadata: Record<string, unknown>,
  sessionId: string,
  ref: GoalRef,
  now: number,
): GoalRef {
  const current = expectCurrent(metadata, sessionId, ref)
  const cleared: GoalRef = { id: current.id, revision: current.revision + 1 }
  append(metadata, { kind: 'goal/change', version: 1, operation: 'clear', cleared, clearedAt: now })
  activations.delete(sessionId)
  return cleared
}

/**
 * Drop continuation authority without touching durable phase or revision.
 *
 * Used on session load, resume and fork: the goal keeps saying what it is,
 * while this process is no longer allowed to act on it unattended.
 */
export function disarmGoal(sessionId: string): void {
  if (activations.has(sessionId)) activations.set(sessionId, 'disarmed')
}

/** Test seam: forget every process-local activation. */
export function resetGoalActivations(): void {
  activations.clear()
}

/**
 * Admit the next continuation round.
 *
 * Returns the attribution the round's prompt carries, or undefined when the
 * goal is not eligible — not active, not armed, or out of capacity. Human
 * turns never call this, which is what keeps them from consuming the cap.
 */
export function admitGoalRound(
  metadata: Record<string, unknown>,
  sessionId: string,
  now: number,
): GoalMessageSource | undefined {
  const current = getGoal(metadata, sessionId)
  if (!current) return undefined
  if (current.phase !== 'active' || current.activation !== 'armed') return undefined
  if (current.roundsStarted >= current.maxGoalRounds) return undefined

  const round = current.roundsStarted + 1
  const changes = readGoalChanges(metadata)
  const folded = foldGoalChanges(changes)
  append(metadata, {
    kind: 'goal/change',
    version: 1,
    // A round re-commits the same phase under the next revision; the round
    // counter is what actually moves.
    operation: 'round',
    goal: snapshotOf({ ...current, revision: current.revision + 1 }),
    roundsStarted: round,
    createdAt: folded.createdAt ?? now,
    updatedAt: now,
  })
  return { kind: 'goal', goalId: current.id, revision: current.revision + 1, round }
}

// ── internals ──────────────────────────────────────────────────────────

/**
 * Append one change, compacting rather than truncating when the log is full.
 *
 * Dropping the oldest entries would be wrong here even though every entry
 * carries a whole snapshot: the strict fold verifies a revision chain, and a
 * log whose head has been cut is indistinguishable from one that was tampered
 * with. Because each entry IS a complete snapshot, the correct compaction is to
 * collapse the surviving prefix into the one snapshot it folds to and keep
 * appending from there — bounded metadata, intact chain, no lost current state.
 */
function append(metadata: Record<string, unknown>, change: GoalChange): void {
  const existing = readGoalChanges(metadata)
  const next = [...existing, change]
  // Reject inconsistent candidates before changing durable in-memory history.
  foldGoalChanges(next)
  if (next.length <= MAX_GOAL_CHANGES) {
    metadata[GOAL_CHANGES_KEY] = next
    return
  }
  const keep = next.slice(-(MAX_GOAL_CHANGES - 1))
  const baseline = baselineFor(next.slice(0, next.length - keep.length))
  const compacted = baseline ? [baseline, ...keep] : keep
  foldGoalChanges(compacted)
  metadata[GOAL_CHANGES_KEY] = compacted
}

/** The single snapshot change that a compacted prefix folds to, if any. */
function baselineFor(prefix: readonly GoalChange[]): GoalSnapshotChange | undefined {
  const folded = foldGoalChanges(prefix)
  if (!folded.goal) return undefined
  return {
    kind: 'goal/change',
    version: 1,
    operation: 'edit',
    goal: folded.goal,
    roundsStarted: folded.roundsStarted,
    createdAt: folded.createdAt ?? 0,
    updatedAt: folded.updatedAt ?? 0,
  }
}

/**
 * Reduce a live view back to exactly the fields the durable log may carry.
 *
 * `GoalView` extends the snapshot with derived and process-local values —
 * `roundsStarted`, timestamps, and above all `activation`, which exists
 * precisely because it must NOT survive a restart. Spreading a view into a
 * change event writes all of them into the transcript, and a later fold would
 * then read a persisted activation as though a human had authorised it.
 */
function snapshotOf(goal: GoalSnapshot): GoalSnapshot {
  const criteria = copyCriteria(goal.criteria, true)
  return {
    id: goal.id,
    revision: goal.revision,
    objective: goal.objective,
    ...(goal.currentMilestone === undefined ? {} : { currentMilestone: goal.currentMilestone }),
    phase: goal.phase,
    ...(goal.blockedReason ? { blockedReason: goal.blockedReason } : {}),
    maxGoalRounds: goal.maxGoalRounds,
    ...(goal.maxDurationMs === undefined ? {} : { maxDurationMs: goal.maxDurationMs }),
    ...(goal.maxTotalTokens === undefined ? {} : { maxTotalTokens: goal.maxTotalTokens }),
    ...(criteria === undefined ? {} : { criteria }),
  }
}

function commit(
  metadata: Record<string, unknown>,
  sessionId: string,
  operation: Exclude<GoalOperation, 'create' | 'clear'>,
  goal: GoalSnapshot,
  activation: GoalActivation | undefined,
  now: number,
): GoalView {
  const folded = foldGoalChanges(readGoalChanges(metadata))
  append(metadata, {
    kind: 'goal/change',
    version: 1,
    operation,
    goal: snapshotOf({ ...goal, revision: goal.revision + 1 }),
    roundsStarted: folded.roundsStarted,
    createdAt: folded.createdAt ?? now,
    updatedAt: now,
  })
  if (activation) activations.set(sessionId, activation)
  return getGoal(metadata, sessionId)!
}

function expectCurrent(
  metadata: Readonly<Record<string, unknown>>,
  sessionId: string,
  ref: GoalRef,
): GoalView {
  const current = getGoal(metadata, sessionId)
  if (!current) throw new GoalError('no current goal', 'GOAL_NOT_FOUND')
  if (current.id !== ref.id) {
    throw new GoalError(`goal "${ref.id}" is not the current goal`, 'GOAL_NOT_FOUND')
  }
  if (current.revision !== ref.revision) {
    throw new GoalError(
      `stale revision ${ref.revision}; current is ${current.revision}`,
      'GOAL_STALE_REVISION',
    )
  }
  return current
}

function assertPhase(
  current: GoalSnapshot,
  operation: GoalOperation,
  allowed: readonly GoalPhase[],
): void {
  if (allowed.includes(current.phase)) return
  throw new GoalError(
    `cannot ${operation} goal "${current.id}" from phase "${current.phase}"; expected ${allowed.join(' or ')}`,
    'GOAL_INVALID_TRANSITION',
  )
}

const withPhase = (goal: GoalSnapshot, phase: GoalPhase): GoalSnapshot => {
  // A phase that is no longer blocked must not keep carrying its reason.
  const { blockedReason: _dropped, ...rest } = goal
  return { ...rest, phase }
}

function requireCriteria(value: readonly GoalCriterionSpec[]): readonly GoalCriterion[] {
  if (!Array.isArray(value) || value.length > MAX_GOAL_CRITERIA) {
    throw new GoalError(`criteria must contain at most ${MAX_GOAL_CRITERIA} items`, 'GOAL_INVALID_CRITERIA')
  }
  const seen = new Set<string>()
  const criteria = value.map((criterion, index) => {
    if (criterion === null || typeof criterion !== 'object') {
      throw new GoalError(`criterion ${index + 1} must be an object`, 'GOAL_INVALID_CRITERIA')
    }
    const id = requireCriterionId((criterion as GoalCriterionSpec).id)
    if (seen.has(id)) throw new GoalError(`criterion id "${id}" is duplicated`, 'GOAL_INVALID_CRITERIA')
    seen.add(id)
    const description = requireCriterionDescription((criterion as GoalCriterionSpec).description)
    return { id, description }
  })
  requireCriteriaSize(criteria)
  return criteria
}

function requireCriterionId(value: unknown): string {
  if (typeof value !== 'string') throw new GoalError('criterion id must be a string', 'GOAL_INVALID_CRITERIA')
  const id = value.trim()
  if (!id || id.length > MAX_CRITERION_ID_CHARS) {
    throw new GoalError(`criterion id must be non-empty and at most ${MAX_CRITERION_ID_CHARS} characters`, 'GOAL_INVALID_CRITERIA')
  }
  return id
}

function requireCriterionDescription(value: unknown): string {
  if (typeof value !== 'string') throw new GoalError('criterion description must be a string', 'GOAL_INVALID_CRITERIA')
  const description = value.trim()
  if (!description || description.length > MAX_CRITERION_DESCRIPTION_CHARS) {
    throw new GoalError(`criterion description must be non-empty and at most ${MAX_CRITERION_DESCRIPTION_CHARS} characters`, 'GOAL_INVALID_CRITERIA')
  }
  return description
}

function mergeCriteria(
  specs: readonly GoalCriterion[],
  previous: readonly GoalCriterion[] | undefined,
  preserveEvidence: boolean,
): readonly GoalCriterion[] {
  const prior = new Map(previous?.map(criterion => [criterion.id, criterion]) ?? [])
  return specs.map(spec => {
    const old = prior.get(spec.id)
    return preserveEvidence && old?.description === spec.description && old.evidence !== undefined
      ? { ...spec, evidence: copyEvidence(old.evidence) }
      : spec
  })
}

function copyCriteria(
  criteria: readonly GoalCriterion[] | undefined,
  preserveEvidence: boolean,
): readonly GoalCriterion[] | undefined {
  if (criteria === undefined) return undefined
  const copy = criteria.map(criterion => ({
    id: criterion.id,
    description: criterion.description,
    ...(preserveEvidence && criterion.evidence ? { evidence: copyEvidence(criterion.evidence) } : {}),
  }))
  requireCriteriaSize(copy)
  return copy
}

function requireCriteriaSize(criteria: readonly GoalCriterion[]): void {
  const bytes = new TextEncoder().encode(JSON.stringify(criteria)).byteLength
  if (bytes > MAX_CRITERIA_JSON_BYTES) {
    throw new GoalError(`criteria must serialize to at most ${MAX_CRITERIA_JSON_BYTES} UTF-8 bytes`, 'GOAL_INVALID_CRITERIA')
  }
}

function copyEvidence(evidence: GoalCriterionEvidence): GoalCriterionEvidence {
  if (evidence.kind === 'user-decision') {
    return {
      kind: 'user-decision',
      decisionId: evidence.decisionId,
      summary: evidence.summary,
      recordedAt: evidence.recordedAt,
    }
  }
  return {
    toolCallId: evidence.toolCallId,
    summary: evidence.summary,
    recordedAt: evidence.recordedAt,
  }
}

function requireEvidence(value: GoalCriterionEvidence): GoalCriterionEvidence {
  if (value === null || typeof value !== 'object') {
    throw new GoalError('criterion evidence must be an object', 'GOAL_INVALID_EVIDENCE')
  }
  const summary = typeof value.summary === 'string' ? value.summary.trim() : ''
  if (!summary || summary.length > MAX_EVIDENCE_SUMMARY_CHARS) {
    throw new GoalError(`evidence summary must be non-empty and at most ${MAX_EVIDENCE_SUMMARY_CHARS} characters`, 'GOAL_INVALID_EVIDENCE')
  }
  if (!Number.isFinite(value.recordedAt)) {
    throw new GoalError('evidence recordedAt must be finite', 'GOAL_INVALID_EVIDENCE')
  }
  if (value.kind === 'user-decision') {
    const decisionId = typeof value.decisionId === 'string' ? value.decisionId.trim() : ''
    if (!decisionId || decisionId.length > MAX_EVIDENCE_DECISION_ID_CHARS || value.toolCallId !== undefined) {
      throw new GoalError(`evidence decisionId must be non-empty and at most ${MAX_EVIDENCE_DECISION_ID_CHARS} characters`, 'GOAL_INVALID_EVIDENCE')
    }
    return { kind: 'user-decision', decisionId, summary, recordedAt: value.recordedAt }
  }
  if (value.kind !== undefined && value.kind !== 'tool-result') {
    throw new GoalError('evidence kind must be user-decision or tool-result', 'GOAL_INVALID_EVIDENCE')
  }
  if (value.decisionId !== undefined) {
    throw new GoalError('tool evidence cannot include decisionId', 'GOAL_INVALID_EVIDENCE')
  }
  const toolCallId = typeof value.toolCallId === 'string' ? value.toolCallId.trim() : ''
  if (!toolCallId || toolCallId.length > MAX_EVIDENCE_TOOL_CALL_ID_CHARS) {
    throw new GoalError(`evidence toolCallId must be non-empty and at most ${MAX_EVIDENCE_TOOL_CALL_ID_CHARS} characters`, 'GOAL_INVALID_EVIDENCE')
  }
  return { toolCallId, summary, recordedAt: value.recordedAt }
}

function validateEvidenceTransition(previous: GoalSnapshot, next: GoalSnapshot): void {
  if (previous.objective !== next.objective || previous.maxGoalRounds !== next.maxGoalRounds) {
    rejectLog('evidence must preserve objective and max rounds')
  }
  if (previous.maxDurationMs !== next.maxDurationMs) rejectLog('evidence must preserve duration')
  if (previous.maxTotalTokens !== next.maxTotalTokens) rejectLog('evidence must preserve total token cap')
  if (!sameOptionalBlockReason(previous.blockedReason, next.blockedReason)) {
    rejectLog('evidence must preserve blocked reason')
  }
  const before = previous.criteria
  const after = next.criteria
  if (before === undefined || after === undefined || before.length !== after.length) {
    rejectLog('evidence must preserve declared criteria')
  }
  let evidenceChanges = 0
  let changedEvidence: GoalCriterionEvidence | undefined
  for (let index = 0; index < before.length; index += 1) {
    const oldCriterion = before[index]!
    const newCriterion = after[index]!
    if (oldCriterion.id !== newCriterion.id || oldCriterion.description !== newCriterion.description) {
      rejectLog('evidence must preserve criterion specifications')
    }
    if (oldCriterion.evidence !== undefined && newCriterion.evidence === undefined) {
      rejectLog('evidence cannot remove criterion evidence')
    }
    if (!sameEvidence(oldCriterion.evidence, newCriterion.evidence)) {
      evidenceChanges += 1
      changedEvidence = newCriterion.evidence
    }
  }
  if (evidenceChanges !== 1) rejectLog('evidence must change exactly one criterion')
  const allowedPhases: readonly GoalPhase[] = changedEvidence?.kind === 'user-decision'
    ? ['active', 'paused', 'blocked']
    : ['active']
  if (!allowedPhases.includes(previous.phase) || !allowedPhases.includes(next.phase)) {
    rejectLog(changedEvidence?.kind === 'user-decision'
      ? 'user-decision evidence recorded against an invalid phase'
      : 'tool evidence recorded against a non-active goal')
  }
  if (previous.phase !== next.phase) rejectLog('evidence must preserve phase')
}

function validateEditCriteriaTransition(previous: GoalSnapshot, next: GoalSnapshot): void {
  if (previous.objective !== next.objective) {
    if (previous.criteria !== undefined && next.criteria === undefined) {
      rejectLog('objective edit must preserve the declared criteria')
    }
    assertNoCriteriaEvidence(next)
    return
  }
  if (previous.criteria === undefined) {
    assertNoCriteriaEvidence(next)
    return
  }
  if (next.criteria === undefined) rejectLog('edit must preserve declared criteria')
  const prior = new Map(previous.criteria.map(criterion => [criterion.id, criterion]))
  for (const criterion of next.criteria) {
    const old = prior.get(criterion.id)
    if (old?.description === criterion.description) {
      if (!sameEvidence(old.evidence, criterion.evidence)) rejectLog('edit must preserve unchanged criterion evidence')
    } else if (criterion.evidence !== undefined) {
      rejectLog('edit must leave new or changed criteria pending')
    }
  }
}

function assertNoCriteriaEvidence(goal: GoalSnapshot): void {
  if (goal.criteria?.some(criterion => criterion.evidence !== undefined)) {
    rejectLog('new goals cannot carry criterion evidence')
  }
}

function assertCompleteCriteria(goal: GoalSnapshot): void {
  if (goal.criteria?.some(criterion => criterion.evidence === undefined)) {
    throw new GoalError('all goal criteria require verified evidence before completion', 'GOAL_INVALID_TRANSITION')
  }
}

function sameCriteria(
  left: readonly GoalCriterion[] | undefined,
  right: readonly GoalCriterion[] | undefined,
): boolean {
  if (left === undefined || right === undefined) return left === right
  if (left.length !== right.length) return false
  return left.every((criterion, index) => {
    const other = right[index]!
    return criterion.id === other.id
      && criterion.description === other.description
      && sameEvidence(criterion.evidence, other.evidence)
  })
}

function sameOptionalBlockReason(left: GoalBlockReason | undefined, right: GoalBlockReason | undefined): boolean {
  return left?.code === right?.code && left?.message === right?.message
}

function sameEvidence(left: GoalCriterionEvidence | undefined, right: GoalCriterionEvidence | undefined): boolean {
  if (left?.kind === 'user-decision' || right?.kind === 'user-decision') {
    return left?.kind === 'user-decision'
      && right?.kind === 'user-decision'
      && left.decisionId === right.decisionId
      && left.summary === right.summary
      && left.recordedAt === right.recordedAt
  }
  return left?.toolCallId === right?.toolCallId
    && left?.summary === right?.summary
    && left?.recordedAt === right?.recordedAt
}

function requireObjective(value: string): string {
  const objective = value.trim()
  if (!objective) throw new GoalError('objective must be a non-empty string', 'GOAL_INVALID_OBJECTIVE')
  return objective.length > MAX_OBJECTIVE_CHARS
    ? `${objective.slice(0, MAX_OBJECTIVE_CHARS - 1)}…`
    : objective
}

function requireCurrentMilestone(value: unknown): string {
  if (typeof value !== 'string') throw new GoalError('current milestone must be a string', 'GOAL_INVALID_MILESTONE')
  const milestone = value.trim()
  if (!milestone || milestone.length > MAX_CURRENT_MILESTONE_CHARS) {
    throw new GoalError(`current milestone must be non-empty and at most ${MAX_CURRENT_MILESTONE_CHARS} characters`, 'GOAL_INVALID_MILESTONE')
  }
  return milestone
}

function requireMaxRounds(value: number): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new GoalError('max_goal_rounds must be a positive integer', 'GOAL_INVALID_MAX_ROUNDS')
  }
  return value
}

function requireMaxDuration(value: number): number {
  if (!Number.isSafeInteger(value) || value < 1 || value > MAX_GOAL_DURATION_MS) {
    throw new GoalError('max_duration_ms must be a positive safe integer', 'GOAL_INVALID_DURATION')
  }
  return value
}

function requireMaxTotalTokens(value: number): number {
  if (!Number.isSafeInteger(value) || value < 1 || value > MAX_GOAL_TOTAL_TOKENS) {
    throw new GoalError('max_total_tokens must be a positive safe integer', 'GOAL_INVALID_TOTAL_TOKENS')
  }
  return value
}

function requireDeadline(createdAt: number, maxDurationMs: number): number {
  if (!Number.isSafeInteger(createdAt) || createdAt < 0) {
    throw new GoalError('goal creation time must be a non-negative safe integer', 'GOAL_INVALID_DURATION')
  }
  const deadline = createdAt + maxDurationMs
  if (!Number.isSafeInteger(deadline)) {
    throw new GoalError('goal deadline exceeds the safe integer range', 'GOAL_INVALID_DURATION')
  }
  return deadline
}

function validateGoalDuration(goal: GoalSnapshot, createdAt: number): void {
  if (goal.maxDurationMs === undefined) return
  requireMaxDuration(goal.maxDurationMs)
  requireDeadline(createdAt, goal.maxDurationMs)
}

function requireBlockReason(reason: GoalBlockReason): GoalBlockReason {
  const message = reason.message?.trim() ?? ''
  if (!message) throw new GoalError('blocked_reason must explain the blocker', 'GOAL_INVALID_BLOCK_REASON')
  const code = reason.code?.trim() || 'model-reported'
  return { code, message }
}

function isGoalChange(value: unknown): value is GoalChange {
  if (value === null || typeof value !== 'object') return false
  const record = value as Record<string, unknown>
  if (record.kind !== 'goal/change' || record.version !== 1) return false
  if (record.operation === 'clear') return isRef(record.cleared) && typeof record.clearedAt === 'number'
  if (record.operation !== 'create'
    && record.operation !== 'edit'
    && record.operation !== 'pause'
    && record.operation !== 'resume'
    && record.operation !== 'complete'
    && record.operation !== 'block'
    && record.operation !== 'round'
    && record.operation !== 'evidence'
    && record.operation !== 'milestone') return false
  return isSnapshot(record.goal)
    && typeof record.roundsStarted === 'number'
    && Number.isFinite(record.roundsStarted)
    && typeof record.createdAt === 'number'
    && Number.isFinite(record.createdAt)
    && typeof record.updatedAt === 'number'
    && Number.isFinite(record.updatedAt)
}

function isGoalChangeEnvelope(value: unknown): boolean {
  if (value === null || typeof value !== 'object') return false
  const record = value as Record<string, unknown>
  return record.kind === 'goal/change' && record.version === 1
}

function isRef(value: unknown): value is GoalRef {
  if (value === null || typeof value !== 'object') return false
  const record = value as Record<string, unknown>
  return typeof record.id === 'string' && typeof record.revision === 'number'
}

function isSnapshot(value: unknown): value is GoalSnapshot {
  if (!isRef(value)) return false
  const record = value as unknown as Record<string, unknown>
  if (typeof record.objective !== 'string' || typeof record.phase !== 'string' || typeof record.maxGoalRounds !== 'number') {
    return false
  }
  if (Object.prototype.hasOwnProperty.call(record, 'currentMilestone')
    && (typeof record.currentMilestone !== 'string'
      || record.currentMilestone !== record.currentMilestone.trim()
      || !record.currentMilestone
      || record.currentMilestone.length > MAX_CURRENT_MILESTONE_CHARS)) return false
  if (Object.prototype.hasOwnProperty.call(record, 'maxDurationMs')
    && (typeof record.maxDurationMs !== 'number' || !Number.isSafeInteger(record.maxDurationMs) || record.maxDurationMs < 1)) {
    return false
  }
  if (Object.prototype.hasOwnProperty.call(record, 'maxTotalTokens')
    && (typeof record.maxTotalTokens !== 'number' || !Number.isSafeInteger(record.maxTotalTokens) || record.maxTotalTokens < 1)) {
    return false
  }
  if (!Object.prototype.hasOwnProperty.call(record, 'criteria')) return true
  return isCriteria(record.criteria)
}

function isCriteria(value: unknown): value is readonly GoalCriterion[] {
  if (!Array.isArray(value) || value.length > MAX_GOAL_CRITERIA) return false
  const seen = new Set<string>()
  for (const criterion of value) {
    if (criterion === null || typeof criterion !== 'object') return false
    const record = criterion as Record<string, unknown>
    if (typeof record.id !== 'string' || typeof record.description !== 'string') return false
    const id = record.id.trim()
    const description = record.description.trim()
    if (record.id !== id || record.description !== description) return false
    if (!id || id.length > MAX_CRITERION_ID_CHARS || !description || description.length > MAX_CRITERION_DESCRIPTION_CHARS) return false
    if (seen.has(id)) return false
    seen.add(id)
    if (Object.prototype.hasOwnProperty.call(record, 'evidence') && !isEvidence(record.evidence)) return false
  }
  try {
    requireCriteriaSize(value)
  } catch {
    return false
  }
  return true
}

function isEvidence(value: unknown): value is GoalCriterionEvidence {
  if (value === null || typeof value !== 'object') return false
  const record = value as Record<string, unknown>
  const summary = typeof record.summary === 'string' ? record.summary.trim() : ''
  if (record.summary !== summary || summary.length === 0 || summary.length > MAX_EVIDENCE_SUMMARY_CHARS
    || typeof record.recordedAt !== 'number' || !Number.isFinite(record.recordedAt)) return false
  if (record.kind === 'user-decision') {
    const decisionId = typeof record.decisionId === 'string' ? record.decisionId.trim() : ''
    return record.decisionId === decisionId
      && decisionId.length > 0
      && decisionId.length <= MAX_EVIDENCE_DECISION_ID_CHARS
      && !Object.prototype.hasOwnProperty.call(record, 'toolCallId')
  }
  if (record.kind !== undefined && record.kind !== 'tool-result') return false
  if (Object.prototype.hasOwnProperty.call(record, 'decisionId')) return false
  const toolCallId = typeof record.toolCallId === 'string' ? record.toolCallId.trim() : ''
  return record.toolCallId === toolCallId
    && toolCallId.length > 0
    && toolCallId.length <= MAX_EVIDENCE_TOOL_CALL_ID_CHARS
}
