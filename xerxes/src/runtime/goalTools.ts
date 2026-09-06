// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Model-facing goal control: `get_goal`, `create_goal`, `update_goal`.
 *
 * Replaces inferring the objective's lifecycle from the model's prose. The old
 * guard grepped assistant text for English phrases — "objective met", "all
 * tests pass" — which meant a model writing in another language, or simply
 * phrasing success differently, could never end a goal, while an innocent
 * "should I continue?" was classified as premature stopping. Lifecycle is now
 * something the model states through a typed call that either succeeds or
 * returns a reason.
 *
 * The authority split follows DeepSeek Harness's goal tools
 * (github.com/deepseek-ai/deepseek-harness, MIT): creating, editing, pausing
 * and resuming require a direct human turn, while completing and blocking are
 * additionally reachable from the goal's own continuation round. Subagents get
 * none of it. Written against Xerxes's tool registry; no source is reproduced.
 *
 * Declared criteria require a record linked to a successful session tool call.
 * The host checks execution outcome; the model explains relevance. This is
 * an auditable claim, not evaluator-backed certification or a command whitelist.
 */

import { ValidationError } from '../core/errors.js'
import { successfulGoalEvidence } from './goalEvidence.js'
import type { ToolExecutionContext } from '../executors/toolRegistry.js'
import { ToolRegistry } from '../executors/toolRegistry.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'
import {
  DEFAULT_MAX_GOAL_ROUNDS,
  GoalError,
  blockGoal,
  completeGoal,
  createGoal,
  editGoal,
  getGoal,
  pauseGoal,
  resumeGoal,
  recordGoalEvidence,
  setGoalMilestone,
  type GoalCriterionSpec,
  type GoalView,
} from './goalDomain.js'

/**
 * Consecutive rounds a blocker must persist before the model may self-block.
 *
 * A mechanical floor, not a judgement: the model still decides whether the
 * condition is genuinely the same one, but it cannot declare a blocker on its
 * first frustration.
 */
export const DEFAULT_BLOCKED_AFTER_CONSECUTIVE_ROUNDS = 3

export interface GoalToolHost {
  /** Session whose metadata owns the goal. */
  sessionId(context: ToolExecutionContext): string
  /** Mutable metadata for that session. */
  metadata(context: ToolExecutionContext): Record<string, unknown>
  /** Whether this turn was opened by a direct human message rather than machinery. */
  isHumanTurn(context: ToolExecutionContext): boolean
  /** The current goal round when this turn is one, else undefined. */
  currentRound(context: ToolExecutionContext): number | undefined
  /** A completed execution from this session, resolved by the host rather than model-supplied data. */
  evidenceExecution?(context: ToolExecutionContext, toolCallId: string): unknown
  goalCreated?(context: ToolExecutionContext, goal: GoalView): void
  tokenUsage?(context: ToolExecutionContext, goal: GoalView): unknown
  validateResume?(context: ToolExecutionContext, goal: GoalView): void
  now?(): number
}

export interface GoalToolOptions {
  readonly blockedAfterConsecutiveRounds?: number
}

const goalView = (goal: GoalView | undefined) =>
  goal === undefined
    ? { goal: null }
    : {
        goal: {
          id: goal.id,
          revision: goal.revision,
          objective: goal.objective,
          ...(goal.currentMilestone === undefined ? {} : { currentMilestone: goal.currentMilestone }),
          phase: goal.phase,
          roundsStarted: goal.roundsStarted,
          maxGoalRounds: goal.maxGoalRounds,
          ...(goal.maxTotalTokens === undefined ? {} : { maxTotalTokens: goal.maxTotalTokens }),
          ...(goal.maxDurationMs === undefined ? {} : { maxDurationMs: goal.maxDurationMs, deadlineAt: goal.createdAt + goal.maxDurationMs }),
          ...(goal.criteria ? { criteria: goal.criteria } : {}),
          ...(goal.blockedReason ? { blockedReason: goal.blockedReason } : {}),
        },
        activation: goal.activation,
      }

function criterionSchema() {
  return { type: 'array', maxItems: 32, description: 'Explicit completion criteria; create or edit only. Keep stable IDs when unchanged.',
    items: { type: 'object', additionalProperties: false, required: ['id', 'description'], properties: {
      id: { type: 'string', maxLength: 80 }, description: { type: 'string', maxLength: 1000 },
    } } }
}

function criteriaInput(value: unknown): readonly GoalCriterionSpec[] | undefined {
  if (value === undefined) return undefined
  if (!Array.isArray(value) || value.length > 32) throw new ValidationError('criteria', 'must be an array of at most 32 criteria')
  return value.map(item => {
    if (!item || typeof item !== 'object' || Array.isArray(item) || typeof item.id !== 'string' || typeof item.description !== 'string'
      || Object.keys(item).some(key => key !== 'id' && key !== 'description')) throw new ValidationError('criteria', 'each criterion must contain only id and description')
    return { id: item.id, description: item.description }
  })
}

function milestoneInput(inputs: JsonObject): string | null | undefined {
  const value = inputs.current_milestone
  if (value === undefined || value === null) return value
  if (typeof value !== 'string') throw new ValidationError('current_milestone', 'must be text or null to clear')
  return value
}

export const GOAL_TOOL_DEFINITIONS: readonly ToolDefinition[] = Object.freeze([
  {
    type: 'function',
    function: {
      name: 'get_goal',
      description:
        'Read the current session goal, or null when there is none. Returns the compare-and-set id and '
        + 'revision that update_goal requires, the durable phase, admitted and capped rounds, any blocker '
        + 'reason, and whether this process may continue the goal automatically.',
      parameters: { type: 'object', additionalProperties: false, properties: {} },
    },
  },
  {
    type: 'function',
    function: {
      name: 'create_goal',
      description:
        'Create one long-running completion objective for this session. Infer goal intent from a direct '
        + 'human request in any language; do not create a goal for routine single-turn work.',
      parameters: {
        type: 'object',
        additionalProperties: false,
        required: ['objective'],
        properties: {
          objective: { type: 'string', description: 'The completion objective, as the human stated it.' },
          current_milestone: { type: 'string', maxLength: 1000, description: 'Optional current work milestone. Progress context only, not completion evidence.' },
          criteria: criterionSchema(),
          max_duration_ms: { type: 'integer', minimum: 1, description: 'Optional wall-time limit in milliseconds from goal creation, including paused time. Set only from the human request.' },
          max_total_tokens: { type: 'integer', minimum: 1, description: 'Optional total counted token admission cap. Starts with provider calls after goal creation; includes descendants and cache tokens. Already admitted concurrent calls can exceed the cap. Set only from the human request.' },
          max_goal_rounds: {
            type: 'integer',
            description: `Total automatic continuation rounds allowed. Defaults to ${DEFAULT_MAX_GOAL_ROUNDS}.`,
          },
        },
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'update_goal',
      description:
        'Change the current goal. Call get_goal first and copy its exact goal_id and revision. '
        + 'Replacements belong only to action "edit"; blocked_reason is required only for action "blocked".',
      parameters: {
        type: 'object',
        additionalProperties: false,
        required: ['goal_id', 'revision', 'action'],
        properties: {
          goal_id: { type: 'string', description: 'Exact id from get_goal.' },
          revision: { type: 'integer', description: 'Exact revision from get_goal.' },
          action: {
            type: 'string',
            enum: ['edit', 'pause', 'resume', 'complete', 'blocked', 'record_evidence', 'milestone'],
            description: 'Lifecycle transition to apply.',
          },
          objective: { type: 'string', description: 'Replacement objective; action "edit" only.' },
          current_milestone: { type: ['string', 'null'], maxLength: 1000, description: 'Current work milestone; milestone or edit action only. Null clears it. Does not change objective, budgets, evidence, or phase.' },
          criteria: criterionSchema(),
          criterion_id: { type: 'string', description: 'Criterion receiving evidence; record_evidence only.' },
          tool_call_id: { type: 'string', description: 'Exact completed successful tool call in this session; record_evidence only.' },
          evidence_summary: { type: 'string', description: 'Explain what this result establishes for the criterion. Relevance is your assessment, not an automatic certification.' },
          max_goal_rounds: { type: 'integer', description: 'Replacement round cap; action "edit" only.' },
          max_duration_ms: { type: 'integer', minimum: 1, description: 'Replacement wall-time limit from original creation; action "edit" only. Requires human authorization.' },
          max_total_tokens: { type: 'integer', minimum: 1, description: 'Replacement total token admission cap; edit only, preserves recorded spend and requires human authorization.' },
          blocked_reason: {
            type: 'string',
            description: 'The concrete condition that persists; action "blocked" only.',
          },
        },
      },
    },
  },
])

/** Model guidance, with the configured threshold interpolated. */
export function goalPolicyPrompt(blockedAfterConsecutiveRounds: number): string {
  return [
    '[Goal policy]',
    'Use the goal tools for one long-running completion objective in the current session. create_goal may',
    'infer goal intent from a direct human request in any language; do not create a goal for routine',
    'single-turn work. Call get_goal before update_goal and copy its exact goal_id and revision.',
    'Declare concrete criteria when creating a goal. Attach evidence with record_evidence using the exact',
    'tool_call_id of a successful completed call in this session and explain its relevance. Every declared',
    'criterion needs evidence before completion. Execution success is not automatic proof of relevance.',
    'Keep the current milestone up to date with action milestone and current_milestone. This records',
    'what you are working on; it does not prove a criterion or change the goal objective or budget.',
    'The user can accept a criterion with a decision note in F10. Such evidence is labelled user-decision',
    'in get_goal. You cannot create a user decision; record_evidence only records tool-result evidence.',
    'After a session resume or fork an active goal is disarmed: when a human asks to continue or resume, in any',
    'wording or language, use update_goal action resume to rearm it. Mark complete only when the objective',
    'is actually achieved and THIS turn ran the check that proves it — not your recollection of an earlier',
    'round, and not a plausible argument that it must be true. If you have not run that check yet, run it',
    'before calling update_goal. Mark blocked only after the same blocking',
    `condition has persisted for at least ${blockedAfterConsecutiveRounds} consecutive goal rounds, and`,
    'describe that concrete condition in blocked_reason; difficulty, uncertainty, or useful remaining work',
    'is not blocked.',
  ].join('\n')
}

/** Register the three goal tools against a host-owned live session. */
export function registerGoalTools(
  registry: ToolRegistry,
  host: GoalToolHost,
  options: GoalToolOptions = {},
  agentId = 'default',
): void {
  const threshold = options.blockedAfterConsecutiveRounds ?? DEFAULT_BLOCKED_AFTER_CONSECUTIVE_ROUNDS
  if (!Number.isSafeInteger(threshold) || threshold < 1) {
    throw new ValidationError('blockedAfterConsecutiveRounds', 'must be a positive integer', threshold)
  }
  const now = () => host.now?.() ?? Date.now()
  const view = (goal: GoalView | undefined) => goalView(goal)
  const capabilities = {
    concurrencySafe: false,
    defer: false,
    destructive: false,
    openWorld: false,
    readOnly: false,
  }

  const byName: Record<string, (inputs: JsonObject, context: ToolExecutionContext) => unknown> = {
    get_goal: (_inputs, context) => {
      assertMainAgent(context)
      const goal = getGoal(host.metadata(context), host.sessionId(context))
      return { ...view(goal), ...(goal && host.tokenUsage ? { token_usage: host.tokenUsage(context, goal) } : {}) }
    },

    create_goal: (inputs, context) => {
      assertMainAgent(context)
      // A goal commits the session to autonomous work, so only a human may open
      // one. `Agent.followup()`-style machinery must not inherit that authority.
      assertHumanAuthority(host, context, 'create_goal')
      const objective = requiredString(inputs, 'objective')
      const maxGoalRounds = optionalInteger(inputs, 'max_goal_rounds')
      const maxDurationMs = optionalInteger(inputs, 'max_duration_ms')
      const maxTotalTokens = optionalInteger(inputs, 'max_total_tokens')
      const criteria = criteriaInput(inputs.criteria)
      const currentMilestone = milestoneInput(inputs)
      if (currentMilestone === null) throw new ValidationError('current_milestone', 'omit it when creating a goal without a milestone')
      return wrap(() => {
        const goal = createGoal(
          host.metadata(context),
          host.sessionId(context),
          { objective, ...(maxGoalRounds === undefined ? {} : { maxGoalRounds }), ...(maxDurationMs === undefined ? {} : { maxDurationMs }), ...(maxTotalTokens === undefined ? {} : { maxTotalTokens }), ...(criteria === undefined ? {} : { criteria }), ...(currentMilestone === undefined ? {} : { currentMilestone }) },
          now(),
        )
        host.goalCreated?.(context, goal)
        return view(goal)
      })
    },

    update_goal: (inputs, context) => {
      assertMainAgent(context)
      const action = requiredString(inputs, 'action')
      if (inputs.current_milestone !== undefined && action !== 'edit' && action !== 'milestone') throw new ValidationError('current_milestone', 'is only accepted for action edit or milestone')
      if (action === 'milestone' && Object.keys(inputs).some(key => !['goal_id', 'revision', 'action', 'current_milestone'].includes(key))) throw new ValidationError('action', 'milestone changes only current_milestone')
      if (inputs.criteria !== undefined && action !== 'edit') throw new ValidationError('criteria', 'is only accepted for action edit')
      if (inputs.max_duration_ms !== undefined && action !== 'edit') throw new ValidationError('max_duration_ms', 'is only accepted for action edit')
      if (inputs.max_total_tokens !== undefined && action !== 'edit') throw new ValidationError('max_total_tokens', 'is only accepted for action edit')
      const ref = { id: requiredString(inputs, 'goal_id'), revision: requiredIntegerField(inputs, 'revision') }
      const metadata = host.metadata(context)
      const sessionId = host.sessionId(context)

      if (action === 'edit' || action === 'pause' || action === 'resume') {
        assertHumanAuthority(host, context, `update_goal action ${action}`)
      }

      return wrap(() => {
        switch (action) {
          case 'edit': {
            const objective = optionalString(inputs, 'objective')
            const maxGoalRounds = optionalInteger(inputs, 'max_goal_rounds')
            const maxDurationMs = optionalInteger(inputs, 'max_duration_ms')
            const maxTotalTokens = optionalInteger(inputs, 'max_total_tokens')
            const criteria = criteriaInput(inputs.criteria)
            const currentMilestone = milestoneInput(inputs)
            return view(editGoal(metadata, sessionId, ref, {
              ...(objective === undefined ? {} : { objective }),
              ...(maxGoalRounds === undefined ? {} : { maxGoalRounds }),
              ...(maxDurationMs === undefined ? {} : { maxDurationMs }),
              ...(maxTotalTokens === undefined ? {} : { maxTotalTokens }),
              ...(criteria === undefined ? {} : { criteria }),
              ...(currentMilestone === undefined ? {} : { currentMilestone }),
            }, now()))
          }
          case 'milestone': {
            assertConcludeAuthority(host, context, 'update milestone', expectCurrentGoal(metadata, sessionId, ref))
            const currentMilestone = milestoneInput(inputs)
            if (currentMilestone === undefined) throw new ValidationError('current_milestone', 'is required for action milestone; use null to clear')
            return view(setGoalMilestone(metadata, sessionId, ref, currentMilestone, now()))
          }
          case 'record_evidence': {
            assertConcludeAuthority(host, context, 'record evidence', expectCurrentGoal(metadata, sessionId, ref))
            const toolCallId = requiredString(inputs, 'tool_call_id')
            if (!successfulGoalEvidence(host.evidenceExecution?.(context, toolCallId), toolCallId)) throw new GoalError('Evidence must reference a completed successful tool call in this session; missing, failed, denied or pending results cannot satisfy a criterion', 'GOAL_INVALID_TRANSITION')
            return view(recordGoalEvidence(metadata, sessionId, ref, requiredString(inputs, 'criterion_id'), {
              toolCallId, summary: requiredString(inputs, 'evidence_summary'), recordedAt: now(),
            }, now()))
          }
          case 'pause':
            return view(pauseGoal(metadata, sessionId, ref, now()))
          case 'resume':
            host.validateResume?.(context, expectCurrentGoal(metadata, sessionId, ref))
            return view(resumeGoal(metadata, sessionId, ref, now()))
          case 'complete': {
            assertConcludeAuthority(host, context, 'complete', expectCurrentGoal(metadata, sessionId, ref))
            // Declared criteria are checked in the domain. Legacy goals without
            // criteria retain their existing policy-based completion behavior.
            const completed = completeGoal(metadata, sessionId, ref, now())
            return withWrapup(host, context, view(completed), completed.objective)
          }
          case 'blocked': {
            assertConcludeAuthority(host, context, 'blocked', expectCurrentGoal(metadata, sessionId, ref))
            const round = host.currentRound(context)
            if (round !== undefined && round < threshold) {
              throw new GoalError(
                `blocked is rejected before round ${threshold}; this is round ${round}. Keep working, or `
                + 'report the concrete condition again once it has actually persisted',
                'GOAL_INVALID_TRANSITION',
              )
            }
            const message = requiredString(inputs, 'blocked_reason')
            const blocked = blockGoal(metadata, sessionId, ref, { code: 'model-reported', message }, now())
            return withWrapup(host, context, view(blocked), blocked.objective, message)
          }
          default:
            throw new ValidationError('action', 'must be edit, pause, resume, complete, blocked, record_evidence, or milestone', action)
        }
      })
    },
  }

  for (const definition of GOAL_TOOL_DEFINITIONS) {
    const name = definition.function.name
    registry.replace(definition, async (inputs, context) => byName[name]!(inputs, context), agentId, capabilities)
  }
}

/**
 * Turn a domain rejection into a tool result the model can act on.
 *
 * Deliberately not thrown: a stale revision or a refused transition is
 * information the model should read and retry against, not a turn failure.
 */
function wrap(operation: () => unknown): unknown {
  try {
    return operation()
  } catch (error) {
    if (error instanceof GoalError) return { ok: false, code: error.code, error: error.message }
    throw error
  }
}

/**
 * Attach the closing-message instruction to a terminal update from an
 * autonomous round.
 *
 * The obvious implementation is to stop the turn the moment a goal reaches
 * `complete` or `blocked`, and that is what the guard this replaces effectively
 * did — the run simply ended, and whatever the model had been about to say to
 * the user was never said. So the run's last visible act was a tool call, and
 * the person had to reconstruct the outcome from the transcript.
 *
 * The round driver already refuses to open another round for a terminal goal,
 * so nothing needs stopping here. Instead the model gets one more inference
 * with an explicit brief: report the outcome to the user, grounded in what this
 * session actually established. A human-driven conclusion gets no such
 * instruction — that turn is already a conversation, and the person is right
 * there to ask.
 */
function withWrapup(
  host: GoalToolHost,
  context: ToolExecutionContext,
  result: unknown,
  objective: string,
  blockedReason?: string,
): unknown {
  if (host.currentRound(context) === undefined) return result
  return { ...(result as Record<string, unknown>), wrapup: goalWrapupInstruction(objective, blockedReason) }
}

const WRAPUP_GROUNDING =
  'Report only what earlier rounds and tool results in this session actually establish; '
  + 'when a detail is not in the session, say so instead of inventing it. '

/**
 * The closing brief for a goal that ended on its own.
 *
 * Exported so the wording is testable without driving a whole turn — the exact
 * text matters, because it is the only thing standing between a finished
 * objective and a run that ends on a silent tool call.
 */
export function goalWrapupInstruction(objective: string, blockedReason?: string): string {
  const heading = `Objective: ${JSON.stringify(objective)}\n`
  if (blockedReason === undefined) {
    return '<goal_complete>\n'
      + heading
      + 'The goal is marked complete and this autonomous run is ending. Write the closing '
      + 'message to the user now: state the outcome, summarize what was done and how it was '
      + 'verified, and point to the concrete results (files, commits, or other artifacts). '
      + WRAPUP_GROUNDING
      + 'Note anything the user should review or do next. Address the user directly. Do not '
      + "call any more tools in this run; further work waits for the user's next instruction.\n"
      + '</goal_complete>'
  }
  return '<goal_blocked>\n'
    + heading
    + `Blocked: ${JSON.stringify(blockedReason)}\n`
    + 'The goal is marked blocked and this autonomous run is ending. Write the closing '
    + 'message to the user now: state what has been completed so far, describe the concrete '
    + 'blocking condition and what you tried, and say exactly what you need from the user to '
    + 'continue. '
    + WRAPUP_GROUNDING
    + 'Address the user directly. Do not call any more tools in this run; further work '
    + "waits for the user's next instruction.\n"
    + '</goal_blocked>'
}

function assertMainAgent(context: ToolExecutionContext): void {
  const kind = String(context.metadata.session_kind ?? '').toLowerCase()
  const subagentId = String(context.metadata.subagent_id ?? '').trim()
  if (kind === 'subagent' || subagentId) {
    throw new ValidationError('context', 'only the main agent may read or change the session goal', kind || subagentId)
  }
}

function assertHumanAuthority(host: GoalToolHost, context: ToolExecutionContext, what: string): void {
  if (host.isHumanTurn(context)) return
  throw new ValidationError(
    'authority',
    `${what} requires a direct human turn; an automatic goal round may only complete or block`,
    'non-human turn',
  )
}

/**
 * The live goal a compare-and-set ref names, or a refusal the model can act on.
 *
 * Read here rather than trusting the ref alone: the authority check needs the
 * goal's CURRENT round, and the ref only proves the caller knew the revision.
 */
function expectCurrentGoal(
  metadata: Record<string, unknown>,
  sessionId: string,
  ref: { readonly id: string; readonly revision: number },
): GoalView {
  const goal = getGoal(metadata, sessionId)
  if (!goal) throw new GoalError('no current goal', 'GOAL_NOT_FOUND')
  if (goal.id !== ref.id) throw new GoalError(`goal "${ref.id}" is not the current goal`, 'GOAL_NOT_FOUND')
  if (goal.revision !== ref.revision) {
    throw new GoalError(
      `stale revision ${ref.revision}; current is ${goal.revision}`,
      'GOAL_STALE_REVISION',
    )
  }
  return goal
}

function assertConcludeAuthority(
  host: GoalToolHost,
  context: ToolExecutionContext,
  what: string,
  goal: GoalView,
): void {
  if (host.isHumanTurn(context)) return
  // Not merely "some round": the goal's OWN current round. A turn opened for an
  // earlier round, or for a goal that has since been edited, is carrying stale
  // authority — exactly the case where a concluding claim is least trustworthy,
  // because the round that made it was working from a different objective.
  if (host.currentRound(context) === goal.roundsStarted) return
  throw new ValidationError(
    'authority',
    `${what} requires a direct human turn or the goal's own current continuation round`,
    'unattributed turn',
  )
}

function requiredString(inputs: JsonObject, field: string): string {
  const value = inputs[field]
  if (typeof value !== 'string' || !value.trim()) {
    throw new ValidationError(field, 'is required and must be a non-empty string', value)
  }
  return value.trim()
}

function optionalString(inputs: JsonObject, field: string): string | undefined {
  const value = inputs[field]
  // Strict-schema models emit empty-string fillers for fields they mean to
  // omit; treating those as a real replacement would blank the objective.
  if (typeof value !== 'string' || !value.trim()) return undefined
  return value.trim()
}

function requiredIntegerField(inputs: JsonObject, field: string): number {
  const value = inputs[field]
  if (typeof value !== 'number' || !Number.isSafeInteger(value)) {
    throw new ValidationError(field, 'is required and must be an integer', value)
  }
  return value
}

function optionalInteger(inputs: JsonObject, field: string): number | undefined {
  const value = inputs[field]
  // Zero is the numeric filler equivalent of the empty string above.
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value === 0) return undefined
  return value
}
