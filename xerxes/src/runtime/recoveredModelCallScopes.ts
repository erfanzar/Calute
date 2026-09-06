// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ModelCallBinding, ModelCallScope } from '../llms/callBudget.js'
import { getGoal, readGoalChanges } from './goalDomain.js'
import { GoalTokenBudget } from './goalTokenBudget.js'
import type { GoalTokenLedger } from './goalTokenLedger.js'

interface RecoverySession {
  readonly id: string
  readonly metadata: Record<string, unknown>
}

const MAX_BINDING_ID_CHARS = 256

export interface RecoveredModelCallScopeOptions {
  readonly readSession: (sessionId: string) => RecoverySession | undefined
  readonly ledger: GoalTokenLedger | undefined
  readonly ownerId: string | undefined
}

/** Restore only explicitly serializable scopes, refusing ambiguous ownership. */
export function restoreRecoveredModelCallScopes(
  bindings: readonly ModelCallBinding[] | undefined,
  sourceSessionId: string | undefined,
  options: RecoveredModelCallScopeOptions,
): readonly ModelCallScope[] {
  if (bindings !== undefined && (!Array.isArray(bindings) || bindings.length > 64)) throw new Error('Invalid model-call recovery bindings')
  const source = typeof sourceSessionId === 'string' ? sourceSessionId.trim() : ''
  if (!source) throw new Error('Recovered model-call scopes require a source session')
  const session = options.readSession(source)
  if (!session || session.id !== source) throw new Error('Recovered model-call source session is unavailable')
  if (source.length > MAX_BINDING_ID_CHARS) throw new Error('Recovered model-call source session is too long')
  if (bindings?.length === 0) return Object.freeze([])
  if (bindings === undefined) {
    if (getGoal(session.metadata, session.id) || readGoalChanges(session.metadata).length > 0) {
      throw new Error('Recovered legacy subagent has unknown goal ownership; respawn it from the current goal turn')
    }
    return Object.freeze([])
  }
  if (!options.ledger || typeof options.ownerId !== 'string' || !options.ownerId.trim()) {
    throw new Error('Recovered goal scopes require a durable token ledger and current daemon owner')
  }
  const restored: ModelCallScope[] = []
  for (const binding of bindings) {
    if (!binding || typeof binding !== 'object' || binding.kind !== 'goal'
      || typeof binding.sessionId !== 'string' || !binding.sessionId || binding.sessionId.trim() !== binding.sessionId
      || binding.sessionId.length > MAX_BINDING_ID_CHARS
      || typeof binding.goalId !== 'string' || !binding.goalId || binding.goalId.trim() !== binding.goalId
      || binding.goalId.length > MAX_BINDING_ID_CHARS) {
      throw new Error('Recovered model-call scope is unrecoverable')
    }
    if (binding.sessionId !== source) throw new Error('Recovered goal scope belongs to a different session')
    const goal = getGoal(session.metadata, session.id)
    if (!goal || goal.id !== binding.goalId || goal.phase !== 'active' || goal.activation !== 'armed') {
      throw new Error('Recovered goal ownership is no longer active; resume the original goal if still current, otherwise dispatch new work')
    }
    restored.push(new GoalTokenBudget(
      () => options.readSession(source), options.ledger, options.ownerId, source, binding.goalId,
    ))
  }
  return Object.freeze(restored)
}
