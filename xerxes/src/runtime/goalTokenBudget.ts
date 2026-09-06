// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ModelCallBinding, ModelCallReceipt, ModelCallScope } from '../llms/callBudget.js'
import { blockGoal, getGoal, type GoalView } from './goalDomain.js'
import type { GoalTokenLedger } from './goalTokenLedger.js'

interface GoalSession { id: string; metadata: Record<string, unknown> }

/** Dynamic for a parent turn; delegated copies retain their original goal identity. */
export class GoalTokenBudget implements ModelCallScope {
  readonly scopeKey: string
  private failure: Error | undefined
  private engagedGoalId: string | undefined
  constructor(private readonly readSession: () => GoalSession | undefined,
    private readonly ledger: GoalTokenLedger | undefined, private readonly ownerId: string,
    private readonly sessionId: string, private readonly boundGoalId?: string) {
    this.scopeKey = `goal:${sessionId}`
  }
  get recoveryBinding(): ModelCallBinding {
    return this.boundGoalId === undefined
      ? { kind: 'unrecoverable' }
      : { kind: 'goal', sessionId: this.sessionId, goalId: this.boundGoalId }
  }
  get maximum(): undefined { return undefined }
  get used(): number { return 0 }
  get exhausted(): boolean { return this.failure !== undefined }
  get tokenFailure(): Error | undefined { return this.failure }
  get available(): boolean { try { this.assertAdmission(); return true } catch { return false } }

  private current(): { session: GoalSession; goal: GoalView } | undefined {
    const session = this.readSession()
    const goal = session && getGoal(session.metadata, session.id)
    if (goal?.id === this.engagedGoalId && goal?.phase === 'blocked' && goal.blockedReason?.code === 'token-budget') {
      throw new Error(goal.blockedReason.message)
    }
    if (this.boundGoalId && (!session || session.id !== this.sessionId || goal?.id !== this.boundGoalId
      || goal.phase !== 'active' || goal.activation !== 'armed')) throw new Error('Delegated goal ownership is no longer active; dispatch new work from the current goal')
    if (!session || session.id !== this.sessionId || !goal || goal.phase !== 'active' || goal.activation !== 'armed') return
    this.engagedGoalId = goal.id
    return { session, goal }
  }

  private checked() {
    if (this.failure) throw this.failure
    const current = this.current()
    if (!current) return
    const { session, goal } = current
    if (!this.ledger) {
      if (goal.maxTotalTokens !== undefined) throw new Error('Goal token ledger is unavailable; configure the durable daemon ledger')
      return
    }
    if (!this.ledger.inspect(session.id, goal.id)) {
      // A goal predating tracking has unknown prior spend, never a zero baseline.
      this.ledger.initialize(session.id, goal.id, false)
    }
    this.ledger.assertAdmission(session.id, goal.id, this.ownerId, goal.maxTotalTokens)
    return { ...current, ledger: this.ledger }
  }

  private reject(error: unknown): never {
    this.failure = error instanceof Error ? error : new Error(String(error))
    const session = this.readSession()
    const goal = session && getGoal(session.metadata, session.id)
    if (session && goal?.phase === 'active' && (!this.boundGoalId || goal.id === this.boundGoalId)) {
      blockGoal(session.metadata, session.id, goal, { code: 'token-budget', message: this.failure.message }, Date.now())
    }
    throw this.failure
  }

  assertAdmission(): void { try { this.checked() } catch (error) { this.reject(error) } }
  charge(): ModelCallReceipt {
    try {
      const current = this.checked()
      if (!current) return () => {}
      const receipt = current.ledger.admit(current.session.id, current.goal.id, this.ownerId, current.goal.maxTotalTokens)
      return (usage, completed) => {
        try { receipt(usage, completed) } catch (error) { this.reject(error) }
      }
    } catch (error) { return this.reject(error) }
  }

  fork(): GoalTokenBudget | undefined {
    const current = this.current()
    return current ? new GoalTokenBudget(this.readSession, this.ledger, this.ownerId, this.sessionId, current.goal.id) : undefined
  }
}
