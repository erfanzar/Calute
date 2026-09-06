// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { blockGoal, getGoal, goalTimeRemainingMs } from './goalDomain.js'

/** Watches a live goal, including one created while its parent turn is running. */
export class GoalTimeGuard {
  private readonly controller = new AbortController()
  private timer: ReturnType<typeof setTimeout> | undefined
  private disposed = false
  readonly signal = this.controller.signal

  constructor(private readonly read: () => { metadata: Record<string, unknown>; id: string } | undefined,
    private readonly now: () => number = Date.now) { this.refresh() }

  refresh(): void {
    if (this.timer) clearTimeout(this.timer)
    this.timer = undefined
    if (this.disposed || this.signal.aborted) return
    try {
      const session = this.read()
      const goal = session && getGoal(session.metadata, session.id)
      const remaining = goal?.phase === 'active' && goal.activation === 'armed'
        ? goalTimeRemainingMs(goal, this.now()) : undefined
      if (remaining === 0 && session && goal) {
        blockGoal(session.metadata, session.id, goal, {
          code: 'time-limit', message: 'Goal wall-time limit expired. Raise the duration limit and resume to continue.',
        }, this.now())
        this.controller.abort(new Error('Goal wall-time limit expired'))
        return
      }
      this.timer = setTimeout(() => this.refresh(), Math.min(remaining ?? 1000, 1000))
      this.timer.unref?.()
    } catch (error) {
      // Invalid durable state cannot authorize continued unattended work.
      this.controller.abort(error)
    }
  }

  dispose(): void {
    this.disposed = true
    if (this.timer) clearTimeout(this.timer)
    this.timer = undefined
  }
}
