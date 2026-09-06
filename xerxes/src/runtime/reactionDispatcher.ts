// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { ReactionClaim, ReactionMailbox, ReactionUsage } from './reactionMailbox.js'
import { ReactionExecutionError } from './reactionUsage.js'

export interface ReactionExecutor {
  /** Acquire the same session admission used by human turns before invoking work. */
  admit(owner: string, work: () => Promise<void>): Promise<void>
  /** Must settle only after provider, tools and child cancellation have settled. */
  run(claim: ReactionClaim, signal: AbortSignal): Promise<ReactionUsage | void>
}

/** Event-triggered dispatcher. No polling and no detached timeout races. */
export class ReactionDispatcher {
  private readonly active = new Map<string, { controller: AbortController; promise: Promise<void>; requested: boolean; runId?: string; reaction?: AbortController }>()
  private closed = false
  private readonly unsubscribe: () => void
  constructor(private readonly mailbox: ReactionMailbox, private readonly executor: ReactionExecutor) {
    this.unsubscribe = mailbox.subscribeCancellation((owner, runId) => {
      const active = this.active.get(owner)
      if (!active) return
      if (runId === undefined) active.controller.abort(new Error('Reaction cancelled by user'))
      else if (active.runId === runId) active.reaction?.abort(new Error('Reaction cancelled by user'))
    })
  }

  dispatch(owner: string): Promise<void> {
    if (this.closed) return Promise.reject(new Error('Reaction dispatcher is closed'))
    const previous = this.active.get(owner)
    if (previous) { previous.requested = true; return previous.promise }
    const controller = new AbortController()
    // Reserve synchronously before invoking any host code that can re-enter.
    const promise = Promise.resolve().then(async () => {
      try {
        let claimed: boolean
        let firstFailure: { error: unknown } | undefined
        do {
          claimed = false
          const draining = this.active.get(owner)
          if (draining) draining.requested = false
          await this.executor.admit(owner, async () => {
            if (controller.signal.aborted || this.closed) return
            const claim = this.mailbox.claim(owner)
            if (!claim) return
            claimed = true
            const reaction = new AbortController()
            const signal = AbortSignal.any([controller.signal, reaction.signal])
            const active = this.active.get(owner)
            if (active) { active.runId = claim.runId; active.reaction = reaction }
            const timeout = setTimeout(() => reaction.abort(new Error("Reaction deadline exceeded")), Math.max(0, claim.deadline - Date.now()))
            try {
              if (!this.mailbox.isAuthorized(claim)) {
                this.mailbox.settle(claim, "cancelled")
                return
              }
              let usage: ReactionUsage | void
              let failure: { error: unknown } | undefined
              try { usage = await this.executor.run(claim, signal) }
              catch (error) {
                failure = { error }
                firstFailure ??= failure
                usage = error instanceof ReactionExecutionError ? error.usage : undefined
              }
              // Settle only after cleanup. A persistence error must escape rather
              // than allowing another claim to bypass uncertain ownership.
              this.mailbox.settle(claim, signal.aborted || !this.mailbox.isAuthorized(claim) ? "cancelled" : failure ? "failed" : "completed",
                failure ? failure.error instanceof Error ? failure.error.message : String(failure.error) : undefined, usage || undefined)
            } finally {
              clearTimeout(timeout)
              if (active?.reaction === reaction) { delete active.reaction; delete active.runId }
            }
          })
        } while ((claimed || this.active.get(owner)?.requested) && !controller.signal.aborted && !this.closed)
        if (firstFailure) throw firstFailure.error
      } finally {
        // Release in this continuation: a separate promise.finally creates a
        // microtask window in which a new offer can join an already-ended drain.
        if (this.active.get(owner)?.controller === controller) this.active.delete(owner)
      }
    })
    this.active.set(owner, { controller, promise, requested: false })
    return promise
  }

  /** Recover persisted-but-not-offered events only after the host loads their owner. */
  reconcile(owner: string, evidenceCursor: (runId: string) => number): Promise<void> {
    if (this.closed) return Promise.reject(new Error('Reaction dispatcher is closed'))
    for (const runId of this.mailbox.recoverableRuns(owner)) {
      const cursor = evidenceCursor(runId)
      if (cursor > 0) this.mailbox.offer(owner, runId, cursor)
    }
    return this.dispatch(owner)
  }

  cancel(owner: string, runId?: string): void {
    this.mailbox.cancel(owner, runId)
  }

  async close(): Promise<void> {
    this.closed = true
    for (const { controller } of this.active.values()) controller.abort(new Error('Reaction dispatcher stopping'))
    // Preserve ownership until actual cleanup. The host may bound its shutdown
    // wait separately, but must not treat that bound as completed cancellation.
    await Promise.allSettled([...this.active.values()].map(entry => entry.promise))
    this.unsubscribe()
  }
}
