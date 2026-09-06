// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Serializes work per session while allowing independent sessions to run in
 * parallel. Human work has precedence over queued background work, but never
 * interrupts work that has already started.
 */

export type SessionOperationPriority = 'human' | 'background'

interface Deferred<T> {
  readonly promise: Promise<T>
  readonly resolve: (value: T | PromiseLike<T>) => void
  readonly reject: (reason?: unknown) => void
}

interface Entry {
  readonly operation: () => Promise<unknown>
  readonly priority: SessionOperationPriority
  readonly deferred: Deferred<unknown>
}

interface State {
  readonly entries: Entry[]
  running: boolean
}

const CLOSED_MESSAGE = 'Session operation queue is closed; no new session work can be admitted'
const QUEUED_MESSAGE = 'Session operation was cancelled because the session operation queue closed'

export class SessionOperationQueue {
  private readonly states = new Map<string, State>()
  private readonly drainWaiters: Array<() => void> = []
  private closed = false

  run<T>(
    key: string,
    operation: () => Promise<T>,
    priority: SessionOperationPriority = 'human',
  ): Promise<T> {
    if (!key.trim()) return Promise.reject(new Error('Session operation key must be non-empty'))
    if (priority !== 'human' && priority !== 'background') return Promise.reject(new Error('Session operation priority must be human or background'))
    if (this.closed) return Promise.reject(new Error(CLOSED_MESSAGE))
    const deferred = createDeferred<T>()
    const state = this.states.get(key) ?? { entries: [], running: false }
    state.entries.push({ operation: async () => operation(), priority, deferred: deferred as Deferred<unknown> })
    this.states.set(key, state)
    // Delaying the first pump lets all synchronous callers for this key enter
    // the queue, so a human request can precede a background request admitted
    // in the same turn. A running operation is never preempted.
    if (!state.running) queueMicrotask(() => this.pump(key, state))
    return deferred.promise
  }

  has(key: string): boolean {
    const state = this.states.get(key)
    return state !== undefined && (state.running || state.entries.length > 0)
  }

  hasHumanPending(key: string): boolean {
    return this.states.get(key)?.entries.some(entry => entry.priority === 'human') === true
  }

  close(): void {
    if (this.closed) return
    this.closed = true
    for (const [key, state] of this.states) {
      for (const entry of state.entries.splice(0)) entry.deferred.reject(new Error(QUEUED_MESSAGE))
      if (!state.running) this.states.delete(key)
    }
    this.resolveDrainIfIdle()
  }

  async drain(): Promise<void> {
    if (this.isIdle()) return
    await new Promise<void>(resolve => this.drainWaiters.push(resolve))
  }

  private pump(key: string, state: State): void {
    if (state.running) return
    if (!state.entries.length) {
      this.states.delete(key)
      this.resolveDrainIfIdle()
      return
    }
    const index = state.entries.findIndex(entry => entry.priority === 'human')
    const entry = state.entries.splice(index < 0 ? 0 : index, 1)[0]!
    state.running = true
    void Promise.resolve()
      .then(entry.operation)
      .then(entry.deferred.resolve, entry.deferred.reject)
      .finally(() => {
        state.running = false
        if (state.entries.length) queueMicrotask(() => this.pump(key, state))
        else {
          this.states.delete(key)
          this.resolveDrainIfIdle()
        }
      })
      // The caller receives the operation's rejection through deferred; this
      // terminal cleanup chain must never create an unhandled rejection.
      .catch(() => undefined)
  }

  private isIdle(): boolean {
    return this.states.size === 0
  }

  private resolveDrainIfIdle(): void {
    if (!this.isIdle()) return
    const waiters = this.drainWaiters.splice(0)
    for (const resolve of waiters) resolve()
  }
}

function createDeferred<T>(): Deferred<T> {
  let resolve!: Deferred<T>['resolve']
  let reject!: Deferred<T>['reject']
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, resolve, reject }
}
