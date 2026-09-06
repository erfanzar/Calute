// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { ReactionMailbox } from '../src/runtime/reactionMailbox.js'
import { ReactionDispatcher } from '../src/runtime/reactionDispatcher.js'

function pending() { let resolve!: () => void; const promise = new Promise<void>(r => { resolve = r }); return { promise, resolve } }
function setup() {
  const mailbox = new ReactionMailbox(':memory:')
  mailbox.configure({ owner: 'owner', runId: 'watch', expiresAt: Date.now() + 60_000, maxReactions: 3, maxDurationMs: 100 })
  mailbox.offer('owner', 'watch', 1)
  return mailbox
}

test('queued cancellation prevents claiming and never invokes the provider', async () => {
  const mailbox = setup()
  const gate = pending()
  let calls = 0
  const dispatcher = new ReactionDispatcher(mailbox, { admit: async (_owner, work) => { await gate.promise; await work() }, run: async () => { calls++ } })
  try {
    const work = dispatcher.dispatch('owner')
    dispatcher.cancel('owner')
    gate.resolve()
    await work
    expect(calls).toBe(0)
    expect(mailbox.unresolved('owner')).toEqual([])
    expect(mailbox.claim('owner')).toBeUndefined()
  } finally { await dispatcher.close(); mailbox.close() }
})

test('deadline abort does not release ownership while the executor still runs', async () => {
  const mailbox = setup()
  const started = pending()
  const cleanup = pending()
  let signal: AbortSignal | undefined
  let calls = 0
  const dispatcher = new ReactionDispatcher(mailbox, { admit: async (_owner, work) => work(), run: async (_claim, abort) => {
    signal = abort; calls++; started.resolve(); await cleanup.promise
  } })
  try {
    const work = dispatcher.dispatch('owner')
    await started.promise
    await Bun.sleep(130)
    expect(signal?.aborted).toBe(true)
    mailbox.offer('owner', 'watch', 2)
    expect(dispatcher.dispatch('owner')).toBe(work)
    expect(mailbox.claim('owner')).toBeUndefined()
    expect(calls).toBe(1)
    cleanup.resolve()
    await work
    expect(mailbox.unresolved('owner')).toEqual([])
    expect(calls).toBe(2)
    expect(mailbox.claim('owner')).toBeUndefined()
  } finally { cleanup.resolve(); await dispatcher.close(); mailbox.close() }
})

test('executor failure is observable and settles its claim without retrying its evidence', async () => {
  const mailbox = setup()
  const dispatcher = new ReactionDispatcher(mailbox, { admit: async (_owner, work) => work(), run: async () => { throw new Error('provider down') } })
  try {
    await expect(dispatcher.dispatch('owner')).rejects.toThrow('provider down')
    expect(mailbox.unresolved('owner')).toEqual([])
    expect(mailbox.claim('owner')).toBeUndefined()
  } finally { await dispatcher.close(); mailbox.close() }
})

test('events offered during execution are drained after releasing and reacquiring admission', async () => {
  const mailbox = setup()
  const seen: number[] = []
  let admissions = 0
  const dispatcher = new ReactionDispatcher(mailbox, {
    admit: async (_owner, work) => { admissions++; await work() },
    run: async claim => {
      seen.push(claim.throughSequence)
      if (seen.length === 1) mailbox.offer('owner', 'watch', 4)
    },
  })
  try {
    await dispatcher.dispatch('owner')
    expect(seen).toEqual([1, 4])
    expect(admissions).toBe(3)
    expect(mailbox.unresolved('owner')).toEqual([])
  } finally { await dispatcher.close(); mailbox.close() }
})

test('per-watch revocation interrupts only the matching active reaction', async () => {
  const mailbox = setup()
  mailbox.configure({ owner: 'owner', runId: 'other', expiresAt: Date.now() + 60_000, maxReactions: 1, maxDurationMs: 5000 })
  const started = pending()
  const cleanup = pending()
  let signal: AbortSignal | undefined
  const dispatcher = new ReactionDispatcher(mailbox, { admit: async (_owner, work) => work(), run: async (_claim, abort) => {
    signal = abort; started.resolve(); await cleanup.promise
  } })
  try {
    const work = dispatcher.dispatch('owner')
    await started.promise
    mailbox.cancel('owner', 'other')
    expect(signal?.aborted).toBe(false)
    mailbox.cancel('other-owner', 'watch')
    expect(signal?.aborted).toBe(false)
    mailbox.cancel('owner', 'watch')
    expect(signal?.aborted).toBe(true)
    expect(mailbox.unresolved('owner')).toHaveLength(1)
    cleanup.resolve()
    await work
    expect(mailbox.unresolved('owner')).toEqual([])
    expect(mailbox.offer('owner', 'watch', 2)).toBe(false)
  } finally { cleanup.resolve(); await dispatcher.close(); mailbox.close() }
})

test('reconciliation offers previously unqueued durable events once and respects cancellation', async () => {
  const mailbox = new ReactionMailbox(':memory:')
  for (const runId of ['watch', 'stopped']) mailbox.configure({ owner: 'owner', runId, expiresAt: Date.now() + 60000, maxReactions: 3, maxDurationMs: 1000 })
  mailbox.cancel('owner', 'stopped')
  const seen: number[] = []
  const dispatcher = new ReactionDispatcher(mailbox, { admit: async (_owner, work) => work(), run: async claim => { seen.push(claim.throughSequence) } })
  try {
    expect(mailbox.claim('owner')).toBeUndefined()
    await dispatcher.reconcile('owner', id => { expect(id).toBe('watch'); return 4 })
    await dispatcher.reconcile('owner', () => 4)
    expect(seen).toEqual([4])
    expect(mailbox.inspect('owner', 'watch')).toMatchObject({ attempts: 1, pendingEvents: 0 })
    await dispatcher.reconcile('owner', () => 5)
    expect(seen).toEqual([4, 5])
  } finally { await dispatcher.close(); mailbox.close() }
})

test('reconciliation never reads evidence or starts work while an executor is unresolved', async () => {
  const mailbox = setup()
  const claim = mailbox.claim('owner')!
  let read = false, ran = false
  const dispatcher = new ReactionDispatcher(mailbox, { admit: async (_owner, work) => work(), run: async () => { ran = true } })
  try {
    await dispatcher.reconcile('owner', () => { read = true; return 3 })
    expect(read).toBe(false)
    expect(ran).toBe(false)
    expect(mailbox.unresolved('owner')).toEqual([claim])
  } finally { mailbox.settle(claim, 'cancelled'); await dispatcher.close(); mailbox.close() }
})

for (const failure of ['cancel', 'throw'] as const) test(`one watch's ${failure} does not strand another queued watch`, async () => {
  const mailbox = setup()
  mailbox.configure({ owner: 'owner', runId: 'next', expiresAt: Date.now() + 70_000, maxReactions: 1, maxDurationMs: 5000 })
  const started = pending()
  const cleanup = pending()
  const seen: string[] = []
  let currentSignal: AbortSignal | undefined
  const dispatcher = new ReactionDispatcher(mailbox, {
    admit: async (_owner, work) => work(),
    run: async (claim, signal) => {
      seen.push(claim.runId)
      if (claim.runId === 'watch') {
        currentSignal = signal
        started.resolve()
        await cleanup.promise
        if (failure === 'throw') throw new Error('first provider failed')
      } else expect(signal.aborted).toBe(false)
    },
  })
  try {
    const work = dispatcher.dispatch('owner')
    const observed = work.catch(error => error)
    await started.promise
    mailbox.offer('owner', 'next', 1)
    if (failure === 'cancel') dispatcher.cancel('owner', 'watch')
    expect(currentSignal?.aborted).toBe(failure === 'cancel')
    expect(seen).toEqual(['watch'])
    expect(mailbox.claim('owner')).toBeUndefined()
    cleanup.resolve()
    const result = await observed
    if (failure === 'throw') expect(result.message).toBe('first provider failed')
    expect(seen).toEqual(['watch', 'next'])
    expect(mailbox.inspect('owner', 'next')?.lastOutcome).toBe('completed')
  } finally { cleanup.resolve(); await dispatcher.close(); mailbox.close() }
})

test('completion arriving after an empty admission check still wakes the owner', async () => {
  const mailbox = new ReactionMailbox(':memory:')
  mailbox.configure({ owner: 'owner', runId: 'late-watch', expiresAt: Date.now() + 60000, maxReactions: 1, maxDurationMs: 1000 })
  const seen: string[] = []
  let admissions = 0
  let joined: Promise<void> | undefined
  const dispatcher = new ReactionDispatcher(mailbox, {
    admit: async (_owner, work) => {
      await work()
      if (++admissions === 1) {
        mailbox.offer('owner', 'late-watch', 1)
        joined = dispatcher.dispatch('owner')
      }
    },
    run: async claim => { seen.push(claim.runId) },
  })
  try {
    const first = dispatcher.dispatch('owner')
    await first
    expect(joined).toBe(first)
    expect(seen).toEqual(['late-watch'])
    expect(mailbox.inspect('owner', 'late-watch')).toMatchObject({ attempts: 1, pendingEvents: 0 })
  } finally { await dispatcher.close(); mailbox.close() }
})

test('a completion in the promise settlement microtask starts a fresh drain', async () => {
  const mailbox = new ReactionMailbox(':memory:')
  mailbox.configure({ owner: 'owner', runId: 'settling-watch', expiresAt: Date.now() + 60000, maxReactions: 1, maxDurationMs: 1000 })
  let admissions = 0, calls = 0
  let joined: Promise<void> | undefined
  const dispatcher = new ReactionDispatcher(mailbox, {
    admit: async (_owner, work) => {
      await work()
      if (++admissions === 1) queueMicrotask(() => queueMicrotask(() => {
        mailbox.offer('owner', 'settling-watch', 1)
        joined = dispatcher.dispatch('owner')
      }))
    },
    run: async () => { calls++ },
  })
  try {
    await dispatcher.dispatch('owner')
    await joined
    expect(calls).toBe(1)
    expect(mailbox.inspect('owner', 'settling-watch')).toMatchObject({ attempts: 1, pendingEvents: 0 })
  } finally { await dispatcher.close(); mailbox.close() }
})
