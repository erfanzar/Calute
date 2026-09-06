// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { DeliveryOutbox } from '../src/cron/outbox.js'
test('outbox survives reopen, scopes payloads and never implicitly retries uncertain delivery', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'cron-outbox-'))
  try {
    const outbox = new DeliveryOutbox(join(dir, 'delivery.sqlite'))
    const id = outbox.enqueue('job', { platform: 'channel' }, 'result', 'archive.md')
    expect(new DeliveryOutbox(outbox.path).inspect('job', id)?.content).toBe('result')
    expect(outbox.inspect('other', id)).toBeNull()
    let calls = 0
    await expect(outbox.send('job', id, () => { calls++; throw new Error('lost acknowledgement') })).rejects.toThrow('lost acknowledgement')
    expect(outbox.inspect('job', id)?.state).toBe('uncertain')
    await expect(outbox.send('job', id, () => { calls++ })).rejects.toThrow('reconcile')
    expect(calls).toBe(1)
  } finally { rmSync(dir, { recursive: true, force: true }) }
})
test('outbox keeps concurrent sends exclusive and sent receipts idempotent', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'cron-outbox-'))
  try {
    const outbox = new DeliveryOutbox(join(dir, 'delivery.sqlite'))
    const id = outbox.enqueue('job', { platform: 'channel' }, 'result', 'archive.md')
    let finish!: () => void
    let calls = 0
    const sending = outbox.send('job', id, async () => { calls++; await new Promise<void>(resolve => { finish = resolve }) })
    await expect(new DeliveryOutbox(outbox.path).send('job', id, () => { calls++ })).rejects.toThrow('reconcile')
    finish(); await sending
    await outbox.send('job', id, () => { calls++ })
    expect(calls).toBe(1)
    expect(outbox.inspect('job', id)?.state).toBe('sent')
  } finally { rmSync(dir, { recursive: true, force: true }) }
})

test('explicit reconciliation permits retry but stale decisions cannot reset a newer attempt', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'cron-outbox-'))
  try {
    const outbox = new DeliveryOutbox(join(dir, 'delivery.sqlite'))
    const id = outbox.enqueue('job', { platform: 'channel' }, 'result', 'archive.md')
    await expect(outbox.send('job', id, () => { throw new Error('uncertain') })).rejects.toThrow()
    expect(outbox.list('other')).toEqual([])
    expect(outbox.list('job')[0]).not.toHaveProperty('content')
    expect(() => outbox.reconcile('other', id, 1, 'retry')).toThrow()
    outbox.reconcile('job', id, 1, 'retry')
    await expect(outbox.send('job', id, () => { throw new Error('uncertain again') })).rejects.toThrow()
    expect(() => outbox.reconcile('job', id, 1, 'retry')).toThrow()
    outbox.reconcile('job', id, 2, 'sent')
    expect(outbox.inspect('job', id)?.state).toBe('sent')
  } finally { rmSync(dir, { recursive: true, force: true }) }
})

test('exited senders recover as uncertain while stale completion cannot settle a retry', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'cron-outbox-recovery-'))
  try {
    const path = join(dir, 'delivery.sqlite')
    const first = new DeliveryOutbox(path, { pid: 12345 })
    const id = first.enqueue('job', { platform: 'test' }, 'output', 'archive')
    let finishOld!: () => void
    const old = first.send('job', id, async () => { await new Promise<void>(resolve => { finishOld = resolve }) })
    const recovered = new DeliveryOutbox(path, { isAlive: () => false })
    expect(recovered.inspect('job', id)?.state).toBe('uncertain')
    expect(recovered.inspect('job', id)?.error).toContain('outcome is unknown')
    recovered.reconcile('job', id, 1, 'retry')
    let finishNew!: () => void
    const current = recovered.send('job', id, async () => { await new Promise<void>(resolve => { finishNew = resolve }) })
    finishOld()
    await expect(old).rejects.toThrow('claim changed')
    expect(recovered.inspect('job', id)?.state).toBe('sending')
    finishNew(); await current
    expect(recovered.inspect('job', id)?.state).toBe('sent')
    expect(recovered.inspect('job', id)?.attempts).toBe(2)
  } finally { rmSync(dir, { recursive: true, force: true }) }
})
test('live or reused process IDs remain fenced on reopen', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'cron-outbox-live-'))
  try {
    const outbox = new DeliveryOutbox(join(dir, 'delivery.sqlite'))
    const id = outbox.enqueue('job', { platform: 'test' }, 'output', 'archive')
    let finish!: () => void
    const active = outbox.send('job', id, async () => { await new Promise<void>(resolve => { finish = resolve }) })
    const reopened = new DeliveryOutbox(outbox.path, { isAlive: () => true })
    expect(reopened.inspect('job', id)?.state).toBe('sending')
    expect(() => reopened.reconcile('job', id, 1, 'retry')).toThrow()
    finish(); await active
  } finally { rmSync(dir, { recursive: true, force: true }) }
})
