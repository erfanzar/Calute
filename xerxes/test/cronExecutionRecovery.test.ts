// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, spyOn, test } from 'bun:test'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { CronJob, JobStore, resumedCronMetadata } from '../src/cron/jobs.js'
import { CronScheduler } from '../src/cron/scheduler.js'

for (const fault of ['receipt', 'removal'] as const) test(`completed execution is not retried after ${fault} persistence fails`, async () => {
  const directory = mkdtempSync(join(tmpdir(), 'cron-recovery-'))
  const errorLog = spyOn(console, 'error').mockImplementation(() => {})
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(new CronJob({ id: 'job', prompt: 'work', oneshot: true, nextRunAt: '2026-09-01T09:00:00Z' }))
    let executions = 0
    let failWrites = false
    const update = store.update.bind(store)
    const remove = store.remove.bind(store)
    store.update = (...args) => { if (failWrites && fault === 'receipt') throw new Error('disk unavailable'); return update(...args) }
    store.remove = (...args) => { if (failWrites) throw new Error('disk unavailable'); return remove(...args) }
    const scheduler = new CronScheduler(store, () => { executions++; failWrites = true; return 'done' })
    await scheduler.tick(new Date('2026-09-01T09:00:00Z'))
    failWrites = false
    // New scheduler and store simulate loss of every in-memory execution flag.
    const reopened = new JobStore(store.path)
    const restarted = new CronScheduler(reopened, () => { executions++; return 'duplicate' })
    await restarted.tick(new Date('2026-09-02T09:00:00Z'))
    const recovered = reopened.get('job')!
    expect(executions).toBe(1)
    expect(recovered.paused).toBe(true)
    expect(recovered.metadata.execution_recovery_required).toBe(true)
    expect(recovered.metadata.retry_count).toBeUndefined()
    expect(recovered.metadata.execution_receipt).toMatchObject({ state: fault === 'receipt' ? 'running' : 'completed' })
    // Explicit review/resume is the only way to discard the execution fence.
    reopened.update('job', { paused: false, metadata: resumedCronMetadata(recovered) })
    await restarted.tick(new Date('2026-09-03T09:00:00Z'))
    expect(executions).toBe(2)
    expect(reopened.get('job')).toBeUndefined()
  } finally { errorLog.mockRestore(); rmSync(directory, { recursive: true, force: true }) }
})

test('intent receipt is durable before execution and normal model failures retain retry behavior', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'cron-intent-'))
  const errorLog = spyOn(console, 'error').mockImplementation(() => {})
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(new CronJob({ id: 'job', prompt: 'work', oneshot: true, nextRunAt: '2026-09-01T09:00:00Z' }))
    const scheduler = new CronScheduler(store, () => {
      expect(new JobStore(store.path).get('job')?.metadata.execution_receipt).toMatchObject({ state: 'running' })
      throw new Error('provider unavailable')
    })
    await scheduler.tick(new Date('2026-09-01T09:00:00Z'))
    const job = store.get('job')!
    expect(job.metadata.execution_receipt).toBeUndefined()
    expect(job.metadata.retry_count).toBe(1)
    expect(job.nextRunAt).toBe('2026-09-01T09:01:00.000Z')
  } finally { errorLog.mockRestore(); rmSync(directory, { recursive: true, force: true }) }
})

test('a failed recurring advance leaves an execution fence instead of repeating the due occurrence', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'cron-advance-'))
  const warning = spyOn(console, 'warn').mockImplementation(() => {})
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(new CronJob({ id: 'job', prompt: 'work', schedule: '0 9 * * *', nextRunAt: '2026-09-01T09:00:00Z' }))
    const update = store.update.bind(store)
    store.update = (...args) => { if ('nextRunAt' in args[1]) throw new Error('advance unavailable'); return update(...args) }
    let executions = 0
    const scheduler = new CronScheduler(store, () => { executions++; return 'done' })
    await scheduler.tick(new Date('2026-09-01T09:00:00Z'))
    store.update = update
    await new CronScheduler(new JobStore(store.path), () => { executions++; return 'duplicate' }).tick(new Date('2026-09-01T09:01:00Z'))
    expect(executions).toBe(1)
    expect(store.get('job')?.paused).toBe(true)
    expect(store.get('job')?.metadata.execution_recovery_required).toBe(true)
  } finally { warning.mockRestore(); rmSync(directory, { recursive: true, force: true }) }
})
