// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test, spyOn } from 'bun:test'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { CronJob, JobStore } from '../src/cron/jobs.js'
import { CronScheduler } from '../src/cron/scheduler.js'

test.each(['success', 'failure'] as const)('schedule lifetime limit persists through %s and restart', async outcome => {
  const dir = mkdtempSync(join(tmpdir(), 'schedule-limit-'))
  const log = spyOn(console, 'error').mockImplementation(() => {})
  try {
    const path = join(dir, 'jobs.json'), store = new JobStore(path)
    store.add(new CronJob({ id: 'job', prompt: 'check', intervalSeconds: 1, maxRuns: 2, nextRunAt: '2026-09-01T00:00:00Z' }))
    let runs = 0
    const runner = () => { runs++; expect(new JobStore(path).get('job')?.runsStarted).toBe(runs); if (outcome === 'failure') throw new Error('failed'); return 'done' }
    await new CronScheduler(store, runner).tick(new Date('2026-09-01T00:00:00Z'))
    const reopened = new JobStore(path), scheduler = new CronScheduler(reopened, runner)
    await scheduler.runNow(reopened.get('job')!, async () => { runs++; return 'manual' })
    await scheduler.tick(new Date('2026-09-02T00:00:00Z'))
    expect(runs).toBe(2)
    expect(reopened.get('job')).toMatchObject({ runsStarted: 2, paused: true })
    await expect(scheduler.runNow(reopened.get('job')!, async () => 'unexpected')).rejects.toThrow('limit reached')
    reopened.update('job', { maxRuns: 3, paused: false })
    await scheduler.runNow(reopened.get('job')!, async () => 'third')
    expect(reopened.get('job')?.runsStarted).toBe(3)
  } finally { log.mockRestore(); rmSync(dir, { recursive: true, force: true }) }
})

test('schedule cannot execute if admission counter persistence fails', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'schedule-count-write-'))
  try {
    const store = new JobStore(join(dir, 'jobs.json'))
    const job = store.add(new CronJob({ id: 'job', prompt: 'check', maxRuns: 1 }))
    store.update = () => { throw new Error('disk failed') }
    let ran = false
    await expect(new CronScheduler(store, () => '').runNow(job, async () => { ran = true })).rejects.toThrow('disk failed')
    expect(ran).toBe(false)
  } finally { rmSync(dir, { recursive: true, force: true }) }
})

test('cancelling an admitted run consumes its durable attempt', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'schedule-count-cancel-'))
  try {
    const store = new JobStore(join(dir, 'jobs.json'))
    const job = store.add(new CronJob({ id: 'job', prompt: 'check', maxRuns: 1 }))
    const scheduler = new CronScheduler(store, () => '')
    const pending = scheduler.runNow(job, async signal => { signal.throwIfAborted(); return 'unexpected' })
    const failure = pending.catch(error => error)
    expect(scheduler.cancel(job.id)).toBe(true)
    expect(await failure).toBeInstanceOf(Error)
    await scheduler.waitForIdle()
    const reloaded = new JobStore(store.path).get(job.id)!
    expect(reloaded.runsStarted).toBe(1)
    await expect(scheduler.runNow(reloaded, async () => 'unexpected')).rejects.toThrow('limit reached')
  } finally { rmSync(dir, { recursive: true, force: true }) }
})
