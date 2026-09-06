// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { CronJob, JobStore } from '../src/cron/jobs.js'
import { CronScheduler } from '../src/cron/scheduler.js'

test('expiry pauses an overdue schedule after restart without consuming an attempt', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'schedule-expiry-'))
  try {
    const path = join(dir, 'jobs.json'), store = new JobStore(path)
    store.add(new CronJob({ id: 'job', prompt: 'check', intervalSeconds: 1, nextRunAt: '2026-09-01T00:00:00Z', expiresAt: '2026-09-01T00:00:01.500Z' }))
    let calls = 0
    await new CronScheduler(store, () => { calls++; return 'done' }).tick(new Date('2026-09-01T00:00:00Z'))
    const reopened = new JobStore(path), scheduler = new CronScheduler(reopened, () => { calls++; return 'unexpected' })
    await scheduler.tick(new Date('2026-09-01T00:00:01.500Z'))
    expect(calls).toBe(1)
    expect(reopened.get('job')).toMatchObject({ paused: true, runsStarted: 1, expiresAt: '2026-09-01T00:00:01.500Z' })
    await expect(scheduler.runNow(reopened.get('job')!, async () => 'unexpected')).rejects.toThrow('expired')
    reopened.update('job', { expiresAt: null, paused: false })
    await scheduler.runNow(reopened.get('job')!, async () => 'allowed')
    expect(reopened.get('job')?.runsStarted).toBe(2)
  } finally { rmSync(dir, { recursive: true, force: true }) }
})

test('expiry is validated and normalized without silently correcting calendar dates', () => {
  const options = { id: 'job', prompt: 'check' }
  expect(new CronJob({ ...options, expiresAt: '2099-01-01T03:00:00+03:00' }).expiresAt).toBe('2099-01-01T00:00:00.000Z')
  for (const expiresAt of ['2099-02-29T00:00:00Z', '2099-01-01', 'bad']) expect(() => new CronJob({ ...options, expiresAt })).toThrow()
  expect(() => CronJob.fromRecord({ ...options, expires_at: 100 })).toThrow('ISO timestamp')
})
