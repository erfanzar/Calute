// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test, spyOn } from 'bun:test'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { CronJob, JobStore, resumedCronMetadata } from '../src/cron/jobs.js'
import { CronScheduler } from '../src/cron/scheduler.js'

test('stop conditions round-trip and reject invalid or independent configuration', () => {
  const job = new CronJob({ id: 'job', prompt: 'check', targetSessionId: 'owner', stopCondition: ' Healthy ' })
  expect(CronJob.fromRecord(job.toRecord()).stopCondition).toBe('Healthy')
  for (const stopCondition of ['', 'x'.repeat(4001), 42]) {
    expect(() => CronJob.fromRecord({ ...job.toRecord(), stop_condition: stopCondition })).toThrow('stopCondition')
  }
  expect(() => new CronJob({ id: 'job', prompt: 'check', stopCondition: 'Healthy' })).toThrow('session follow-up')
})
test.each([['success', false], ['failure', false], ['success', true], ['failure', true]] as const)('completion remains stopped through scheduler %s and restart (oneshot %s)', async (outcome, oneshot) => {
  const dir = mkdtempSync(join(tmpdir(), 'followup-condition-'))
  const quiet = spyOn(console, 'error').mockImplementation(() => {})
  try {
    const path = join(dir, 'jobs.json'), store = new JobStore(path)
    store.add(new CronJob({ id: 'job', prompt: 'check', targetSessionId: 'owner', stopCondition: 'Healthy', ...(oneshot ? { oneshot: true } : { intervalSeconds: 1 }), maxRuns: 10, nextRunAt: '2026-09-01T00:00:00Z' }))
    let runs = 0
    const runner = () => {
      runs++
      const current = store.get('job')!
      store.update('job', { paused: true, metadata: { ...current.metadata, followup_completion: { source: 'model_reported', evidence: 'Healthy endpoint' } } })
      if (outcome === 'failure') throw new Error('provider failed after acknowledgement')
      return 'done'
    }
    await new CronScheduler(store, runner).tick(new Date('2026-09-01T00:00:00Z'))
    const reopened = new JobStore(path), scheduler = new CronScheduler(reopened, runner)
    expect(reopened.get('job')).toMatchObject({ paused: true, nextRunAt: undefined, runsStarted: 1, metadata: { followup_completion: { evidence: 'Healthy endpoint' } } })
    await scheduler.tick(new Date('2026-09-02T00:00:00Z'))
    await expect(scheduler.runNow(reopened.get('job')!, async () => 'unexpected')).rejects.toThrow('explicitly resume')
    expect(runs).toBe(1)
    const current = reopened.get('job')!
    reopened.update('job', { paused: false, metadata: resumedCronMetadata(current) })
    await scheduler.runNow(reopened.get('job')!, async () => 'rearmed')
    expect(reopened.get('job')?.runsStarted).toBe(2)
  } finally { quiet.mockRestore(); rmSync(dir, { recursive: true, force: true }) }
})
