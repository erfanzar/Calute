// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { CronJob, JobStore } from '../src/cron/jobs.js'

test('removal refuses a schedule changed after inspection, including a new execution receipt', () => {
  const directory = mkdtempSync(join(tmpdir(), 'cron-remove-revision-'))
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    const job = store.add(new CronJob({ id: 'guarded', prompt: 'work' }))
    const revision = Bun.hash(JSON.stringify(job.toRecord())).toString(16)
    store.update(job.id, { metadata: { execution_receipt: { state: 'running' } } })
    expect(() => store.remove(job.id, revision)).toThrow('Schedule changed')
    expect(store.get(job.id)?.metadata.execution_receipt).toEqual({ state: 'running' })
    const current = store.get(job.id)!
    expect(store.remove(job.id, Bun.hash(JSON.stringify(current.toRecord())).toString(16))).toBe(true)
  } finally { rmSync(directory, { recursive: true, force: true }) }
})

for (const contents of ['{torn', '{}', '[null]', '[{"id":"broken"}]', '[{"id":"same","prompt":"a"},{"id":"same","prompt":"b"}]']) {
  test(`cron store preserves invalid data during reads and mutations: ${contents}`, () => {
    const directory = mkdtempSync(join(tmpdir(), 'cron-integrity-'))
    const path = join(directory, 'jobs.json')
    try {
      writeFileSync(path, contents)
      const store = new JobStore(path)
      expect(() => store.listJobs()).toThrow()
      expect(() => store.add(new CronJob({ id: 'new', prompt: 'work' }))).toThrow()
      expect(() => store.update('broken', { paused: true })).toThrow()
      expect(() => store.remove('broken')).toThrow()
      expect(readFileSync(path, 'utf8')).toBe(contents)
    } finally { rmSync(directory, { recursive: true, force: true }) }
  })
}

test('cron invalid updates fail before persistence and valid legacy records remain usable', () => {
  const directory = mkdtempSync(join(tmpdir(), 'cron-integrity-'))
  const path = join(directory, 'jobs.json')
  try {
    writeFileSync(path, '[{"id":"legacy","prompt":"work"}]')
    const store = new JobStore(path)
    const before = readFileSync(path, 'utf8')
    expect(() => store.update('legacy', { prompt: '' })).toThrow()
    expect(() => store.add(new CronJob({ id: '', prompt: 'work' }))).toThrow()
    expect(readFileSync(path, 'utf8')).toBe(before)
    expect(store.update('legacy', { paused: true })?.paused).toBe(true)
    expect(new JobStore(path).get('legacy')?.prompt).toBe('work')
  } finally { rmSync(directory, { recursive: true, force: true }) }
})

test('cron refuses to recreate a store deleted after opening', () => {
  const directory = mkdtempSync(join(tmpdir(), 'cron-integrity-'))
  const path = join(directory, 'jobs.json')
  try {
    const store = new JobStore(path)
    store.add(new CronJob({ id: 'existing', prompt: 'work' }))
    rmSync(path)
    expect(() => store.add(new CronJob({ id: 'new', prompt: 'work' }))).toThrow()
    expect(() => readFileSync(path)).toThrow()
    expect(() => new JobStore(directory)).toThrow()
  } finally { rmSync(directory, { recursive: true, force: true }) }
})

test('cron writer lock rejects contention and releases after an executor exits', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'cron-writer-'))
  const path = join(directory, 'jobs.json')
  const store = new JobStore(path)
  store.add(new CronJob({ id: 'existing', prompt: 'work' }))
  const child = Bun.spawn([process.execPath, '--eval', `import { Database } from 'bun:sqlite'; const db = new Database(process.argv[1]); db.exec('BEGIN IMMEDIATE'); console.log('locked'); await Bun.sleep(60000);`, `${path}.writer.sqlite`], { stdout: 'pipe', stderr: 'pipe' })
  try {
    const reader = child.stdout.getReader()
    const ready = await reader.read()
    expect(new TextDecoder().decode(ready.value)).toContain('locked')
    reader.releaseLock()
    expect(() => store.add(new CronJob({ id: 'blocked', prompt: 'work' }))).toThrow()
    expect(store.listJobs().map(job => job.id)).toEqual(['existing'])
    child.kill('SIGKILL')
    await child.exited
    store.add(new CronJob({ id: 'after', prompt: 'work' }))
    expect(store.listJobs().map(job => job.id)).toEqual(['existing', 'after'])
  } finally { child.kill(); await child.exited; rmSync(directory, { recursive: true, force: true }) }
})

test('cron revision check inside the writer transaction preserves newer edits', () => {
  const directory = mkdtempSync(join(tmpdir(), 'cron-revision-'))
  const path = join(directory, 'jobs.json')
  try {
    const first = new JobStore(path)
    const original = first.add(new CronJob({ id: 'shared', prompt: 'original' }))
    const revision = Bun.hash(JSON.stringify(original.toRecord())).toString(16)
    new JobStore(path).update('shared', { prompt: 'newer' })
    expect(() => first.update('shared', { prompt: 'stale' }, revision)).toThrow('Schedule changed')
    expect(first.get('shared')?.prompt).toBe('newer')
  } finally { rmSync(directory, { recursive: true, force: true }) }
})
