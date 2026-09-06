// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { Scheduler } from '../src/runtime/scheduler.js'
import { JobStore } from '../src/cron/jobs.js'
import { migrateScheduledTrigger, previewScheduleMigration } from '../src/cron/migration.js'

test('preview reports unsupported dependencies without disabling any source', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'schedule-preview-'))
  try {
    const source = new Scheduler({ directory })
    const target = new JobStore(join(directory, 'jobs.json'))
    for (const id of ['simple', 'dependent']) {
      await source.createTrigger({ id, owner: 'user', schedule: { kind: 'interval', intervalSeconds: 60 },
        payload: { id, objective: 'Review', creatorId: 'user', dependencies: id === 'dependent' ? ['other'] : [] } })
    }
    const preview = await previewScheduleMigration(source, directory)
    expect(preview.find(item => item.id === 'simple')).toMatchObject({ supported: true, enabled: true, destination: null })
    expect(preview.find(item => item.id === 'dependent')).toMatchObject({ supported: false, enabled: true, reason: 'Dependent trigger tasks require explicit workflow migration' })
    await expect(migrateScheduledTrigger(source, target, 'dependent', directory)).rejects.toThrow('workflow migration')
    expect([...(await source.load()).triggers.values()].every(trigger => trigger.enabled && !trigger.migratedTo)).toBe(true)
    expect(target.listJobs()).toHaveLength(0)
  } finally { await rm(directory, { recursive: true, force: true }) }
})
test('migration fences the source and idempotently imports a paused interval job', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'schedule-migrate-'))
  try {
    const source = new Scheduler({ directory })
    const target = new JobStore(join(directory, 'jobs.json'))
    const definition = { id: 'old', owner: 'user', schedule: { kind: 'interval' as const, intervalSeconds: 30 }, payload: { id: 'task', objective: 'Review code', creatorId: 'user', dependencies: [] } }
    await source.createTrigger(definition)
    const id = await migrateScheduledTrigger(source, target, 'old', directory)
    expect(target.get(id)?.intervalSeconds).toBe(30)
    expect(target.get(id)?.paused).toBe(true)
    expect(await source.evaluate()).toEqual([])
    await expect(source.enableTrigger('old')).rejects.toThrow('migrated')
    await expect(source.createTrigger(definition)).rejects.toThrow('migrated')
    await expect(source.removeTrigger('old')).rejects.toThrow('recovery record')
    target.update(id, { prompt: 'Reviewed settings' })
    expect(await migrateScheduledTrigger(source, target, 'old', directory)).toBe(id)
    expect(target.listJobs()).toHaveLength(1)
    expect(target.get(id)?.prompt).toBe('Reviewed settings')
  } finally { await rm(directory, { recursive: true, force: true }) }
})
test('failed transfer stays fenced and can retry only its recorded destination', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'schedule-migrate-failure-'))
  try {
    const source = new Scheduler({ directory })
    await source.createTrigger({ id: 'old', owner: 'user', schedule: { kind: 'interval', intervalSeconds: 30 }, payload: { id: 'task', objective: 'Review', creatorId: 'user', dependencies: [] } })
    await expect(source.migrateTrigger('old', 'target', async () => { throw new Error('disk full') })).rejects.toThrow('disk full')
    const restored = new Scheduler({ directory })
    expect((await restored.load()).triggers.get('old')?.enabled).toBe(false)
    expect(await restored.evaluate()).toEqual([])
    await expect(restored.migrateTrigger('old', 'other', async () => {})).rejects.toThrow('another destination')
    let transferred = false
    await restored.migrateTrigger('old', 'target', async () => { transferred = true })
    expect(transferred).toBe(true)
  } finally { await rm(directory, { recursive: true, force: true }) }
})
