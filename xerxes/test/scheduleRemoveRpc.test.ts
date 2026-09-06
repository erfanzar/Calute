// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, realpath, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { CronJob, JobStore } from '../src/cron/jobs.js'
import { DaemonServer } from '../src/daemon/server.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { requestDaemonControl } from '../src/daemon/controlClient.js'

test('schedule.remove rejects unfinished execution receipts and preserves archived evidence', async () => {
  const directory = await realpath(await mkdtemp(join(tmpdir(), 'schedule-remove-')))
  const store = new JobStore(join(directory, 'jobs.json'))
  const socketPath = join(directory, 'daemon.sock')
  const archive = join(directory, 'archive', 'guarded', 'result.md')
  await Bun.write(archive, 'retained evidence')
  store.add(new CronJob({ id: 'guarded', prompt: 'work', paused: true, projectRoot: directory,
    metadata: { execution_receipt: { state: 'running', occurrence: 'fixture' } } }))
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') })
  const server = new DaemonServer({ runtime, socketPath, projectDirectory: directory,
    cronStoreFactory: () => store, cronLeasePath: join(directory, 'lease'), cronArchiveDirectory: join(directory, 'archive') })
  await server.start()
  try {
    expect(await requestDaemonControl(socketPath, 'schedule.remove', { schedule_id: 'guarded' })).toMatchObject({ ok: false })
    expect(store.get('guarded')).toBeDefined()
    store.update('guarded', { metadata: {} })
    expect(await requestDaemonControl(socketPath, 'schedule.remove', { schedule_id: 'guarded' })).toEqual({ ok: true, schedule_id: 'guarded', removed: true })
    expect(store.get('guarded')).toBeUndefined()
    expect(await Bun.file(archive).text()).toBe('retained evidence')
  } finally { await server.stop(); await rm(directory, { recursive: true, force: true }) }
})
