// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { clearGoal, createGoal, getGoal } from '../src/runtime/goalDomain.js'
import { DaemonTranscriptStore } from '../src/session/daemonTranscript.js'

test('goal-only sessions persist, list, reload, and retain cleared history', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-goal-only-'))
  try {
    const first = new InMemoryDaemonRuntime(undefined, { sessionDirectory: directory, currentProjectDirectory: directory })
    const session = await first.openSession('goal-only')
    const created = createGoal(session.metadata, session.id, { objective: 'remember this goal' }, 1_000)
    expect(created.phase).toBe('active')
    await first.flushSessions()

    const store = new DaemonTranscriptStore({ directory, currentProjectDirectory: directory })
    expect((await store.list()).map(transcript => transcript.sessionId)).toContain(session.id)

    const reloaded = new InMemoryDaemonRuntime(undefined, { sessionDirectory: directory, currentProjectDirectory: directory })
    const restored = await reloaded.openSession(session.id, undefined, { resume: true, cwd: directory })
    expect(getGoal(restored.metadata, restored.id)?.objective).toBe('remember this goal')

    const cleared = clearGoal(restored.metadata, restored.id, getGoal(restored.metadata, restored.id)!, 2_000)
    expect(cleared.revision).toBe(2)
    await reloaded.flushSessions()

    const afterClear = new InMemoryDaemonRuntime(undefined, { sessionDirectory: directory, currentProjectDirectory: directory })
    const restoredAfterClear = await afterClear.openSession(session.id, undefined, { resume: true, cwd: directory })
    expect(getGoal(restoredAfterClear.metadata, restoredAfterClear.id)).toBeUndefined()
    expect((await store.list()).map(transcript => transcript.sessionId)).toContain(session.id)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('malformed goal history is rejected while empty sessions remain phantom-free', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-goal-only-invalid-'))
  try {
    const empty = new InMemoryDaemonRuntime(undefined, { sessionDirectory: directory, currentProjectDirectory: directory })
    const emptySession = await empty.openSession('empty-only')
    await empty.flushSessions()
    const store = new DaemonTranscriptStore({ directory, currentProjectDirectory: directory })
    expect(await store.list()).toHaveLength(0)
    expect(emptySession.messages).toHaveLength(0)

    const runtime = new InMemoryDaemonRuntime(undefined, { sessionDirectory: directory, currentProjectDirectory: directory })
    const session = await runtime.openSession('invalid-goal')
    createGoal(session.metadata, session.id, { objective: 'will be corrupted' }, 1_000)
    await runtime.flushSessions()
    const path = store.pathFor(session.id)
    const raw = JSON.parse(await readFile(path, 'utf8')) as Record<string, unknown>
    raw.metadata = { goal_changes: [{ kind: 'goal/change', version: 1, operation: 'evidence' }] }
    await writeFile(path, JSON.stringify(raw), 'utf8')
    await expect(store.load(session.id)).rejects.toThrow()
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
