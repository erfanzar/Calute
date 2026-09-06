// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, spyOn, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonTranscriptStore } from '../src/session/daemonTranscript.js'

test('context saves and deletion cannot overlap or recreate a deleted transcript', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-controls-race-'))
  const store = new DaemonTranscriptStore({ directory, currentProjectDirectory: directory })
  const runtime = new InMemoryDaemonRuntime(undefined, { transcriptStore: store, currentProjectDirectory: directory })
  const session = await runtime.openSession('controls-race')
  session.messages.push({ role: 'user', content: 'hello' }, { role: 'assistant', content: 'done' })
  session.turnCount = 1
  await runtime.flushSessions()
  const entered = Promise.withResolvers<void>(), release = Promise.withResolvers<void>()
  const originalSave = store.save.bind(store)
  const save = spyOn(store, 'save').mockImplementation(async (...args) => { entered.resolve(); await release.promise; return originalSave(...args) })
  const controls = { version: 1 as const, revision: 1, pins: [{ scope: 'project' as const, path: 'MEMORY.md', content: 'fact' }], excluded: [] }
  try {
    const pending = runtime.saveSessionContextControls(session.sessionKey, controls)
    await entered.promise
    await expect(runtime.deleteSavedSession(session.id)).rejects.toThrow('persistence operation')
    release.resolve(); await pending; save.mockRestore()
    const deleting = Promise.withResolvers<void>(), allowDelete = Promise.withResolvers<void>()
    const originalRemove = store.remove.bind(store)
    const remove = spyOn(store, 'remove').mockImplementation(async id => { deleting.resolve(); await allowDelete.promise; return originalRemove(id) })
    try {
      const deletion = runtime.deleteSavedSession(session.id)
      await deleting.promise
      await expect(runtime.saveSessionContextControls(session.sessionKey, { ...controls, revision: 2 })).rejects.toThrow('persistence operation')
      allowDelete.resolve(); expect(await deletion).toBe(true)
      expect(await store.load(session.id)).toBeUndefined()
      expect(runtime.listSessions()).toHaveLength(0)
    } finally { allowDelete.resolve(); remove.mockRestore() }
  } finally { release.resolve(); save.mockRestore(); await rm(directory, { recursive: true, force: true }) }
})

test('an independent stale writer cannot overwrite newer controls with unchanged messages', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-controls-stale-'))
  try {
    const first = new InMemoryDaemonRuntime(undefined, { sessionDirectory: directory, currentProjectDirectory: directory })
    const session = await first.openSession('first-writer')
    session.messages.push({ role: 'user', content: 'hello' }, { role: 'assistant', content: 'done' })
    session.turnCount = 1
    await first.flushSessions()
    const second = new InMemoryDaemonRuntime(undefined, { sessionDirectory: directory, currentProjectDirectory: directory })
    await second.openSession(session.id, session.agentId, { resume: true, cwd: session.cwd })
    const controls = { version: 1 as const, revision: 1, pins: [{ scope: 'project' as const, path: 'MEMORY.md', content: 'new fact' }], excluded: [] }
    await first.saveSessionContextControls(session.sessionKey, controls)
    await expect(second.flushSessions()).rejects.toThrow('changed in another writer')
    const third = new InMemoryDaemonRuntime(undefined, { sessionDirectory: directory, currentProjectDirectory: directory })
    const restored = await third.openSession(session.id, session.agentId, { resume: true, cwd: session.cwd })
    expect(restored.metadata.context_controls).toEqual(controls)
  } finally { await rm(directory, { recursive: true, force: true }) }
})
