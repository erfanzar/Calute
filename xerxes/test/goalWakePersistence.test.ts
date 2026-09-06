// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { claimGoalWake, queueGoalWake, readGoalWake } from '../src/runtime/goalWake.js'
import { DaemonTranscriptStore } from '../src/session/daemonTranscript.js'

test('valid queued and running wake receipts survive save and reload', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-wake-persist-'))
  try {
    const runtime = new InMemoryDaemonRuntime(undefined, { sessionDirectory: directory, currentProjectDirectory: directory })
    const session = await runtime.openSession('wake-persist')
    session.messages.push({ role: 'user', content: 'start' }, { role: 'assistant', content: 'started' })
    session.turnCount = 1
    const queued = queueGoalWake(session.metadata, session.id, 'goal-1', 2, 100)
    await runtime.flushSessions()
    const store = new DaemonTranscriptStore({ directory, currentProjectDirectory: directory })
    const loaded = await store.load(session.id)
    expect(loaded && readGoalWake(loaded.metadata, session.id)).toEqual(queued)

    const resumed = new InMemoryDaemonRuntime(undefined, { sessionDirectory: directory, currentProjectDirectory: directory })
    const restored = await resumed.openSession(session.id, undefined, { resume: true, cwd: directory })
    expect(readGoalWake(restored.metadata, session.id)).toEqual(queued)
    const running = claimGoalWake(restored.metadata, session.id, queued.id, 'owner-1', 1, 110)
    await resumed.flushSessions()
    const reloadedRunning = new InMemoryDaemonRuntime(undefined, { sessionDirectory: directory, currentProjectDirectory: directory })
    const restoredRunning = await reloadedRunning.openSession(session.id, undefined, { resume: true, cwd: directory })
    expect(readGoalWake(restoredRunning.metadata, session.id)).toEqual(running)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('mismatched or malformed wake metadata is denied at save and reload boundaries', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-wake-invalid-'))
  try {
    const runtime = new InMemoryDaemonRuntime(undefined, { sessionDirectory: directory, currentProjectDirectory: directory })
    const session = await runtime.openSession('wake-invalid')
    session.messages.push({ role: 'user', content: 'start' }, { role: 'assistant', content: 'started' })
    session.turnCount = 1
    session.metadata.goal_wake = {
      version: 1,
      id: crypto.randomUUID(),
      sessionId: 'foreign-session',
      goalId: 'goal-1',
      revision: 1,
      state: 'queued',
      queuedAt: 1,
    }
    await expect(runtime.flushSessions()).rejects.toThrow()
    session.metadata.goal_wake = { version: 1, state: 'queued' }
    await expect(runtime.flushSessions()).rejects.toThrow()
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('branch metadata strips the source wake without mutating the source receipt', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-wake-branch-'))
  try {
    const runtime = new InMemoryDaemonRuntime(undefined, { sessionDirectory: directory, currentProjectDirectory: directory })
    const source = await runtime.openSession('wake-source')
    source.messages.push({ role: 'user', content: 'start' }, { role: 'assistant', content: 'started' })
    source.turnCount = 1
    const queued = queueGoalWake(source.metadata, source.id, 'goal-1', 1, 1)

    const branch = await runtime.openSession('wake-branch')
    branch.messages = [...source.messages]
    branch.turnCount = source.turnCount
    branch.metadata = {
      ...source.metadata,
      forked_from: source.id,
      parent_session_id: source.id,
    }
    await runtime.flushSessions()

    expect(readGoalWake(source.metadata, source.id)).toEqual(queued)
    expect(branch.metadata.goal_wake).toBeUndefined()
    const store = new DaemonTranscriptStore({ directory, currentProjectDirectory: directory })
    const persistedBranch = await store.load(branch.id)
    expect(persistedBranch?.metadata.goal_wake).toBeUndefined()
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
