// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { RunHistory } from '../src/runtime/runHistory.js'

const input = (ownerSessionId: string) => ({ ownerSessionId, workspace: '/workspace', kind: 'monitor' as const, sourceId: 'socket-source', title: 'socket watch' })

test('persists and reloads a websocket monitor source', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-websocket-monitor-'))
  try {
    const history = new RunHistory(join(directory, 'runs.sqlite'))
    let runId: string
    try {
      const run = history.startMonitor(input('session-a'), { trigger: 'output', match: 'error', expiresAt: Date.now() + 10_000, source: { kind: 'websocket', url: 'wss://example.test/events' } })
      runId = run.id
    } finally { history.close() }
    const reopened = new RunHistory(join(directory, 'runs.sqlite'))
    try { expect(reopened.monitorConfiguration('session-a', runId!)?.source).toEqual({ kind: 'websocket', url: 'wss://example.test/events' }) }
    finally { reopened.close() }
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('rejects invalid websocket sources and trigger combinations', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-websocket-monitor-invalid-'))
  try {
    const history = new RunHistory(join(directory, 'runs.sqlite'))
    try {
      expect(() => history.startMonitor(input('session-a'), { trigger: 'change', match: 'x', expiresAt: Date.now() + 1000, source: { kind: 'websocket', url: 'wss://example.test' } })).toThrow()
      expect(() => history.startMonitor(input('session-a'), { trigger: 'output', match: 'x', expiresAt: Date.now() + 1000, source: { kind: 'websocket', url: 'https://example.test' } })).toThrow()
      expect(() => history.startMonitor(input('session-a'), { trigger: 'output', match: 'x', expiresAt: Date.now() + 1000, source: { kind: 'websocket', url: 'wss://user:pass@example.test/path?secret=1' } })).toThrow()
    } finally { history.close() }
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('keeps websocket monitor records owner-scoped', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-websocket-monitor-owner-'))
  try {
    const history = new RunHistory(join(directory, 'runs.sqlite'))
    try {
      const run = history.startMonitor(input('session-a'), { trigger: 'output', match: 'x', expiresAt: Date.now() + 1000, source: { kind: 'websocket', url: 'ws://localhost:9000' } })
      expect(history.monitorConfiguration('session-b', run.id)).toBeUndefined()
    } finally { history.close() }
  } finally { await rm(directory, { recursive: true, force: true }) }
})
