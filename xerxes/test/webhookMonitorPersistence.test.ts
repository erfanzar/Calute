// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { RunHistory } from '../src/runtime/runHistory.js'

test('persists and reloads webhook monitor source across restart', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-webhook-monitor-'))
  const path = join(directory, 'runs.sqlite')
  const input = { ownerSessionId: 'session-a', workspace: '/repo', kind: 'monitor' as const, sourceId: 'deploy', title: 'Watch Webhook deploy' }
  const history = new RunHistory(path)
  const run = history.startMonitor(input, { trigger: 'output', match: 'error', expiresAt: Date.now() + 10_000, source: { kind: 'webhook', name: 'deploy' } })
  history.close()
  const reopened = new RunHistory(path)
  try { expect(reopened.monitorConfiguration('session-a', run.id)?.source).toEqual({ kind: 'webhook', name: 'deploy' }) }
  finally { reopened.close() }
})

test('rejects invalid webhook names and trigger combinations', () => {
  const history = new RunHistory(':memory:')
  const input = { ownerSessionId: 'session-a', workspace: '/repo', kind: 'monitor' as const, sourceId: 'bad', title: 'Watch Webhook bad' }
  try {
    expect(() => history.startMonitor(input, { trigger: 'change', match: 'x', expiresAt: Date.now() + 1000, source: { kind: 'webhook', name: 'deploy' } })).toThrow()
    expect(() => history.startMonitor(input, { trigger: 'output', match: 'x', expiresAt: Date.now() + 1000, source: { kind: 'webhook', name: 'bad.name' } })).toThrow()
  } finally { history.close() }
})
