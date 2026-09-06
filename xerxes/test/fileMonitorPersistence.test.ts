// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Database } from 'bun:sqlite'
import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { RunHistory } from '../src/runtime/runHistory.js'

function input(sourceId: string) {
  return { ownerSessionId: 'session-a', workspace: '/workspace', kind: 'monitor' as const, sourceId, title: 'watch' }
}

test('persists terminal and file monitor sources and preserves old absent sources', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-file-monitor-'))
  const path = join(directory, 'runs.sqlite')
  try {
    const history = new RunHistory(path)
    const terminal = history.startMonitor(input('terminal-a'), { trigger: 'output', match: 'error', expiresAt: Date.now() + 10_000, source: { kind: 'terminal', terminalId: 'terminal-a' } })
    const file = history.startMonitor(input('file-a'), { trigger: 'change', match: 'changed', expiresAt: Date.now() + 10_000, source: { kind: 'file', path: 'src/app.ts', workspace: '/workspace' } })
    const legacy = history.startMonitor(input('legacy'), { trigger: 'completion', match: 'done', expiresAt: Date.now() + 10_000 })
    history.close()
    const reopened = new RunHistory(path)
    try {
      expect(reopened.monitorConfiguration('session-a', terminal.id)?.source).toEqual({ kind: 'terminal', terminalId: 'terminal-a' })
      expect(reopened.monitorConfiguration('session-a', file.id)?.source).toEqual({ kind: 'file', path: 'src/app.ts', workspace: '/workspace' })
      expect(reopened.monitorConfiguration('session-a', legacy.id)?.source).toBeUndefined()
    } finally { reopened.close() }
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('rejects invalid source and malformed stored source', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-file-monitor-invalid-'))
  const path = join(directory, 'runs.sqlite')
  try {
    const history = new RunHistory(path)
    try {
      expect(() => history.startMonitor(input('bad'), { trigger: 'change', match: 'x', expiresAt: Date.now() + 1000, source: { kind: 'terminal', terminalId: 't' } })).toThrow()
      const run = history.startMonitor(input('stored-bad'), { trigger: 'output', match: 'x', expiresAt: Date.now() + 1000 })
      const db = new Database(path)
      try { db.query('UPDATE monitor_configurations SET source_json=? WHERE run_id=?').run('{"kind":"future"}', run.id) }
      finally { db.close() }
      const reopened = new RunHistory(path)
      try { expect(() => reopened.monitorConfiguration('session-a', run.id)).toThrow('Invalid stored monitor source') }
      finally { reopened.close() }
    } finally { history.close() }
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('migrates an old monitor table with no source column', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-file-monitor-old-schema-'))
  const path = join(directory, 'runs.sqlite')
  try {
    const db = new Database(path)
    try {
      db.exec(`CREATE TABLE run_history (id TEXT PRIMARY KEY, owner TEXT NOT NULL, workspace TEXT NOT NULL, kind TEXT NOT NULL, source TEXT NOT NULL, title TEXT NOT NULL, state TEXT NOT NULL, started INTEGER NOT NULL, ended INTEGER, output TEXT NOT NULL DEFAULT '', truncated INTEGER NOT NULL DEFAULT 0, error TEXT, revision INTEGER NOT NULL DEFAULT 1, acknowledged INTEGER NOT NULL DEFAULT 1, pid INTEGER NOT NULL); CREATE TABLE monitor_configurations (run_id TEXT PRIMARY KEY, trigger TEXT NOT NULL, match TEXT NOT NULL, expires INTEGER NOT NULL);`)
      db.query('INSERT INTO run_history(id,owner,workspace,kind,source,title,state,started,pid) VALUES(?,?,?,?,?,?,?,?,?)').run('old-run', 'session-a', '/workspace', 'monitor', 'terminal-a', 'watch', 'succeeded', Date.now(), process.pid)
      db.query('INSERT INTO monitor_configurations(run_id,trigger,match,expires) VALUES(?,?,?,?)').run('old-run', 'output', 'error', Date.now() + 10_000)
    } finally { db.close() }
    const history = new RunHistory(path)
    expect(history.monitorConfiguration('session-a', 'old-run')).toEqual({ trigger: 'output', match: 'error', expiresAt: expect.any(Number) })
    history.close()
  } finally { await rm(directory, { recursive: true, force: true }) }
})
