// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { Database } from 'bun:sqlite'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { RunHistory } from '../src/runtime/runHistory.js'

const input = { ownerSessionId: 'owner', workspace: '/repo', kind: 'schedule' as const, sourceId: 'cron-1', title: 'Run tests' }

test('run completion and versioned acknowledgement survive reconnect and are owner scoped', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-runs-'))
  const path = join(directory, 'runs.sqlite')
  let history = new RunHistory(path)
  try {
    const run = history.start(input)
    expect(history.list('owner', { unreadOnly: true })).toEqual([])
    const completed = history.finish('owner', run.id, 'succeeded', { output: 'tests passed' })
    expect(completed.unread).toBe(true)
    expect(history.inspect('other', run.id)).toBeUndefined()
    expect(() => history.acknowledge('other', run.id, completed.revision)).toThrow()
    expect(() => history.acknowledge('owner', run.id, run.revision)).toThrow('changed')
    history.close()
    history = new RunHistory(path)
    expect(history.list('owner', { unreadOnly: true })).toHaveLength(1)
    history.acknowledge('owner', run.id, completed.revision)
    history.close()
    history = new RunHistory(path)
    expect(history.list('owner', { unreadOnly: true })).toEqual([])
    expect(history.inspect('owner', run.id)?.output).toBe('tests passed')
    expect(() => history.finish('owner', run.id, 'failed')).toThrow('different outcome')
  } finally { history.close(); rmSync(directory, { recursive: true, force: true }) }
})

test('restart marks dead owners interrupted while preserving live runs and bounded output', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-runs-recovery-'))
  const path = join(directory, 'runs.sqlite')
  let history = new RunHistory(path, { pid: 99999, isAlive: () => true })
  try {
    const dead = history.start(input)
    history.close()
    history = new RunHistory(path, { isAlive: pid => pid === process.pid })
    expect(history.inspect('owner', dead.id)).toMatchObject({ state: 'interrupted', unread: true })
    const live = history.start(input)
    const second = new RunHistory(path)
    try { expect(second.inspect('owner', live.id)?.state).toBe('running') } finally { second.close() }
    const finished = history.finish('owner', live.id, 'failed', { output: 'x'.repeat(90_000), error: 'exit 1' })
    expect(finished.output.length).toBe(64_000)
    expect(finished.outputTruncated).toBe(true)
    expect(() => history.list('owner', { limit: 99999 })).toThrow()
  } finally { history.close(); rmSync(directory, { recursive: true, force: true }) }
})

test('completion listeners fire once after persistence and keep quiet foreground successes out of the inbox', () => {
  const history = new RunHistory(':memory:')
  const seen: string[] = []
  const unsubscribe = history.subscribe(run => {
    expect(history.inspect(run.ownerSessionId, run.id)?.state).toBe(run.state)
    seen.push(run.id)
  })
  try {
    const run = history.start(input)
    history.checkpointOutput('owner', run.id, 'partial output')
    history.finish('owner', run.id, 'failed', { error: 'failure' })
    history.finish('owner', run.id, 'failed')
    expect(history.inspect('owner', run.id)?.output).toBe('partial output')
    expect(seen).toEqual([run.id])
    const quiet = history.start(input)
    history.finish('owner', quiet.id, 'succeeded', { notify: false })
    expect(seen).toEqual([run.id])
    unsubscribe()
    history.finish('owner', history.start(input).id, 'failed')
    expect(seen).toEqual([run.id])
  } finally { history.close() }
})

test('monitor evidence replays after reopening with stable cursors and revision-aware attention', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-event-replay-'))
  const path = join(directory, 'runs.sqlite')
  let history = new RunHistory(path)
  try {
    const run = history.start({ ...input, kind: 'monitor' })
    const first = { sequence: 1, text: 'error first', at: 100 }
    expect(history.appendEvent('owner', run.id, first, first.text)).toBe(true)
    const revision = history.inspect('owner', run.id)!.revision
    expect(history.list('owner', { unreadOnly: true })).toHaveLength(1)
    history.acknowledge('owner', run.id, revision)
    expect(history.appendEvent('owner', run.id, first, first.text)).toBe(false)
    expect(history.inspect('owner', run.id)?.unread).toBe(false)
    history.appendEvent('owner', run.id, { sequence: 2, text: 'error second', at: 101 }, 'second tail')
    expect(() => history.acknowledge('owner', run.id, revision)).toThrow('changed')
    history.close()
    history = new RunHistory(path)
    const page = history.events('owner', run.id, 0, 1)
    expect(page).toEqual({ events: [first], nextCursor: 1, hasMore: true })
    expect(history.events('owner', run.id, page.nextCursor)).toEqual({
      events: [{ sequence: 2, text: 'error second', at: 101 }], nextCursor: 2, hasMore: false,
    })
    expect(history.events('owner', run.id, 2).events).toEqual([])
    expect(history.inspect('owner', run.id)?.output).toBe('second tail')
    expect(() => history.events('other', run.id)).toThrow('Unknown run')
    expect(() => history.appendEvent('other', run.id, first, 'spoof')).toThrow('Unknown run')
    expect(() => history.events('owner', run.id, -1)).toThrow('cursor')
  } finally { history.close(); rmSync(directory, { recursive: true, force: true }) }
})

test('event conflicts and gaps cannot overwrite evidence or update its displayed tail', () => {
  const history = new RunHistory(':memory:')
  try {
    const run = history.start({ ...input, kind: 'monitor' })
    history.appendEvent('owner', run.id, { sequence: 1, text: 'original', at: 1 }, 'original')
    expect(() => history.appendEvent('owner', run.id, { sequence: 1, text: 'conflict', at: 1 }, 'wrong')).toThrow('Conflicting')
    expect(() => history.appendEvent('owner', run.id, { sequence: 3, text: 'gap', at: 3 }, 'wrong')).toThrow('gap')
    expect(history.inspect('owner', run.id)?.output).toBe('original')
    history.finish('owner', run.id, 'cancelled')
    expect(() => history.appendEvent('owner', run.id, { sequence: 2, text: 'too late', at: 2 }, 'wrong')).toThrow('completed')
    expect(history.events('owner', run.id).events).toHaveLength(1)
    expect(history.inspect('owner', run.id)?.output).toBe('original')
  } finally { history.close() }
})

test('schedule history filters before limiting and keeps workspace isolation', () => {
  let time = 0
  const history = new RunHistory(':memory:', { now: () => ++time })
  try {
    const wanted = history.start(input)
    for (let i = 0; i < 120; i++) history.start({ ...input, sourceId: `other-${i}` })
    history.start({ ...input, kind: 'agent' })
    history.start({ ...input, workspace: '/foreign' })
    expect(history.listWorkspace('/repo', { sourceId: 'cron-1', kind: 'schedule', limit: 1 }).map(run => run.id)).toEqual([wanted.id])
    expect(history.list('stranger', { sourceId: 'cron-1', kind: 'schedule' })).toEqual([])
  } finally { history.close() }
})

test('run pages preserve timestamp ties and remain stable when newer results arrive', () => {
  let now = 100
  const history = new RunHistory(':memory:', { now: () => now })
  try {
    for (let i = 0; i < 205; i++) history.start({ ...input, title: String(i) })
    const expected = history.list('owner', { limit: 500 }).map(run => run.id)
    const first = history.list('owner')
    const last = first.at(-1)!
    now = 200
    history.start({ ...input, title: 'New arrival' })
    history.start({ ...input, ownerSessionId: 'other' })
    const second = history.list('owner', { before: { startedAt: last.startedAt, id: last.id } })
    const next = second.at(-1)!
    const third = history.list('owner', { before: { startedAt: next.startedAt, id: next.id } })
    expect([...first, ...second, ...third].map(run => run.id)).toEqual(expected)
    expect(third).toHaveLength(5)
    expect(() => history.list('owner', { before: { startedAt: NaN, id: 'x' } })).toThrow('cursor')
  } finally { history.close() }
})

test('state and kind filters apply before paging and preserve owner isolation', () => {
  let now = 1
  const history = new RunHistory(':memory:', { now: () => now++ })
  try {
    const failed = history.start({ ...input, kind: 'agent', title: 'Old failure' })
    history.finish('owner', failed.id, 'failed')
    const other = history.start({ ...input, ownerSessionId: 'other', kind: 'agent' })
    history.finish('other', other.id, 'failed')
    for (let i = 0; i < 120; i++) history.start(input)
    expect(history.list('owner', { kind: 'agent', state: 'failed' }).map(run => run.id)).toEqual([failed.id])
    expect(history.listWorkspace('/repo', { state: 'failed' })).toHaveLength(2)
    expect(history.list('owner', { state: 'running', limit: 10 })).toHaveLength(10)
  } finally { history.close() }
})

test('attempt usage persists independently, is owner scoped, and finishes atomically with the outcome', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-run-usage-'))
  const path = join(directory, 'runs.sqlite')
  let history = new RunHistory(path)
  const usage = { input_tokens: 12, output_tokens: 4, measured_calls: 1, settled_calls: 2, pending_calls: 0, complete: false }
  try {
    const first = history.start(input)
    const observed: unknown[] = []
    history.subscribe(run => observed.push(run.tokenUsage))
    expect(() => history.finish('other', first.id, 'succeeded', { tokenUsage: usage })).toThrow('Unknown run')
    expect(() => history.finish('owner', first.id, 'succeeded', { tokenUsage: { ...usage, complete: true } })).toThrow('completeness')
    expect(history.inspect('owner', first.id)?.state).toBe('running')
    history.finish('owner', first.id, 'failed', { tokenUsage: usage, error: 'provider failed', output: 'partial evidence' })
    expect(observed).toEqual([usage])
    const second = history.start(input)
    history.finish('owner', second.id, 'cancelled', { tokenUsage: { ...usage, input_tokens: 30 } })
    // Repeated finalization cannot overwrite the original attempt evidence.
    history.finish('owner', first.id, 'failed', { tokenUsage: { ...usage, input_tokens: 999 } })
    history.close()
    history = new RunHistory(path)
    expect(history.inspect('owner', first.id)?.tokenUsage).toEqual(usage)
    expect(history.inspect('owner', first.id)?.output).toBe('partial evidence')
    expect(history.inspect('owner', second.id)?.tokenUsage?.input_tokens).toBe(30)
    expect(history.listWorkspace('/elsewhere')).toEqual([])
  } finally { history.close(); rmSync(directory, { recursive: true, force: true }) }
})

test('legacy run databases migrate without fabricating token usage', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-run-migration-'))
  const path = join(directory, 'runs.sqlite')
  const db = new Database(path)
  db.exec(`CREATE TABLE run_history (
    id TEXT PRIMARY KEY, owner TEXT NOT NULL, workspace TEXT NOT NULL, kind TEXT NOT NULL,
    source TEXT NOT NULL, title TEXT NOT NULL, state TEXT NOT NULL, started INTEGER NOT NULL,
    ended INTEGER, output TEXT NOT NULL DEFAULT '', truncated INTEGER NOT NULL DEFAULT 0,
    error TEXT, revision INTEGER NOT NULL DEFAULT 1, acknowledged INTEGER NOT NULL DEFAULT 1,
    pid INTEGER NOT NULL
  ); INSERT INTO run_history(id,owner,workspace,kind,source,title,state,started,pid)
    VALUES('old','owner','/repo','schedule','cron-1','Old attempt','succeeded',1,1);`)
  db.close()
  const history = new RunHistory(path)
  try {
    expect(history.inspect('owner', 'old')?.tokenUsage).toBeNull()
    expect(history.inspect('owner', 'old')?.state).toBe('succeeded')
    const run = history.start(input)
    const usage = { input_tokens: 0, output_tokens: 0, measured_calls: 0, settled_calls: 1, pending_calls: 0, complete: false }
    expect(history.finish('owner', run.id, 'cancelled', { tokenUsage: usage }).tokenUsage).toEqual(usage)
  } finally { history.close(); rmSync(directory, { recursive: true, force: true }) }
})

test('abrupt owner death retains settled usage and the pending call on recovery', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-usage-crash-'))
  const path = join(directory, 'runs.sqlite')
  const script = `
    import { RunHistory } from ${JSON.stringify(join(import.meta.dir, '../src/runtime/runHistory.ts'))};
    import { ModelCallBudget } from ${JSON.stringify(join(import.meta.dir, '../src/llms/callBudget.ts'))};
    const history = new RunHistory(${JSON.stringify(path)});
    const run = history.start(${JSON.stringify(input)});
    const budget = new ModelCallBudget(undefined, usage => history.checkpointUsage('owner', run.id, usage));
    budget.charge()({inputTokens: 17, outputTokens: 3});
    budget.charge();
    console.log(run.id);
    setInterval(() => {}, 1000);
  `
  const child = Bun.spawn([process.execPath, '-e', script], { stdout: 'pipe', stderr: 'pipe' })
  let history: RunHistory | undefined
  try {
    const reader = child.stdout.getReader()
    const first = await reader.read()
    reader.releaseLock()
    const id = new TextDecoder().decode(first.value).trim()
    expect(id).toMatch(/^[a-f0-9-]{36}$/)
    child.kill(9)
    await child.exited
    history = new RunHistory(path)
    const recovered = history.inspect('owner', id)
    expect(recovered?.state).toBe('interrupted')
    expect(recovered?.tokenUsage).toEqual({ input_tokens: 17, output_tokens: 3, measured_calls: 1, settled_calls: 1, pending_calls: 1, complete: false })
    expect(() => history!.checkpointUsage('owner', id, { input_tokens: 0, output_tokens: 0, measured_calls: 0, settled_calls: 0, pending_calls: 0, complete: true })).toThrow('completed')
  } finally { child.kill(9); await child.exited; history?.close(); rmSync(directory, { recursive: true, force: true }) }
}, 10000)

test.each(['different-command', '', 'original-command'])('recovery checks live PID command identity: %s', observed => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-run-identity-'))
  const path = join(directory, 'runs.sqlite')
  let history = new RunHistory(path, { pid: 12345, isAlive: () => true, commandOf: () => 'original-command' })
  try {
    const run = history.start(input)
    history.checkpointOutput('owner', run.id, 'partial evidence')
    history.close()
    history = new RunHistory(path, { isAlive: () => true, commandOf: pid => pid === 12345 ? observed : 'new-owner' })
    expect(history.inspect('owner', run.id)).toMatchObject({ state: observed === 'different-command' ? 'interrupted' : 'running', output: 'partial evidence' })
    if (observed === 'different-command') expect(history.inspect('owner', run.id)?.error).toContain('identity changed')
  } finally { history.close(); rmSync(directory, { recursive: true, force: true }) }
})

test.each(['start-one', 'start-two', ''])('recovery checks same-command process start identity: %s', observed => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-run-birth-'))
  const path = join(directory, 'runs.sqlite')
  let history = new RunHistory(path, { pid: 12345, isAlive: () => true, commandOf: () => 'same-daemon', startOf: () => 'start-one' })
  try {
    const run = history.start(input)
    history.checkpointOutput('owner', run.id, 'retained output')
    history.close()
    history = new RunHistory(path, { isAlive: () => true, commandOf: () => 'same-daemon', startOf: () => observed })
    expect(history.inspect('owner', run.id)).toMatchObject({ state: observed === 'start-two' ? 'interrupted' : 'running', output: 'retained output' })
    if (observed === 'start-two') {
      expect(history.inspect('owner', run.id)?.error).toContain('different start time')
      expect(history.inspect('owner', run.id)?.unread).toBe(true)
    }
  } finally { history.close(); rmSync(directory, { recursive: true, force: true }) }
})

test('latest outcome is scoped to owner, kind and source and survives restart without exposing output', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-latest-outcome-'))
  const path = join(directory, 'runs.sqlite')
  let now = 100
  let history = new RunHistory(path, { now: () => now })
  try {
    const first = history.start(input)
    history.finish('owner', first.id, 'failed', { output: 'private diagnostic output' })
    now = 200
    const latest = history.start(input)
    history.finish('owner', latest.id, 'succeeded')
    now = 300
    history.start({ ...input, ownerSessionId: 'other' })
    history.start({ ...input, kind: 'terminal' })
    history.start({ ...input, sourceId: 'other-source' })
    const expected = { id: latest.id, state: 'succeeded', startedAt: 200, endedAt: 200 } as const
    expect(history.latestOutcome('owner', 'cron-1', 'schedule')).toEqual(expected)
    expect(history.latestOutcome('unknown', 'cron-1', 'schedule')).toBeNull()
    history.close()
    history = new RunHistory(path)
    expect(history.latestOutcome('owner', 'cron-1', 'schedule')).toEqual(expected)
  } finally { history.close(); rmSync(directory, { recursive: true, force: true }) }
})
