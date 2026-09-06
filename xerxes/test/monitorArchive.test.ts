// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { RunHistory } from '../src/runtime/runHistory.js'
import { ReactionMailbox } from '../src/runtime/reactionMailbox.js'
import { TerminalRegistry } from '../src/runtime/terminalRegistry.js'
import { TerminalMonitors } from '../src/runtime/terminalMonitors.js'

test('recovered watches remain inspectable with bounded evidence and cancellable reaction grants', () => {
  const root = mkdtempSync(join(tmpdir(), 'xerxes-watch-recovery-'))
  const path = join(root, 'runs.sqlite')
  const history = new RunHistory(path)
  const mailbox = new ReactionMailbox(':memory:')
  const terminals = new TerminalRegistry()
  const source = terminals.open({ ownerSessionId: 'owner', id: 'source', command: 'build', cwd: root, kind: 'background' })
  const original = new TerminalMonitors(terminals, history, undefined, undefined, mailbox)
  let recoveredHistory: RunHistory | undefined
  try {
    const pending = original.start('owner', { terminalId: source.id, match: 'error', reaction: { maxReactions: 3, maxDurationMs: 1000 } })
    for (let index = 0; index < 25; index++) source.append(`error ${index}\n`)
    const completed = original.start('owner', { terminalId: source.id, match: 'ready', maxEvents: 1 })
    source.append('ready\n')
    recoveredHistory = new RunHistory(path, { isAlive: () => false })
    const recovered = new TerminalMonitors(new TerminalRegistry(), recoveredHistory, undefined, undefined, mailbox)
    expect(recovered.list('owner')).toHaveLength(2)
    const restored = recovered.inspect('owner', pending.id)
    expect(restored).toMatchObject({ terminalId: source.id, trigger: 'output', match: 'error', state: 'interrupted', stopAction: 'cancel-reactions', droppedEvents: 5 })
    expect(restored.events).toHaveLength(20)
    expect(restored.events[0]?.text).toContain('error 5')
    expect(restored.error).toContain('Owning process exited')
    expect(recovered.inspect('owner', completed.id)).toMatchObject({ state: 'archived', match: 'ready' })
    expect(recovered.list('stranger')).toEqual([])
    expect(() => recovered.inspect('stranger', pending.id)).toThrow('Unknown monitor')
    expect(() => recovered.stop('stranger', pending.id)).toThrow('Unknown monitor')
    recovered.stop('owner', pending.id)
    expect(mailbox.inspect('owner', pending.id)?.state).toBe('cancelled')
    expect(recovered.inspect('owner', pending.id).stopAction).toBeNull()
    expect(terminals.inspect('owner', source.id)?.running).toBe(true)
  } finally {
    original.close(); recoveredHistory?.close(); history.close(); mailbox.close()
    rmSync(root, { recursive: true, force: true })
  }
})

test('another live daemon watch is displayed as detached and cannot be stopped through this host', () => {
  const history = new RunHistory(':memory:')
  const terminals = new TerminalRegistry()
  terminals.open({ ownerSessionId: 'owner', id: 'source', command: 'build', cwd: '/repo', kind: 'background' })
  const original = new TerminalMonitors(terminals, history)
  try {
    const watch = original.start('owner', { terminalId: 'source', trigger: 'completion' })
    const other = new TerminalMonitors(new TerminalRegistry(), history)
    expect(other.inspect('owner', watch.id)).toMatchObject({ state: 'detached', stopAction: null })
    expect(() => other.stop('owner', watch.id)).toThrow('another daemon')
    // Reaction runs have no watch configuration and must not appear as watches.
    history.start({ ownerSessionId: 'owner', workspace: '/repo', kind: 'monitor', sourceId: 'claim', title: 'Monitor reaction' })
    expect(other.list('owner')).toHaveLength(1)
  } finally { original.close(); history.close() }
})
