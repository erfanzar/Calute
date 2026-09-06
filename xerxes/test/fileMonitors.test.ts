// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import type { FileMonitorChange, FileMonitorSource } from '../src/runtime/fileMonitorSource.js'
import { RunHistory } from '../src/runtime/runHistory.js'
import { TerminalRegistry } from '../src/runtime/terminalRegistry.js'
import { TerminalMonitors } from '../src/runtime/terminalMonitors.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { registerMonitorTools } from '../src/tools/monitorTools.js'
import { ReactionMailbox } from '../src/runtime/reactionMailbox.js'

function fixture() {
  const history = new RunHistory(':memory:')
  const events: string[] = [], errors: unknown[] = []
  const subscriptions: Array<{ change: (event: FileMonitorChange) => void; error: (error: unknown) => void; closed: boolean }> = []
  const source: FileMonitorSource = { async open(workspace, path, change, error, signal) {
    signal?.throwIfAborted()
    const item = { change, error, closed: false }
    subscriptions.push(item)
    return { workspace, path: `${workspace}/${path}`, close: () => { item.closed = true } }
  } }
  const monitors = new TerminalMonitors(new TerminalRegistry(), history, (_watch, event) => events.push(event.text), error => errors.push(error), undefined, { source, resolveWorkspace: owner => `/repo/${owner}` })
  return { history, monitors, subscriptions, events, errors, close() { monitors.close(); history.close() } }
}

test('file watches retain bounded events, isolate owners, and detach at the event limit', async () => {
  const f = fixture()
  try {
    const watch = await f.monitors.startFile('owner', { path: 'build.log', maxEvents: 2 })
    expect(watch.source).toEqual({ kind: 'file', path: '/repo/owner/build.log', workspace: '/repo/owner' })
    expect(f.events).toEqual([])
    const source = f.subscriptions[0]!
    source.change({ identity: 'changed:1', text: 'changed' })
    source.change({ identity: 'changed:1', text: 'duplicate' })
    source.change({ identity: 'deleted:1', text: 'deleted' })
    source.change({ identity: 'recreated:2', text: 'late event' })
    source.error(new Error('late error'))
    expect(f.events).toEqual(['changed', 'deleted'])
    expect(f.monitors.inspect('owner', watch.id).state).toBe('limit-reached')
    expect(source.closed).toBe(true)
    expect(f.history.events('owner', watch.id).events.map(event => event.text)).toEqual(f.events)
    expect(() => f.monitors.inspect('other', watch.id)).toThrow('Unknown monitor')
    expect(() => f.monitors.stop('other', watch.id)).toThrow('Unknown monitor')
    expect(f.monitors.list('other')).toEqual([])
    expect(f.errors).toEqual([])
  } finally { f.close() }
})

test('file watch failures and expiry release sources and retain their outcomes', async () => {
  const f = fixture()
  try {
    const failed = await f.monitors.startFile('owner', { path: 'a' })
    f.subscriptions[0]!.error(new Error('parent replaced'))
    expect(f.monitors.inspect('owner', failed.id).state).toBe('failed')
    expect(f.history.inspect('owner', failed.id)?.error).toBe('parent replaced')
    expect(f.subscriptions[0]!.closed).toBe(true)
    const expiring = await f.monitors.startFile('owner', { path: 'b', durationMs: 100 })
    await Bun.sleep(150)
    expect(f.monitors.inspect('owner', expiring.id).state).toBe('expired')
    expect(f.subscriptions[1]!.closed).toBe(true)
    const interrupted = await f.monitors.startFile('owner', { path: 'c' })
    f.monitors.close()
    const restored = new TerminalMonitors(new TerminalRegistry(), f.history)
    try {
      const view = restored.inspect('owner', interrupted.id)
      expect(view.state).toBe('interrupted')
      expect(view.terminalId).toBe('')
      expect(view.sourceStatus).toContain('Changes during downtime were not observed')
      expect(view.source?.kind).toBe('file')
    } finally { restored.close() }
  } finally { f.close() }
})

test('pending file attachments count against capacity and owner disposal aborts them', async () => {
  const history = new RunHistory(':memory:')
  let opened = 0, aborted = 0
  const source: FileMonitorSource = { open(_workspace, _path, _change, _error, signal) {
    opened++
    return new Promise((_resolve, reject) => {
      signal!.addEventListener('abort', () => { aborted++; reject(new Error('attachment aborted')) }, { once: true })
    })
  } }
  const monitors = new TerminalMonitors(new TerminalRegistry(), history, undefined, undefined, undefined, { source, resolveWorkspace: () => '/repo' })
  try {
    const pending = Array.from({ length: 16 }, () => monitors.startFile('owner', { path: 'a' }).catch(error => String(error)))
    await expect(monitors.startFile('owner', { path: 'b' })).rejects.toThrow('Session monitor limit')
    expect(opened).toBe(16)
    monitors.disposeOwner('owner')
    expect((await Promise.all(pending)).every(error => typeof error === 'string' && error.includes('attachment aborted'))).toBe(true)
    expect(aborted).toBe(16)
    expect(monitors.list('owner')).toEqual([])
  } finally { monitors.close(); history.close() }
})

test('native file tool uses trusted ownership and forbids autonomous reaction escalation', async () => {
  const f = fixture()
  const registry = new ToolRegistry()
  registerMonitorTools(registry, f.monitors)
  const call = { id: 'watch', type: 'function' as const, function: { name: 'monitor_file', arguments: { file_path: 'a' } } }
  try {
    await expect(registry.execute(call, { metadata: {} })).rejects.toThrow('authenticated session')
    const result = JSON.parse(await registry.execute(call, { sessionId: 'owner', metadata: {} }))
    expect(result.source.workspace).toBe('/repo/owner')
    await expect(registry.execute({ ...call, function: { ...call.function, arguments: { file_path: 'a', react: true } } }, { sessionId: 'owner', metadata: {} })).rejects.toThrow('direct user turn')
    expect(f.subscriptions).toHaveLength(1)
    f.monitors.stop('owner', result.id)
    expect(f.subscriptions[0]!.closed).toBe(true)
  } finally { f.close() }
})

test('file reactions require explicit grants, obey attempt limits, and revoke queued work on stop', async () => {
  const history = new RunHistory(':memory:')
  const mailbox = new ReactionMailbox(':memory:')
  const callbacks: Array<(event: FileMonitorChange) => void> = []
  const source: FileMonitorSource = { async open(workspace, path, change) {
    callbacks.push(change)
    return { workspace, path: `${workspace}/${path}`, close() {} }
  } }
  const monitors = new TerminalMonitors(new TerminalRegistry(), history, (watch, event) => mailbox.offer(watch.owner, watch.id, event.sequence), undefined, mailbox, { source, resolveWorkspace: () => '/repo' })
  const registry = new ToolRegistry()
  registerMonitorTools(registry, monitors)
  try {
    await monitors.startFile('owner', { path: 'notification' })
    callbacks[0]!({ identity: '1', text: 'changed' })
    expect(mailbox.claim('owner')).toBeUndefined()
    const reactive = JSON.parse(await registry.execute({ id: 'r', type: 'function', function: { name: 'monitor_file', arguments: { file_path: 'reactive', react: true, max_reactions: 1 } } }, { sessionId: 'owner', metadata: { goal_turn_human: true } }))
    callbacks[1]!({ identity: '1', text: 'changed' })
    expect(mailbox.claim('other')).toBeUndefined()
    const claim = mailbox.claim('owner')!
    expect(claim.runId).toBe(reactive.id)
    expect(claim.throughSequence).toBe(1)
    mailbox.settle(claim, 'completed')
    callbacks[1]!({ identity: '2', text: 'changed again' })
    expect(mailbox.claim('owner')).toBeUndefined()
    const pending = await monitors.startFile('owner', { path: 'cancel', reaction: { maxReactions: 1, maxDurationMs: 1000 } })
    callbacks[2]!({ identity: '1', text: 'changed' })
    monitors.stop('owner', pending.id)
    expect(mailbox.claim('owner')).toBeUndefined()
  } finally { monitors.close(); mailbox.close(); history.close() }
})
