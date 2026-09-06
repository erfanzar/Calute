// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import type { WebSocketMonitorEvent, WebSocketMonitorSource } from '../src/runtime/websocketMonitorSource.js'
import { RunHistory } from '../src/runtime/runHistory.js'
import { ReactionMailbox } from '../src/runtime/reactionMailbox.js'
import { TerminalRegistry } from '../src/runtime/terminalRegistry.js'
import { TerminalMonitors } from '../src/runtime/terminalMonitors.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { registerMonitorTools } from '../src/tools/monitorTools.js'

function fixture() {
  const history = new RunHistory(':memory:'), mailbox = new ReactionMailbox(':memory:')
  const subscriptions: Array<{ emit: (event: WebSocketMonitorEvent) => void; status: (status: string) => void; error: (error: unknown) => void; closed: boolean }> = []
  const errors: unknown[] = []
  const source: WebSocketMonitorSource = { async open(url, emit, status, error, signal) {
    signal?.throwIfAborted()
    const item = { emit, status, error, closed: false }
    subscriptions.push(item)
    status('Connected')
    return { url, close() { item.closed = true } }
  } }
  const monitors = new TerminalMonitors(new TerminalRegistry(), history, (watch, event) => mailbox.offer(watch.owner, watch.id, event.sequence), error => errors.push(error), mailbox, undefined, { source, resolveWorkspace: owner => `/repo/${owner}` })
  return { history, mailbox, monitors, subscriptions, errors, close() { monitors.close(); mailbox.close(); history.close() } }
}

test('WebSocket watches ignore irrelevant frames and deduplicate matching events before reactions', async () => {
  const f = fixture()
  try {
    const watch = await f.monitors.startWebSocket('owner', { url: 'wss://example.test/feed', match: 'ERROR', reaction: { maxReactions: 1, maxDurationMs: 1000 } })
    const source = f.subscriptions[0]!
    for (let i = 0; i < 5000; i++) source.emit({ kind: 'message', identity: String(i), text: 'healthy' })
    expect(f.history.events('owner', watch.id).events).toEqual([])
    expect(f.mailbox.claim('owner')).toBeUndefined()
    source.emit({ kind: 'message', identity: 'failure', text: 'build error' })
    source.emit({ kind: 'message', identity: 'failure', text: 'build error' })
    expect(f.monitors.inspect('owner', watch.id).events).toHaveLength(1)
    const claim = f.mailbox.claim('owner')!
    expect(claim.throughSequence).toBe(1)
    f.mailbox.settle(claim, 'completed')
    expect(f.mailbox.claim('owner')).toBeUndefined()
    expect(() => f.monitors.inspect('other', watch.id)).toThrow('Unknown monitor')
  } finally { f.close() }
})

test('WebSocket gaps are retained, status is visible, and limits close sources without late rewrites', async () => {
  const f = fixture()
  try {
    const watch = await f.monitors.startWebSocket('owner', { url: 'wss://example.test/feed', match: 'error', maxEvents: 2 })
    const source = f.subscriptions[0]!
    source.status('Reconnecting; messages may be missed')
    source.emit({ kind: 'gap', identity: 'gap:1', text: 'Disconnected; missed messages are unavailable' })
    expect(f.monitors.inspect('owner', watch.id).sourceStatus).toContain('Reconnecting')
    source.emit({ kind: 'message', identity: 'failure', text: 'error ' + 'x'.repeat(64000) })
    expect(f.monitors.inspect('owner', watch.id).state).toBe('limit-reached')
    expect(f.monitors.inspect('owner', watch.id).events[1]!.text.length).toBeLessThanOrEqual(8192)
    expect(f.monitors.inspect('owner', watch.id).events[0]!.text).toContain('Observation gap')
    expect(source.closed).toBe(true)
    source.error(new Error('late'))
    source.status('late connected')
    expect(f.monitors.inspect('owner', watch.id).state).toBe('limit-reached')
    expect(f.errors).toEqual([])
    expect(f.mailbox.claim('owner')).toBeUndefined()
  } finally { f.close() }
})

test('WebSocket tool creates an owner-scoped watch and shutdown preserves interruption evidence', async () => {
  const f = fixture(), tools = new ToolRegistry()
  registerMonitorTools(tools, f.monitors)
  const call = { id: 'watch', type: 'function' as const, function: { name: 'monitor_websocket', arguments: { websocket_url: 'wss://example.test/feed', match: 'error' } } }
  try {
    await expect(tools.execute(call, { metadata: {} })).rejects.toThrow('authenticated session')
    const watch = JSON.parse(await tools.execute(call, { sessionId: 'owner', metadata: {} }))
    expect(watch.source).toEqual({ kind: 'websocket', url: 'wss://example.test/feed' })
    await expect(tools.execute({ ...call, function: { ...call.function, arguments: { ...call.function.arguments, react: true } } }, { sessionId: 'owner', metadata: {} })).rejects.toThrow('direct user turn')
    f.monitors.close()
    expect(f.subscriptions[0]!.closed).toBe(true)
    const restored = new TerminalMonitors(new TerminalRegistry(), f.history)
    try {
      expect(restored.inspect('owner', watch.id)).toMatchObject({ state: 'interrupted', terminalId: '', source: watch.source })
      expect(restored.inspect('owner', watch.id).sourceStatus).toContain('Messages during downtime were not observed')
    } finally { restored.close() }
  } finally { f.close() }
})

test('WebSocket source failure releases attachment and queued reactions', async () => {
  const f = fixture()
  try {
    const watch = await f.monitors.startWebSocket('owner', { url: 'wss://example.test/feed', match: 'error', reaction: { maxReactions: 2, maxDurationMs: 1000 } })
    f.subscriptions[0]!.emit({ kind: 'message', identity: '1', text: 'error' })
    f.subscriptions[0]!.error(new Error('Reconnect attempts exhausted'))
    expect(f.monitors.inspect('owner', watch.id)).toMatchObject({ state: 'failed', error: 'Reconnect attempts exhausted' })
    expect(f.subscriptions[0]!.closed).toBe(true)
    expect(f.mailbox.claim('owner')).toBeUndefined()
    expect(f.history.inspect('owner', watch.id)?.state).toBe('failed')
  } finally { f.close() }
})
