// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import type { WebhookMonitorSource } from '../src/runtime/webhookMonitorSource.js'
import { RunHistory } from '../src/runtime/runHistory.js'
import { TerminalRegistry } from '../src/runtime/terminalRegistry.js'
import { TerminalMonitors } from '../src/runtime/terminalMonitors.js'

function fixture() {
  const history = new RunHistory(':memory:')
  const subscriptions: Array<{ emit: (event: { text: string; identity: string }) => void; error: (error: unknown) => void; closed: boolean }> = []
  const source: WebhookMonitorSource = {
    list: () => [{ name: 'deploy' }, { name: 'alerts_1' }],
    async open(name, emit, error, signal) {
      signal?.throwIfAborted()
      const item = { emit, error, closed: false }
      subscriptions.push(item)
      return { name, close: () => { item.closed = true } }
    },
  }
  const monitors = new TerminalMonitors(new TerminalRegistry(), history, undefined, undefined, undefined, undefined, undefined, { source, resolveWorkspace: owner => `/repo/${owner}` })
  return { history, monitors, subscriptions, close: () => { monitors.close(); history.close() } }
}

test('webhook watches expose sources, match messages, deduplicate identities, and enforce ownership', async () => {
  const f = fixture()
  try {
    expect(f.monitors.webhookSources('owner')).toEqual([{ name: 'deploy' }, { name: 'alerts_1' }])
    const watch = await f.monitors.startWebhook('owner', { name: 'deploy', match: 'error', maxEvents: 2 })
    const source = f.subscriptions[0]!
    for (let i = 0; i < 1000; i++) source.emit({ text: 'healthy', identity: String(i) })
    source.emit({ text: 'build error', identity: 'same' })
    source.emit({ text: 'build error', identity: 'same' })
    source.emit({ text: 'runtime error', identity: 'other' })
    expect(f.monitors.inspect('owner', watch.id).events.map(event => event.text)).toEqual(['[Webhook message] build error', '[Webhook message] runtime error'])
    expect(f.monitors.inspect('owner', watch.id).state).toBe('limit-reached')
    expect(source.closed).toBe(true)
    expect(() => f.monitors.inspect('other', watch.id)).toThrow('Unknown monitor')
  } finally { f.close() }
})

test('webhook stop and cancellation close the subscription and preserve an interruption gap', async () => {
  const f = fixture()
  try {
    const watch = await f.monitors.startWebhook('owner', { name: 'deploy', match: 'event', durationMs: 1_000 })
    expect(f.monitors.stop('owner', watch.id).state).toBe('stopped')
    expect(f.subscriptions[0]!.closed).toBe(true)
    const controller = new AbortController()
    controller.abort(new Error('cancelled'))
    await expect(f.monitors.startWebhook('owner', { name: 'deploy', match: 'event', signal: controller.signal })).rejects.toThrow('cancelled')
  } finally { f.close() }
})
