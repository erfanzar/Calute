// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createHmac } from 'node:crypto'
import { connect, type Socket } from 'node:net'
import { mkdir, mkdtemp, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { RunHistory } from '../src/runtime/runHistory.js'
import { TerminalRegistry } from '../src/runtime/terminalRegistry.js'
import { TerminalMonitors } from '../src/runtime/terminalMonitors.js'
import { WebhookMonitorHub } from '../src/runtime/webhookMonitorSource.js'
import { registerMonitorTools } from '../src/tools/monitorTools.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'

class SocketClient {
  private buffer = ''
  private readonly frames: Array<Record<string, unknown>> = []
  private readonly waiters: Array<() => void> = []
  private constructor(private readonly socket: Socket) {
    socket.setEncoding('utf8')
    socket.on('data', chunk => {
      this.buffer += String(chunk)
      let index = this.buffer.indexOf('\n')
      while (index >= 0) {
        const line = this.buffer.slice(0, index).trim()
        this.buffer = this.buffer.slice(index + 1)
        if (line) this.frames.push(JSON.parse(line) as Record<string, unknown>)
        index = this.buffer.indexOf('\n')
      }
      for (const wake of this.waiters.splice(0)) wake()
    })
  }
  static connect(path: string): Promise<SocketClient> {
    return new Promise((resolve, reject) => {
      const socket = connect(path)
      socket.once('connect', () => resolve(new SocketClient(socket)))
      socket.once('error', reject)
    })
  }
  send(id: number, method: string, params: Record<string, unknown>): void {
    this.socket.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`)
  }
  async response(id: number): Promise<Record<string, unknown>> {
    const deadline = Date.now() + 3_000
    for (;;) {
      const index = this.frames.findIndex(frame => frame.id === id)
      if (index >= 0) return this.frames.splice(index, 1)[0]!
      if (Date.now() >= deadline) throw new Error(`response ${id} not received`)
      await new Promise<void>(resolve => { this.waiters.push(resolve); setTimeout(resolve, 20) })
    }
  }
  close(): void { this.socket.destroy() }
}

function result(frame: Record<string, unknown>): Record<string, unknown> { return (frame.result ?? {}) as Record<string, unknown> }

async function waitFor<T>(read: () => T | Promise<T>, predicate: (value: T) => boolean): Promise<T> {
  const deadline = Date.now() + 3_000
  let value = await read()
  while (!predicate(value)) {
    if (Date.now() >= deadline) throw new Error(`condition not met: ${JSON.stringify(value)}`)
    await Bun.sleep(30)
    value = await read()
  }
  return value
}

const secret = 'offline-rpc-webhook-test-secret-1234567890'
function signed(body: string, id: string): RequestInit {
  const timestamp = String(Math.floor(Date.now() / 1000))
  return { method: 'POST', body, headers: { 'x-xerxes-timestamp': timestamp, 'x-xerxes-delivery-id': id,
    'x-xerxes-signature': 'sha256=' + createHmac('sha256', secret).update(`${timestamp}.${id}.${body}`).digest('hex') } }
}

test('daemon exposes configured webhook sources, signed events and ownership without exposing secrets', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-webhook-rpc-'))
  const workspace = join(root, 'workspace')
  await mkdir(workspace)
  const history = new RunHistory(join(root, 'runs.sqlite'))
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: workspace, sessionDirectory: join(root, 'sessions') })
  const terminals = new TerminalRegistry({ runHistory: history })
  const hub = new WebhookMonitorHub({ sources: [{ name: 'build', secret }], port: 0 })
  const monitors = new TerminalMonitors(terminals, history, undefined, undefined, undefined, undefined, undefined, {
    source: hub, resolveWorkspace: owner => runtime.listSessions().find(session => session.id === owner)?.cwd ?? (() => { throw new Error('missing owner') })()
  })
  const socketPath = join(root, 'daemon.sock')
  const server = new DaemonServer({ socketPath, projectDirectory: workspace, runtime, runHistory: history, terminalRegistry: terminals, monitors, monitorWebhookServer: hub, autoTitle: false })
  let owner: SocketClient | undefined, foreign: SocketClient | undefined
  try {
    await server.start()
    owner = await SocketClient.connect(socketPath)
    foreign = await SocketClient.connect(socketPath)
    owner.send(1, 'initialize', { session_key: 'owner', project_dir: workspace })
    foreign.send(2, 'initialize', { session_key: 'foreign', project_dir: workspace })
    await owner.response(1); await foreign.response(2)
    owner.send(3, 'monitor.sources', {})
    expect(result(await owner.response(3))).toEqual({ ok: true, webhooks: [{ name: 'build' }] })
    owner.send(4, 'monitor.create', { source_kind: 'webhook', webhook_name: 'build', file_path: 'bad', match: 'error' })
    expect(result(await owner.response(4)).ok).toBe(false)
    owner.send(5, 'monitor.create', { source_kind: 'webhook', webhook_name: 'build', match: 'error', duration_seconds: 60 })
    const created = result(await owner.response(5))
    expect(created.ok).toBe(true)
    const id = String((created.monitor as Record<string, unknown>).id)
    const tools = new ToolRegistry()
    registerMonitorTools(tools, monitors)
    const sessionId = String((created.monitor as Record<string, unknown>).owner)
    const discover = { id: 'sources', type: 'function' as const, function: { name: 'list_monitor_sources', arguments: {} } }
    await expect(tools.execute(discover, { metadata: {} })).rejects.toThrow('authenticated session')
    expect(JSON.parse(await tools.execute(discover, { sessionId, metadata: {} }))).toEqual({ webhooks: [{ name: 'build' }] })
    const react = { id: 'react', type: 'function' as const, function: { name: 'monitor_webhook', arguments: { webhook_name: 'build', match: 'error', react: true } } }
    await expect(tools.execute(react, { sessionId, metadata: {} })).rejects.toThrow('direct user turn')
    expect(JSON.stringify(created)).not.toContain(secret)
    const endpoint = new URL('/monitors/build', hub.url!).href
    expect((await fetch(endpoint, { method: 'POST', body: 'error unsigned' })).status).toBe(401)
    expect((await fetch(endpoint, signed('info', 'irrelevant'))).status).toBe(202)
    expect((await fetch(endpoint, signed('error build failed', 'failure'))).status).toBe(202)
    expect((await fetch(endpoint, signed('error build failed', 'failure'))).status).toBe(200)
    owner.send(6, 'monitor.inspect', { monitor_id: id })
    const watch = result(await owner.response(6)).monitor as Record<string, unknown>
    expect(watch.source).toEqual({ kind: 'webhook', name: 'build' })
    expect(watch.events).toEqual([expect.objectContaining({ text: expect.stringContaining('error build failed') })])
    foreign.send(7, 'monitor.stop', { monitor_id: id })
    expect((await foreign.response(7)).error).toBeDefined()
    owner.send(8, 'monitor.stop', { monitor_id: id })
    expect(result(await owner.response(8))).toMatchObject({ ok: true, monitor: { state: 'stopped' } })
    expect((await fetch(endpoint, signed('error after stop', 'after-stop'))).status).toBe(410)
    owner.send(9, 'monitor.create', { source_kind: 'webhook', webhook_name: 'build', match: 'error' })
    const second = result(await owner.response(9)).monitor as Record<string, unknown>
    owner.close(); foreign.close()
    await server.stop()
    expect(hub.url).toBeUndefined()
    expect(monitors.list(String(second.owner)).find(row => row.id === second.id)?.state).toBe('interrupted')
  } finally {
    owner?.close(); foreign?.close(); monitors.close(); await server.stop(); history.close()
    await rm(root, { recursive: true, force: true })
  }
})
