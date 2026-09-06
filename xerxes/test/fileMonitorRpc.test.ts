// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { connect, type Socket } from 'node:net'
import { mkdir, mkdtemp, realpath, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { RunHistory } from '../src/runtime/runHistory.js'
import { TerminalRegistry } from '../src/runtime/terminalRegistry.js'
import { TerminalMonitors } from '../src/runtime/terminalMonitors.js'
import { nativeFileMonitorSource } from '../src/runtime/fileMonitorSource.js'

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
      const waiters = this.waiters.splice(0)
      for (const wake of waiters) wake()
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

async function waitFor<T>(read: () => T | Promise<T>, predicate: (value: T) => boolean): Promise<T> {
  const deadline = Date.now() + 3_000
  let value = await read()
  while (!predicate(value)) {
    if (Date.now() >= deadline) throw new Error(`condition not met: ${JSON.stringify(value)}`)
    await Bun.sleep(40)
    value = await read()
  }
  return value
}

function result(frame: Record<string, unknown>): Record<string, unknown> {
  return (frame.result ?? {}) as Record<string, unknown>
}

test('file monitor RPC watches real workspace changes with authenticated ownership', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-file-monitor-rpc-'))
  const workspace = join(root, 'workspace')
  await mkdir(workspace, { recursive: true })
  const file = join(workspace, 'watched.txt')
  const outside = join(root, 'outside.txt')
  await writeFile(file, 'before')
  await writeFile(outside, 'outside')
  const history = new RunHistory(join(root, 'runs.sqlite'))
  const terminals = new TerminalRegistry({ runHistory: history })
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: workspace, sessionDirectory: join(root, 'sessions') })
  const monitors = new TerminalMonitors(terminals, history, undefined, undefined, undefined, {
    source: nativeFileMonitorSource,
    resolveWorkspace: owner => {
      const session = runtime.listSessions().find(candidate => candidate.id === owner)
      if (!session) throw new Error('File monitor owner session is unavailable')
      return session.cwd
    },
  })
  const socketPath = join(root, 'daemon.sock')
  const server = new DaemonServer({ socketPath, projectDirectory: workspace, runtime, runHistory: history, terminalRegistry: terminals, monitors, autoTitle: false })
  await server.start()
  const owner = await SocketClient.connect(socketPath)
  const foreign = await SocketClient.connect(socketPath)
  try {
    owner.send(1, 'initialize', { session_key: 'monitor-owner', project_dir: workspace })
    foreign.send(2, 'initialize', { session_key: 'monitor-foreign', project_dir: workspace })
    const initialized = await owner.response(1)
    await foreign.response(2)
    expect(result(initialized).ok).toBe(true)

    owner.send(3, 'monitor.create', { source_kind: 'file', file_path: 'watched.txt', trigger: 'change', duration_seconds: 60 })
    const created = result(await owner.response(3))
    expect(created.ok).toBe(true)
    const monitor = created.monitor as Record<string, unknown>
    expect(monitor).toMatchObject({ terminalId: '', trigger: 'change', source: { kind: 'file', path: await realpath(file), workspace: await realpath(workspace) } })
    const id = String(monitor.id)

    await writeFile(file, 'after')
    const inspected = await waitFor(async () => {
      owner.send(4, 'monitor.inspect', { monitor_id: id })
      const response = result(await owner.response(4))
      return response.monitor as Record<string, unknown> | undefined
    }, value => Array.isArray(value?.events) && value.events.length === 1)
    if (!inspected) throw new Error('monitor inspection missing after file change')
    expect(inspected.events).toEqual([expect.objectContaining({ text: expect.stringContaining('"event":"changed"') })])

    foreign.send(5, 'monitor.inspect', { monitor_id: id })
    const foreignInspect = await foreign.response(5)
    expect(foreignInspect.error ?? result(foreignInspect).error).toBeDefined()
    foreign.send(51, 'monitor.stop', { monitor_id: id })
    expect((await foreign.response(51)).error).toBeDefined()

    owner.send(6, 'monitor.stop', { monitor_id: id })
    expect(result(await owner.response(6))).toMatchObject({ ok: true, monitor: { state: 'stopped' } })
    await writeFile(file, 'after-stop')
    await Bun.sleep(150)
    owner.send(7, 'monitor.inspect', { monitor_id: id })
    expect(result(await owner.response(7)).monitor).toMatchObject({ state: 'stopped', events: [expect.anything()] })

    owner.send(8, 'monitor.create', { source_kind: 'file', file_path: '../outside.txt', trigger: 'change', duration_seconds: 60 })
    const invalid = await owner.response(8)
    const failure = invalid.error ?? result(invalid).error
    expect(typeof failure === 'object' && failure !== null ? (failure as { message?: unknown }).message : failure).toMatch(/workspace|outside|traversal|regular/i)
  } finally {
    owner.close(); foreign.close()
    await server.stop()
    monitors.close()
    history.close()
    await rm(root, { recursive: true, force: true })
  }
})
