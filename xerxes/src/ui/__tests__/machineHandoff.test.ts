// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { EventEmitter } from 'node:events'
import { writeFileSync, existsSync } from 'node:fs'
import type { spawn } from 'node:child_process'
import { describe, expect, it, vi } from 'vitest'
import { connectRemoteMachine, parseRemoteMachine, remoteMachineCommand } from '../lib/machineHandoff.js'

const machine = { alias: 'gpu', target: 'me@server', workspacePath: "/work/it's a project; $(touch nope)" }
const child = () => {
  const event = Object.assign(new EventEmitter(), { stdout: new EventEmitter(), stderr: new EventEmitter(), kill: vi.fn() })
  event.kill.mockImplementation(() => { queueMicrotask(() => event.emit('close', null)); return true })
  return event
}

describe('local TUI over SSH', () => {
  it('forwards a private socket and launches the renderer locally, then cleans up', async () => {
    const setup = child(), tunnel = child(), local = child()
    let socket = ''
    const onSessionId = vi.fn()
    const launch = vi.fn((binary: string, args: string[], options?: { env?: NodeJS.ProcessEnv }) => {
      if (args.includes('-L')) {
        socket = args[args.indexOf('-L') + 1]!.split(':')[0]!
        writeFileSync(socket, '')
        return tunnel
      }
      if (binary === 'ssh') {
        queueMicrotask(() => { setup.stdout.emit('data', 'XERXES_REMOTE_READY {"socketPath":"/remote/rpc.sock","projectDir":"/remote/project"}\n'); setup.emit('close', 0) })
        return setup
      }
      writeFileSync(options!.env!.XERXES_TUI_ACTIVE_SESSION_FILE!, JSON.stringify({ session_id: 'remote-session-1' }))
      queueMicrotask(() => local.emit('close', 0))
      return local
    })
    const suspend = vi.fn(async (action: () => Promise<void>) => action())
    await connectRemoteMachine(machine, { spawnProcess: launch as unknown as typeof spawn, suspend, tuiEntry: '/local/entry.js', onSessionId })
    expect(onSessionId).toHaveBeenCalledWith('remote-session-1')
    expect(launch.mock.calls[0]?.[1]).not.toContain('-t')
    expect(launch.mock.calls[1]?.[1]).toContain('-N')
    expect(launch).toHaveBeenLastCalledWith(process.execPath, ['/local/entry.js'], expect.objectContaining({ stdio: 'inherit', env: expect.objectContaining({ XERXES_REMOTE_SOCKET: socket, XERXES_PROJECT_DIR: '/remote/project', XERXES_TUI_RESUME: '' }) }))
    expect(suspend).toHaveBeenCalledOnce()
    expect(tunnel.kill).toHaveBeenCalledWith('SIGTERM')
    expect(existsSync(socket)).toBe(false)
    expect(remoteMachineCommand(machine)).toContain('XERXES_REMOTE_READY')
  })
  it('rejects malformed machine responses before spawning', () => {
    for (const target of ['-oProxyCommand=bad', 'server;bad', 'server\nother']) expect(() => parseRemoteMachine({ ...machine, target })).toThrow('Invalid remote')
    expect(() => parseRemoteMachine({ ...machine, workspacePath: 'relative' })).toThrow()
  })
  it('restores the original renderer when the live tunnel disconnects', async () => {
    const setup = child(), tunnel = child(), local = child()
    let restored = false
    const launch = (binary: string, args: string[]) => {
      if (args.includes('-L')) { writeFileSync(args[args.indexOf('-L') + 1]!.split(':')[0]!, ''); return tunnel }
      if (binary === 'ssh') {
        queueMicrotask(() => { setup.stdout.emit('data', 'XERXES_REMOTE_READY {"socketPath":"/remote/rpc.sock","projectDir":"/remote"}\n'); setup.emit('close', 0) })
        return setup
      }
      queueMicrotask(() => tunnel.emit('close', 255))
      return local
    }
    await expect(connectRemoteMachine(machine, { spawnProcess: launch as unknown as typeof spawn, suspend: async action => { try { await action() } finally { restored = true } } })).rejects.toThrow('SSH tunnel closed')
    expect(local.kill).toHaveBeenCalledWith('SIGTERM')
    expect(restored).toBe(true)
  })
  it('reports setup errors without suspending the local renderer', async () => {
    const suspend = vi.fn()
    const launch = () => { const proc = child(); queueMicrotask(() => proc.emit('error', new Error('missing ssh'))); return proc }
    await expect(connectRemoteMachine(machine, { spawnProcess: launch as unknown as typeof spawn, suspend })).rejects.toThrow('missing ssh')
    expect(suspend).not.toHaveBeenCalled()
  })
  it('cancels before setup without starting any process', async () => {
    const controller = new AbortController(); controller.abort()
    const launch = vi.fn()
    await expect(connectRemoteMachine(machine, { signal: controller.signal, spawnProcess: launch as unknown as typeof spawn })).rejects.toThrow('cancelled')
    expect(launch).not.toHaveBeenCalled()
  })
})

it('retries transient drops and resumes the last remote session', async () => {
  const { reconnectRemoteMachine } = await import('../lib/machineHandoff.js')
  let calls = 0
  const connect = vi.fn(async (_machine, options) => {
    if (++calls === 1) { options?.onSessionId?.('session-123'); throw new Error('SSH tunnel closed (255). Connection reset') }
  }) as unknown as typeof connectRemoteMachine
  const wait = vi.fn(async () => {})
  const progress = vi.fn()
  await reconnectRemoteMachine(machine, { onProgress: progress }, connect, wait)
  expect(connect).toHaveBeenLastCalledWith(machine, expect.objectContaining({ resumeSessionId: 'session-123' }))
  expect(wait).toHaveBeenCalledWith(2000, undefined)
  expect(progress).toHaveBeenCalledWith(expect.stringContaining('Retry 1/3'))
})
it('bounds retries and does not retry authentication failures or cancellation', async () => {
  const { reconnectRemoteMachine } = await import('../lib/machineHandoff.js')
  const connect = vi.fn(async () => { throw new Error('SSH tunnel closed (255)') })
  const wait = vi.fn(async () => {})
  await expect(reconnectRemoteMachine(machine, {}, connect, wait)).rejects.toThrow('SSH tunnel closed')
  expect(connect).toHaveBeenCalledTimes(4)
  const denied = vi.fn(async () => { throw new Error('SSH tunnel closed (255). Permission denied') })
  await expect(reconnectRemoteMachine(machine, {}, denied, wait)).rejects.toThrow('Permission denied')
  expect(denied).toHaveBeenCalledOnce()
  const controller = new AbortController()
  const cancelWait = async () => { controller.abort(); controller.signal.throwIfAborted() }
  connect.mockClear()
  await expect(reconnectRemoteMachine(machine, { signal: controller.signal }, connect, cancelWait)).rejects.toThrow()
  expect(connect).toHaveBeenCalledOnce()
})

it('does not launch a local TUI if the tunnel drops while suspending the parent renderer', async () => {
  const setup = child(), tunnel = child(), local = child()
  const launch = vi.fn((binary: string, args: string[]) => {
    if (args.includes('-L')) { writeFileSync(args[args.indexOf('-L') + 1]!.split(':')[0]!, ''); return tunnel }
    if (binary === 'ssh') {
      queueMicrotask(() => { setup.stdout.emit('data', 'XERXES_REMOTE_READY {"socketPath":"/remote/rpc.sock","projectDir":"/remote"}\n'); setup.emit('close', 0) })
      return setup
    }
    queueMicrotask(() => local.emit('close', 0)); return local
  })
  await expect(connectRemoteMachine(machine, { spawnProcess: launch as unknown as typeof spawn, suspend: async action => { tunnel.emit('close', 255); await action() } })).rejects.toThrow('SSH tunnel closed')
  expect(launch).toHaveBeenCalledTimes(2)
})
it('cancels remote setup promptly even if SSH ignores termination', async () => {
  const setup = child()
  setup.kill.mockImplementation(() => true)
  const controller = new AbortController()
  const work = connectRemoteMachine(machine, { signal: controller.signal, spawnProcess: (() => { queueMicrotask(() => controller.abort()); return setup }) as unknown as typeof spawn }).then(() => 'success', () => 'cancelled')
  try {
    const result = await Promise.race([work, new Promise<string>(resolve => setTimeout(() => resolve('hung'), 100))])
    expect(result).toBe('cancelled')
  } finally { setup.emit('close', null); await work }
})
