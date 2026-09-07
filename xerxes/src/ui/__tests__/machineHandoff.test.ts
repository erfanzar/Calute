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
    const launch = vi.fn((binary: string, args: string[]) => {
      if (args.includes('-L')) {
        socket = args[args.indexOf('-L') + 1]!.split(':')[0]!
        writeFileSync(socket, '')
        return tunnel
      }
      if (binary === 'ssh') {
        queueMicrotask(() => { setup.stdout.emit('data', 'XERXES_REMOTE_READY {"socketPath":"/remote/rpc.sock","projectDir":"/remote/project"}\n'); setup.emit('close', 0) })
        return setup
      }
      queueMicrotask(() => local.emit('close', 0))
      return local
    })
    const suspend = vi.fn(async (action: () => Promise<void>) => action())
    await connectRemoteMachine(machine, { spawnProcess: launch as unknown as typeof spawn, suspend, tuiEntry: '/local/entry.js' })
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
