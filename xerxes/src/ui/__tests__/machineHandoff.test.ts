// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { EventEmitter } from 'node:events'
import type { spawn } from 'node:child_process'
import { describe, expect, it, vi } from 'vitest'
import { connectRemoteMachine, parseRemoteMachine, remoteMachineCommand } from '../lib/machineHandoff.js'

const machine = { alias: 'gpu', target: 'me@server', workspacePath: "/work/it's a project; $(touch nope)" }

describe('machine SSH handoff', () => {
  it('quotes remote paths and never uses a local shell', async () => {
    const child = new EventEmitter()
    const launch = vi.fn(() => { queueMicrotask(() => child.emit('close', 0, null)); return child })
    const suspend = vi.fn(async (run: () => Promise<void>) => { await run() })
    await connectRemoteMachine(machine, { spawnProcess: launch as unknown as typeof spawn, suspend })
    expect(launch).toHaveBeenCalledWith('ssh', ['-t', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', '--', machine.target, remoteMachineCommand(machine)], { stdio: 'inherit' })
    expect(remoteMachineCommand(machine)).toContain('exec "${SHELL:-/bin/sh}" -lc')
    expect(suspend).toHaveBeenCalledOnce()
  })
  it('rejects malformed machine responses before spawning', async () => {
    for (const target of ['-oProxyCommand=bad', 'server;bad', 'server\nother']) {
      expect(() => parseRemoteMachine({ ...machine, target })).toThrow('Invalid remote')
    }
    expect(() => parseRemoteMachine({ ...machine, workspacePath: 'relative' })).toThrow()
  })
  it('restores suspended terminal on SSH errors and cancellation', async () => {
    let restored = false
    const child = Object.assign(new EventEmitter(), { kill: vi.fn(() => { queueMicrotask(() => child.emit('close', null, 'SIGTERM')); return true }) })
    const controller = new AbortController()
    const run = connectRemoteMachine(machine, {
      signal: controller.signal,
      spawnProcess: (() => child) as unknown as typeof spawn,
      suspend: async action => { try { await action() } finally { restored = true } }
    })
    controller.abort()
    await expect(run).rejects.toThrow('cancelled')
    expect(child.kill).toHaveBeenCalledWith('SIGTERM')
    expect(restored).toBe(true)
    const broken = new EventEmitter()
    await expect(connectRemoteMachine(machine, {
      spawnProcess: (() => { queueMicrotask(() => broken.emit('error', new Error('missing ssh'))); return broken }) as unknown as typeof spawn,
      suspend: async action => action()
    })).rejects.toThrow('missing ssh')
  })
})
