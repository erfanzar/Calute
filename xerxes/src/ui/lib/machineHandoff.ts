// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { spawn } from 'node:child_process'

import { withTerminalSuspended } from './terminalRuntime.opentui.js'

export interface RemoteMachine {
  alias: string
  target: string
  workspacePath: string
}

export function parseRemoteMachine(value: unknown): RemoteMachine {
  if (!value || typeof value !== 'object') throw new Error('Invalid remote machine response')
  const row = value as Record<string, unknown>
  if (typeof row.alias !== 'string' || !/^[a-zA-Z0-9][a-zA-Z0-9_.-]*$/.test(row.alias) ||
      typeof row.target !== 'string' || !/^[a-zA-Z0-9][a-zA-Z0-9_.@:-]*$/.test(row.target) ||
      typeof row.workspacePath !== 'string' || !row.workspacePath.startsWith('/') || /[\0\r\n]/.test(row.workspacePath)) {
    throw new Error('Invalid remote machine response')
  }
  return { alias: row.alias, target: row.target, workspacePath: row.workspacePath }
}

const quoteShell = (value: string): string => `'${value.replaceAll("'", "'\\''")}'`

export function remoteMachineCommand(machine: RemoteMachine): string {
  const validated = parseRemoteMachine(machine)
  const script = `cd ${quoteShell(validated.workspacePath)} && exec xerxes`
  return `exec "\${SHELL:-/bin/sh}" -lc ${quoteShell(script)}`
}

/** SSH owns the terminal only while the local renderer is suspended. */
export async function connectRemoteMachine(
  machine: RemoteMachine,
  options: {
    signal?: AbortSignal
    spawnProcess?: typeof spawn
    suspend?: (run: () => Promise<void>) => Promise<void>
  } = {}
): Promise<void> {
  const command = remoteMachineCommand(machine)
  if (options.signal?.aborted) throw new Error('Remote connection cancelled')
  await (options.suspend ?? withTerminalSuspended)(() => new Promise<void>((resolve, reject) => {
    const child = (options.spawnProcess ?? spawn)('ssh', ['-t', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', '--', machine.target, command], { stdio: 'inherit' })
    const abort = () => child.kill('SIGTERM')
    const cleanup = () => options.signal?.removeEventListener('abort', abort)
    options.signal?.addEventListener('abort', abort, { once: true })
    if (options.signal?.aborted) abort()
    child.once('error', error => { cleanup(); reject(error) })
    child.once('close', (code, signal) => {
      cleanup()
      if (options.signal?.aborted) reject(new Error('Remote connection cancelled'))
      else if (code === 0) resolve()
      else reject(new Error(`SSH exited ${signal ? `with signal ${signal}` : `with code ${code ?? 'unknown'}`}. Check the host and that Xerxes is installed there.`))
    })
  }))
}
