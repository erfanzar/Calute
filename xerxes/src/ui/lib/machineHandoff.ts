// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { mkdtemp, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { setTimeout as delay } from 'node:timers/promises'

import { withTerminalSuspended } from './terminalRuntime.opentui.js'
import { remoteBootstrapScript } from './remoteBootstrap.js'

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
  return `exec sh -c ${quoteShell(remoteBootstrapScript(validated.workspacePath, 'daemon'))}`
}

/** Start the remote daemon, forward its private socket, and run a LOCAL TUI. */
export async function connectRemoteMachine(
  machine: RemoteMachine,
  options: {
    signal?: AbortSignal
    spawnProcess?: typeof spawn
    suspend?: (run: () => Promise<void>) => Promise<void>
    tuiEntry?: string
  } = {}
): Promise<void> {
  const command = remoteMachineCommand(machine)
  if (options.signal?.aborted) throw new Error('Remote connection cancelled')
  const launch = options.spawnProcess ?? spawn
  const ssh = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3']
  const directory = await mkdtemp(join(tmpdir(), 'xr-'))
  const socket = join(directory, 'rpc.sock')
  try {
    const output = await new Promise<string>((resolve, reject) => {
      const child = launch('ssh', [...ssh, '-T', '--', machine.target, command], { stdio: ['ignore', 'pipe', 'pipe'] })
      let stdout = '', stderr = ''
      const abort = () => child.kill('SIGTERM')
      const timer = setTimeout(() => { child.kill('SIGTERM'); reject(new Error('Remote setup timed out; inspect ~/.xerxes/remote-runtime/setup.log.')) }, 300_000)
      options.signal?.addEventListener('abort', abort, { once: true })
      if (options.signal?.aborted) abort()
      child.stdout?.on('data', chunk => { stdout = (stdout + String(chunk)).slice(-65536) })
      child.stderr?.on('data', chunk => { stderr = (stderr + String(chunk)).slice(-8192) })
      const cleanup = () => { clearTimeout(timer); options.signal?.removeEventListener('abort', abort) }
      child.once('error', error => { cleanup(); reject(error) })
      child.once('close', code => {
        cleanup()
        if (options.signal?.aborted) reject(new Error('Remote connection cancelled'))
        else if (code !== 0) reject(new Error(`Remote setup failed (${code}): ${stderr || stdout}`))
        else resolve(stdout)
      })
    })
    const ready = output.split('\n').find(line => line.startsWith('XERXES_REMOTE_READY '))
    if (!ready) throw new Error('Remote setup did not return a daemon address.')
    const remote: unknown = JSON.parse(ready.slice('XERXES_REMOTE_READY '.length))
    if (!remote || typeof remote !== 'object' || !('socketPath' in remote) || !('projectDir' in remote) ||
      typeof remote.socketPath !== 'string' || !remote.socketPath.startsWith('/') || /[:\r\n\0]/u.test(remote.socketPath) ||
      typeof remote.projectDir !== 'string' || !remote.projectDir.startsWith('/') || /[\r\n\0]/u.test(remote.projectDir)) throw new Error('Invalid remote daemon address.')
    // Own this connection: multiplexed -N can exit successfully after handing
    // the forwarding to an unrelated persistent master, defeating cleanup.
    const tunnel = launch('ssh', [...ssh, '-S', 'none', '-o', 'ControlMaster=no', '-o', 'ForkAfterAuthentication=no', '-N', '-T', '-o', 'ExitOnForwardFailure=yes', '-L', `${socket}:${remote.socketPath}`, '--', machine.target], { stdio: ['ignore', 'ignore', 'pipe'] })
    let failure: Error | undefined
    let local: ReturnType<typeof spawn> | undefined
    let closing = false
    let stderr = ''
    tunnel.stderr?.on('data', chunk => { stderr = (stderr + String(chunk)).slice(-8192) })
    const failed = (error: Error) => { if (!closing) { failure = error; local?.kill('SIGTERM') } }
    tunnel.once('error', failed)
    tunnel.once('close', code => failed(new Error(`SSH tunnel closed (${code}). ${stderr}`)))
    const abort = () => { failure = new Error('Remote connection cancelled'); tunnel.kill('SIGTERM'); local?.kill('SIGTERM') }
    options.signal?.addEventListener('abort', abort, { once: true })
    try {
      const deadline = Date.now() + 15000
      while (!existsSync(socket)) {
        if (options.signal?.aborted) abort()
        if (failure) throw failure
        if (Date.now() >= deadline) throw new Error('SSH tunnel startup timed out.')
        await delay(25)
      }
      if (failure) throw failure
      await (options.suspend ?? withTerminalSuspended)(() => new Promise<void>((resolve, reject) => {
        local = launch(process.execPath, [options.tuiEntry ?? process.argv[1]!], {
          stdio: 'inherit',
          env: { ...process.env, XERXES_REMOTE_SOCKET: socket, XERXES_PROJECT_DIR: remote.projectDir as string,
            XERXES_REMOTE_LABEL: machine.alias,
            XERXES_CWD: remote.projectDir as string, XERXES_TUI_RESUME: '', XERXES_TUI_QUERY: '', XERXES_TUI_ACTIVE_SESSION_FILE: '' }
        })
        local.once('error', reject)
        local.once('close', code => failure ? reject(failure) : code === 0 ? resolve() : reject(new Error(`Local remote-workspace TUI exited (${code}).`)))
        if (options.signal?.aborted) abort()
      }))
    } finally {
      closing = true
      options.signal?.removeEventListener('abort', abort)
      tunnel.kill('SIGTERM')
    }
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
}
