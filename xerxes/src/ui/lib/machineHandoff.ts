// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
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
    resumeSessionId?: string
    onSessionId?: (id: string) => void
    onProgress?: (message: string) => void
  } = {}
): Promise<void> {
  const command = remoteMachineCommand(machine)
  if (options.signal?.aborted) throw new Error('Remote connection cancelled')
  const launch = options.spawnProcess ?? spawn
  const ssh = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3']
  const directory = await mkdtemp(join(tmpdir(), 'xr-'))
  const socket = join(directory, 'rpc.sock')
  const sessionFile = join(directory, "active-session")
  try {
    options.onProgress?.("Checking remote runtime…")
    const output = await new Promise<string>((resolve, reject) => {
      const child = launch('ssh', [...ssh, '-T', '--', machine.target, command], { stdio: ['ignore', 'pipe', 'pipe'] })
      let stdout = '', stderr = ''
      let killTimer: ReturnType<typeof setTimeout> | undefined
      const stop = (error: Error) => {
        cleanup()
        child.kill('SIGTERM')
        killTimer = setTimeout(() => child.kill('SIGKILL'), 1000)
        killTimer.unref?.()
        reject(error)
      }
      const abort = () => stop(new Error('Remote connection cancelled'))
      const timer = setTimeout(() => stop(new Error('Remote setup timed out; inspect ~/.xerxes/remote-runtime/setup.log.')), 300_000)
      const cleanup = () => { clearTimeout(timer); options.signal?.removeEventListener('abort', abort) }
      options.signal?.addEventListener('abort', abort, { once: true })
      if (options.signal?.aborted) abort()
      child.stdout?.on('data', chunk => { stdout = (stdout + String(chunk)).slice(-65536) })
      child.stderr?.on('data', chunk => { stderr = (stderr + String(chunk)).slice(-8192) })
      child.once('error', error => { cleanup(); clearTimeout(killTimer); reject(error) })
      child.once('close', code => {
        cleanup()
        clearTimeout(killTimer)
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
    options.onProgress?.("Opening SSH tunnel…")
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
      options.onProgress?.("Opening remote workspace…")
      await (options.suspend ?? withTerminalSuspended)(() => new Promise<void>((resolve, reject) => {
        // Suspending the parent can yield; the tunnel may die in that gap.
        if (failure) { reject(failure); return }
        if (options.signal?.aborted) { reject(new Error('Remote connection cancelled')); return }
        local = launch(process.execPath, [options.tuiEntry ?? process.argv[1]!], {
          stdio: 'inherit',
          env: { ...process.env, XERXES_REMOTE_SOCKET: socket, XERXES_PROJECT_DIR: remote.projectDir as string,
            XERXES_REMOTE_LABEL: machine.alias,
            XERXES_CWD: remote.projectDir as string, XERXES_TUI_RESUME: options.resumeSessionId ?? '', XERXES_TUI_QUERY: '', XERXES_TUI_ACTIVE_SESSION_FILE: sessionFile }
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
    try {
      const saved: unknown = JSON.parse(await readFile(sessionFile, 'utf8'))
      if (saved && typeof saved === 'object' && 'session_id' in saved && typeof saved.session_id === 'string' && /^[a-zA-Z0-9_-]{1,128}$/.test(saved.session_id)) options.onSessionId?.(saved.session_id)
    } catch (error) {
      if (!(error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT')) options.onProgress?.('Could not recover the session ID; use the remote session picker.')
    }
    await rm(directory, { recursive: true, force: true })
  }
}

/** Retry transport interruptions, never authentication or host-key failures. */
export function retryableRemoteFailure(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)
  if (/permission denied|authentication|host key|host identification|cancelled|canceled/i.test(message)) return false
  return /SSH tunnel closed|connection (?:reset|refused|closed)|timed out|network is unreachable|no route to host|broken pipe/i.test(message)
}

export async function reconnectRemoteMachine(
  machine: RemoteMachine,
  options: NonNullable<Parameters<typeof connectRemoteMachine>[1]> = {},
  connect = connectRemoteMachine,
  wait: (ms: number, signal?: AbortSignal) => Promise<void> = async (ms, signal) => { await delay(ms, undefined, { signal }) },
): Promise<void> {
  let resumeSessionId = options.resumeSessionId
  for (let attempt = 0; ; attempt++) {
    options.signal?.throwIfAborted()
    try {
      await connect(machine, { ...options, resumeSessionId, onSessionId: id => { resumeSessionId = id; options.onSessionId?.(id) } })
      return
    } catch (error) {
      if (options.signal?.aborted || attempt >= 3 || !retryableRemoteFailure(error)) throw error
      const seconds = [2, 5, 10][attempt]!
      options.onProgress?.(`Connection interrupted. Retry ${attempt + 1}/3 in ${seconds}s · Esc cancel`)
      await wait(seconds * 1000, options.signal)
    }
  }
}
