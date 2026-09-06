// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { join } from 'node:path'
import { readFile } from 'node:fs/promises'
import { xerxesHome } from '../daemon/paths.js'
import { executeCommand } from '../tools/processTools.js'
import { WorkspacePathResolver } from '../tools/pathSafety.js'

export interface WorkspaceSetup { command: string[]; timeoutMs: number }
export interface WorkspaceSetupResult { status: 'running' | 'completed' | 'failed'; command: string[]; startedAt: string; finishedAt?: string; stdout?: string; stderr?: string; error?: string; truncated?: boolean }
export async function loadWorkspaceSetup(repository: string, path = join(xerxesHome(), 'workspace-setup.json')): Promise<WorkspaceSetup | undefined> {
  const text = await readFile(path, 'utf8').catch(error => { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return null; throw error })
  if (text === null) return undefined
  if (Buffer.byteLength(text) > 65536) throw new Error('Workspace setup configuration exceeds 64 KiB: ' + path)
  const document: unknown = JSON.parse(text)
  if (!document || typeof document !== 'object' || Array.isArray(document)) throw new Error('Workspace setup configuration must be an object: ' + path)
  const workspaces = (document as Record<string, unknown>).workspaces
  if (!workspaces || typeof workspaces !== 'object' || Array.isArray(workspaces)) throw new Error('Workspace setup requires a workspaces map: ' + path)
  const value = (workspaces as Record<string, unknown>)[repository]
  if (value === undefined) return undefined
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid workspace setup for ' + repository)
  const row = value as Record<string, unknown>
  if (!Array.isArray(row.command) || !row.command.length || row.command.length > 64 || row.command.some(arg => typeof arg !== 'string' || arg.includes('\0') || arg.length > 8192)) throw new Error('Workspace setup command must be a bounded argv array')
  if (!row.command[0] || /[\s;&|`$<>]/.test(row.command[0])) throw new Error('Workspace setup requires one executable; pass arguments separately')
  const timeoutMs = row.timeout_ms ?? 120000
  if (typeof timeoutMs !== 'number' || !Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 120000) throw new Error('Workspace setup timeout_ms must be between 1 and 120000')
  return { command: [...row.command], timeoutMs }
}
export async function runWorkspaceSetup(setup: WorkspaceSetup, cwd: string, signal: AbortSignal | undefined, save: (result: WorkspaceSetupResult) => Promise<void>) {
  const initial: WorkspaceSetupResult = { status: 'running', command: setup.command, startedAt: new Date().toISOString() }
  await save(initial)
  let latest = initial
  try {
    const result = await executeCommand({ cmd: setup.command[0]!, args: setup.command.slice(1), timeout_ms: setup.timeoutMs, max_output_chars: 4000 }, new WorkspacePathResolver(cwd), signal)
    if (!('exitCode' in result)) throw new Error('Workspace setup unexpectedly backgrounded')
    const saved: WorkspaceSetupResult = { ...initial, status: result.exitCode === 0 && !result.timedOut ? 'completed' : 'failed', finishedAt: new Date().toISOString(), stdout: result.stdout, stderr: result.stderr, truncated: result.truncated }
    latest = saved
    if (saved.status === 'failed') saved.error = result.timedOut ? 'Workspace setup timed out' : 'Workspace setup exited with code ' + result.exitCode
    await save(saved)
    if (saved.error) throw new Error(saved.error)
  } catch (error) {
    // Cancellation/execution failures also remain visible after daemon restart.
    await save({ ...latest, status: 'failed', finishedAt: new Date().toISOString(), error: String(error) })
    throw error
  }
}
