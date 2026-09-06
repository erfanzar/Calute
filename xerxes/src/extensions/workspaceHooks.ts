// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { resolve } from 'node:path'
import { HookRunner } from './hooks.js'
import { loadShellHookConfigSync, registerShellHooks, resolveHookPoint, ShellHookExecutionError } from './shellHooks.js'

export interface WorkspaceHookInspection {
  readonly recent: readonly WorkspaceHookResult[]
  readonly workspace: string
  readonly workspaceTrusted: boolean
  readonly loadedAt: string
  readonly sources: readonly string[]
  readonly errors: readonly string[]
  readonly hooks: readonly { event: string; command: string; matcher: string | null; timeoutMs: number; blocking: boolean }[]
}
export interface WorkspaceHookResult {
  readonly failureKind?: 'timeout' | 'exit' | 'execution'
  readonly exitCode?: number
  readonly hookIndex: number
  readonly event: string
  readonly status: 'completed' | 'denied' | 'failed'
  readonly at: string
  readonly durationMs: number
}
const inspections = new WeakMap<HookRunner, WorkspaceHookInspection>()
export function inspectWorkspaceHooks(runner: HookRunner): WorkspaceHookInspection | undefined { return inspections.get(runner) }

export function workspaceHookFailures(inspection: WorkspaceHookInspection, event?: string) {
  const point = event === undefined ? undefined : resolveHookPoint(event, 'Hook failures')
  const results = inspection.recent.filter(result => result.status !== 'completed' && (point === undefined || result.event === point)).slice().reverse()
  return { event: point ?? null, retainedExecutions: inspection.recent.length, failed: results.filter(result => result.status === 'failed').length,
    denied: results.filter(result => result.status === 'denied').length, results }
}

/** Selection preview only: never executes shell commands or predicts their verdicts. */
export function previewWorkspaceHooks(inspection: WorkspaceHookInspection, event: string, toolName = '') {
  const point = resolveHookPoint(event, 'Hook preview')
  if (toolName.length > 512 || /[\r\n\0]/.test(toolName)) throw new Error('Preview tool name must be at most 512 characters without line breaks')
  const hooks = inspection.hooks.filter(hook => hook.event === point).map((hook, index) => ({
    ...hook,
    index: index + 1,
    matches: hook.matcher === null || new RegExp(hook.matcher).test(toolName),
  }))
  return { event: point, toolName, executed: false as const, hooks, matched: hooks.filter(hook => hook.matches).length }
}

/** Each workspace gets its own trusted configuration and shell cwd. Cache is bounded. */
export function workspaceShellHooks(options: {
  readonly home: string
  readonly allowWorkspace: boolean
  readonly reportError: (message: string) => void
}): (cwd: string) => HookRunner {
  const runners = new Map<string, HookRunner>()
  return cwd => {
    const root = resolve(cwd)
    const existing = runners.get(root)
    if (existing) {
      runners.delete(root)
      runners.set(root, existing)
      return existing
    }
    const loaded = loadShellHookConfigSync({ home: options.home, allowWorkspace: options.allowWorkspace, workspaceRoot: root })
    for (const error of loaded.errors) options.reportError(error)
    const runner = new HookRunner()
    const recent: WorkspaceHookResult[] = []
    const registered = new Map<string, number>()
    registerShellHooks({ register(point, callback) {
      const index = registered.get(point) ?? 0
      registered.set(point, index + 1)
      const matcher = loaded.hooks[point]?.[index]?.matcher
      runner.register(point, async payload => {
        if (matcher !== undefined && !new RegExp(matcher).test(typeof payload.toolName === 'string' ? payload.toolName : '')) return callback(payload)
        const start = performance.now()
        let status: WorkspaceHookResult['status'] = 'failed'
        let failureKind: WorkspaceHookResult['failureKind']
        let exitCode: number | undefined
        try {
          const result = await callback(payload)
          status = point === 'tool_permission_check' && result && typeof result === 'object' && 'allow' in result && result.allow === false ? 'denied' : 'completed'
          return result
        } catch (error) {
          failureKind = error instanceof ShellHookExecutionError ? error.kind : 'execution'
          exitCode = error instanceof ShellHookExecutionError ? error.exitCode : undefined
          throw error
        } finally {
          recent.push(Object.freeze({ event: point, hookIndex: index + 1, status, at: new Date().toISOString(), durationMs: Math.max(0, Math.round(performance.now() - start)), ...(failureKind ? { failureKind } : {}), ...(exitCode === undefined ? {} : { exitCode }) }))
          if (recent.length > 100) recent.shift()
        }
      })
    } }, loaded.hooks, { cwd: root })
    inspections.set(runner, Object.freeze({ workspace: root, workspaceTrusted: options.allowWorkspace,
      loadedAt: new Date().toISOString(), sources: loaded.sources, errors: loaded.errors,
      get recent() { return Object.freeze([...recent]) },
      hooks: Object.freeze(Object.entries(loaded.hooks).flatMap(([event, specs]) => specs.map(spec => Object.freeze({
        event, command: spec.command, matcher: spec.matcher ?? null, timeoutMs: spec.timeout_ms ?? 60_000,
        blocking: event === 'tool_permission_check',
      })))),
    }))
    runners.set(root, runner)
    if (runners.size > 32) runners.delete(runners.keys().next().value!)
    return runner
  }
}
