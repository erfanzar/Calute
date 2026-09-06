// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, mkdir, rm, realpath } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { workspaceShellHooks, workspaceHookFailures } from '../src/extensions/workspaceHooks.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { HookRunner } from '../src/extensions/hooks.js'

test.skipIf(process.platform === 'win32')('workspace shell hooks select separate configs and working directories during concurrent turns', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-workspace-hooks-'))
  try {
    const home = join(root, 'home'), a = join(root, 'a'), b = join(root, 'b')
    await Promise.all([home, a, b].map(path => mkdir(path)))
    await Bun.write(join(home, 'config.json'), JSON.stringify({ hooks: { on_turn_start: [{ command: 'pwd > user-hook.txt' }] } }))
    await Bun.write(join(a, 'xerxes.json'), JSON.stringify({ hooks: { on_turn_start: [{ command: 'printf A > workspace-hook.txt' }] } }))
    await Bun.write(join(b, 'xerxes.json'), JSON.stringify({ hooks: { on_turn_start: [{ command: 'printf B > workspace-hook.txt' }] } }))
    const errors: string[] = []
    const forWorkspace = workspaceShellHooks({ home, allowWorkspace: true, reportError: error => errors.push(error) })
    await Promise.all([a, b].map(cwd => forWorkspace(cwd).run('on_turn_start', {})))
    expect((await Bun.file(join(a, 'user-hook.txt')).text()).trim()).toBe(await realpath(a))
    expect((await Bun.file(join(b, 'user-hook.txt')).text()).trim()).toBe(await realpath(b))
    expect(await Bun.file(join(a, 'workspace-hook.txt')).text()).toBe('A')
    expect(await Bun.file(join(b, 'workspace-hook.txt')).text()).toBe('B')
    expect(errors).toEqual([])
    expect(forWorkspace(a)).toBe(forWorkspace(a))
    await rm(join(b, 'workspace-hook.txt'))
    const untrusted = workspaceShellHooks({ home, allowWorkspace: false, reportError: error => errors.push(error) })
    await untrusted(b).run('on_turn_start', {})
    expect(await Bun.file(join(b, 'workspace-hook.txt')).exists()).toBe(false)
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('session lifecycle hooks resolve using the opened and evicted session project', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-lifecycle-hooks-'))
  const seen: string[] = []
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: root, sessionDirectory: join(root, 'sessions'),
    hookRunnerForSession: session => {
      const hooks = new HookRunner()
      hooks.register('on_session_start', () => { seen.push('start:' + session.cwd) })
      hooks.register('on_session_end', () => { seen.push('end:' + session.cwd) })
      return hooks
    } })
  try {
    const other = join(root, 'other'); await mkdir(other)
    await runtime.openSession('other', undefined, { cwd: other })
    runtime.evictSession('other')
    expect(seen).toEqual(['start:' + other, 'end:' + other])
  } finally { await runtime.shutdown(); await rm(root, { recursive: true, force: true }) }
})

test('inspection reports cached configuration and trust without executing hooks', async () => {
  const { inspectWorkspaceHooks } = await import('../src/extensions/workspaceHooks.js')
  const root = await mkdtemp(join(tmpdir(), 'xerxes-hook-inspect-'))
  try {
    const home = join(root, 'home'); await mkdir(home)
    const config = join(home, 'config.json')
    await Bun.write(config, JSON.stringify({ hooks: { PreToolUse: [{ command: 'touch must-not-exist', matcher: 'ReadFile', timeout_ms: 500 }] } }))
    await Bun.write(join(root, 'xerxes.json'), '{invalid')
    const factory = workspaceShellHooks({ home, allowWorkspace: false, reportError: () => {} })
    const runner = factory(root)
    const inspection = inspectWorkspaceHooks(runner)!
    expect(inspection.sources).toEqual([config])
    expect(inspection.workspaceTrusted).toBe(false)
    expect(inspection.errors).toEqual([])
    expect(inspection.hooks).toMatchObject([{ event: 'tool_permission_check', blocking: true, matcher: 'ReadFile', timeoutMs: 500 }])
    expect(await Bun.file(join(root, 'must-not-exist')).exists()).toBe(false)
    await Bun.write(config, '{}')
    expect(inspectWorkspaceHooks(factory(root))).toBe(inspection)
    const trusted = workspaceShellHooks({ home, allowWorkspace: true, reportError: () => {} })
    expect(inspectWorkspaceHooks(trusted(root))?.errors[0]).toContain(join(root, 'xerxes.json'))
  } finally { await rm(root, { recursive: true, force: true }) }
})

test.skipIf(process.platform === 'win32')('hook inspection records bounded outcomes without inputs or output and excludes unmatched hooks', async () => {
  const { inspectWorkspaceHooks } = await import('../src/extensions/workspaceHooks.js')
  const root = await mkdtemp(join(tmpdir(), 'xerxes-hook-results-'))
  try {
    const home = join(root, 'home'); await mkdir(home)
    await Bun.write(join(home, 'config.json'), JSON.stringify({ hooks: {
      on_turn_start: [{ command: 'printf private-output' }],
      tool_permission_check: [{ command: 'exit 2', matcher: '^ReadFile$' }],
      on_turn_end: [{ command: 'exit 1' }],
      on_session_start: [{ command: 'sleep 1', timeout_ms: 20 }],
    } }))
    const runner = workspaceShellHooks({ home, allowWorkspace: false, reportError: () => {} })(root)
    await runner.run('tool_permission_check', { toolName: 'WriteFile' })
    expect(inspectWorkspaceHooks(runner)?.recent).toHaveLength(0)
    await runner.run('on_turn_start', { secret: 'private-input' })
    await runner.run('tool_permission_check', { toolName: 'ReadFile' })
    await runner.run('on_turn_end', {})
    const snapshot = inspectWorkspaceHooks(runner)!.recent
    expect(snapshot.map(item => item.status)).toEqual(['completed', 'denied', 'failed'])
    expect(snapshot[2]).toMatchObject({ failureKind: 'exit', exitCode: 1 })
    expect(JSON.stringify(snapshot)).not.toContain('private-')
    await runner.run('on_session_start', {})
    expect(inspectWorkspaceHooks(runner)?.recent.at(-1)).toMatchObject({ status: 'failed', failureKind: 'timeout' })
    expect(inspectWorkspaceHooks(runner)?.recent.at(-1)?.exitCode).toBeUndefined()
    const failures = workspaceHookFailures(inspectWorkspaceHooks(runner)!)
    expect(failures).toMatchObject({ failed: 2, denied: 1, retainedExecutions: 4 })
    expect(failures.results.map(row => row.status)).toEqual(['failed', 'failed', 'denied'])
    expect(workspaceHookFailures(inspectWorkspaceHooks(runner)!, 'PreToolUse')).toMatchObject({ failed: 0, denied: 1 })
    expect(() => workspaceHookFailures(inspectWorkspaceHooks(runner)!, 'typo')).toThrow('Hook failures')
    expect(JSON.stringify(failures)).not.toContain('private-')
    await Promise.all(Array.from({ length: 101 }, () => runner.run('on_turn_start', {})))
    expect(inspectWorkspaceHooks(runner)?.recent).toHaveLength(100)
    expect(snapshot).toHaveLength(3)
    expect(workspaceHookFailures(inspectWorkspaceHooks(runner)!).results).toEqual([])
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('hook preview resolves aliases, preserves execution order, and never executes commands', async () => {
  const { inspectWorkspaceHooks, previewWorkspaceHooks } = await import('../src/extensions/workspaceHooks.js')
  const root = await mkdtemp(join(tmpdir(), 'xerxes-hook-preview-'))
  try {
    const home = join(root, 'home'); await mkdir(home)
    await Bun.write(join(home, 'config.json'), JSON.stringify({ hooks: { PreToolUse: [
      { command: 'touch should-not-exist', matcher: '^ReadFile$', timeout_ms: 200 },
      { command: 'exit 2', matcher: '^WriteFile$' },
      { command: 'exit 0' },
    ] } }))
    await Bun.write(join(root, 'xerxes.json'), JSON.stringify({ hooks: { PreToolUse: [{ command: 'touch untrusted' }] } }))
    const runner = workspaceShellHooks({ home, allowWorkspace: false, reportError: () => {} })(root)
    const inspection = inspectWorkspaceHooks(runner)!
    const preview = previewWorkspaceHooks(inspection, 'PreToolUse', 'ReadFile')
    expect(preview).toMatchObject({ event: 'tool_permission_check', executed: false, matched: 2, hooks: [{ index: 1, matches: true, blocking: true, timeoutMs: 200 }, { index: 2, matches: false }, { index: 3, matches: true }] })
    expect(previewWorkspaceHooks(inspection, 'on_turn_end').hooks).toEqual([])
    expect(() => previewWorkspaceHooks(inspection, 'typo')).toThrow('Hook preview')
    expect(inspection.recent).toEqual([])
    expect(await Bun.file(join(root, 'should-not-exist')).exists()).toBe(false)
    expect(await Bun.file(join(root, 'untrusted')).exists()).toBe(false)
  } finally { await rm(root, { recursive: true, force: true }) }
})
