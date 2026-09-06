// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { BackgroundCommandManager } from '../src/tools/backgroundCommands.js'
import { registerProcessTools } from '../src/tools/processTools.js'
import { WorkspacePathResolver } from '../src/tools/pathSafety.js'
import { RunHistory } from '../src/runtime/runHistory.js'
import { ReactionMailbox } from '../src/runtime/reactionMailbox.js'
import { TerminalRegistry } from '../src/runtime/terminalRegistry.js'
import { TerminalMonitors } from '../src/runtime/terminalMonitors.js'
import { ProcessRegistry } from '../src/runtime/processRegistry.js'
import type { JsonObject } from '../src/types/toolCalls.js'

const owner = { sessionId: 'owner', metadata: { goal_turn_human: true } }
function exec(registry: ToolRegistry, args: JsonObject) {
  return registry.execute({ id: crypto.randomUUID(), type: 'function', function: { name: 'exec_command', arguments: args } }, owner)
}

test.each([true, false])('background completion offers one durable owner reaction (explicit background: %s)', async immediate => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-completion-'))
  const history = new RunHistory(':memory:')
  const mailbox = new ReactionMailbox(':memory:')
  const terminals = new TerminalRegistry({ runHistory: history })
  const background = new BackgroundCommandManager(new ProcessRegistry(), terminals)
  let resolve!: () => void
  const delivered = new Promise<void>(done => { resolve = done })
  const monitors = new TerminalMonitors(terminals, history, (watch, event) => {
    mailbox.offer(watch.owner, watch.id, event.sequence)
    resolve()
  }, undefined, mailbox)
  const registry = new ToolRegistry()
  registerProcessTools(registry, new WorkspacePathResolver(root), background, terminals,
    (session, id) => monitors.start(session, { terminalId: id, trigger: 'completion', durationMs: 5000, maxEvents: 1, reaction: { maxReactions: 1, maxDurationMs: 1000 } }))
  let timer: ReturnType<typeof setTimeout> | undefined
  try {
    const result = JSON.parse(await exec(registry, { cmd: process.execPath, args: ['-e', 'await Bun.sleep(100); console.log("finished evidence")'], run_in_background: immediate, timeout_ms: 1, notify_on_completion: true }))
    expect(result.procId).toBeString()
    expect(result.completion_watch.id).toBeString()
    await Promise.race([delivered, new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('completion not delivered')), 4000) })])
    const claim = mailbox.claim('owner')
    expect(claim?.runId).toBe(result.completion_watch.id)
    expect(mailbox.claim('other-owner')).toBeUndefined()
    expect(history.events('owner', claim!.runId).events[0]?.text).toContain('finished evidence')
    mailbox.settle(claim!, 'completed')
    expect(mailbox.claim('owner')).toBeUndefined()
  } finally {
    if (timer) clearTimeout(timer)
    await background.disposeAll()
    monitors.close(); mailbox.close(); history.close()
    await rm(root, { recursive: true, force: true })
  }
})

test('unsupported completion fails before spawning, and watch admission failure retains the process handle', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-completion-error-'))
  const background = new BackgroundCommandManager()
  const registry = new ToolRegistry()
  const paths = new WorkspacePathResolver(root)
  try {
    registerProcessTools(registry, paths, background)
    const args = { cmd: process.execPath, args: ['-e', 'await Bun.sleep(1000)'], run_in_background: true, notify_on_completion: true }
    await expect(exec(registry, args)).rejects.toThrow('not enabled by this host')
    expect(background.listForOwner('owner')).toHaveLength(0)
    const supported = new ToolRegistry()
    registerProcessTools(supported, paths, background, undefined, () => { throw new Error('Session monitor limit reached') })
    await expect(supported.execute({ id: 'reaction', type: 'function', function: { name: 'exec_command', arguments: args } }, { sessionId: 'owner', metadata: { goal_turn_human: false } })).rejects.toThrow('direct user turn')
    expect(background.listForOwner('owner')).toHaveLength(0)
    const result = JSON.parse(await exec(supported, args))
    expect(result.completion_watch_error).toContain('monitor limit')
    expect(background.listForOwner('owner')[0]?.procId).toBe(result.procId)
  } finally { await background.disposeAll(); await rm(root, { recursive: true, force: true }) }
})

test('foreground completion and opted-out background commands do not schedule a follow-up', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-completion-optout-'))
  const background = new BackgroundCommandManager()
  const registry = new ToolRegistry()
  let watches = 0
  registerProcessTools(registry, new WorkspacePathResolver(root), background, undefined, () => {
    watches++; return { id: 'unexpected', expiresAt: Date.now() + 1000 }
  })
  try {
    const foreground = JSON.parse(await exec(registry, { cmd: process.execPath, args: ['-e', 'console.log("ready")'], notify_on_completion: true }))
    expect(foreground.stdout).toContain('ready')
    expect(foreground.completion_watch).toBeUndefined()
    const started = JSON.parse(await exec(registry, { cmd: process.execPath, args: ['-e', 'await Bun.sleep(1000)'], run_in_background: true }))
    expect(started.procId).toBeString()
    expect(watches).toBe(0)
  } finally { await background.disposeAll(); await rm(root, { recursive: true, force: true }) }
})
