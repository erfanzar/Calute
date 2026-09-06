// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import type { AgentDefinition } from '../src/agents/definitions.js'
import { replacePersistedSubagentSnapshots, persistedSubagentSnapshotValues } from '../src/agents/subagentPersistence.js'
import type { SubagentWorktreePort } from '../src/agents/subagentManager.js'
import { DaemonSubagentEventBus } from '../src/daemon/subagentEvents.js'
import { createNativeSubagentHost } from '../src/daemon/subagentHost.js'
import { recoverSubagentSnapshots } from '../src/daemon/subagentCoordinator.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import type { SpawnedAgentSnapshot } from '../src/operators/subagents.js'
import { getActiveSession } from '../src/runtime/sessionContext.js'

const definition: AgentDefinition = { name: 'worker', description: 'workspace test', source: 'test', model: '',
  systemPrompt: 'Use report_workspace once.', allowedTools: null, excludeTools: [], tools: [], isolation: '', maxDepth: 3 }
interface Report { cwd: string; projectRoot: unknown; sentinel: string }
class WorkspaceClient implements LlmClient {
  calls = 0
  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.calls++
    if (!request.messages.some(message => message.role === 'tool')) {
      yield { toolCalls: [{ id: crypto.randomUUID(), type: 'function', function: { name: 'report_workspace', arguments: {} } }] }
    } else yield { content: 'workspace recorded' }
  }
}
function makeHost(client: WorkspaceClient, cwd: string, reports: Report[], resolveSourceWorkspace?: (source: string) => string,
  worktreeForWorkspace?: (workspace: string) => SubagentWorktreePort) {
  const tools = new ToolRegistry()
  tools.register({ type: 'function', function: { name: 'report_workspace', description: 'Read local sentinel',
    parameters: { type: 'object', properties: {}, additionalProperties: false } } }, async (_call, context) => {
    const active = getActiveSession<{ cwd: string }>()
    if (!active) throw new Error('No active child workspace')
    const report = { cwd: active.cwd, projectRoot: context.metadata.project_root,
      sentinel: await Bun.file(join(active.cwd, 'sentinel.txt')).text() }
    reports.push(report)
    return JSON.stringify(report)
  })
  return createNativeSubagentHost({ agentDefinitions: new Map([['worker', definition]]), cwd, eventBus: new DaemonSubagentEventBus(),
    llm: client, model: 'test-model', permissionMode: 'accept-all', toolExecutor: tools, tools: tools.definitions(),
    ...(resolveSourceWorkspace ? { resolveSourceWorkspace } : {}), ...(worktreeForWorkspace ? { worktreeForWorkspace } : {}) })
}
async function directories() {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-workspace-recovery-'))
  const a = join(root, 'a'), b = join(root, 'b'), tree = join(root, 'tree')
  for (const [path, value] of [[a, 'A'], [b, 'B'], [tree, 'TREE']] as const) {
    await mkdir(path)
    await Bun.write(join(path, 'sentinel.txt'), value)
  }
  return { root, a, b, tree }
}
function recovered(workspace?: string): SpawnedAgentSnapshot {
  return { id: 'recovered-child', name: 'child', title: 'Child', agentId: 'worker', promptProfile: 'worker', sourceAgentId: 'parent',
    createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), status: 'interrupted', closed: false, queueSize: 0,
    lastInput: 'read sentinel', ...(workspace === undefined ? {} : { workspace }) }
}

test('restart and live retry read original project files even after the source moves', async () => {
  const { root, a, b } = await directories()
  let source = a
  const reports: Report[] = [], client = new WorkspaceClient()
  const first = makeHost(client, b, reports, () => source)
  try {
    const task = await first.managerPort.spawn({ message: 'read sentinel', promptProfile: 'worker', sourceAgentId: 'parent' })
    await first.managerPort.wait([task.id], 5000)
    expect(reports).toEqual([{ cwd: a, projectRoot: a, sentinel: 'A' }])
    source = b
    await first.retry(task.id, { sourceAgentId: 'parent', message: 'retry' })
    await first.managerPort.wait([task.id], 5000)
    const metadata: Record<string, unknown> = {}
    replacePersistedSubagentSnapshots(metadata, first.managerPort.listHandles())
    const snapshots = recoverSubagentSnapshots([], 'parent', persistedSubagentSnapshotValues(metadata))
    expect(snapshots[0]?.workspace).toBe(a)
    await first.manager.shutdown()
    const after: Report[] = [], next = makeHost(new WorkspaceClient(), b, after, () => source)
    try {
      next.turnCoordinator.restore?.('parent', snapshots)
      const retry = await next.retry(task.id, { sourceAgentId: 'parent', message: 'read again' })
      await next.managerPort.wait([retry.id], 5000)
      expect(after).toEqual([{ cwd: a, projectRoot: a, sentinel: 'A' }])
    } finally { await next.manager.shutdown() }
  } finally { await first.manager.shutdown(); await rm(root, { recursive: true, force: true }) }
})

test('recovered isolated task allocates under its original project and tools run inside the new worktree', async () => {
  const { root, a, b, tree } = await directories()
  const calls: string[] = [], reports: Report[] = []
  const native = makeHost(new WorkspaceClient(), b, reports, () => b, workspace => {
    calls.push(workspace)
    return { async create() { return { branch: 'test', path: tree } }, async isClean() { return true }, async remove() {} }
  })
  try {
    native.turnCoordinator.restore?.('parent', [{ ...recovered(a), rules: ['permission:accept-all', 'isolation:worktree'] }])
    const task = await native.retry('recovered-child', { sourceAgentId: 'parent' })
    await native.managerPort.wait([task.id], 5000)
    expect(calls).toEqual([a])
    expect(reports).toEqual([{ cwd: tree, projectRoot: a, sentinel: 'TREE' }])
  } finally { await native.manager.shutdown(); await rm(root, { recursive: true, force: true }) }
})

test('malformed workspace and unavailable legacy owner reject before provider execution', async () => {
  const client = new WorkspaceClient()
  const native = makeHost(client, process.cwd(), [], () => { throw new Error('source session missing') })
  try {
    native.turnCoordinator.restore?.('parent', [recovered('')])
    await expect(native.retry('recovered-child', { sourceAgentId: 'parent' })).rejects.toThrow('malformed')
    native.turnCoordinator.restore?.('parent', [{ ...recovered(), id: 'legacy-child', name: 'legacy' }])
    await expect(native.retry('legacy-child', { sourceAgentId: 'parent' })).rejects.toThrow('source session missing')
    expect(client.calls).toBe(0)
  } finally { await native.manager.shutdown() }
})
