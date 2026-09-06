// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, rm, realpath } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { BUILTIN_AGENTS } from '../src/agents/definitions.js'
import { JobStore } from '../src/cron/jobs.js'
import { DaemonServer } from '../src/daemon/server.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import { createNativeSubagentHost } from '../src/daemon/subagentHost.js'
import { DaemonSubagentEventBus } from '../src/daemon/subagentEvents.js'
import { RunHistory } from '../src/runtime/runHistory.js'
import type { LlmClient } from '../src/llms/client.js'

test.each([2, 3, 4, undefined])('schedule limit %i includes actual native child turns and reports exhaustion', async maximum => {
  const directory = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-model-budget-')))
  const store = new JobStore(join(directory, 'jobs.json'))
  const history = new RunHistory(join(directory, 'runs.sqlite'))
  let parentCalls = 0, childCalls = 0, titleCalls = 0
  const child: LlmClient = { async *stream() { childCalls++; yield { content: 'child evidence', usage: { inputTokens: 1, outputTokens: 1 } } } }
  const host = createNativeSubagentHost({ agentDefinitions: BUILTIN_AGENTS, cwd: directory, eventBus: new DaemonSubagentEventBus(), llm: child, model: 'gpt-4o', permissionMode: 'accept-all', tools: [], toolExecutor: { async execute() { return '' } } })
  const parent: LlmClient = { async *stream() {
    parentCalls++
    const liveUsage = history.listWorkspace(directory).find(run => run.kind === "schedule")?.tokenUsage
    expect(liveUsage).toEqual(expect.objectContaining({ input_tokens: parentCalls === 1 ? 0 : 2, pending_calls: 1, complete: false }))
    if (parentCalls === 1) yield { toolCalls: [{ id: 'delegate', type: 'function', function: { name: 'ReadFile', arguments: { path: 'fixture' } } }], usage: { inputTokens: 1, outputTokens: 1 } }
    else yield { content: 'verified result', usage: { inputTokens: 1, outputTokens: 1 } }
  } }
  const runner = new AgentTurnRunner({ llm: parent, model: 'gpt-4o', permissionMode: 'accept-all', tools: [{ type: 'function', function: { name: 'ReadFile', description: 'Fixture evidence', parameters: { type: 'object', properties: { path: { type: 'string' } } } } }], toolExecutor: { async execute() {
    const task = await host.managerPort.spawn({ message: 'Produce fixture evidence', promptProfile: 'coder', title: 'Evidence' })
    await host.managerPort.wait([task.id], 1000)
    expect(host.managerPort.listHandles()).toContainEqual(expect.objectContaining({ id: task.id, status: 'completed' }))
    return 'child evidence'
  } } })
  const runtime = new InMemoryDaemonRuntime(runner, { currentProjectDirectory: directory, model: 'gpt-4o', sessionDirectory: join(directory, 'sessions') })
  const caller = await runtime.openSession('caller')
  const server = new DaemonServer({ titleClientFactory: () => ({ async *stream() { titleCalls++; yield { content: "Evidence gathered" } } }), socketPath: join(directory, 'daemon.sock'), runtime, runHistory: history, projectDirectory: directory, cronStoreFactory: () => store, cronLeasePath: join(directory, 'lease'), cronArchiveDirectory: join(directory, 'archive') })
  await server.start()
  try {
    const created = await server.scheduleToolRequest(caller.id, 'create', { prompt: 'Gather evidence', schedule: '0 9 * * *', paused: true, max_model_calls: maximum }) as { job: { id: string; max_model_calls: number | null } };
    const jobId = created.job.id
    expect(created.job.max_model_calls).toBe(maximum ?? null)
    expect(new JobStore(store.path).get(jobId)?.maxModelCalls).toBe(maximum)
    const work = server.scheduleToolRequest(caller.id, 'run', { schedule_id: jobId })
    if (maximum === 2) await expect(work).rejects.toThrow('Model call budget exhausted')
    else expect((await work).ok).toBe(true)
    expect(childCalls).toBe(1)
    expect(titleCalls).toBe(maximum === 4 || maximum === undefined ? 1 : 0)
    expect(parentCalls).toBe(maximum === 2 ? 1 : 2)
    expect(store.get(jobId)?.metadata.model_call_usage).toEqual({ used: maximum ?? 4, maximum: maximum ?? null, exhausted: maximum === 2 })
    expect(store.get(jobId)?.metadata.token_usage).toEqual(expect.objectContaining({ input_tokens: maximum === 2 ? 2 : 3, output_tokens: maximum === 2 ? 2 : 3, measured_calls: maximum === 2 ? 2 : 3, complete: maximum === 2 || maximum === 3 }))
    expect(history.listWorkspace(directory).find(run => run.kind === "schedule")?.tokenUsage).toEqual(expect.objectContaining({ input_tokens: maximum === 2 ? 2 : 3, output_tokens: maximum === 2 ? 2 : 3, complete: maximum === 2 || maximum === 3 }))
    expect(history.listWorkspace(directory).find(run => run.kind === 'schedule')?.state).toBe(maximum === 2 ? 'failed' : 'succeeded')
  } finally { await server.stop(); await host.manager.shutdown(); history.close(); await rm(directory, { recursive: true, force: true }) }
})

test('daemon persists lifetime tokens across native attempts and projects admission state', async () => {
  const directory = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-lifetime-tokens-')))
  const store = new JobStore(join(directory, 'jobs.json'))
  const history = new RunHistory(join(directory, 'runs.sqlite'))
  let calls = 0
  const llm: LlmClient = { async *stream() { calls++; yield { content: 'checked', usage: { inputTokens: 2, cacheReadTokens: 5, outputTokens: 3 } } } }
  const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({ llm, model: 'gpt-4o', tools: [], toolExecutor: { async execute() { return '' } } }), { currentProjectDirectory: directory, model: 'gpt-4o', sessionDirectory: join(directory, 'sessions') })
  const caller = await runtime.openSession('caller')
  const server = new DaemonServer({ socketPath: join(directory, 'daemon.sock'), runtime, runHistory: history, projectDirectory: directory, cronStoreFactory: () => store, cronLeasePath: join(directory, 'lease'), cronArchiveDirectory: join(directory, 'archive') })
  await server.start()
  try {
    const created = await server.scheduleToolRequest(caller.id, 'create', { prompt: 'Check', schedule: '0 9 * * *', paused: true, max_model_calls: 1, max_total_tokens: 20 }) as { job: { id: string } }
    const id = created.job.id
    for (let attempt = 1; attempt <= 2; attempt++) {
      await server.scheduleToolRequest(caller.id, 'run', { schedule_id: id })
      const list = await server.scheduleToolRequest(caller.id, 'list', {}) as { jobs: Array<{ id: string; revision: string; token_budget: unknown }> }
      expect(list.jobs.find(job => job.id === id)?.token_budget).toEqual({ used: attempt * 10, complete: true, maximum: 20, blocked: attempt === 2 })
      expect(new JobStore(store.path).get(id)?.metadata.total_token_usage).toMatchObject({ attempt, usage: { input_tokens: attempt * 7, output_tokens: attempt * 3, complete: true } })
    }
    await expect(server.scheduleToolRequest(caller.id, 'run', { schedule_id: id })).rejects.toThrow('Total token budget')
    expect(calls).toBe(2)
    expect(store.get(id)?.runsStarted).toBe(2)
  } finally { await server.stop(); history.close(); await rm(directory, { recursive: true, force: true }) }
})
