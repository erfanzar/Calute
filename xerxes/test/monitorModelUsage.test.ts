// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, realpath, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { BUILTIN_AGENTS } from '../src/agents/definitions.js'
import { JobStore } from '../src/cron/jobs.js'
import { DaemonServer } from '../src/daemon/server.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import { createNativeSubagentHost } from '../src/daemon/subagentHost.js'
import { DaemonSubagentEventBus } from '../src/daemon/subagentEvents.js'
import { completeLlm, type LlmClient } from '../src/llms/client.js'
import { RunHistory } from '../src/runtime/runHistory.js'
import { ReactionMailbox } from '../src/runtime/reactionMailbox.js'
import { TerminalMonitors } from '../src/runtime/terminalMonitors.js'
import { TerminalRegistry } from '../src/runtime/terminalRegistry.js'

test.each([[true, false, false, undefined], [false, false, false, undefined], [true, true, false, undefined], [true, false, true, undefined], [true, false, false, 28]] as const)('monitor usage covers child=%s title=%s provider-failure=%s', async (childUsage, autoTitle, providerFailure, maximum) => {
  const directory = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-monitor-usage-')))
  const history = new RunHistory(join(directory, 'runs.sqlite'))
  const mailbox = new ReactionMailbox(join(directory, 'reactions.sqlite'))
  const terminals = new TerminalRegistry()
  let releaseTitle!: () => void
  const titleGate = new Promise<void>(resolve => { releaseTitle = resolve })
  let parentCalls = 0, childCalls = 0, auxiliaryCalls = 0, titleCalls = 0
  const host = createNativeSubagentHost({ agentDefinitions: BUILTIN_AGENTS, cwd: directory, eventBus: new DaemonSubagentEventBus(), model: 'gpt-4o', permissionMode: 'accept-all', tools: [],
    llm: { async *stream() { childCalls++; yield { content: 'child evidence', ...(childUsage ? { usage: { inputTokens: 5, outputTokens: 1 } } : {}) } } },
    toolExecutor: { async execute() { return '' } } })
  const auxiliary: LlmClient = { async *stream() { auxiliaryCalls++; yield { content: 'auxiliary result', usage: { inputTokens: 7, outputTokens: 2 } } } }
  const llm: LlmClient = { async *stream() {
    parentCalls++
    const claim = mailbox.unresolved(owner.id)[0]!
    expect(mailbox.inspect(owner.id, claim.runId)?.usage).toMatchObject({ inputTokens: parentCalls === 1 ? 0 : childUsage ? 22 : 17, complete: false })
    expect(history.listWorkspace(directory).find(run => run.title.startsWith('Monitor reaction:'))?.tokenUsage?.pending_calls).toBe(1)
    if (parentCalls === 1) yield { toolCalls: [{ id: 'delegate', type: 'function', function: { name: 'ReadFile', arguments: { path: 'fixture' } } }], usage: { inputTokens: 10, outputTokens: 3 } }
    else {
      yield { content: 'verified monitor result', usage: { inputTokens: 10, outputTokens: 3 } }
      if (providerFailure) throw new Error('stream request failed (401): unauthorized')
    }
  } }
  const runner = new AgentTurnRunner({ llm, model: 'gpt-4o', permissionMode: 'accept-all', tools: [{ type: 'function', function: { name: 'ReadFile', description: 'Fixture', parameters: { type: 'object', properties: { path: { type: 'string' } } } } }], toolExecutor: { async execute() {
    const task = await host.managerPort.spawn({ message: 'Collect evidence', promptProfile: 'coder', title: 'Evidence' })
    await host.managerPort.wait([task.id], 1000)
    await completeLlm(auxiliary, { model: 'gpt-4o', messages: [] })
    return 'evidence collected'
  } } })
  const runtime = new InMemoryDaemonRuntime(runner, { currentProjectDirectory: directory, model: 'gpt-4o', sessionDirectory: join(directory, 'sessions') })
  const owner = await runtime.openSession('owner')
  if (!autoTitle) owner.metadata.title = 'Existing task'
  const monitors = new TerminalMonitors(terminals, history, (watch, event) => server.notifyMonitorEvent(watch, event), undefined, mailbox)
  const server = new DaemonServer({ runtime, runHistory: history, reactionMailbox: mailbox, monitors, terminalRegistry: terminals, projectDirectory: directory, socketPath: join(directory, 'daemon.sock'), cronLeasePath: join(directory, 'lease'), cronStoreFactory: () => new JobStore(join(directory, 'jobs.json')),
    titleClientFactory: () => ({ async *stream() { titleCalls++; await titleGate; yield { content: 'Monitor evidence', usage: { inputTokens: 4, outputTokens: 1 } } } }) })
  await server.start()
  try {
    const terminal = terminals.open({ ownerSessionId: owner.id, id: 'build', cwd: directory, command: 'fixture', kind: 'background' })
    const watch = monitors.start(owner.id, { terminalId: 'build', match: 'error', reaction: { maxReactions: 1, maxDurationMs: 5000, ...(maximum ? { maxTotalTokens: maximum } : {}) } })
    terminal.append('error: fixture failed\n')
    const deadline = Date.now() + 3000
    while (!['completed', 'failed', 'cancelled'].includes(mailbox.inspect(owner.id, watch.id)?.lastOutcome ?? '') && Date.now() < deadline) await Bun.sleep(10)
    expect(parentCalls).toBe(maximum ? 1 : 2)
    expect(childCalls).toBe(1)
    expect(auxiliaryCalls).toBe(1)
    expect(titleCalls).toBe(autoTitle ? 1 : 0)
    const usage = { inputTokens: maximum ? 22 : childUsage ? 32 : 27, outputTokens: maximum ? 6 : childUsage ? 9 : 8, complete: childUsage && !autoTitle && !providerFailure && !maximum }
    expect(mailbox.inspect(owner.id, watch.id)).toMatchObject({ lastOutcome: providerFailure || maximum ? 'failed' : 'completed', usage })
    expect(history.list(owner.id).find(run => run.title.startsWith('Monitor reaction:'))).toMatchObject({ state: providerFailure || maximum ? 'failed' : 'succeeded', tokenUsage: { input_tokens: usage.inputTokens, output_tokens: usage.outputTokens, complete: usage.complete, settled_calls: maximum ? 3 : 4, pending_calls: autoTitle ? 1 : 0 } })
  } finally { releaseTitle(); monitors.close(); await server.stop(); await host.manager.shutdown(); mailbox.close(); history.close(); await rm(directory, { recursive: true, force: true }) }
})
