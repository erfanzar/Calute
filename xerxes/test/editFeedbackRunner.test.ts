// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import type { DaemonSession } from '../src/daemon/runtime.js'
import { EditFeedback } from '../src/runtime/editFeedback.js'
import { EditDiagnostics } from '../src/runtime/editDiagnostics.js'
import type { LlmDelta } from '../src/llms/client.js'

test('real turn loop captures edit baseline and persists diagnostic feedback with the session', async () => {
  const session: DaemonSession = {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: process.cwd(), extra: {},
    id: 'edit-feedback-session', interactionMode: 'code', sessionKey: 'edit-feedback', lastActive: 0,
    messages: [], metadata: {}, model: 'fixture', planMode: false, status: 'working',
    thinkingContent: [], toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0,
    turnCount: 0, workspace: '/tmp/agents/default',
  }
  let calls = 0; let edited = false; const order: string[] = []
  const runner = new AgentTurnRunner({
    model: 'fixture', permissionMode: 'accept-all', editDiagnostics: true,
    llm: { async *stream(): AsyncGenerator<LlmDelta> {
      if (++calls === 1) yield { toolCalls: [{ id: 'write', type: 'function', function: { name: 'WriteFile', arguments: { file_path: 'source.ts', content: 'changed' } } }] }
      else yield { content: 'Edit complete.' }
    } },
    tools: [{ type: 'function', function: { name: 'WriteFile', description: 'Fixture', parameters: { type: 'object' } } }],
    toolExecutor: { execute: async () => { order.push('write'); edited = true; return 'ok' } },
    createEditFeedback: cwd => new EditFeedback(cwd, new EditDiagnostics(cwd, { fileExists: () => false }), {
      execute: async () => { order.push(edited ? 'after' : 'before'); return { fresh: true, diagnostics: edited ? [{ message: 'new semantic error', severity: 1, range: { start: { line: 1, character: 0 }, end: { line: 1, character: 1 } } }] : [] } },
    }),
  })
  for await (const _event of runner.run(session, 'Edit source.ts', new AbortController().signal)) { /* Consume real loop. */ }
  expect(order).toEqual(['before', 'write', 'after'])
  expect(session.messages.some(message => typeof message.content === 'string' && message.content.includes('source.ts:2:1 error: new semantic error'))).toBe(true)
})
