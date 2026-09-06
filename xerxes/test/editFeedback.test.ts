// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { EditDiagnostics } from '../src/runtime/editDiagnostics.js'
import { EditFeedback } from '../src/runtime/editFeedback.js'

const call = { id: 'edit', type: 'function' as const, function: { name: 'WriteFile', arguments: { file_path: 'file.ts', content: 'changed' } } }
const context = { metadata: {} }
const item = (message: string) => ({ message, severity: 1, range: { start: { line: 0, character: 2 }, end: { line: 0, character: 3 } } })
const noChecker = () => new EditDiagnostics('/workspace', { fileExists: () => false })

test('captures LSP baseline before edit and reports only new version-matched findings', async () => {
  const order: string[] = []; let edited = false
  const feedback = new EditFeedback('/workspace', noChecker(), { execute: async () => { order.push(edited ? 'after' : 'before'); return { fresh: true, diagnostics: edited ? [item('existing'), item('new')] : [item('existing')] } } })
  await feedback.wrap({ execute: async () => { order.push('edit'); edited = true; return 'ok' } }).execute(call, context)
  const report = await feedback.report()
  expect(order).toEqual(['before', 'edit', 'after'])
  expect(report).toContain('1 new problem(s)'); expect(report).toContain('file.ts:1:3 error: new'); expect(report).not.toContain('existing')
})

test('unconfirmed LSP results never suppress independent checker findings', async () => {
  let runs = 0
  const checker = new EditDiagnostics('/workspace', { checker: { source: 'tsc', command: ['fake'] }, includeWorkspaceRisk: false,
    commandRunner: () => ({ exitCode: runs++ ? 2 : 0, stdout: runs > 1 ? 'file.ts(2,3): error TS123: checker finding' : '' }) })
  const feedback = new EditFeedback('/workspace', checker, { execute: async () => { throw new Error('server unavailable') } })
  await feedback.wrap({ execute: async () => 'ok' }).execute(call, context)
  const report = await feedback.report()
  expect(report).toContain('checker finding'); expect(report).toContain('Diagnostics unconfirmed')
})

test('missing baseline labels current diagnostics without claiming the edit introduced them', async () => {
  let count = 0
  const feedback = new EditFeedback('/workspace', noChecker(), { execute: async () => ++count === 1 ? { fresh: false, diagnostics: [] } : { fresh: true, diagnostics: [item('problem')] } })
  await feedback.wrap({ execute: async () => 'ok' }).execute(call, context)
  expect(await feedback.report()).toContain('current problem(s) (pre-edit baseline unconfirmed)')
})

test('read-only and failed edits do not produce post-edit reports', async () => {
  let requests = 0
  const feedback = new EditFeedback('/workspace', noChecker(), { execute: async () => { requests++; return { fresh: true, diagnostics: [] } } })
  await feedback.wrap({ execute: async () => 'ok' }).execute({ ...call, function: { ...call.function, name: 'ReadFile' } }, context)
  expect(requests).toBe(0); expect(await feedback.report()).toBe('')
  await expect(feedback.wrap({ execute: async () => { throw new Error('write failed') } }).execute(call, context)).rejects.toThrow('write failed')
  expect(await feedback.report()).toBe(''); expect(requests).toBe(1)
})

test('cancellation after baseline prevents mutation', async () => {
  const controller = new AbortController(); let edited = false
  const feedback = new EditFeedback('/workspace', noChecker(), { execute: async () => { controller.abort(); return { fresh: false, diagnostics: [] } } })
  await expect(feedback.wrap({ execute: async () => { edited = true; return 'ok' } }).execute(call, context, controller.signal)).rejects.toThrow('cancelled')
  expect(edited).toBe(false)
})
