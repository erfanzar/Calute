// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { ModelCallBudget, withModelCallBudget, chargeModelCall, withIndependentModelCallBudget, optionalModelCallAvailable, assertModelCallBudget } from '../src/llms/callBudget.js'
import { completeLlm, type LlmClient } from '../src/llms/client.js'
import { createAgentState } from '../src/streaming/events.js'
import { runTurn } from '../src/streaming/loop.js'

test('concurrent descendants share admission, isolated scopes do not, and closed scopes reject late work', async () => {
  const budget = new ModelCallBudget(2)
  let late!: () => Promise<void>
  await withModelCallBudget(budget, async () => {
    let release!: () => void
    const pending = new Promise<void>(resolve => { release = resolve }).then(() => chargeModelCall())
    late = async () => { release(); await pending }
    const results = await Promise.allSettled(Array.from({ length: 3 }, async () => { await Promise.resolve(); chargeModelCall() }))
    expect(results.filter(value => value.status === 'fulfilled')).toHaveLength(2)
    expect(budget.used).toBe(2)
    expect(budget.exhausted).toBe(true)
  })
  budget.close()
  await expect(late()).rejects.toThrow('scope has ended')
  const independent = new ModelCallBudget(1)
  withModelCallBudget(independent, chargeModelCall)
  expect(independent.used).toBe(1)
  expect(() => chargeModelCall()).not.toThrow()
})

test('streaming and auxiliary completions share the limit before touching the provider', async () => {
  let calls = 0
  const llm: LlmClient = { async *stream() { calls++; yield { content: 'done', usage: { inputTokens: 1, outputTokens: 1 } } } }
  const budget = new ModelCallBudget(1)
  await withModelCallBudget(budget, async () => {
    await completeLlm(llm, { model: 'test', messages: [{ role: 'user', content: 'auxiliary' }] })
    const events = []
    for await (const event of runTurn({ state: createAgentState(), model: 'test', userMessage: 'main', tools: [] }, { llm, retryDelays: [0, 0], toolExecutor: { async execute() { return '' } } })) events.push(event)
    expect(events.some(event => event.type === 'turn_done')).toBe(true)
    expect(JSON.stringify(events)).toContain('Model call budget exhausted')
    expect(calls).toBe(1)
    expect(budget.used).toBe(1)
  })
})

test('already-cancelled completions do not spend admission', async () => {
  const budget = new ModelCallBudget(1)
  await withModelCallBudget(budget, async () => {
    const signal = AbortSignal.abort(new Error('cancelled'))
    await expect(completeLlm({ async *stream() { throw new Error('must not call') } }, { model: 'test', messages: [] }, signal)).rejects.toThrow('cancelled')
    expect(budget.used).toBe(0)
  })
})

test('usage receipts are idempotent and distinguish partial, missing and pending calls', () => {
  const budget = new ModelCallBudget()
  const first = budget.charge()
  first({ inputTokens: 5, outputTokens: 2 })
  first({ inputTokens: 100, outputTokens: 100 })
  budget.charge()({ inputTokens: 3, outputTokens: 1 }, false)
  budget.charge()({ inputTokens: NaN, outputTokens: -1 })
  const pending = budget.charge()
  expect(budget.usage).toEqual({ input_tokens: 8, output_tokens: 3, measured_calls: 1, settled_calls: 3, pending_calls: 1, complete: false })
  budget.close()
  pending({ inputTokens: 2, outputTokens: 2 })
  expect(budget.usage.pending_calls).toBe(0)
  expect(budget.usage.complete).toBe(false)
})

test('completion cancellation settles unknown usage without inventing zero usage', async () => {
  const budget = new ModelCallBudget()
  const controller = new AbortController()
  const client: LlmClient = { async *stream() {}, async complete() { controller.abort(new Error('cancelled')); return new Promise(() => {}) } }
  await withModelCallBudget(budget, async () => {
    await expect(completeLlm(client, { model: 'test', messages: [] }, controller.signal)).rejects.toThrow('cancelled')
  })
  expect(budget.usage).toEqual({ input_tokens: 0, output_tokens: 0, measured_calls: 0, settled_calls: 1, pending_calls: 0, complete: false })
})

test('stream retry accounting preserves partial usage from the failed attempt', async () => {
  const budget = new ModelCallBudget()
  let calls = 0
  const llm: LlmClient = { async *stream() {
    calls++
    yield { usage: { inputTokens: calls === 1 ? 7 : 11, outputTokens: 2 } }
    if (calls === 1) throw new Error('transient connection drop')
    yield { content: 'done' }
  } }
  await withModelCallBudget(budget, async () => {
    for await (const _event of runTurn({ state: createAgentState(), model: 'test', userMessage: 'main', tools: [] }, { llm, retryDelays: [0], toolExecutor: { async execute() { return '' } } })) { /* consume */ }
  })
  expect(calls).toBe(2)
  expect(budget.usage).toEqual({ input_tokens: 18, output_tokens: 4, measured_calls: 1, settled_calls: 2, pending_calls: 0, complete: false })
})

test('checkpoint failure prevents provider execution and cannot be retried around', async () => {
  let providerCalls = 0
  let writes = 0
  const budget = new ModelCallBudget(undefined, () => { writes++; throw new Error('disk full') })
  await withModelCallBudget(budget, async () => {
    const llm: LlmClient = { async *stream() { providerCalls++; yield { content: 'must not run' } } }
    const events = []
    for await (const event of runTurn({ state: createAgentState(), model: 'test', userMessage: 'main', tools: [] }, { llm, retryDelays: [0, 0], toolExecutor: { async execute() { return '' } } })) events.push(event)
    expect(JSON.stringify(events)).toContain('Could not persist model usage checkpoint')
    await expect(completeLlm(llm, { model: 'test', messages: [] })).rejects.toThrow('checkpoint')
  })
  expect(providerCalls).toBe(0)
  expect(writes).toBe(1)
  expect(budget.available).toBe(false)
})

test('late receipts cannot checkpoint after finalization', () => {
  const snapshots: unknown[] = []
  const budget = new ModelCallBudget(undefined, usage => snapshots.push(usage))
  const receipt = budget.charge()
  budget.close()
  receipt({ inputTokens: 2, outputTokens: 1 })
  expect(snapshots).toHaveLength(1)
  expect(snapshots[0]).toEqual(expect.objectContaining({ pending_calls: 1, complete: false }))
})

test('nested provider calls consume parent and child budgets and cannot bypass the parent limit', async () => {
  const parent = new ModelCallBudget(1), child = new ModelCallBudget(5)
  let calls = 0
  const llm: LlmClient = { async *stream() { calls++; yield { content: 'done', usage: { inputTokens: 7, outputTokens: 3 } } } }
  await withModelCallBudget(parent, () => withModelCallBudget(child, async () => {
    await completeLlm(llm, { model: 'test', messages: [] })
    await expect(completeLlm(llm, { model: 'test', messages: [] })).rejects.toThrow('budget exhausted')
  }))
  expect(calls).toBe(1)
  for (const budget of [parent, child]) expect(budget.usage).toMatchObject({ input_tokens: 7, output_tokens: 3, settled_calls: 1, pending_calls: 0, complete: true })
})

test('child admission denial does not consume an available parent slot', () => {
  const parent = new ModelCallBudget(3), child = new ModelCallBudget(1)
  child.charge()({ inputTokens: 1, outputTokens: 0 })
  expect(() => withModelCallBudget(parent, () => withModelCallBudget(child, chargeModelCall))).toThrow('budget exhausted')
  expect(parent.used).toBe(0)
})

test('reentering the same budget does not double charge usage', () => {
  const budget = new ModelCallBudget(3)
  withModelCallBudget(budget, () => withModelCallBudget(budget, () => chargeModelCall()?.({ inputTokens: 5, outputTokens: 2 })))
  expect(budget.used).toBe(1)
  expect(budget.usage.input_tokens).toBe(5)
})

test('nested receipts update other scopes even when a usage checkpoint fails', () => {
  let checkpoints = 0
  const parent = new ModelCallBudget(2, () => { if (++checkpoints > 1) throw new Error('store unavailable') })
  const child = new ModelCallBudget(2)
  withModelCallBudget(parent, () => withModelCallBudget(child, () => {
    const receipt = chargeModelCall()!
    expect(() => receipt({ inputTokens: 9, outputTokens: 4 })).toThrow('checkpoint')
    expect(child.usage).toMatchObject({ input_tokens: 9, output_tokens: 4, pending_calls: 0, complete: true })
    expect(optionalModelCallAvailable()).toBe(false)
    expect(() => assertModelCallBudget()).toThrow('checkpoint')
  }))
})

test('failed inner admission settles earlier scope counters without calling a provider', () => {
  const parent = new ModelCallBudget(3)
  const child = new ModelCallBudget(3, () => { throw new Error('disk full') })
  expect(() => withModelCallBudget(parent, () => withModelCallBudget(child, chargeModelCall))).toThrow('checkpoint')
  expect(parent.usage).toMatchObject({ settled_calls: 1, pending_calls: 0, complete: false })
  expect(child.persistenceError).toBeDefined()
})

test('a closed parent prevents optional and required work inside a fresh child scope', () => {
  const parent = new ModelCallBudget(3), child = new ModelCallBudget(3)
  parent.close()
  withModelCallBudget(parent, () => withModelCallBudget(child, () => {
    expect(optionalModelCallAvailable()).toBe(false)
    expect(() => chargeModelCall()).toThrow('scope has ended')
    expect(() => assertModelCallBudget()).toThrow('budget exhausted')
  }))
  expect(child.used).toBe(0)
})


test('a separately admitted host run does not inherit the completed notifying run', () => {
  const previous = new ModelCallBudget(1), next = new ModelCallBudget(1)
  previous.close()
  withModelCallBudget(previous, () => withIndependentModelCallBudget(next, () => {
    expect(optionalModelCallAvailable()).toBe(true)
    chargeModelCall()?.({ inputTokens: 2, outputTokens: 1 })
  }))
  expect(previous.used).toBe(0)
  expect(next.usage).toMatchObject({ input_tokens: 2, output_tokens: 1, complete: true })
})
