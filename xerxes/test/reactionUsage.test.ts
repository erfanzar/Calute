// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { ReactionChildUsage, ReactionExecutionError } from '../src/runtime/reactionUsage.js'
import { ReactionMailbox } from '../src/runtime/reactionMailbox.js'
import { ReactionDispatcher } from '../src/runtime/reactionDispatcher.js'

test('reaction child usage counts new children once and excludes preexisting agents', () => {
  const meter = new ReactionChildUsage()
  const parent = { inputTokens: 100, outputTokens: 20, complete: true }
  expect(meter.addTo(parent)).toEqual(parent)
  meter.observe({ agent_id: 'old', input_tokens: 900, output_tokens: 50 })
  meter.observe({ agent_id: 'new', task_index: 1, event: { type: 'turn_begin' } })
  for (const input of [30, 30, 20, 45]) meter.observe({ agent_id: 'new', task_index: 1, input_tokens: input, output_tokens: 7 })
  meter.observe({ agent_id: 'second', event: { type: 'turn_begin' }, input_tokens: 8, output_tokens: 2 })
  expect(meter.addTo(parent)).toEqual({ inputTokens: 153, outputTokens: 29, complete: false })
  meter.observe({ agent_id: 'new', task_index: 1, input_tokens: -1, output_tokens: NaN })
  expect(meter.addTo(parent)).toEqual({ inputTokens: 153, outputTokens: 29, complete: false })
})

for (const cancelled of [false, true]) test(`reaction ${cancelled ? 'cancellation' : 'failure'} retains measured usage`, async () => {
  const mailbox = new ReactionMailbox(':memory:')
  mailbox.configure({ owner: 'owner', runId: 'watch', expiresAt: Date.now() + 60_000, maxReactions: 2, maxDurationMs: 5000 })
  mailbox.offer('owner', 'watch', 1)
  const dispatcher = new ReactionDispatcher(mailbox, {
    admit: async (_owner, work) => work(),
    run: async () => {
      if (cancelled) mailbox.cancel('owner')
      throw new ReactionExecutionError(new Error('provider stopped'), { inputTokens: 120, outputTokens: 30, complete: false })
    },
  })
  try {
    await expect(dispatcher.dispatch('owner')).rejects.toThrow('provider stopped')
    expect(mailbox.inspect('owner', 'watch')).toMatchObject({ lastOutcome: cancelled ? 'cancelled' : 'failed', usage: { inputTokens: 120, outputTokens: 30, complete: false } })
  } finally { await dispatcher.close(); mailbox.close() }
})
