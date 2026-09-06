// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { beginScheduleTokenUsage, scheduleTokenState } from '../src/cron/tokenUsage.js'
import { ModelCallBudget, assertModelCallBudget, withModelCallBudget } from '../src/llms/callBudget.js'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { CronJob, JobStore } from '../src/cron/jobs.js'
import { CronScheduler } from '../src/cron/scheduler.js'

test.each(['exhausted', 'unknown'] as const)('scheduler refuses %s lifetime usage before admission after restart', async kind => {
  const dir = mkdtempSync(join(tmpdir(), 'schedule-tokens-'))
  try {
    const path = join(dir, 'jobs.json'), store = new JobStore(path)
    const usage = new ModelCallBudget()
    usage.charge()({ inputTokens: 7, outputTokens: 3 })
    store.add(new CronJob({ id: 'budget', prompt: 'check', intervalSeconds: 60, maxTotalTokens: 10,
      runsStarted: 1, metadata: kind === 'exhausted' ? { total_token_usage: beginScheduleTokenUsage(undefined, 1)(usage.usage) } : {} }))
    const reopened = new JobStore(path)
    let calls = 0
    const scheduler = new CronScheduler(reopened, () => { calls++; return 'unexpected' })
    await expect(scheduler.runNow(reopened.get('budget')!, async () => { calls++; return 'unexpected' })).rejects.toThrow('Total token budget')
    expect(calls).toBe(0)
    expect(reopened.get('budget')).toMatchObject({ runsStarted: 1, paused: true, maxTotalTokens: 10 })
    reopened.update('budget', { max_total_tokens: null, paused: false })
    await scheduler.runNow(reopened.get('budget')!, async () => { calls++; return 'allowed' })
    expect(calls).toBe(1)
    expect(new JobStore(path).get('budget')?.runsStarted).toBe(2)
  } finally { rmSync(dir, { recursive: true, force: true }) }
})

test('checkpoints replace the attempt contribution and include cache tokens across restart', () => {
  const aggregate = beginScheduleTokenUsage(undefined, 1)
  const first = new ModelCallBudget()
  first.charge()({ inputTokens: 2, outputTokens: 3, cacheReadTokens: 10, cacheCreationTokens: 5 })
  const snapshot = aggregate(first.usage)
  expect(aggregate(first.usage)).toEqual(snapshot)
  expect(scheduleTokenState(snapshot, 1)).toEqual({ used: 20, complete: true })
  const next = beginScheduleTokenUsage(JSON.parse(JSON.stringify(snapshot)), 2)
  const second = new ModelCallBudget()
  const settle = second.charge()
  expect(scheduleTokenState(next(second.usage), 2)).toEqual({ used: 20, complete: false })
  settle({ inputTokens: 3, outputTokens: 7 })
  expect(scheduleTokenState(next(second.usage), 2)).toEqual({ used: 30, complete: true })
  expect(() => beginScheduleTokenUsage(snapshot, 1)).toThrow('Stale')
  expect(() => next({ ...second.usage, input_tokens: -1 })).toThrow('Invalid')
})

test('missing historical attempts and partial receipts remain unknown', () => {
  expect(scheduleTokenState(undefined, 0).complete).toBe(true)
  expect(scheduleTokenState(undefined, 1).complete).toBe(false)
  const budget = new ModelCallBudget()
  budget.charge()({ inputTokens: 2, outputTokens: 3 }, false)
  const partial = beginScheduleTokenUsage(undefined, 1)(budget.usage)
  const empty = new ModelCallBudget().usage
  expect(scheduleTokenState(beginScheduleTokenUsage(partial, 2)(empty), 2)).toEqual({ used: 5, complete: false })
  expect(scheduleTokenState(beginScheduleTokenUsage(undefined, 2)(empty), 2).complete).toBe(false)
})

test('total admission includes prior usage and allows only already pending calls to overshoot', () => {
  const budget = new ModelCallBudget(undefined, undefined, { maximum: 20, priorTokens: 10, priorComplete: true })
  const first = budget.charge(), concurrent = budget.charge()
  first({ inputTokens: 1, cacheReadTokens: 8, outputTokens: 1 })
  expect(() => budget.charge()).toThrow('exhausted')
  concurrent({ inputTokens: 5, outputTokens: 5 })
  expect(budget.used).toBe(2)
  expect(budget.usage.input_tokens + budget.usage.output_tokens).toBe(20)
  expect(() => withModelCallBudget(budget, assertModelCallBudget)).toThrow('exhausted')
})

test('unknown usage and accounting overflow fail closed', () => {
  const prior = new ModelCallBudget(undefined, undefined, { maximum: 20, priorTokens: 0, priorComplete: false })
  expect(() => prior.charge()).toThrow('incomplete')
  const budget = new ModelCallBudget(undefined, undefined, { maximum: 20, priorTokens: 0, priorComplete: true })
  budget.charge()()
  expect(() => budget.charge()).toThrow('incomplete')
  const overflow = new ModelCallBudget()
  expect(() => overflow.charge()({ inputTokens: Number.MAX_SAFE_INTEGER, outputTokens: 1 })).toThrow('overflow')
  expect(() => withModelCallBudget(overflow, assertModelCallBudget)).toThrow('overflow')
  expect(overflow.usage.complete).toBe(false)
})
