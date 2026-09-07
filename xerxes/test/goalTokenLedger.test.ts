// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { Database } from 'bun:sqlite'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { GoalTokenLedger } from '../src/runtime/goalTokenLedger.js'

const usage = { inputTokens: 10, outputTokens: 5 }

function temporaryLedger(): { readonly path: string; readonly directory: string } {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-goal-tokens-'))
  return { directory, path: join(directory, 'tokens.sqlite') }
}

test('ledger requires initialization and survives restart with cache accounting', () => {
  const temporary = temporaryLedger()
  let ledger = new GoalTokenLedger(temporary.path)
  try {
    expect(ledger.inspect('session', 'goal')).toBeNull()
    expect(() => ledger.admit('session', 'goal', 'owner', 100)).toThrow('not initialized')
    ledger.initialize('session', 'goal')
    const receipt = ledger.admit('session', 'goal', 'owner', 100)
    receipt({ ...usage, cacheReadTokens: 3, cacheCreationTokens: 2 })
    expect(ledger.inspect('session', 'goal')).toEqual({
      inputTokens: 15, outputTokens: 5, measuredCalls: 1, settledCalls: 1, pendingCalls: 0, complete: true,
    })
    ledger.close()
    ledger = new GoalTokenLedger(temporary.path)
    expect(ledger.inspect('session', 'goal')).toMatchObject({ inputTokens: 15, outputTokens: 5, complete: true })
    expect(() => ledger.admit('session', 'goal', 'owner', 20)).toThrow('exhausted')
    const raised = ledger.admit('session', 'goal', 'owner', 21)
    raised(usage)
  } finally {
    ledger.close()
    rmSync(temporary.directory, { recursive: true, force: true })
  }
})

test('same-owner pending admissions may overshoot, while another capped owner is fenced', () => {
  const ledger = new GoalTokenLedger(':memory:')
  try {
    ledger.initialize('session', 'goal')
    const first = ledger.admit('session', 'goal', 'owner-a', 10)
    const second = ledger.admit('session', 'goal', 'owner-a', 10)
    expect(() => ledger.admit('session', 'goal', 'owner-b', 10)).toThrow('another process')
    const uncapped = ledger.admit('session', 'goal', 'owner-b')
    first({ inputTokens: 8, outputTokens: 4 })
    second({ inputTokens: 8, outputTokens: 4 })
    uncapped({ inputTokens: 0, outputTokens: 0 })
    expect(ledger.inspect('session', 'goal')).toMatchObject({ inputTokens: 16, outputTokens: 8, measuredCalls: 3, settledCalls: 3, complete: true })
    expect(() => ledger.admit('session', 'goal', 'owner-a', 24)).toThrow('exhausted')
  } finally { ledger.close() }
})

test('failed and missing usage block capped work while uncapped tracking continues', () => {
  const ledger = new GoalTokenLedger(':memory:')
  try {
    ledger.initialize('session', 'goal')
    const failed = ledger.admit('session', 'goal', 'owner', 100)
    failed(undefined, false)
    expect(ledger.inspect('session', 'goal')).toEqual({
      inputTokens: 0, outputTokens: 0, measuredCalls: 0, settledCalls: 1, pendingCalls: 0, complete: false,
    })
    expect(() => ledger.admit('session', 'goal', 'owner', 100)).toThrow('incomplete')
    const uncapped = ledger.admit('session', 'goal', 'owner')
    uncapped(usage)
    expect(() => failed(usage)).not.toThrow()
    expect(() => ledger.admit('session', 'goal', 'owner', 100)).toThrow('incomplete')
  } finally { ledger.close() }
})

test('receipt settlement is idempotent and rejects use after close', () => {
  const ledger = new GoalTokenLedger(':memory:')
  ledger.initialize('session', 'goal')
  const receipt = ledger.admit('session', 'goal', 'owner', 100)
  receipt(usage)
  receipt({ inputTokens: 90, outputTokens: 90 })
  ledger.close()
  expect(() => receipt(usage)).toThrow('closed')
  expect(() => ledger.inspect('session', 'goal')).toThrow('closed')
})

test('reopened pending calls fail closed for a different owner', () => {
  const temporary = temporaryLedger()
  let ledger = new GoalTokenLedger(temporary.path)
  try {
    ledger.initialize('session', 'goal')
    ledger.admit('session', 'goal', 'old-owner', 100)
    ledger.close()
    ledger = new GoalTokenLedger(temporary.path)
    expect(() => ledger.admit('session', 'goal', 'new-owner', 100)).toThrow('another process')
    expect(() => ledger.admit('session', 'goal', 'old-owner', 100)).not.toThrow()
  } finally {
    ledger.close()
    rmSync(temporary.directory, { recursive: true, force: true })
  }
})

test('incomplete baselines can be tracked uncapped but cannot admit capped work', () => {
  const ledger = new GoalTokenLedger(':memory:')
  try {
    ledger.initialize('session', 'goal', false)
    expect(ledger.inspect('session', 'goal')?.complete).toBe(false)
    expect(() => ledger.admit('session', 'goal', 'owner', 100)).toThrow('baseline')
    const receipt = ledger.admit('session', 'goal', 'owner')
    receipt(usage)
    expect(ledger.inspect('session', 'goal')).toMatchObject({ complete: false, measuredCalls: 1 })
  } finally { ledger.close() }
})

test('completed=false preserves known partial usage while remaining incomplete', () => {
  const ledger = new GoalTokenLedger(':memory:')
  try {
    ledger.initialize('session', 'goal')
    const receipt = ledger.admit('session', 'goal', 'owner', 100)
    receipt({ ...usage, cacheReadTokens: 2 }, false)
    expect(ledger.inspect('session', 'goal')).toEqual({
      inputTokens: 12, outputTokens: 5, measuredCalls: 0, settledCalls: 1, pendingCalls: 0, complete: false,
    })
    expect(() => ledger.admit('session', 'goal', 'owner', 100)).toThrow('incomplete')
    expect(() => ledger.admit('session', 'goal', 'owner')).not.toThrow()
  } finally { ledger.close() }
})

test('invalid usage, identifiers, and aggregate overflow fail without settling', () => {
  const ledger = new GoalTokenLedger(':memory:')
  try {
    ledger.initialize('session', 'goal')
    const invalid = ledger.admit('session', 'goal', 'owner', Number.MAX_SAFE_INTEGER)
    expect(() => invalid({ inputTokens: 1.5, outputTokens: 0 })).toThrow('input tokens')
    expect(ledger.inspect('session', 'goal')).toMatchObject({ pendingCalls: 0, settledCalls: 1, measuredCalls: 0, complete: false })
    expect(() => ledger.assertAdmission('session', 'goal', 'owner', Number.MAX_SAFE_INTEGER)).toThrow('incomplete')
    const overflow = ledger.admit('session', 'goal', 'owner')
    expect(() => overflow({ inputTokens: Number.MAX_SAFE_INTEGER, outputTokens: 1 })).toThrow('overflow')
    expect(ledger.inspect('session', 'goal')).toMatchObject({ pendingCalls: 0, settledCalls: 2, measuredCalls: 0, complete: false })
    expect(() => ledger.assertAdmission('session', 'goal', 'owner', Number.MAX_SAFE_INTEGER)).toThrow('incomplete')
    expect(() => ledger.initialize('bad\n', 'goal')).toThrow('sessionId')
    expect(() => ledger.admit('session', 'goal', 'owner', 0)).toThrow('maximum')
    expect(() => ledger.admit('x'.repeat(8193), 'goal', 'owner')).toThrow('sessionId')
  } finally { ledger.close() }
})

test('corrupt persisted rows fail closed', () => {
  const temporary = temporaryLedger()
  let ledger = new GoalTokenLedger(temporary.path)
  ledger.initialize('session', 'goal')
  ledger.close()
  const db = new Database(temporary.path)
  db.query(`INSERT INTO goal_token_calls(
    receipt_id,session_id,goal_id,owner_id,state,measured,input_tokens,cache_read_tokens,cache_creation_tokens,output_tokens
  ) VALUES('corrupt','session','goal','owner','settled',0,1,NULL,NULL,NULL)`).run()
  db.close()
  expect(() => new GoalTokenLedger(temporary.path)).toThrow('partial')
  rmSync(temporary.directory, { recursive: true, force: true })
})

test('unrelated corruption does not poison a healthy hot-path lookup', () => {
  const temporary = temporaryLedger()
  const ledger = new GoalTokenLedger(temporary.path)
  const db = new Database(temporary.path)
  try {
    ledger.initialize('healthy-session', 'healthy-goal')
    db.query(`INSERT INTO goal_token_calls(
      receipt_id,session_id,goal_id,owner_id,state,measured,input_tokens,cache_read_tokens,cache_creation_tokens,output_tokens
    ) VALUES('unrelated-corrupt','other-session','other-goal','owner','settled',0,1,NULL,NULL,NULL)`).run()
    expect(ledger.inspect('healthy-session', 'healthy-goal')).toEqual({
      inputTokens: 0, outputTokens: 0, measuredCalls: 0, settledCalls: 0, pendingCalls: 0, complete: true,
    })
    expect(() => ledger.admit('healthy-session', 'healthy-goal', 'owner', 100)).not.toThrow()
  } finally {
    db.close()
    ledger.close()
    rmSync(temporary.directory, { recursive: true, force: true })
  }
})

test('uncapped goals continue past two million tokens while explicit caps still reject', () => {
  const ledger = new GoalTokenLedger(':memory:')
  try {
    ledger.initialize('session', 'goal')
    ledger.admit('session', 'goal', 'owner')({ inputTokens: 2090726, outputTokens: 0 })
    expect(() => ledger.admit('session', 'goal', 'owner', 2000000)).toThrow('exhausted')
    expect(() => ledger.admit('session', 'goal', 'owner')({ inputTokens: 1, outputTokens: 1 })).not.toThrow()
  } finally { ledger.close() }
})
