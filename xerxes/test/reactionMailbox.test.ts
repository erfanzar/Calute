// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { Database } from 'bun:sqlite'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { ReactionMailbox } from '../src/runtime/reactionMailbox.js'

const policy = { owner: 'owner', runId: 'watch', expiresAt: 10_000, maxReactions: 2, maxDurationMs: 1000 }

test('bursts coalesce, new events wait for the active claim, and attempts are bounded', () => {
  const mailbox = new ReactionMailbox(':memory:', () => 100)
  try {
    mailbox.configure(policy)
    expect(mailbox.claim('owner')).toBeUndefined()
    mailbox.offer('owner', 'watch', 1)
    mailbox.offer('owner', 'watch', 5)
    const first = mailbox.claim('owner')!
    expect(first).toMatchObject({ fromSequence: 1, throughSequence: 5, deadline: 1100 })
    expect(mailbox.isAuthorized(first)).toBe(true)
    mailbox.offer('owner', 'watch', 9)
    expect(mailbox.claim('owner')).toBeUndefined()
    mailbox.settle(first, 'completed')
    const second = mailbox.claim('owner')!
    expect(second).toMatchObject({ fromSequence: 6, throughSequence: 9 })
    mailbox.settle(second, 'failed', 'provider failure')
    expect(mailbox.offer('owner', 'watch', 10)).toBe(false)
    expect(mailbox.claim('owner')).toBeUndefined()
  } finally { mailbox.close() }
})

test('cancellation revokes authority without releasing an unsettled executor', () => {
  let now = 100
  const mailbox = new ReactionMailbox(':memory:', () => now)
  try {
    mailbox.configure(policy)
    mailbox.configure({ ...policy, runId: 'second' })
    mailbox.offer('owner', 'watch', 1)
    const claim = mailbox.claim('owner')!
    now = 2000
    expect(mailbox.isAuthorized(claim)).toBe(false)
    mailbox.offer('owner', 'second', 1)
    expect(mailbox.claim('owner')).toBeUndefined()
    mailbox.cancel('owner')
    mailbox.settle(claim, 'interrupted')
    expect(mailbox.claim('owner')).toBeUndefined()
    expect(mailbox.offer('owner', 'second', 2)).toBe(false)
    expect(() => mailbox.settle(claim, 'completed')).toThrow('differently')
  } finally { mailbox.close() }
})

test('reopen preserves claims and two connections cannot admit overlapping owner work', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-mailbox-'))
  const path = join(directory, 'mailbox.sqlite')
  let first = new ReactionMailbox(path, () => 100)
  const second = new ReactionMailbox(path, () => 100)
  try {
    first.configure(policy)
    first.offer('owner', 'watch', 2)
    const claim = first.claim('owner')!
    expect(second.claim('owner')).toBeUndefined()
    first.close()
    first = new ReactionMailbox(path, () => 5000)
    expect(first.unresolved('owner')).toEqual([claim])
    expect(first.claim('owner')).toBeUndefined()
    expect(first.unresolved('other')).toEqual([])
    expect(first.offer('other', 'watch', 3)).toBe(false)
    expect(() => first.settle({ ...claim, owner: 'other' }, 'completed')).toThrow('Unknown')
    first.settle(claim, 'interrupted', 'executor confirmed stopped')
    first.offer('owner', 'watch', 3)
    expect(first.claim('owner')?.fromSequence).toBe(3)
  } finally { first.close(); second.close(); rmSync(directory, { recursive: true, force: true }) }
})

test('reaction health distinguishes queued, active, unsettled cancellation and exhausted budgets', () => {
  let now = 100
  const mailbox = new ReactionMailbox(':memory:', () => now)
  try {
    mailbox.configure({ ...policy, maxReactions: 1 })
    expect(mailbox.inspect('other', 'watch')).toBeUndefined()
    expect(mailbox.inspect('owner', 'watch')?.state).toBe('waiting')
    mailbox.offer('owner', 'watch', 2)
    expect(mailbox.inspect('owner', 'watch')).toMatchObject({ state: 'queued', pendingEvents: 2, attempts: 0 })
    const claim = mailbox.claim('owner')!
    expect(mailbox.inspect('owner', 'watch')).toMatchObject({ state: 'running', activeClaimId: claim.id, attempts: 1, pendingEvents: 0 })
    now = 2000
    expect(mailbox.inspect('owner', 'watch')?.state).toBe('awaiting-cleanup')
    mailbox.settle(claim, 'failed', 'provider unavailable')
    expect(mailbox.inspect('owner', 'watch')).toMatchObject({ state: 'exhausted', activeClaimId: null, lastOutcome: 'failed', lastError: 'provider unavailable' })
    mailbox.configure({ ...policy, runId: 'another' })
    mailbox.offer('owner', 'another', 1)
    const next = mailbox.claim('owner')!
    mailbox.cancel('owner', 'another')
    expect(mailbox.inspect('owner', 'another')?.state).toBe('cancelling')
    mailbox.settle(next, 'cancelled')
    expect(mailbox.inspect('owner', 'another')?.state).toBe('cancelled')
  } finally { mailbox.close() }
})

test('restart records a dead executor as interrupted and revokes all its session grants', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-reaction-recovery-'))
  const path = join(directory, 'mailbox.sqlite')
  let mailbox = new ReactionMailbox(path, () => 100, { pid: 900001, isAlive: () => true })
  try {
    mailbox.configure(policy)
    mailbox.configure({ ...policy, runId: 'second' })
    mailbox.offer('owner', 'watch', 1)
    const claim = mailbox.claim('owner')!
    mailbox.offer('owner', 'second', 2)
    mailbox.close()
    mailbox = new ReactionMailbox(path, () => 200, { isAlive: () => false })
    expect(mailbox.unresolved('owner')).toEqual([])
    expect(mailbox.inspect('owner', 'watch')).toMatchObject({ state: 'cancelled', lastOutcome: 'interrupted', attempts: 1 })
    expect(mailbox.inspect('owner', 'watch')?.lastError).toContain('effects may be incomplete')
    expect(mailbox.inspect('owner', 'second')?.state).toBe('cancelled')
    expect(mailbox.claim('owner')).toBeUndefined()
    expect(() => mailbox.settle(claim, 'completed')).toThrow('differently')
  } finally { mailbox.close(); rmSync(directory, { recursive: true, force: true }) }
})

test('live or reused executor PIDs remain fenced even after the deadline', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-reaction-live-'))
  const path = join(directory, 'mailbox.sqlite')
  const first = new ReactionMailbox(path, () => 100, { pid: 900002, isAlive: () => true })
  try {
    first.configure(policy)
    first.offer('owner', 'watch', 1)
    const claim = first.claim('owner')!
    const next = new ReactionMailbox(path, () => 5000, { isAlive: () => true })
    try {
      expect(next.unresolved('owner')).toEqual([claim])
      expect(next.inspect('owner', 'watch')?.state).toBe('awaiting-cleanup')
      expect(next.claim('owner')).toBeUndefined()
    } finally { next.close() }
  } finally { first.close(); rmSync(directory, { recursive: true, force: true }) }
})

test('legacy claims without executor identity remain unresolved after schema migration', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-reaction-legacy-'))
  const path = join(directory, 'mailbox.sqlite')
  const legacy = new Database(path)
  legacy.exec(`CREATE TABLE reaction_claims (
    id TEXT PRIMARY KEY, owner TEXT NOT NULL, run_id TEXT NOT NULL,
    first INTEGER NOT NULL, last INTEGER NOT NULL, deadline INTEGER NOT NULL,
    outcome TEXT NOT NULL, error TEXT);
    INSERT INTO reaction_claims VALUES('old','owner','watch',1,1,50,'claimed',NULL);`)
  legacy.close()
  const mailbox = new ReactionMailbox(path, () => 100, { isAlive: () => false })
  try {
    mailbox.configure(policy)
    mailbox.offer('owner', 'watch', 2)
    expect(mailbox.unresolved('owner')[0]?.id).toBe('old')
    expect(mailbox.inspect('owner', 'watch')?.state).toBe('awaiting-cleanup')
    expect(mailbox.claim('owner')).toBeUndefined()
  } finally { mailbox.close(); rmSync(directory, { recursive: true, force: true }) }
})

test('usage is charged once per settled claim and missing observations remain incomplete', () => {
  const mailbox = new ReactionMailbox(':memory:', () => 100)
  try {
    mailbox.configure(policy)
    mailbox.offer('owner', 'watch', 1)
    const first = mailbox.claim('owner')!
    expect(mailbox.inspect('owner', 'watch')?.usage.complete).toBe(false)
    mailbox.settle(first, 'completed', undefined, { inputTokens: 120, outputTokens: 30, complete: true })
    mailbox.settle(first, 'completed', undefined, { inputTokens: 120, outputTokens: 30, complete: true })
    expect(mailbox.inspect('owner', 'watch')?.usage).toEqual({ inputTokens: 120, outputTokens: 30, complete: true })
    mailbox.offer('owner', 'watch', 2)
    const second = mailbox.claim('owner')!
    expect(() => mailbox.settle(second, 'completed', undefined, { inputTokens: -1, outputTokens: 0, complete: true })).toThrow('usage')
    mailbox.settle(second, 'failed', 'no usage returned')
    expect(mailbox.inspect('owner', 'watch')?.usage).toEqual({ inputTokens: 120, outputTokens: 30, complete: false })
  } finally { mailbox.close() }
})

test('live usage checkpoints are monotonic, claim scoped, and preserved by unknown settlement', () => {
  const mailbox = new ReactionMailbox(':memory:', () => 100)
  try {
    mailbox.configure(policy)
    mailbox.offer('owner', 'watch', 1)
    const claim = mailbox.claim('owner')!
    const usage = { inputTokens: 12, outputTokens: 3, complete: true }
    mailbox.checkpointUsage(claim, usage)
    mailbox.checkpointUsage(claim, usage)
    expect(mailbox.inspect('owner', 'watch')?.usage).toEqual({ ...usage, complete: false })
    expect(() => mailbox.checkpointUsage({ ...claim, owner: 'other' }, usage)).toThrow('claim')
    expect(() => mailbox.checkpointUsage({ ...claim, throughSequence: 2 }, usage)).toThrow('claim')
    expect(() => mailbox.checkpointUsage(claim, { ...usage, inputTokens: 1 })).toThrow('decreasing')
    expect(() => mailbox.settle(claim, 'completed', undefined, { ...usage, inputTokens: 1 })).toThrow('decrease')
    mailbox.cancel('owner', 'watch')
    mailbox.checkpointUsage(claim, { ...usage, inputTokens: 15 })
    mailbox.settle(claim, 'cancelled', 'executor ended without final counters')
    expect(mailbox.inspect('owner', 'watch')?.usage).toEqual({ inputTokens: 15, outputTokens: 3, complete: false })
    expect(() => mailbox.checkpointUsage(claim, usage)).toThrow('claim')
  } finally { mailbox.close() }
})

test('abrupt executor death preserves the authoritative reaction aggregate without replay', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-mailbox-crash-'))
  const path = join(directory, 'mailbox.sqlite')
  const script = `
    import { ReactionMailbox } from ${JSON.stringify(join(import.meta.dir, '../src/runtime/reactionMailbox.ts'))};
    const mailbox = new ReactionMailbox(${JSON.stringify(path)});
    mailbox.configure({owner:'owner',runId:'watch',expiresAt:Date.now()+60000,maxReactions:2,maxDurationMs:5000});
    mailbox.offer('owner','watch',1);
    const claim = mailbox.claim('owner');
    mailbox.checkpointUsage(claim,{inputTokens:23,outputTokens:7,complete:true});
    console.log('checkpoint persisted');
    setInterval(() => {},1000);
  `
  const child = Bun.spawn([process.execPath, '-e', script], { stdout: 'pipe', stderr: 'pipe' })
  let mailbox: ReactionMailbox | undefined
  try {
    const reader = child.stdout.getReader()
    const ready = await reader.read()
    reader.releaseLock()
    expect(new TextDecoder().decode(ready.value)).toContain('checkpoint persisted')
    child.kill(9); await child.exited
    mailbox = new ReactionMailbox(path)
    expect(mailbox.inspect('owner', 'watch')).toMatchObject({ state: 'cancelled', lastOutcome: 'interrupted', usage: { inputTokens: 23, outputTokens: 7, complete: false } })
    expect(mailbox.claim('owner')).toBeUndefined()
    mailbox.close(); mailbox = new ReactionMailbox(path)
    expect(mailbox.inspect('owner', 'watch')?.usage.inputTokens).toBe(23)
  } finally { child.kill(9); await child.exited; mailbox?.close(); rmSync(directory, { recursive: true, force: true }) }
}, 10000)

test.each(['completed', 'failed', 'cancelled', 'interrupted'] as const)('settled %s one-shot reactions release grant capacity without deleting evidence', outcome => {
  const mailbox = new ReactionMailbox(':memory:', () => 100)
  try {
    for (let index = 0; index < 20; index++) {
      const runId = `completion-${index}`
      mailbox.configure({ ...policy, runId, maxReactions: 1 })
      mailbox.offer('owner', runId, 1)
      const claim = mailbox.claim('owner')!
      expect(claim.runId).toBe(runId)
      mailbox.settle(claim, outcome)
    }
    expect(mailbox.inspect('owner', 'completion-0')).toMatchObject({ state: 'exhausted', attempts: 1, lastOutcome: outcome })
    expect(mailbox.claim('owner')).toBeUndefined()
  } finally { mailbox.close() }
})

test.each(['live', 'cancelled', 'expired'] as const)('unsettled %s execution still occupies grant capacity', state => {
  let now = 100
  const mailbox = new ReactionMailbox(':memory:', () => now)
  try {
    mailbox.configure({ ...policy, runId: 'active', maxReactions: 1, expiresAt: 200 })
    mailbox.offer('owner', 'active', 1)
    const claim = mailbox.claim('owner')!
    if (state === 'cancelled') mailbox.cancel('owner', 'active')
    if (state === 'expired') now = 300
    for (let index = 0; index < 15; index++) mailbox.configure({ ...policy, runId: `waiting-${index}` })
    expect(() => mailbox.configure({ ...policy, runId: 'overflow' })).toThrow('limit reached')
    expect(mailbox.unresolved('owner')).toHaveLength(1)
    mailbox.settle(claim, 'cancelled')
    expect(() => mailbox.configure({ ...policy, runId: 'replacement' })).not.toThrow()
    expect(mailbox.inspect('owner', 'active')?.lastOutcome).toBe('cancelled')
  } finally { mailbox.close() }
})

test.each(['exhausted', 'unknown'] as const)('persistent reaction token threshold rejects %s usage without starving another watch', kind => {
  const directory = mkdtempSync(join(tmpdir(), 'reaction-token-limit-'))
  const path = join(directory, 'mailbox.sqlite')
  let mailbox = new ReactionMailbox(path, () => 100)
  try {
    mailbox.configure({ ...policy, maxTotalTokens: 10 })
    mailbox.offer('owner', 'watch', 1)
    const claim = mailbox.claim('owner')!
    expect(claim.tokenBudget).toEqual({ maximum: 10, priorTokens: 0, priorComplete: true })
    mailbox.settle(claim, 'completed', undefined, { inputTokens: 7, outputTokens: 3, complete: kind === 'exhausted' })
    mailbox.close()
    mailbox = new ReactionMailbox(path, () => 100)
    mailbox.offer('owner', 'watch', 2)
    expect(mailbox.claim('owner')).toBeUndefined()
    expect(mailbox.inspect('owner', 'watch')).toMatchObject({ state: 'exhausted', attempts: 1, pendingEvents: 1, tokenBudget: { maximum: 10, blocked: true } })
    mailbox.configure({ ...policy, runId: 'other' })
    mailbox.offer('owner', 'other', 1)
    expect(mailbox.claim('owner')?.runId).toBe('other')
  } finally { mailbox.close(); rmSync(directory, { recursive: true, force: true }) }
})

test('reaction claims carry measured prior consumption for provider admission', () => {
  const mailbox = new ReactionMailbox(':memory:', () => 100)
  try {
    mailbox.configure({ ...policy, maxTotalTokens: 20 })
    mailbox.offer('owner', 'watch', 1)
    mailbox.settle(mailbox.claim('owner')!, 'completed', undefined, { inputTokens: 7, outputTokens: 3, complete: true })
    mailbox.offer('owner', 'watch', 2)
    expect(mailbox.claim('owner')?.tokenBudget).toEqual({ maximum: 20, priorTokens: 10, priorComplete: true })
  } finally { mailbox.close() }
})

test('token-exhausted policies release capacity while retaining accounting', () => {
  const mailbox = new ReactionMailbox(':memory:', () => 100)
  try {
    for (let index = 0; index < 20; index++) {
      const runId = `watch-${index}`
      mailbox.configure({ ...policy, runId, maxTotalTokens: 1 })
      mailbox.offer('owner', runId, 1)
      const claim = mailbox.claim('owner')!
      mailbox.settle(claim, 'completed', undefined, { inputTokens: 1, outputTokens: 0, complete: true })
    }
    expect(mailbox.inspect('owner', 'watch-0')).toMatchObject({ usage: { inputTokens: 1, complete: true }, tokenBudget: { blocked: true } })
  } finally { mailbox.close() }
})

test('guarded policy edits preserve spend and cursors, reject active/stale edits, and survive restart', () => {
  const directory = mkdtempSync(join(tmpdir(), 'reaction-policy-edit-'))
  const path = join(directory, 'mailbox.sqlite')
  let mailbox = new ReactionMailbox(path, () => 100)
  try {
    mailbox.configure({ ...policy, maxTotalTokens: 10 })
    const initial = mailbox.inspect('owner', 'watch')!.policy!
    mailbox.offer('owner', 'watch', 1)
    expect(() => mailbox.updatePolicy('owner', 'watch', initial)).toThrow('changed')
    const claim = mailbox.claim('owner')!
    expect(() => mailbox.updatePolicy('owner', 'watch', mailbox.inspect('owner', 'watch')!.policy!)).toThrow('active reaction')
    mailbox.settle(claim, 'completed', undefined, { inputTokens: 7, outputTokens: 3, complete: true })
    mailbox.offer('owner', 'watch', 2)
    const before = mailbox.inspect('owner', 'watch')!
    expect(() => mailbox.updatePolicy('other', 'watch', before.policy!)).toThrow('Unknown')
    mailbox.updatePolicy('owner', 'watch', { ...before.policy!, maxReactions: 3, maxTotalTokens: 30 })
    expect(() => mailbox.updatePolicy('owner', 'watch', before.policy!)).toThrow('changed')
    mailbox.close(); mailbox = new ReactionMailbox(path, () => 100)
    expect(mailbox.inspect('owner', 'watch')).toMatchObject({ attempts: 1, pendingEvents: 1, usage: { inputTokens: 7, outputTokens: 3 }, policy: { maxReactions: 3, maxTotalTokens: 30 } })
    const next = mailbox.claim('owner')!
    expect(next).toMatchObject({ fromSequence: 2, throughSequence: 2, tokenBudget: { priorTokens: 10, maximum: 30 } })
    mailbox.settle(next, 'completed', undefined, { inputTokens: 2, outputTokens: 1, complete: true })
    mailbox.cancel('owner', 'watch')
    expect(() => mailbox.updatePolicy('owner', 'watch', mailbox.inspect('owner', 'watch')!.policy!)).toThrow('Cancelled')
  } finally { mailbox.close(); rmSync(directory, { recursive: true, force: true }) }
})
