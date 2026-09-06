// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Database } from 'bun:sqlite'
import { chmodSync, mkdirSync } from 'node:fs'
import { dirname } from 'node:path'
import type { ModelCallReceipt } from '../llms/callBudget.js'

export type { ModelCallReceipt } from '../llms/callBudget.js'

const MAX_ID_LENGTH = 8192
const MAX_TOKEN = Number.MAX_SAFE_INTEGER

class ReceiptAccountingError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ReceiptAccountingError'
  }
}

export interface GoalTokenSummary {
  readonly inputTokens: number
  readonly outputTokens: number
  readonly measuredCalls: number
  readonly settledCalls: number
  readonly pendingCalls: number
  readonly complete: boolean
}

interface LedgerRow {
  session_id: string
  goal_id: string
  baseline_complete: number
}

interface CallRow {
  receipt_id: string
  session_id: string
  goal_id: string
  owner_id: string
  state: string
  measured: number
  input_tokens: number | null
  cache_read_tokens: number | null
  cache_creation_tokens: number | null
  output_tokens: number | null
}

interface StoredUsage {
  readonly inputTokens: number
  readonly cacheReadTokens: number
  readonly cacheCreationTokens: number
  readonly outputTokens: number
}

/** Durable, fail-closed token accounting for one goal's provider calls. */
export class GoalTokenLedger {
  private readonly db: Database
  private closed = false

  constructor(readonly path: string) {
    if (typeof path !== 'string' || !path || (path !== ':memory:' && path.includes('\0'))) {
      throw new Error('Invalid goal token ledger path')
    }
    if (path !== ':memory:') mkdirSync(dirname(path), { recursive: true, mode: 0o700 })
    this.db = new Database(path, { create: true, strict: true })
    try {
      if (path !== ':memory:') chmodSync(path, 0o600)
      this.db.exec(`PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;
        CREATE TABLE IF NOT EXISTS goal_token_ledgers (
          session_id TEXT NOT NULL,
          goal_id TEXT NOT NULL,
          baseline_complete INTEGER NOT NULL CHECK (baseline_complete IN (0, 1)),
          PRIMARY KEY (session_id, goal_id)
        ) STRICT;
        CREATE TABLE IF NOT EXISTS goal_token_calls (
          receipt_id TEXT PRIMARY KEY,
          session_id TEXT NOT NULL,
          goal_id TEXT NOT NULL,
          owner_id TEXT NOT NULL,
          state TEXT NOT NULL CHECK (state IN ('pending', 'settled')),
          measured INTEGER NOT NULL CHECK (measured IN (0, 1)),
          input_tokens INTEGER,
          cache_read_tokens INTEGER,
          cache_creation_tokens INTEGER,
          output_tokens INTEGER,
          FOREIGN KEY (session_id, goal_id) REFERENCES goal_token_ledgers(session_id, goal_id),
          CHECK (state = 'pending' AND measured = 0 AND input_tokens IS NULL AND cache_read_tokens IS NULL AND cache_creation_tokens IS NULL AND output_tokens IS NULL
            OR state = 'settled' AND (measured = 0 OR (input_tokens IS NOT NULL AND cache_read_tokens IS NOT NULL AND cache_creation_tokens IS NOT NULL AND output_tokens IS NOT NULL)))
        ) STRICT;
        CREATE INDEX IF NOT EXISTS goal_token_calls_lookup ON goal_token_calls(session_id, goal_id);
      `)
      this.validateSchema()
      this.validatePersistedRows()
    } catch (error) {
      this.closed = true
      this.db.close()
      throw error
    }
  }

  /** Create a ledger baseline exactly once; an incomplete baseline remains incomplete forever. */
  initialize(sessionId: string, goalId: string, baselineComplete = true): void {
    this.ensureOpen()
    validateId(sessionId, 'sessionId')
    validateId(goalId, 'goalId')
    if (typeof baselineComplete !== 'boolean') throw new Error('Invalid goal token baseline')
    this.db.transaction(() => {
      this.validateTargetRows(sessionId, goalId)
      const existing = this.db.query<LedgerRow, [string, string]>(
        'SELECT session_id,goal_id,baseline_complete FROM goal_token_ledgers WHERE session_id=? AND goal_id=?',
      ).get(sessionId, goalId)
      if (existing) {
        validateLedgerRow(existing)
        if (Boolean(existing.baseline_complete) !== baselineComplete) throw new Error('Goal token ledger already initialized')
        return
      }
      this.db.query('INSERT INTO goal_token_ledgers(session_id,goal_id,baseline_complete) VALUES(?,?,?)')
        .run(sessionId, goalId, baselineComplete ? 1 : 0)
    }).immediate()
  }

  inspect(sessionId: string, goalId: string): GoalTokenSummary | null {
    this.ensureOpen()
    validateId(sessionId, 'sessionId')
    validateId(goalId, 'goalId')
    return this.db.transaction(() => {
      this.validateTargetRows(sessionId, goalId)
      const ledger = this.db.query<LedgerRow, [string, string]>(
        'SELECT session_id,goal_id,baseline_complete FROM goal_token_ledgers WHERE session_id=? AND goal_id=?',
      ).get(sessionId, goalId)
      if (!ledger) return null
      validateLedgerRow(ledger)
      return summarize(ledger, this.readCalls(sessionId, goalId))
    }).deferred()
  }

  /** Admit synchronously persisted provider work and return its idempotent settlement receipt. */
  admit(sessionId: string, goalId: string, ownerId: string, maximum?: number): ModelCallReceipt {
    this.ensureOpen()
    validateId(sessionId, 'sessionId')
    validateId(goalId, 'goalId')
    validateId(ownerId, 'ownerId')
    validateMaximum(maximum)
    const receiptId = crypto.randomUUID()
    this.db.transaction(() => {
      this.checkAdmission(sessionId, goalId, ownerId, maximum)
      this.db.query(`INSERT INTO goal_token_calls(
        receipt_id,session_id,goal_id,owner_id,state,measured,input_tokens,cache_read_tokens,cache_creation_tokens,output_tokens
      ) VALUES(?,?,?,?, 'pending', 0, NULL, NULL, NULL, NULL)`).run(receiptId, sessionId, goalId, ownerId)
    }).immediate()
    let settled = false
    return (usage, completed = true) => {
      this.ensureOpen()
      if (settled) return
      let validUsage: StoredUsage | undefined
      try {
        if (completed !== undefined && typeof completed !== 'boolean') throw new ReceiptAccountingError('Invalid model call completion flag')
        validUsage = usage === undefined ? undefined : validateUsage(usage)
      } catch (error) {
        if (error instanceof ReceiptAccountingError) {
          this.settleUnknownReceipt(receiptId, sessionId, goalId)
          settled = true
        }
        throw error
      }
      try {
        this.db.transaction(() => {
          const row = this.db.query<CallRow, [string, string, string]>(
            'SELECT receipt_id,session_id,goal_id,owner_id,state,measured,input_tokens,cache_read_tokens,cache_creation_tokens,output_tokens FROM goal_token_calls WHERE receipt_id=? AND session_id=? AND goal_id=?',
          ).get(receiptId, sessionId, goalId)
          if (!row) throw new Error('Unknown goal token receipt')
          validateCallRow(row)
          if (row.state !== 'pending') return
          if (validUsage !== undefined) {
            const calls = this.readCalls(sessionId, goalId)
            const ledger = this.db.query<LedgerRow, [string, string]>(
              'SELECT session_id,goal_id,baseline_complete FROM goal_token_ledgers WHERE session_id=? AND goal_id=?',
            ).get(sessionId, goalId)
            if (!ledger) throw new Error('Goal token ledger is not initialized')
            validateLedgerRow(ledger)
            const summary = summarize(ledger, calls)
            let nextInput: number
            let nextOutput: number
            try {
              nextInput = safeAdd(summary.inputTokens, effectiveInput(validUsage), 'Goal token input usage overflow')
              nextOutput = safeAdd(summary.outputTokens, validUsage.outputTokens, 'Goal token output usage overflow')
              safeAdd(nextInput, nextOutput, 'Goal token usage overflow')
            } catch (error) {
              throw new ReceiptAccountingError(error instanceof Error ? error.message : 'Goal token usage overflow')
            }
            this.db.query(`UPDATE goal_token_calls SET state='settled',measured=?,input_tokens=?,cache_read_tokens=?,cache_creation_tokens=?,output_tokens=? WHERE receipt_id=? AND state='pending'`)
              .run(completed === false ? 0 : 1, validUsage.inputTokens, validUsage.cacheReadTokens, validUsage.cacheCreationTokens, validUsage.outputTokens, receiptId)
          } else {
            this.db.query(`UPDATE goal_token_calls SET state='settled',measured=0 WHERE receipt_id=? AND state='pending'`).run(receiptId)
          }
        }).immediate()
      } catch (error) {
        if (error instanceof ReceiptAccountingError) {
          this.settleUnknownReceipt(receiptId, sessionId, goalId)
          settled = true
        }
        throw error
      }
      settled = true
    }
  }

  private settleUnknownReceipt(receiptId: string, sessionId: string, goalId: string): void {
    this.db.transaction(() => {
      const row = this.db.query<CallRow, [string, string, string]>(
        'SELECT receipt_id,session_id,goal_id,owner_id,state,measured,input_tokens,cache_read_tokens,cache_creation_tokens,output_tokens FROM goal_token_calls WHERE receipt_id=? AND session_id=? AND goal_id=?',
      ).get(receiptId, sessionId, goalId)
      if (!row) throw new Error('Unknown goal token receipt')
      validateCallRow(row)
      if (row.state === 'pending') {
        this.db.query(`UPDATE goal_token_calls SET state='settled',measured=0,input_tokens=NULL,cache_read_tokens=NULL,cache_creation_tokens=NULL,output_tokens=NULL WHERE receipt_id=? AND state='pending'`).run(receiptId)
      }
    }).immediate()
  }

  /** Pure durable preflight shared by hosts and the receipt-producing admission path. */
  assertAdmission(sessionId: string, goalId: string, ownerId: string, maximum?: number): void {
    this.ensureOpen()
    validateId(sessionId, 'sessionId')
    validateId(goalId, 'goalId')
    validateId(ownerId, 'ownerId')
    validateMaximum(maximum)
    this.db.transaction(() => {
      this.checkAdmission(sessionId, goalId, ownerId, maximum)
    }).deferred()
  }

  private checkAdmission(sessionId: string, goalId: string, ownerId: string, maximum: number | undefined): void {
    const ledger = this.db.query<LedgerRow, [string, string]>(
      'SELECT session_id,goal_id,baseline_complete FROM goal_token_ledgers WHERE session_id=? AND goal_id=?',
    ).get(sessionId, goalId)
    if (!ledger) throw new Error('Goal token ledger is not initialized')
    validateLedgerRow(ledger)
    const calls = this.readCalls(sessionId, goalId)
    if (maximum !== undefined && calls.some(row => row.state === 'pending' && row.owner_id !== ownerId)) {
      throw new Error('Goal token ledger has a pending call owned by another process')
    }
    if (maximum !== undefined && calls.some(row => row.state === 'settled' && row.measured === 0)) {
      throw new Error('Goal token usage is incomplete; further calls are blocked')
    }
    if (!ledger.baseline_complete && maximum !== undefined) {
      throw new Error('Goal token baseline is incomplete; capped admission is blocked')
    }
    const summary = summarize(ledger, calls)
    if (maximum !== undefined && safeAdd(summary.inputTokens, summary.outputTokens, 'Goal token usage overflow') >= maximum) {
      throw new Error(`Goal token budget exhausted (${summary.inputTokens + summary.outputTokens}/${maximum})`)
    }
  }

  close(): void {
    if (this.closed) return
    this.closed = true
    this.db.close()
  }

  private ensureOpen(): void {
    if (this.closed) throw new Error('Goal token ledger is closed')
  }

  private validateSchema(): void {
    const expected = new Map<string, readonly string[]>([
      ['goal_token_ledgers', ['session_id', 'goal_id', 'baseline_complete']],
      ['goal_token_calls', ['receipt_id', 'session_id', 'goal_id', 'owner_id', 'state', 'measured', 'input_tokens', 'cache_read_tokens', 'cache_creation_tokens', 'output_tokens']],
    ])
    for (const [table, columns] of expected) {
      const actual = this.db.query<{ name: string }, []>(`PRAGMA table_info(${table})`).all().map(row => row.name)
      if (actual.length !== columns.length || actual.some((name, index) => name !== columns[index])) throw new Error(`Invalid goal token ledger schema for ${table}`)
    }
  }

  private readCalls(sessionId?: string, goalId?: string): CallRow[] {
    const rows = sessionId === undefined || goalId === undefined
      ? this.db.query<CallRow, []>('SELECT receipt_id,session_id,goal_id,owner_id,state,measured,input_tokens,cache_read_tokens,cache_creation_tokens,output_tokens FROM goal_token_calls').all()
      : this.db.query<CallRow, [string, string]>('SELECT receipt_id,session_id,goal_id,owner_id,state,measured,input_tokens,cache_read_tokens,cache_creation_tokens,output_tokens FROM goal_token_calls WHERE session_id=? AND goal_id=?').all(sessionId, goalId)
    rows.forEach(validateCallRow)
    return rows
  }

  private validateTargetRows(sessionId: string, goalId: string): void {
    const ledger = this.db.query<LedgerRow, [string, string]>(
      'SELECT session_id,goal_id,baseline_complete FROM goal_token_ledgers WHERE session_id=? AND goal_id=?',
    ).get(sessionId, goalId)
    const calls = this.readCalls(sessionId, goalId)
    if (!ledger) {
      if (calls.length > 0) throw new Error('Goal token call has no ledger')
      return
    }
    validateLedgerRow(ledger)
    summarize(ledger, calls)
  }

  private validatePersistedRows(): void {
    const ledgers = this.db.query<LedgerRow, []>('SELECT session_id,goal_id,baseline_complete FROM goal_token_ledgers').all()
    ledgers.forEach(validateLedgerRow)
    const calls = this.readCalls()
    const ledgerKeys = new Set(ledgers.map(row => `${row.session_id}\u0000${row.goal_id}`))
    for (const row of calls) {
      if (!ledgerKeys.has(`${row.session_id}\u0000${row.goal_id}`)) throw new Error('Goal token call has no ledger')
    }
    for (const ledger of ledgers) summarize(ledger, calls.filter(row => row.session_id === ledger.session_id && row.goal_id === ledger.goal_id))
  }
}

function summarize(ledger: LedgerRow, calls: readonly CallRow[]): GoalTokenSummary {
  let inputTokens = 0
  let outputTokens = 0
  let measuredCalls = 0
  let settledCalls = 0
  let pendingCalls = 0
  for (const row of calls) {
    validateCallRow(row)
    if (row.state === 'pending') {
      pendingCalls++
      continue
    }
    settledCalls++
    if (row.measured === 1) measuredCalls++
    const usage = rowUsage(row)
    if (usage !== undefined) {
      inputTokens = safeAdd(inputTokens, effectiveInput(usage), 'Goal token input usage overflow')
      outputTokens = safeAdd(outputTokens, usage.outputTokens, 'Goal token output usage overflow')
    }
    safeAdd(inputTokens, outputTokens, 'Goal token usage overflow')
  }
  return {
    inputTokens,
    outputTokens,
    measuredCalls,
    settledCalls,
    pendingCalls,
    complete: ledger.baseline_complete === 1 && pendingCalls === 0 && measuredCalls === settledCalls,
  }
}

function rowUsage(row: CallRow): StoredUsage | undefined {
  if (row.input_tokens === null && row.cache_read_tokens === null && row.cache_creation_tokens === null && row.output_tokens === null) return undefined
  if (row.input_tokens === null || row.cache_read_tokens === null || row.cache_creation_tokens === null || row.output_tokens === null) throw new Error('Invalid partial goal token call')
  return {
    inputTokens: row.input_tokens,
    cacheReadTokens: row.cache_read_tokens,
    cacheCreationTokens: row.cache_creation_tokens,
    outputTokens: row.output_tokens,
  }
}

function effectiveInput(usage: StoredUsage): number {
  return safeAdd(safeAdd(usage.inputTokens, usage.cacheReadTokens, 'Goal token input usage overflow'), usage.cacheCreationTokens, 'Goal token input usage overflow')
}

function validateLedgerRow(row: LedgerRow): void {
  validateId(row.session_id, 'stored session id')
  validateId(row.goal_id, 'stored goal id')
  if (!Number.isInteger(row.baseline_complete) || (row.baseline_complete !== 0 && row.baseline_complete !== 1)) throw new Error('Invalid stored goal token baseline')
}

function validateCallRow(row: CallRow): void {
  validateId(row.receipt_id, 'stored receipt id')
  validateId(row.session_id, 'stored session id')
  validateId(row.goal_id, 'stored goal id')
  validateId(row.owner_id, 'stored owner id')
  if (row.state !== 'pending' && row.state !== 'settled') throw new Error('Invalid stored goal token state')
  if (row.measured !== 0 && row.measured !== 1) throw new Error('Invalid stored goal token measured flag')
  if (row.state === 'pending') {
    if (row.measured !== 0 || row.input_tokens !== null || row.cache_read_tokens !== null || row.cache_creation_tokens !== null || row.output_tokens !== null) throw new Error('Invalid pending goal token call')
    return
  }
  const usage = rowUsage(row)
  if (row.measured === 1 && usage === undefined) throw new Error('Invalid measured goal token call')
  if (usage !== undefined) {
    validateToken(usage.inputTokens, 'stored input tokens')
    validateToken(usage.cacheReadTokens, 'stored cache-read tokens')
    validateToken(usage.cacheCreationTokens, 'stored cache-creation tokens')
    validateToken(usage.outputTokens, 'stored output tokens')
    effectiveInput(usage)
  }
}

function validateUsage(usage: { readonly inputTokens: number; readonly outputTokens: number; readonly cacheReadTokens?: number; readonly cacheCreationTokens?: number }): StoredUsage {
  try {
    if (!usage || typeof usage !== 'object') throw new Error('Invalid model call usage')
    const inputTokens = validateToken(usage.inputTokens, 'input tokens')
    const outputTokens = validateToken(usage.outputTokens, 'output tokens')
    const cacheReadTokens = validateToken(usage.cacheReadTokens ?? 0, 'cache-read tokens')
    const cacheCreationTokens = validateToken(usage.cacheCreationTokens ?? 0, 'cache-creation tokens')
    const result = { inputTokens, cacheReadTokens, cacheCreationTokens, outputTokens }
    effectiveInput(result)
    return result
  } catch (error) {
    throw new ReceiptAccountingError(error instanceof Error ? error.message : 'Invalid model call usage')
  }
}

function validateToken(value: number | null | undefined, name: string): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0 || value > MAX_TOKEN) throw new Error(`Invalid ${name}`)
  return value
}

function validateMaximum(maximum: number | undefined): void {
  if (maximum !== undefined) validateToken(maximum, 'maximum token budget')
  if (maximum === 0) throw new Error('Invalid maximum token budget')
}

function validateId(value: unknown, name: string): asserts value is string {
  if (typeof value !== 'string' || !value || value.length > MAX_ID_LENGTH || value.trim() !== value || /[\u0000-\u001f\u007f]/u.test(value)) throw new Error(`Invalid ${name}`)
}

function safeAdd(left: number, right: number, message: string): number {
  const result = left + right
  if (!Number.isSafeInteger(result) || result < 0) throw new Error(message)
  return result
}
