// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ActivityChanges } from '../runtime/activityChanges.js'
import { parseScheduleTime } from './time.js'
import { cronTimezone, dayOffsets, wallInstants, wallTime, zoneFormatter } from './timezone.js'
import { Database } from 'bun:sqlite'
import { realpathSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'

export interface CronJobOptions {
  readonly deliver?: string
  readonly id: string
  readonly lastRunAt?: string
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly nextRunAt?: string
  readonly oneshot?: boolean
  readonly paused?: boolean
  readonly prompt: string
  /** Project directory that owns the job, so a listing can name the repo. */
  readonly projectRoot?: string
  readonly recipient?: string
  readonly timezone?: string
  readonly schedule?: string
  readonly workspaceId?: string
  readonly timeoutMs?: number
  readonly maxRetries?: number
  readonly targetSessionId?: string
  readonly stopCondition?: string
  readonly expiresAt?: string
  readonly maxRuns?: number
  readonly runsStarted?: number
  readonly maxModelCalls?: number
  readonly maxTotalTokens?: number
  readonly intervalSeconds?: number
  readonly missedRunPolicy?: 'coalesce' | 'skip'
  readonly misfireGraceSeconds?: number
}

/** A persisted recurring or one-shot agent prompt. */
export class CronJob {
  deliver: string
  readonly id: string
  lastRunAt: string | undefined
  metadata: Record<string, unknown>
  nextRunAt: string | undefined
  oneshot: boolean
  paused: boolean
  readonly prompt: string
  projectRoot: string | undefined
  recipient: string
  readonly timezone: string
  schedule: string
  workspaceId: string | undefined
  readonly timeoutMs: number | undefined
  readonly maxRetries: number | undefined
  readonly targetSessionId: string | undefined
  readonly stopCondition: string | undefined
  readonly expiresAt: string | undefined
  readonly maxRuns: number | undefined
  readonly runsStarted: number
  readonly maxModelCalls: number | undefined
  readonly maxTotalTokens: number | undefined
  readonly intervalSeconds: number | undefined
  readonly missedRunPolicy: 'coalesce' | 'skip'
  readonly misfireGraceSeconds: number

  constructor(options: CronJobOptions) {
    if (options.missedRunPolicy !== undefined && options.missedRunPolicy !== 'coalesce' && options.missedRunPolicy !== 'skip') throw new Error('missedRunPolicy must be coalesce or skip')
    this.missedRunPolicy = options.missedRunPolicy ?? 'coalesce'
    this.misfireGraceSeconds = boundedOptional(options.misfireGraceSeconds, 1, 86400, 'misfireGraceSeconds') ?? 300
    this.timeoutMs = boundedOptional(options.timeoutMs, 1, 3_600_000, 'timeoutMs')
    this.maxRetries = boundedOptional(options.maxRetries, 0, 10, 'maxRetries')
    if (options.targetSessionId !== undefined && (typeof options.targetSessionId !== 'string' || !/^[a-zA-Z0-9_-]{1,128}$/.test(options.targetSessionId))) throw new Error('Invalid target session ID')
    this.targetSessionId = options.targetSessionId
    if (options.stopCondition !== undefined && (typeof options.stopCondition !== 'string' || !options.stopCondition.trim() || options.stopCondition.length > 4000)) throw new Error('stopCondition must contain 1–4000 characters')
    this.stopCondition = options.stopCondition?.trim()
    if (this.stopCondition && !this.targetSessionId) throw new Error('Stop conditions require a session follow-up')
    this.expiresAt = options.expiresAt === undefined ? undefined : parseScheduleTime(options.expiresAt).toISOString()
    this.maxRuns = boundedOptional(options.maxRuns, 1, 10000, 'maxRuns')
    this.runsStarted = boundedOptional(options.runsStarted, 0, Number.MAX_SAFE_INTEGER, 'runsStarted') ?? 0
    this.maxModelCalls = boundedOptional(options.maxModelCalls, 1, 10000, 'maxModelCalls')
    this.maxTotalTokens = boundedOptional(options.maxTotalTokens, 1, Number.MAX_SAFE_INTEGER, 'maxTotalTokens')
    this.intervalSeconds = boundedOptional(options.intervalSeconds, 1, 86400, 'intervalSeconds')
    if (this.intervalSeconds !== undefined && (options.oneshot || options.schedule?.trim())) throw new Error('Interval schedules cannot also be cron or one-shot')
    this.timezone = cronTimezone(options.timezone)
    this.id = options.id
    this.prompt = options.prompt
    this.schedule = options.schedule ?? ''
    this.deliver = options.deliver ?? 'none'
    this.recipient = options.recipient ?? ''
    this.paused = options.paused ?? false
    this.oneshot = options.oneshot ?? false
    this.lastRunAt = options.lastRunAt
    this.nextRunAt = options.nextRunAt
    this.projectRoot = options.projectRoot
    this.workspaceId = options.workspaceId
    this.metadata = { ...(options.metadata ?? {}) }
  }

  toRecord(): Record<string, unknown> {
    return {
      id: this.id,
      prompt: this.prompt,
      schedule: this.schedule,
      timezone: this.timezone,
      deliver: this.deliver,
      recipient: this.recipient,
      paused: this.paused,
      oneshot: this.oneshot,
      last_run_at: this.lastRunAt ?? null,
      next_run_at: this.nextRunAt ?? null,
      project_root: this.projectRoot ?? null,
      workspace_id: this.workspaceId ?? null,
      metadata: { ...this.metadata },
      timeout_ms: this.timeoutMs ?? null,
      max_retries: this.maxRetries ?? null,
      target_session_id: this.targetSessionId ?? null,
      stop_condition: this.stopCondition ?? null,
      expires_at: this.expiresAt ?? null,
      max_runs: this.maxRuns ?? null,
      runs_started: this.runsStarted,
      max_model_calls: this.maxModelCalls ?? null,
      max_total_tokens: this.maxTotalTokens ?? null,
      interval_seconds: this.intervalSeconds ?? null,
      missed_run_policy: this.missedRunPolicy,
      misfire_grace_seconds: this.misfireGraceSeconds,
    }
  }

  static fromRecord(value: Record<string, unknown>): CronJob {
    const id = stringValue(value.id)
    const prompt = stringValue(value.prompt)
    if (!id || !prompt)
      throw new Error('Cron job records require id and prompt')
    return new CronJob({
      id,
      prompt,
      schedule: stringValue(value.schedule),
      timezone: value.timezone == null ? 'UTC' : value.timezone as string,
      deliver: stringValue(value.deliver) || 'none',
      recipient: stringValue(value.recipient),
      paused: value.paused === true,
      oneshot: value.oneshot === true,
      ...(value.missed_run_policy == null ? {} : { missedRunPolicy: value.missed_run_policy as 'coalesce' | 'skip' }),
      ...(value.misfire_grace_seconds == null ? {} : { misfireGraceSeconds: boundedOptional(value.misfire_grace_seconds, 1, 86400, 'misfire_grace_seconds')! }),
      ...(value.interval_seconds == null ? {} : { intervalSeconds: boundedOptional(value.interval_seconds, 1, 86400, 'interval_seconds')! }),
      ...(value.timeout_ms == null ? {} : { timeoutMs: boundedOptional(value.timeout_ms, 1, 3_600_000, 'timeout_ms')! }),
      ...(value.max_retries == null ? {} : { maxRetries: boundedOptional(value.max_retries, 0, 10, 'max_retries')! }),
      ...(value.target_session_id == null ? {} : { targetSessionId: value.target_session_id as string }),
      ...(value.stop_condition == null ? {} : { stopCondition: value.stop_condition as string }),
      ...(value.expires_at == null ? {} : { expiresAt: typeof value.expires_at === 'string' ? value.expires_at : (() => { throw new Error('expires_at must be an ISO timestamp') })() }),
      ...(value.max_runs == null ? {} : { maxRuns: boundedOptional(value.max_runs, 1, 10000, 'max_runs')! }),
      ...(value.runs_started == null ? {} : { runsStarted: boundedOptional(value.runs_started, 0, Number.MAX_SAFE_INTEGER, 'runs_started')! }),
      ...(value.max_model_calls == null ? {} : { maxModelCalls: boundedOptional(value.max_model_calls, 1, 10000, 'max_model_calls')! }),
      ...(value.max_total_tokens == null ? {} : { maxTotalTokens: boundedOptional(value.max_total_tokens, 1, Number.MAX_SAFE_INTEGER, 'max_total_tokens')! }),
      ...(nullableString(value.last_run_at)
        ? { lastRunAt: nullableString(value.last_run_at) as string }
        : {}),
      ...(nullableString(value.next_run_at)
        ? { nextRunAt: nullableString(value.next_run_at) as string }
        : {}),
      // Records written before jobs carried a project root stay loadable; the
      // field simply reads as undefined.
      ...(nullableString(value.project_root)
        ? { projectRoot: nullableString(value.project_root) as string }
        : {}),
      ...(nullableString(value.workspace_id)
        ? { workspaceId: nullableString(value.workspace_id) as string }
        : {}),
      ...(isRecord(value.metadata) ? { metadata: value.metadata } : {}),
    })
  }
}

export interface JobStoreOptions {
  /**
   * Project directory stamped onto jobs added without one. The store file is
   * shared by every daemon, so a job that does not name its repo cannot later
   * be attributed to one.
   */
  readonly projectRoot?: string
}

/**
 * JSON-backed persistence with an OS-released SQLite writer lock. Mutations
 * fail immediately on contention instead of blocking the daemon event loop.
 */
export class JobStore {
  readonly activityChanges = new ActivityChanges()
  readonly projectRoot: string | undefined

  constructor(readonly path: string, options: JobStoreOptions = {}) {
    this.projectRoot = options.projectRoot
    mkdirSync(dirname(path), { recursive: true })
    try {
      readFileSync(path, 'utf8')
    } catch (error) {
      if (!(error instanceof Error) || !('code' in error) || error.code !== 'ENOENT') throw error
      // Exclusive creation cannot overwrite a store another process created
      // between the failed read and this write.
      try { writeFileSync(path, '[]\n', { encoding: 'utf8', flag: 'wx', mode: 0o600 }) }
      catch (creationError) {
        if (!(creationError instanceof Error) || !('code' in creationError) || creationError.code !== 'EEXIST') throw creationError
      }
    }
  }

  add(job: CronJob): CronJob {
    return this.mutate(() => {
    job.projectRoot ??= this.projectRoot
    CronJob.fromRecord(job.toRecord())
    const records = this.load().filter((record) => record.id !== job.id)
    records.push(job.toRecord())
    this.save(records)
    return job
    })
  }

  get(jobId: string): CronJob | undefined {
    return this.listJobs().find((job) => job.id === jobId)
  }

  listJobs(): CronJob[] {
    return this.load().map(record => CronJob.fromRecord(record))
  }

  newId(): string {
    return crypto.randomUUID().replaceAll('-', '').slice(0, 12)
  }

  remove(jobId: string, expectedRevision?: string): boolean {
    return this.mutate(() => {
    const records = this.load()
    const existing = records.find(record => record.id === jobId)
    if (existing && expectedRevision !== undefined && Bun.hash(JSON.stringify(CronJob.fromRecord(existing).toRecord())).toString(16) !== expectedRevision) {
      throw new Error('Schedule changed; refresh before removing')
    }
    const filtered = records.filter((record) => record.id !== jobId)
    if (filtered.length === records.length) return false
    this.save(filtered)
    return true
    })
  }

  update(
    jobId: string,
    changes: Readonly<Record<string, unknown>>,
    expectedRevision?: string,
  ): CronJob | undefined {
    return this.mutate(() => {
    const records = this.load()
    const position = records.findIndex((record) => record.id === jobId)
    if (position < 0) return undefined
    const existing = records[position]
    if (!existing) return undefined
    if (expectedRevision !== undefined && Bun.hash(JSON.stringify(CronJob.fromRecord(existing).toRecord())).toString(16) !== expectedRevision) throw new Error('Schedule changed; refresh before editing')
    const updated = { ...existing, ...recordToSnakeCase(changes) }
    const job = CronJob.fromRecord(updated)
    records[position] = updated
    this.save(records)
    return job
    })
  }

  private mutate<T>(operation: () => T): T {
    // Canonicalize aliases so every writer uses the same lock. Never delete
    // this sidecar while clients are alive: SQLite releases locks on exit.
    const lock = new Database(`${realpathSync(this.path)}.writer.sqlite`, { create: true })
    try {
      lock.exec('PRAGMA busy_timeout = 0')
      return lock.transaction(operation).immediate()
    } finally { lock.close() }
  }

  private load(): Record<string, unknown>[] {
    const parsed: unknown = JSON.parse(readFileSync(this.path, 'utf8'))
    if (!Array.isArray(parsed)) throw new Error(`Invalid cron job store ${this.path}: expected an array`)
    const ids = new Set<string>()
    return parsed.map((record: unknown, index: number) => {
      if (!isRecord(record)) throw new Error(`Invalid cron job record ${index} in ${this.path}`)
      const job = CronJob.fromRecord(record)
      if (ids.has(job.id)) throw new Error(`Duplicate cron job ${job.id} in ${this.path}`)
      ids.add(job.id)
      return record
    })
  }

  private save(records: readonly Record<string, unknown>[]): void {
    const temporary = `${this.path}.${process.pid}.${crypto.randomUUID()}.tmp`
    try {
      writeFileSync(temporary, `${JSON.stringify(records, null, 2)}\n`, 'utf8')
      renameSync(temporary, this.path)
      this.activityChanges.notify()
    } catch (error) {
      rmSync(temporary, { force: true })
      throw error
    }
  }
}

/** Longest forward search window: one full leap cycle plus a margin day. */
const MAX_SEARCH_DAYS = 366 * 4 + 1

/**
 * Calculate the first UTC minute strictly after `now` matching a five-field cron
 * expression. Iterates days (not minutes) so sparse schedules such as Feb-29 are
 * found across non-leap years. Follows POSIX day-of-month/day-of-week semantics:
 * when both fields are restricted, a match in either is sufficient.
 */
export function nextFireAt(schedule: string, now = new Date(), timezone = 'UTC'): Date {
  const zone = cronTimezone(timezone)
  const formatter = zone === 'UTC' ? undefined : zoneFormatter(zone)
  const parts = schedule.trim().split(/\s+/)
  if (parts.length !== 5)
    throw new Error(
      `expected 5-field cron expression, got ${JSON.stringify(schedule)}`,
    )
  const daySpec = parts[2] ?? ''
  const weekDaySpec = parts[4] ?? ''
  const minutes = [...parseCronField(parts[0] ?? '', 0, 59)].sort(
    (left, right) => left - right,
  )
  const hours = [...parseCronField(parts[1] ?? '', 0, 23)].sort(
    (left, right) => left - right,
  )
  const day = parseCronField(daySpec, 1, 31)
  const month = parseCronField(parts[3] ?? '', 1, 12)
  // Day-of-week accepts 0-7 with 7 as a Sunday alias for 0.
  const weekDay = new Set(
    [...parseCronField(weekDaySpec, 0, 7)].map((value) =>
      value === 7 ? 0 : value,
    ),
  )
  const bothDaysRestricted = daySpec !== '*' && weekDaySpec !== '*'
  const dayMatches = (date: Date): boolean => {
    const domMatch = day.has(date.getUTCDate())
    const dowMatch = weekDay.has(date.getUTCDay())
    return bothDaysRestricted ? domMatch || dowMatch : domMatch && dowMatch
  }

  const earliest = new Date(now)
  earliest.setUTCSeconds(0, 0)
  earliest.setUTCMinutes(earliest.getUTCMinutes() + 1)
  const local = formatter ? wallTime(earliest, formatter) : earliest
  const cursor = new Date(
    Date.UTC(
      local.getUTCFullYear(),
      local.getUTCMonth(),
      local.getUTCDate(),
    ),
  )
  for (let offset = 0; offset <= MAX_SEARCH_DAYS; offset += 1) {
    if (month.has(cursor.getUTCMonth() + 1) && dayMatches(cursor)) {
      const offsets = formatter ? dayOffsets(cursor, formatter) : []
      let best: Date | undefined
      for (const hour of hours) {
        for (const minute of minutes) {
          const candidate = new Date(cursor)
          candidate.setUTCHours(hour, minute, 0, 0)
          if (!formatter) { if (candidate >= earliest) return candidate; continue }
          if (best && candidate.getTime() - Math.max(...offsets) >= best.getTime()) break
          for (const instant of wallInstants(candidate, formatter, offsets)) {
            if (instant >= earliest && (!best || instant < best)) best = instant
          }
        }
      }
      if (best) return best
    }
    cursor.setUTCDate(cursor.getUTCDate() + 1)
  }
  throw new Error(
    `no fire time found within four years for ${JSON.stringify(schedule)}`,
  )
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function nullableString(value: unknown): string | undefined {
  return typeof value === 'string' && value ? value : undefined
}

function parseCronField(spec: string, low: number, high: number): Set<number> {
  if (!spec) throw new Error('empty cron field')
  const values = new Set<number>()
  for (const part of spec.split(',')) {
    const [rangeSpec, stepSpec] = part.split('/', 2)
    const step = stepSpec === undefined ? 1 : integer(stepSpec)
    if (step <= 0) throw new Error(`invalid cron step ${JSON.stringify(part)}`)
    const [start, end] =
      rangeSpec === '*' ? [low, high] : parseRange(rangeSpec ?? '', low, high)
    for (let value = start; value <= end; value += step) values.add(value)
  }
  return values
}

function parseRange(spec: string, low: number, high: number): [number, number] {
  const segments = spec.split('-', 2)
  const start = integer(segments[0] ?? '')
  const end = segments.length === 1 ? start : integer(segments[1] ?? '')
  if (start < low || end > high || start > end)
    throw new Error(`cron value ${JSON.stringify(spec)} outside ${low}-${high}`)
  return [start, end]
}

function integer(value: string): number {
  if (!/^\d+$/.test(value))
    throw new Error(`invalid cron integer ${JSON.stringify(value)}`)
  return Number(value)
}

function recordToSnakeCase(
  changes: Readonly<Record<string, unknown>>,
): Record<string, unknown> {
  const aliases: Record<string, string> = {
    lastRunAt: 'last_run_at',
    nextRunAt: 'next_run_at',
    projectRoot: 'project_root',
    workspaceId: 'workspace_id',
    timeoutMs: 'timeout_ms',
    maxRetries: 'max_retries',
    targetSessionId: 'target_session_id',
    stopCondition: 'stop_condition',
    maxTotalTokens: 'max_total_tokens',
    expiresAt: 'expires_at',
    maxRuns: 'max_runs',
    runsStarted: 'runs_started',
    maxModelCalls: 'max_model_calls',
    intervalSeconds: 'interval_seconds',
    missedRunPolicy: 'missed_run_policy',
    misfireGraceSeconds: 'misfire_grace_seconds',
  }
  return Object.fromEntries(
    Object.entries(changes).map(([key, value]) => [aliases[key] ?? key, value]),
  )
}

/** Explicit operator resume acknowledges possible effects of an interrupted occurrence. */
export function resumedCronMetadata(job: CronJob): Record<string, unknown> {
  const { execution_receipt, execution_recovery_required: _recovery, followup_completion: _completion, ...remaining } = job.metadata
  return execution_receipt == null ? remaining : { ...remaining, previous_execution_receipt: execution_receipt, execution_reviewed_at: new Date().toISOString() }
}

function stringValue(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function boundedOptional(value: unknown, minimum: number, maximum: number, name: string): number | undefined {
  if (value === undefined) return undefined
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < minimum || value > maximum) throw new Error(`${name} must be an integer from ${minimum} to ${maximum}`)
  return value
}
