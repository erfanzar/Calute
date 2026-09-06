// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { daemonPaths, resolveProjectDirectory } from '../daemon/paths.js'
import { requestDaemonControl } from '../daemon/controlClient.js'
import type { JsonRpcPayload } from '../protocol/jsonRpc.js'

export type ScheduleCommandAction = 'create' | 'disable' | 'enable' | 'remove' | 'fire' | 'inspect' | 'cancel' | 'list'

export interface ScheduleCommandRequest {
  (method: string, params: JsonRpcPayload): Promise<JsonRpcPayload>
}

export interface ScheduleCommandOptions {
  readonly action: ScheduleCommandAction
  readonly id?: string
  readonly owner?: string
  readonly schedule?: string
  readonly objective?: string
  readonly deliveryId?: string
  readonly directory?: string
  readonly projectDirectory?: string
  readonly socketPath?: string
  readonly timezone?: string
  readonly paused?: boolean
  readonly signal?: AbortSignal
  readonly request?: ScheduleCommandRequest
}

export interface ScheduleCommandResult {
  readonly ok: boolean
  readonly message?: string
  readonly error?: string
}

export async function runScheduleCommand(options: ScheduleCommandOptions): Promise<ScheduleCommandResult> {
  if (options.directory !== undefined) return { ok: false, error: 'The legacy --directory store is no longer used; use --project-dir or --socket; inspect old records with /schedules legacy and migrate explicitly' }
  if (options.owner !== undefined) return { ok: false, error: 'The legacy --owner is no longer used; daemon control derives ownership from the active project' }
  if (options.deliveryId !== undefined) return { ok: false, error: 'The legacy --delivery-id is unsupported; use the returned schedule ID with schedule fire' }
  try {
    const request = commandRequest(options)
    const socketPath = options.socketPath ?? daemonPaths(options.projectDirectory ?? process.cwd()).socketPath
    const send = options.request ?? ((method: string, params: JsonRpcPayload) => requestDaemonControl(socketPath, method, params, { ...(options.signal ? { signal: options.signal } : {}), timeoutMs: options.action === 'fire' ? 3_630_000 : 30_000 }))
    const response = await send(request.method, request.params)
    if (response.ok !== true) return { ok: false, error: stringValue(response.error) ?? 'Daemon returned a malformed schedule response' }
    validateResponse(options.action, response)
    return { ok: true, message: readableResponse(options.action, response) }
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) }
  }
}

function commandRequest(options: ScheduleCommandOptions): { method: string; params: JsonRpcPayload } {
  switch (options.action) {
    case 'create': {
      if (options.id) throw new Error('New schedules receive an ID; use the returned ID for later actions')
      if (!options.schedule || !options.objective) throw new Error('create requires --schedule and --objective')
      const timing = parseSchedule(options.schedule)
      if (!timing.ok) throw new Error(timing.error)
      return { method: 'schedule.create', params: withProject({ prompt: options.objective, paused: options.paused ?? false, ...(timing.intervalSeconds === undefined ? { schedule: timing.schedule } : { interval_seconds: timing.intervalSeconds }), ...(options.timezone === undefined ? {} : { timezone: options.timezone }) }, options) }
    }
    case 'list': return { method: 'schedule.list', params: withProject({}, options) }
    case 'disable': return { method: 'schedule.pause', params: withProject(scheduleId(options.id), options) }
    case 'enable': return { method: 'schedule.resume', params: withProject(scheduleId(options.id), options) }
    case 'remove': return { method: 'schedule.remove', params: withProject(scheduleId(options.id), options) }
    case 'fire': return { method: 'schedule.run', params: withProject(scheduleId(options.id), options) }
    case 'inspect': return { method: 'schedule.inspect', params: withProject(scheduleId(options.id), options) }
    case 'cancel': return { method: 'schedule.cancel', params: withProject(scheduleId(options.id), options) }
  }
}

function scheduleId(id: string | undefined): Record<string, unknown> {
  const value = id?.trim()
  if (!value) throw new Error('schedule action requires --id')
  return { schedule_id: value }
}

function withProject(params: JsonRpcPayload, options: ScheduleCommandOptions): Record<string, unknown> {
  return { ...params, expected_project_directory: resolveProjectDirectory(options.projectDirectory ?? process.cwd()) }
}

function validateResponse(action: ScheduleCommandAction, response: JsonRpcPayload): void {
  if (action === 'list' && !Array.isArray(response.jobs)) throw new Error('Daemon returned a malformed schedule list')
  const jobs = action === 'list' ? response.jobs as unknown[] : action === 'remove' ? [] : [response.job]
  if (jobs.some(job => !isRecord(job) || !stringValue(job.id) || typeof job.paused !== 'boolean')) throw new Error('Daemon returned a malformed schedule record')
  if (action === 'remove' && (response.removed !== true || !stringValue(response.schedule_id))) throw new Error('Daemon did not confirm schedule removal')
  if (action === 'cancel' && typeof response.requested !== 'boolean') throw new Error('Daemon did not confirm cancellation status')
}

function readableResponse(action: ScheduleCommandAction, response: JsonRpcPayload): string {
  if (action === 'list') {
    const jobs = response.jobs as readonly unknown[]
    if (!jobs.length) return 'No schedules.'
    return jobs.map(job => {
      const item = job as Record<string, unknown>
      const state = item.paused === true ? 'paused' : 'enabled'
      return `${stringValue(item.id) ?? '?'} · ${state}${stringValue(item.next_run_at) ? ` · next ${stringValue(item.next_run_at)}` : ''}`
    }).join('\n')
  }
  if (action === 'remove') return `Removed schedule ${String(response.schedule_id)}`
  const job = response.job as Record<string, unknown>
  if (action === 'inspect') return JSON.stringify(job, null, 2)
  if (action === 'cancel') return response.requested ? `Cancellation requested for ${String(job.id)}; inspect to confirm it has stopped.` : `No active local run for ${String(job.id)}.`
  if (action === 'fire') return `Completed schedule ${String(job.id)}${typeof response.output === 'string' ? '\n' + response.output : ''}`
  return `${action === 'create' ? 'Created' : 'Updated'} schedule ${String(job.id)} · ${job.paused ? 'paused' : 'enabled'}${stringValue(job.next_run_at) ? ` · next ${stringValue(job.next_run_at)}` : ''}`
}

function parseSchedule(schedule: string): { ok: true; schedule: string; intervalSeconds?: undefined } | { ok: true; schedule?: undefined; intervalSeconds: number } | { ok: false; error: string } {
  const value = schedule.trim()
  if (value.startsWith('interval:')) {
    const intervalSeconds = Number(value.slice('interval:'.length))
    if (!Number.isSafeInteger(intervalSeconds) || intervalSeconds < 1 || intervalSeconds > 86_400) return { ok: false, error: 'interval requires an integer from 1 to 86400 seconds' }
    return { ok: true, intervalSeconds }
  }
  if (value.startsWith('webhook:') || value.startsWith('event:')) return { ok: false, error: 'event and webhook schedules are legacy-only; use daemon monitor or webhook configuration' }
  if (value.startsWith('cron:')) {
    const raw = value.slice('cron:'.length)
    const standard = raw.split(/\s+/).filter(Boolean)
    if (standard.length === 5) return { ok: true, schedule: standard.join(' ') }
    const legacy = raw.split('/')
    if (legacy.length > 1 && legacy.length <= 4) {
      if (raw.includes('*/')) return { ok: false, error: 'legacy cron slash-step syntax is ambiguous; use standard five-field cron' }
      const fields = [legacy[0] ?? '*', legacy[1] ?? '*', legacy[2] ?? '*', legacy[3] ?? '*']
      const bounds = [{ min: 0, max: 59 }, { min: 0, max: 23 }, { min: 1, max: 31 }, { min: 0, max: 6 }]
      if (fields[2] !== '*' && fields[3] !== '*') return { ok: false, error: 'legacy cron cannot combine day-of-month and day-of-week; use standard five-field cron' }
      for (let index = 0; index < fields.length; index++) if (!isValidCronField(fields[index]!, bounds[index]!.min, bounds[index]!.max)) return { ok: false, error: `invalid legacy cron field ${JSON.stringify(fields[index])}` }
      return { ok: true, schedule: `${fields[0]} ${fields[1]} ${fields[2]} * ${fields[3]}` }
    }
    return { ok: false, error: 'cron requires a standard five-field expression' }
  }
  const standard = value.split(/\s+/).filter(Boolean)
  if (standard.length === 5) return { ok: true, schedule: standard.join(' ') }
  return { ok: false, error: 'schedule must be interval:<seconds> or a standard five-field cron expression' }
}

/* Legacy parser removed; the daemon receives standard cron text. */
/*
  if (false) {
    const parts = schedule.slice('cron:'.length).split('/')
    const fields = [
      { name: 'minute', value: parts[0], min: 0, max: 59 },
      { name: 'hour', value: parts[1], min: 0, max: 23 },
      { name: 'day', value: parts[2], min: 1, max: 31 },
      { name: 'day-of-week', value: parts[3], min: 0, max: 6 },
    ] as const
    // Reject here, where the user is still looking. A blank field used to
    // become 0 (`Number('')`), so `cron:` silently created an hourly job, and
    // an out-of-range value parsed fine and then never fired — the two worst
    // outcomes for a scheduler, both silent.
    for (const field of fields) {
      if (field.value === undefined) continue
      if (!isValidCronField(field.value, field.min, field.max)) {
        return { ok: false, error: `cron ${field.name} field ${JSON.stringify(field.value)} is not a valid `
          + `${field.min}-${field.max} value, list, range, or step` }
      }
    }
    // Omit blank trailing fields rather than storing undefined-as-present.
    return {
      ok: true,
      schedule: {
        kind: 'cron',
        ...(parts[0] === undefined ? {} : { minute: parts[0] }),
        ...(parts[1] === undefined ? {} : { hour: parts[1] }),
        ...(parts[2] === undefined ? {} : { day: parts[2] }),
        ...(parts[3] === undefined ? {} : { dayOfWeek: parts[3] }),
      },
    }
  }
  return { ok: false, error: 'legacy schedule' }
}
*/

/**
 * Whether one cron field is a usable `*`, step, list, range, or literal.
 *
 * Blank and whitespace are rejected because `Number('')` is 0, which turned a
 * malformed schedule into a real but unrequested one; literals outside the
 * field's range are rejected because they parse fine and then match no clock.
 */
export function isValidCronField(pattern: string, min: number, max: number): boolean {
  if (pattern === '*') return true
  if (pattern.startsWith('*/')) {
    const step = Number(pattern.slice(2))
    return Number.isInteger(step) && step > 0 && step <= max
  }
  const parts = pattern.split(',')
  if (!parts.length) return false
  return parts.every(part => {
    if (part.trim() === '') return false
    const bounds = part.includes('-') ? part.split('-') : [part]
    if (bounds.length > 2) return false
    return bounds.every(bound => {
      if (bound.trim() === '' || !/^\d+$/.test(bound.trim())) return false
      const value = Number(bound)
      return Number.isInteger(value) && value >= min && value <= max
    })
  })
}

function stringValue(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
