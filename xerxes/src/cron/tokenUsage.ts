// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { ModelCallUsage } from '../llms/callBudget.js'

export interface ScheduleTokenUsage { version: 1; attempt: number; usage: ModelCallUsage }
const empty: ModelCallUsage = { input_tokens: 0, output_tokens: 0, measured_calls: 0, settled_calls: 0, pending_calls: 0, complete: true }
const counters = ['input_tokens', 'output_tokens', 'measured_calls', 'settled_calls', 'pending_calls'] as const
const integer = (value: unknown): value is number => typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
export function readScheduleTokenUsage(value: unknown): ScheduleTokenUsage | undefined {
  if (value === undefined) return
  if (!value || typeof value !== 'object') throw new Error('Invalid cumulative schedule token usage')
  const row = value as Record<string, unknown>
  const usage = row.usage as ModelCallUsage | undefined
  if (row.version !== 1 || !integer(row.attempt) || !usage || !counters.every(key => integer(usage[key]))
    || typeof usage.complete !== 'boolean' || usage.measured_calls > usage.settled_calls
    || !Number.isSafeInteger(usage.input_tokens + usage.output_tokens)
    || (usage.complete && (usage.pending_calls !== 0 || usage.measured_calls !== usage.settled_calls))) throw new Error('Invalid cumulative schedule token usage')
  return { version: 1, attempt: row.attempt, usage: { ...usage } }
}
export function scheduleTokenState(value: unknown, attempts: number): { used: number; complete: boolean } {
  const total = readScheduleTokenUsage(value)
  return { used: total ? total.usage.input_tokens + total.usage.output_tokens : 0,
    complete: total ? total.attempt === attempts && total.usage.complete : attempts === 0 }
}
/** Each checkpoint replaces the current attempt contribution, never adds it twice. */
export function beginScheduleTokenUsage(value: unknown, attempt: number): (current: ModelCallUsage) => ScheduleTokenUsage {
  const previous = readScheduleTokenUsage(value)
  if (!integer(attempt) || attempt < 1 || (previous && previous.attempt >= attempt)) throw new Error('Stale schedule token accounting attempt')
  const base = previous?.usage ?? empty
  const priorComplete = previous ? previous.attempt === attempt - 1 && base.complete : attempt === 1
  return current => {
    // Validate the contribution before adding it: negative or inconsistent
    // counters must not be hidden by a larger valid historical total.
    readScheduleTokenUsage({ version: 1, attempt, usage: current })
    const sum = (key: typeof counters[number]) => {
      const result = base[key] + current[key]
      if (!integer(result)) throw new Error('Cumulative schedule token usage overflow')
      return result
    }
    const result: ScheduleTokenUsage = { version: 1, attempt, usage: {
      input_tokens: sum('input_tokens'), output_tokens: sum('output_tokens'), measured_calls: sum('measured_calls'),
      settled_calls: sum('settled_calls'), pending_calls: sum('pending_calls'), complete: priorComplete && current.complete,
    } }
    return readScheduleTokenUsage(result)!
  }
}
