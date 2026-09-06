// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
export interface ScheduleTokens { used: number | null; complete: boolean; maximum: number | null; blocked: boolean }
export function parseScheduleTokens(value: unknown): ScheduleTokens | undefined {
  if (value === undefined) return
  if (!value || typeof value !== 'object') throw new Error('Invalid schedule token budget')
  const row = value as Record<string, unknown>
  const count = (value: unknown) => typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
  if ((row.used !== null && !count(row.used)) || (row.maximum !== null && (!count(row.maximum) || row.maximum === 0))
    || typeof row.complete !== 'boolean' || typeof row.blocked !== 'boolean'
    || (row.complete && row.used === null)) throw new Error('Invalid schedule token budget')
  const used = row.used as number | null, maximum = row.maximum as number | null
  if (row.blocked !== (maximum !== null && (!row.complete || used === null || used >= maximum))) throw new Error('Inconsistent schedule token budget')
  return { used, maximum, complete: row.complete, blocked: row.blocked }
}
export function scheduleTokensLabel(value: ScheduleTokens | undefined): string {
  if (!value) return 'Lifetime tokens: unavailable'
  return `Lifetime tokens: ${value.used ?? 'unknown'} / ${value.maximum ?? 'unlimited'}${value.complete ? '' : ' · incomplete'}${value.blocked ? ' · new calls blocked' : ''}`
}
