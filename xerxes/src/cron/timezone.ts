// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export function cronTimezone(value: string = 'UTC'): string {
  if (typeof value !== 'string' || value.length > 100 || !/^[A-Za-z][A-Za-z0-9_+\/-]*$/.test(value)) throw new Error('Timezone must be an IANA zone name')
  try { return new Intl.DateTimeFormat('en-US', { timeZone: value }).resolvedOptions().timeZone }
  catch { throw new Error(`Unknown timezone: ${value}`) }
}
export function zoneFormatter(zone: string): Intl.DateTimeFormat {
  return new Intl.DateTimeFormat('en-US', { timeZone: zone, year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23' })
}
/** A UTC-shaped date holding a zone's local calendar fields, never an actual instant. */
export function wallTime(instant: Date, formatter: Intl.DateTimeFormat): Date {
  const parts = Object.fromEntries(formatter.formatToParts(instant).map(part => [part.type, part.value]))
  const result = new Date(0)
  result.setUTCFullYear(Number(parts.year), Number(parts.month) - 1, Number(parts.day))
  result.setUTCHours(Number(parts.hour), Number(parts.minute), Number(parts.second), 0)
  return result
}
/** Collect offsets surrounding a local day, including both sides of DST/date-line changes. */
export function dayOffsets(day: Date, formatter: Intl.DateTimeFormat): number[] {
  const offsets = new Set<number>()
  for (let hour = -48; hour <= 48; hour += 6) {
    const instant = new Date(day.getTime() + hour * 3_600_000)
    offsets.add(wallTime(instant, formatter).getTime() - instant.getTime())
  }
  return [...offsets]
}
export function wallInstants(wall: Date, formatter: Intl.DateTimeFormat, offsets: readonly number[]): Date[] {
  return offsets.map(offset => new Date(wall.getTime() - offset))
    .filter(instant => wallTime(instant, formatter).getTime() === wall.getTime())
    .sort((a, b) => a.getTime() - b.getTime())
}
