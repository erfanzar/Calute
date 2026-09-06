// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Parse an explicit instant without Date's silent calendar normalization. */
export function parseScheduleTime(value: string): Date {
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?(Z|[+-](\d{2}):(\d{2}))$/.exec(value)
  if (!match) throw new Error('One-shot time requires an ISO timestamp with timezone')
  const year = Number(match[1]), month = Number(match[2]), day = Number(match[3])
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0)
  const days = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
  if (month < 1 || month > 12 || day < 1 || day > days[month - 1]!
    || Number(match[4]) > 23 || Number(match[5]) > 59 || Number(match[6]) > 59
    || (match[8] !== 'Z' && (Number(match[9]) > 23 || Number(match[10]) > 59))) {
    throw new Error('One-shot time contains an invalid calendar date or timezone offset')
  }
  const date = new Date(value)
  if (!Number.isFinite(date.getTime())) throw new Error('Invalid one-shot time')
  return date
}
