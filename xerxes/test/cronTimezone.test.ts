// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { CronJob, nextFireAt } from '../src/cron/jobs.js'
test('cron skips nonexistent spring times and emits both repeated fall times', () => {
  expect(nextFireAt('30 2 * * *', new Date('2026-03-08T05:00:00Z'), 'America/New_York').toISOString()).toBe('2026-03-09T06:30:00.000Z')
  const first = nextFireAt('30 1 * * *', new Date('2026-11-01T04:00:00Z'), 'America/New_York')
  expect(first.toISOString()).toBe('2026-11-01T05:30:00.000Z')
  expect(nextFireAt('30 1 * * *', first, 'America/New_York').toISOString()).toBe('2026-11-01T06:30:00.000Z')
})
test('timezone cron supports fractional offsets and local calendar constraints', () => {
  expect(nextFireAt('0 9 * * *', new Date('2026-01-01T00:00:00Z'), 'Asia/Kathmandu').toISOString()).toBe('2026-01-01T03:15:00.000Z')
  expect(nextFireAt('0 0 1 1 *', new Date('2025-12-31T12:00:00Z'), 'Pacific/Kiritimati').toISOString()).toBe('2026-12-31T10:00:00.000Z')
  expect(() => nextFireAt('* * * * *', new Date(), 'Invalid/Zone')).toThrow('timezone')
})
test('persisted jobs round-trip timezone and old records default to UTC', () => {
  const job = new CronJob({ id: 'zone', prompt: 'Check', schedule: '0 9 * * *', timezone: 'America/New_York' })
  expect(CronJob.fromRecord(job.toRecord()).timezone).toBe('America/New_York')
  expect(CronJob.fromRecord({ id: 'old', prompt: 'Check' }).timezone).toBe('UTC')
})

test('half-hour DST changes and skipped calendar dates do not normalize cron times', () => {
  const first = nextFireAt('45 1 * * *', new Date('2026-04-04T13:00:00Z'), 'Australia/Lord_Howe')
  expect(first.toISOString()).toBe('2026-04-04T14:45:00.000Z')
  expect(nextFireAt('45 1 * * *', first, 'Australia/Lord_Howe').toISOString()).toBe('2026-04-04T15:15:00.000Z')
  expect(nextFireAt('15 2 * * *', new Date('2026-10-03T13:00:00Z'), 'Australia/Lord_Howe').toISOString()).toBe('2026-10-04T15:15:00.000Z')
  expect(nextFireAt('0 12 * * *', new Date('2011-12-30T00:00:00Z'), 'Pacific/Apia').toISOString()).toBe('2011-12-30T22:00:00.000Z')
})
