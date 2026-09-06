// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { parseScheduleTime } from '../src/cron/time.js'

test.each(['2027-02-29T10:00:00Z', '2100-02-29T10:00:00Z', '2028-04-31T10:00:00Z', '2028-00-01T10:00:00Z', '2028-01-00T10:00:00Z', '2028-01-01T24:00:00Z', '2028-01-01T10:60:00Z', '2028-01-01T10:00:60Z', '2028-01-01T10:00:00+24:00', '2028-01-01T10:00:00+02:60', '2028-01-01', '2028-01-01T10:00:00'])('rejects ambiguous or normalized timing %s', value => {
  expect(() => parseScheduleTime(value)).toThrow()
})
test('valid leap days and explicit offsets preserve the requested instant', () => {
  expect(parseScheduleTime('2000-02-29T00:15:00.1+05:30').toISOString()).toBe('2000-02-28T18:45:00.100Z')
  expect(parseScheduleTime('2028-02-29T23:30:00-02:00').toISOString()).toBe('2028-03-01T01:30:00.000Z')
  expect(parseScheduleTime('0096-02-29T00:00:00Z').toISOString()).toBe('0096-02-29T00:00:00.000Z')
})
