// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { processStartIdentity } from '../src/core/processLiveness.js'

test('invalid PIDs cannot probe process groups or enter a command', () => {
  for (const pid of [0, -1, NaN, Infinity, 1.5, Number.MAX_SAFE_INTEGER + 1]) {
    expect(processStartIdentity(pid)).toBe('')
  }
})

test('current process start identity is stable when the host exposes it', () => {
  const first = processStartIdentity(process.pid)
  // Hosts may restrict process inspection; unknown remains conservative.
  expect(processStartIdentity(process.pid)).toBe(first)
  if (first) expect(first.startsWith(`${process.platform}:`)).toBe(true)
})
