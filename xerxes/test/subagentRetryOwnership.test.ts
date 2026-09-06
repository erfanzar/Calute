// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { resolveOwnedSubagentRetry, resolveSubagentRetryRequest } from '../src/daemon/subagentRetryOwnership.js'
import type { SpawnedAgentSnapshot } from '../src/operators/subagents.js'

function snapshot(id: string, name: string, sourceAgentId: string): SpawnedAgentSnapshot {
  return {
    agentId: 'coder',
    closed: true,
    createdAt: '2026-01-01T00:00:00.000Z',
    id,
    name,
    promptProfile: 'coder',
    queueSize: 0,
    sourceAgentId,
    status: 'completed',
    title: name,
    updatedAt: '2026-01-01T00:00:01.000Z',
  }
}

const ownerA = snapshot('a-id', 'worker', 'session-a')
const ownerB = snapshot('b-id', 'worker', 'session-b')

test('exact id takes precedence and rejects a different owner', () => {
  expect(() => resolveOwnedSubagentRetry([ownerA, ownerB], 'b-id', 'session-a')).toThrow('another session')
})

test('same-name lookup selects the requesting owner', () => {
  expect(resolveOwnedSubagentRetry([ownerA, ownerB], 'worker', 'session-b')).toBe(ownerB)
})

test('ambiguous same-owner names require an id', () => {
  const duplicate = snapshot('a-id-2', 'worker', 'session-a')
  expect(() => resolveOwnedSubagentRetry([ownerA, duplicate], 'worker', 'session-a')).toThrow('ambiguous')
})

test('missing owner and target fail closed', () => {
  expect(() => resolveOwnedSubagentRetry([ownerA], 'a-id', '')).toThrow('owning session')
  expect(() => resolveOwnedSubagentRetry([ownerA], undefined, 'session-a')).toThrow('task id or stable name')
})

test('unknown target reports an actionable owner-scoped miss', () => {
  expect(() => resolveOwnedSubagentRetry([ownerA, ownerB], 'missing', 'session-a')).toThrow('owning session')
})

test('request adapter requires a live session and preserves task and message', () => {
  expect(resolveSubagentRetryRequest(
    { task: ' a-id ', sessionKey: ' key ', message: 'try again' },
    key => key === 'key' ? { id: 'session-a' } : undefined,
  )).toEqual({
    task: 'a-id',
    options: { sourceAgentId: 'session-a', message: 'try again' },
  })
  expect(() => resolveSubagentRetryRequest({ task: 'a-id' }, () => ({ id: 'session-a' }))).toThrow('owning session')
  expect(() => resolveSubagentRetryRequest({ task: 'a-id', sessionKey: 'gone' }, () => undefined)).toThrow('no longer exists')
})
