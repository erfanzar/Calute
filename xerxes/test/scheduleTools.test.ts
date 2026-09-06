// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { registerScheduleTools } from '../src/tools/scheduleTools.js'
import { permissionDisposition } from '../src/streaming/permissions.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

const call = (name: string, args: JsonObject): ToolCall => ({ id: 'schedule-test', type: 'function', function: { name, arguments: args } })
test('schedule tools preserve identity, parameters and cancellation and prohibit background self-scheduling', async () => {
  const registry = new ToolRegistry()
  const requests: unknown[] = []
  registerScheduleTools(registry, async (...args) => { requests.push(args); return { ok: true } })
  await expect(registry.execute(call('list_schedules', {}), { metadata: {} })).rejects.toThrow('authenticated')
  await expect(registry.execute(call('manage_schedule', { action: 'create', prompt: 'Review', paused: true, interval_seconds: 60 }), { sessionId: 'owner', metadata: {} })).rejects.toThrow('direct user turn')
  expect(requests).toHaveLength(0)
  const context = { sessionId: 'owner', metadata: { goal_turn_human: true } }
  const args = { action: 'create', prompt: 'Review', paused: true, interval_seconds: 60 }
  expect(JSON.parse(await registry.execute(call('manage_schedule', args), context))).toEqual({ ok: true })
  expect(requests[0]).toMatchObject(['owner', 'create', args, undefined])
  await registry.execute(call('list_schedules', {}), { sessionId: 'owner', metadata: {} })
  expect(requests[1]).toMatchObject(['owner', 'list', {}, undefined])
  const controller = new AbortController()
  controller.abort(new Error('cancelled'))
  await expect(registry.execute(call('manage_schedule', { action: 'pause', schedule_id: 'job' }), context, controller.signal)).rejects.toThrow()
  expect(requests).toHaveLength(2)
  await registry.execute(call('manage_schedule', { action: 'complete', schedule_id: 'job', evidence: 'Health check passed' }), { sessionId: 'owner', metadata: { goal_turn_human: false } })
  expect(requests[2]).toMatchObject(['owner', 'complete', { action: 'complete', schedule_id: 'job', evidence: 'Health check passed' }, undefined])
})
test('schedule activation uses the same approval boundary as persistent cron', () => {
  for (const action of ['create', 'update', 'resume', 'run']) {
    expect(permissionDisposition(call('manage_schedule', { action }), 'accept-all')).toBe('prompt')
  }
  expect(permissionDisposition(call('manage_schedule', { action: 'inspect' }), 'accept-all')).toBe('allow')
})
