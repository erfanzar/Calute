// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { runScheduleCommand, type ScheduleCommandRequest } from '../src/runtime/scheduleCommand.js'
import type { JsonRpcPayload } from '../src/protocol/jsonRpc.js'

const job = { id: 'generated', paused: false, next_run_at: '2030-01-01T00:00:00.000Z' }

test('schedule commands use unified daemon actions with an explicit project assertion', async () => {
  const calls: Array<{ method: string; params: JsonRpcPayload }> = []
  const request: ScheduleCommandRequest = async (method, params) => {
    calls.push({ method, params })
    return method === 'schedule.list' ? { ok: true, jobs: [job] }
      : method === 'schedule.remove' ? { ok: true, schedule_id: job.id, removed: true }
      : method === 'schedule.cancel' ? { ok: true, job, requested: false }
      : { ok: true, job, output: 'execution evidence' }
  }
  const create = await runScheduleCommand({ action: 'create', schedule: 'interval:60', objective: 'work', projectDirectory: '/tmp', request })
  expect(create).toMatchObject({ ok: true })
  expect(create.message).toContain('Created schedule generated · enabled')
  expect(calls[0]).toMatchObject({ method: 'schedule.create', params: { interval_seconds: 60, prompt: 'work', paused: false } })
  expect(calls[0]?.params.expected_project_directory).toBeDefined()
  for (const [action, method] of [['disable', 'pause'], ['enable', 'resume'], ['remove', 'remove'], ['fire', 'run'], ['inspect', 'inspect'], ['cancel', 'cancel'], ['list', 'list']] as const) {
    const result = await runScheduleCommand({ action, id: job.id, request })
    expect(result.ok).toBe(true)
    expect(calls.at(-1)?.method).toBe(`schedule.${method}`)
    if (action === 'cancel') expect(result.message).toContain('No active local run')
    if (action === 'fire') expect(result.message).toContain('Completed schedule generated\nexecution evidence')
  }
})

test('standard cron steps stay intact and ambiguous legacy or unsupported sources never reach the daemon', async () => {
  const calls: JsonRpcPayload[] = []
  const request: ScheduleCommandRequest = async (_method, params) => { calls.push(params); return { ok: true, job } }
  const create = (schedule: string) => runScheduleCommand({ action: 'create', schedule, objective: 'work', request })
  expect((await create('cron:*/5 * * * *')).ok).toBe(true)
  expect(calls.at(-1)?.schedule).toBe('*/5 * * * *')
  expect((await create('cron:5/14')).ok).toBe(true)
  expect(calls.at(-1)?.schedule).toBe('5 14 * * *')
  for (const value of ['cron:', 'cron:99', 'cron:0/25', 'cron:*/15', 'cron:0/0/1/1', 'interval:Infinity', 'interval:1.5', 'interval:86401', 'event:build', 'webhook:/build']) {
    expect((await create(value)).ok).toBe(false)
  }
  expect(calls).toHaveLength(2)
})

test('legacy options, daemon rejection and uncertain transport failures never report success or retry', async () => {
  let calls = 0
  const request: ScheduleCommandRequest = async () => { calls++; throw new Error('connection closed; outcome unknown') }
  for (const legacy of [{ directory: '/tmp/legacy' }, { owner: 'user' }, { deliveryId: 'delivery' }]) {
    expect((await runScheduleCommand({ action: 'list', ...legacy, request })).ok).toBe(false)
  }
  expect(calls).toBe(0)
  expect(await runScheduleCommand({ action: 'fire', id: job.id, request })).toMatchObject({ ok: false, error: 'connection closed; outcome unknown' })
  expect(calls).toBe(1)
  expect(await runScheduleCommand({ action: 'list', request: async () => ({ ok: true, jobs: [{}] }) })).toMatchObject({ ok: false })
  expect(await runScheduleCommand({ action: 'remove', id: job.id, request: async () => ({ ok: true }) })).toMatchObject({ ok: false })
  expect(await runScheduleCommand({ action: 'list', request: async () => ({ ok: false, error: 'wrong project' }) })).toMatchObject({ ok: false, error: 'wrong project' })
})
