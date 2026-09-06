// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { afterEach, expect, it, vi } from 'vitest'
import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { getOverlayState, patchOverlayState, resetFlowOverlays, resetOverlayState } from '../app/overlayStore.js'
import { ScheduleOverlay } from '../opentui/scheduleOverlay.js'
import { DARK_THEME } from '../theme.js'
const job = { id: 'daily', prompt: 'Review changes', schedule: '0 9 * * *', paused: false, execution_state: 'idle', next_run_at: '2026-09-06T09:00:00Z', metadata: { delivery_state: 'failed', delivery_error: 'Destination offline' } }
afterEach(() => resetOverlayState())
it.each([[220, 65], [110, 35], [60, 24], [40, 18]])('renders schedules at %ix%i', async (width, height) => {
  const rpc = vi.fn(async () => ({ ok: true, jobs: [job] }))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ScheduleOverlay t={DARK_THEME} /></GatewayProvider>, { width, height })
  try {
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('Review changes')
    if (width >= 100) expect(screen.captureCharFrame()).toContain('Destination offline')
    expect(screen.captureCharFrame()).toContain('P pause/resume')
    expect(screen.captureCharFrame()).toContain('Esc')
  } finally { act(() => screen.renderer.destroy()) }
})
it('keeps cancellation accessible while a manual run is pending and restores overlay state', async () => {
  patchOverlayState({ schedules: true }); resetFlowOverlays()
  expect(getOverlayState().schedules).toBe(true)
  let finish!: (result: { ok: boolean }) => void
  const rpc = vi.fn(async (method: string) => {
    if (method === 'schedule.list') return { ok: true, jobs: [job] }
    if (method === 'schedule.run') return new Promise(resolve => { finish = resolve })
    return { ok: true }
  })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ScheduleOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 30 })
  try {
    await screen.flush()
    act(() => screen.mockInput.pressKey('g'))
    await screen.flush()
    act(() => screen.mockInput.pressKey('x'))
    await screen.flush()
    expect(rpc).toHaveBeenCalledWith('schedule.cancel', { schedule_id: 'daily' })
    finish({ ok: true })
    await screen.flush()
    act(() => screen.mockInput.pressKey('p'))
    await screen.flush()
    expect(rpc).toHaveBeenCalledWith('schedule.pause', { schedule_id: 'daily' })
    act(() => screen.mockInput.pressKey('ESCAPE'))
    await vi.waitFor(() => expect(getOverlayState().schedules).toBe(false))
  } finally { act(() => screen.renderer.destroy()) }
})

it('opens selected schedule history and returns to schedules on Escape', async () => {
  patchOverlayState({ schedules: true })
  const rpc = vi.fn(async (method: string) => method === 'schedule.list' ? { ok: true, jobs: [job] } : { ok: true, runs: [] })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ScheduleOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 30 })
  try {
    await screen.flush()
    act(() => screen.mockInput.pressKey('h'))
    await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Schedule history')
    expect(rpc).toHaveBeenCalledWith('run.list', { unread_only: false, scope: 'workspace', source_id: 'daily', kind: 'schedule' })
    act(() => screen.mockInput.pressKey('ESCAPE'))
    await screen.flush()
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Schedules · current workspace') })
    expect(getOverlayState().schedules).toBe(true)
  } finally { act(() => screen.renderer.destroy()) }
})

it('explains uncertain execution before offering resume', async () => {
  const rpc = vi.fn(async () => ({ ok: true, jobs: [{ ...job, paused: true, metadata: { execution_recovery_required: true } }] }))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ScheduleOverlay t={DARK_THEME} /></GatewayProvider>, { width: 150, height: 40 })
  try {
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('Review run output before resuming')
    expect(screen.captureCharFrame()).toContain('may execute again')
    act(() => screen.mockInput.pressKey('p'))
    await screen.flush()
    expect(rpc).toHaveBeenCalledWith('schedule.resume', { schedule_id: 'daily' })
  } finally { act(() => screen.renderer.destroy()) }
})

it('labels partial token usage instead of presenting it as an exact total', async () => {
  const rpc = vi.fn(async () => ({ ok: true, jobs: [{ ...job, metadata: { token_usage: { input_tokens: 8, output_tokens: 3, complete: false } } }] }))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ScheduleOverlay t={DARK_THEME} /></GatewayProvider>, { width: 160, height: 40 })
  try {
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('8 input')
    expect(screen.captureCharFrame()).toContain('some usage unavailable')
  } finally { act(() => screen.renderer.destroy()) }
})
it.each([[220, 65], [40, 18]])('opens scoped follow-ups and a bounded new draft at %ix%i', async (width, height) => {
  patchOverlayState({ loops: true }); resetFlowOverlays()
  expect(getOverlayState().loops).toBe(true)
  const rpc = vi.fn(async () => ({ ok: true, jobs: [], destinations: [], next_run_at: '2099-01-01T00:00:00Z' }))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ScheduleOverlay t={DARK_THEME} followupsOnly /></GatewayProvider>, { width, height })
  try {
    await screen.flush(); await screen.flush()
    expect(rpc).toHaveBeenCalledWith('schedule.list', { scope: 'session' })
    expect(screen.captureCharFrame()).toContain('Follow-ups')
    act(() => screen.mockInput.pressKey('n')); await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).toContain('New conversation follow-up')
    act(() => screen.mockInput.pressKey('ESCAPE')); await screen.flush(); await screen.flush()
    expect(getOverlayState().loops).toBe(true)
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Follow-ups') })
    act(() => screen.mockInput.pressKey('ESCAPE')); await screen.flush(); await screen.flush()
    await vi.waitFor(() => expect(getOverlayState().loops).toBe(false))
  } finally { act(() => screen.renderer.destroy()) }
})
