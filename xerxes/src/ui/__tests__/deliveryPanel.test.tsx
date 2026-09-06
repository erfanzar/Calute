// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { expect, it, vi } from 'vitest'
import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { DeliveryPanel } from '../opentui/deliveryPanel.js'
import { DARK_THEME } from '../theme.js'
const delivery = { id: 'attempt', platform: 'slack', recipient: 'builds', state: 'uncertain', attempts: 2, content: 'Build completed' }
it.each([[150, 40], [40, 18]])('requires an explicit reconciliation decision at %ix%i', async (width, height) => {
  const rpc = vi.fn(async (method: string) => method === 'schedule.deliveries' ? { ok: true, deliveries: [delivery] } : { ok: true, delivery })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><DeliveryPanel t={DARK_THEME} scheduleId="job" onClose={() => {}} /></GatewayProvider>, { width, height })
  try {
    await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Build completed')
    act(() => screen.mockInput.pressKey('s'))
    await screen.flush()
    expect(rpc).not.toHaveBeenCalledWith('schedule.delivery.send', expect.anything())
    act(() => screen.mockInput.pressKey('t'))
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('duplicate message')
    expect(rpc).not.toHaveBeenCalledWith('schedule.delivery.resolve', expect.anything())
    act(() => screen.mockInput.pressKey('RETURN'))
    await screen.flush()
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('schedule.delivery.resolve', { schedule_id: 'job', delivery_id: 'attempt', attempts: 2, decision: 'retry' }))
    expect(rpc).not.toHaveBeenCalledWith('schedule.delivery.send', expect.anything())
  } finally { act(() => screen.renderer.destroy()) }
})
it('keeps send failures visible after refresh', async () => {
  const row = { ...delivery, state: 'pending' }
  const rpc = vi.fn(async (method: string) => method === 'schedule.deliveries' ? { ok: true, deliveries: [row] } : method === 'schedule.delivery.send' ? { ok: false, error: 'Channel unavailable' } : { ok: true, delivery: row })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><DeliveryPanel t={DARK_THEME} scheduleId="job" onClose={() => {}} /></GatewayProvider>, { width: 110, height: 35 })
  try {
    await screen.flush(); await screen.flush()
    act(() => screen.mockInput.pressKey('s'))
    await screen.flush(); await screen.flush()
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Channel unavailable') })
  } finally { act(() => screen.renderer.destroy()) }
})
