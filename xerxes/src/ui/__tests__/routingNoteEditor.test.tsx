// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { expect, it, vi } from 'vitest'
import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { RoutingNoteEditor } from '../opentui/routingNoteEditor.js'
import { DARK_THEME } from '../theme.js'
it.each([[220, 65], [40, 18]])('preserves scoped drafts and failed saves at %ix%i', async (width, height) => {
  const rpc = vi.fn(async (method: string) => method.endsWith('.get') ? { ok: true, routing_note: { note: '', revision: 4 } } : { ok: false, error: 'Routing note changed' })
  const close = vi.fn()
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><RoutingNoteEditor t={DARK_THEME} profile="local" model="small" onClose={close} /></GatewayProvider>, { width, height })
  try {
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).not.toContain('Loading notes') })
    await act(async () => screen.mockInput.typeText('Broad work')); await screen.flush()
    act(() => screen.mockInput.pressKey('TAB')); await screen.flush()
    await act(async () => screen.mockInput.typeText('Quick checks')); await screen.flush()
    act(() => screen.mockInput.pressKey('F2')); await screen.flush()
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('model.routing_note.save', { provider_profile: 'local', model: 'small', note: 'Quick checks', revision: 4 }))
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Routing note changed') })
    expect(screen.captureCharFrame()).toContain('Quick checks')
    act(() => screen.mockInput.pressKey('TAB')); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Broad work')
    act(() => screen.mockInput.pressKey('ESCAPE')); await vi.waitFor(() => expect(close).toHaveBeenCalledTimes(1))
  } finally { act(() => screen.renderer.destroy()) }
})

it('retries failed loading and displays saved text', async () => {
  let fail = true
  const rpc = vi.fn(async (method: string) => method.endsWith('.get')
    ? fail ? { ok: false, error: 'Offline' } : { ok: true, routing_note: { note: '', revision: 0 } }
    : { ok: true, routing_note: { note: 'canonical', revision: 1 } })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><RoutingNoteEditor t={DARK_THEME} profile="local" model="" onClose={() => {}} /></GatewayProvider>, { width: 100, height: 25 })
  try {
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('F5 retry') })
    expect(screen.captureCharFrame()).not.toContain('Loading notes')
    fail = false
    act(() => screen.mockInput.pressKey('F5'))
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).not.toContain('Notes unavailable') })
    await act(async () => screen.mockInput.typeText('draft')); await screen.flush()
    act(() => screen.mockInput.pressKey('F2'))
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('canonical') })
    expect(screen.captureCharFrame()).not.toContain('draft')
  } finally { act(() => screen.renderer.destroy()) }
})
