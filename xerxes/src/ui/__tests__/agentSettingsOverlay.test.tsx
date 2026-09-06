// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { expect, it, vi } from 'vitest'
import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { AgentSettingsOverlay } from '../opentui/agentSettingsOverlay.js'
import { DARK_THEME } from '../theme.js'
import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'

it('selects a provider, discovered model and supported effort with arrows and saves the selections', async () => {
  const rpc = vi.fn(async (method: string) => {
    if (method === 'agent.settings.get') return { ok: true, revision: 0, settings: { light: 'gpt-5' }, profiles: [{ name: 'work', provider: 'openai', model: 'gpt-5' }] }
    if (method === 'agent.settings.options') return { ok: true, reasoning_efforts: ['low', 'high'] }
    if (method === 'fetch_models') return { ok: true, models: ['gpt-5', 'gpt-6-astra'] }
    return { ok: true, revision: 1 }
  })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><AgentSettingsOverlay t={DARK_THEME} /></GatewayProvider>, { width: 80, height: 24 })
  try {
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Provider profiles: work') })
    await act(async () => screen.mockInput.pressKey('ARROW_DOWN'))
    await screen.flush()
    await act(async () => screen.mockInput.pressKey('TAB'))
    await screen.flush()
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('gpt-6-astra') })
    expect(rpc).toHaveBeenCalledWith('fetch_models', { profile_name: 'work' })
    await act(async () => screen.mockInput.pressKey('ARROW_DOWN'))
    await screen.flush()
    await act(async () => screen.mockInput.pressKey('TAB'))
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('(inherit) · low · high') })
    await act(async () => screen.mockInput.pressKey('ARROW_UP'))
    await screen.flush()
    await act(async () => screen.mockInput.pressKey('RETURN'))
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('agent.settings.save', { revision: 0, settings: { default: 'inherit', light: { model: 'gpt-6-astra', provider_profile: 'work', reasoning_effort: 'high' } } }))
  } finally { act(() => screen.renderer.destroy()); resetOverlayState() }
})

it('keeps a custom model after discovery failure and allows refresh without overwriting it', async () => {
  let attempts = 0
  const rpc = vi.fn(async (method: string) => {
    if (method === 'agent.settings.get') return { ok: true, revision: 0, settings: { light: 'custom-model' }, profiles: [] }
    if (method === 'fetch_models') {
      if (++attempts === 1) throw new Error('Provider offline')
      return { ok: true, models: ['available-model'], warning: 'Cached catalog' }
    }
    return { ok: true, revision: 1 }
  })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><AgentSettingsOverlay t={DARK_THEME} /></GatewayProvider>, { width: 80, height: 24 })
  try {
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('custom-model') })
    await act(async () => screen.mockInput.pressKey('TAB'))
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Provider offline') })
    await act(async () => screen.mockInput.pressKey('F5'))
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Cached catalog') })
    expect(attempts).toBe(2)
    await act(async () => screen.mockInput.pressKey('RETURN'))
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('agent.settings.save', { revision: 0, settings: { default: 'inherit', light: { model: 'custom-model' } } }))
  } finally { act(() => screen.renderer.destroy()); resetOverlayState() }
})

it.each([[150, 40], [40, 18]])('edits provider/model/effort settings at %ix%i without losing a rejected draft', async (width, height) => {
  const rpc = vi.fn(async (method: string) => method === 'agent.settings.get' ? { ok: true, revision: 3, settings: { light: { provider_profile: 'work', model: 'fast', reasoning_effort: 'low' }, balanced: 'normal', smart: 'deep' }, profiles: [{ name: 'work', provider: 'openai', model: 'fast' }] } : { ok: false, error: 'Agent settings changed; reload before saving' })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><AgentSettingsOverlay t={DARK_THEME} /></GatewayProvider>, { width, height })
  try {
    await screen.flush()
    await vi.waitFor(() => expect(screen.captureCharFrame()).toContain('LIGHT'))
    act(() => screen.mockInput.pressKey('RETURN'))
    await screen.flush()
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('agent.settings.save', expect.objectContaining({ revision: 3, settings: expect.objectContaining({ light: { provider_profile: 'work', model: 'fast', reasoning_effort: 'low' } }) })))
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).not.toContain('Saving…') })
    patchOverlayState({ agentSettings: true })
    act(() => screen.mockInput.pressKey('ESCAPE'))
    await screen.flush()
    await vi.waitFor(() => expect(getOverlayState().agentSettings).toBe(false))
  } finally { act(() => screen.renderer.destroy()); resetOverlayState() }
})

it('edits the selected model field and saves its provider and effort together', async () => {
  const rpc = vi.fn(async (method: string) => method === 'agent.settings.get'
    ? { ok: true, revision: 0, settings: { light: { provider_profile: 'work', model: '', reasoning_effort: 'high' } }, profiles: [] }
    : { ok: true, revision: 1 })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><AgentSettingsOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 35 })
  try {
    await screen.flush()
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('work') })
    act(() => screen.mockInput.pressKey('TAB'))
    await screen.flush()
    await act(async () => screen.mockInput.typeText('gpt-6-astra'))
    await screen.flush()
    act(() => screen.mockInput.pressKey('RETURN'))
    await screen.flush()
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('agent.settings.save', { revision: 0, settings: { default: 'inherit', light: { provider_profile: 'work', model: 'gpt-6-astra', reasoning_effort: 'high' } } }))
  } finally { act(() => screen.renderer.destroy()); resetOverlayState() }
})

it.each([false, true])('configures default mode and disables a default tier safely (disable=%s)', async disable => {
  const rpc = vi.fn(async (method: string) => method === 'agent.settings.get'
    ? { ok: true, revision: 0, settings: { light: 'fast', smart: 'deep' }, profiles: [] }
    : { ok: true, revision: 1 })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><AgentSettingsOverlay t={DARK_THEME} /></GatewayProvider>, { width: 80, height: 24 })
  try {
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('fast') })
    await act(async () => screen.mockInput.pressKey('F2'))
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Default: light') })
    if (disable) {
      await act(async () => screen.mockInput.pressKey('F4'))
      await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Default: inherit') })
    }
    expect(rpc).not.toHaveBeenCalledWith('agent.settings.save', expect.anything())
    await act(async () => screen.mockInput.pressKey('RETURN'))
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('agent.settings.save', {
      revision: 0, settings: disable ? { default: 'inherit', smart: { model: 'deep' } }
        : { default: 'light', light: { model: 'fast' }, smart: { model: 'deep' } },
    }))
    expect(rpc).not.toHaveBeenCalledWith('agent.settings.options', expect.anything())
  } finally { act(() => screen.renderer.destroy()); resetOverlayState() }
})
