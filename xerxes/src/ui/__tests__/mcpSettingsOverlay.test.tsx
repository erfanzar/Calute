// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { expect, it, vi } from 'vitest'
import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { McpSettingsOverlay } from '../opentui/mcpSettingsOverlay.js'
import { DARK_THEME } from '../theme.js'
import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
const settings = { ok: true, revision: 'a'.repeat(64), servers: [{ name: 'fixture', enabled: true, transport: 'stdio', timeout_ms: null, configured_fields: ['command', 'env'] }] }
it.each([[220, 65], [40, 18]])('preserves failed drafts and closes cleanly at %ix%i', async (width, height) => {
  const rpc = vi.fn(async (method: string) => method === 'mcp.settings.get' ? settings : { ok: false, error: 'Settings changed' })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><McpSettingsOverlay t={DARK_THEME} /></GatewayProvider>, { width, height })
  try {
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('fixture') })
    await act(async () => screen.mockInput.pressKey('RETURN'))
    await act(async () => screen.mockInput.pressKey('s', { ctrl: true }))
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('mcp.settings.save', { name: 'fixture', revision: settings.revision, changes: { enabled: false } }))
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('false') })
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).not.toContain('Saving/loading') })
    patchOverlayState({ mcpSettings: true })
    await act(async () => screen.mockInput.pressKey('ESCAPE'))
    await vi.waitFor(() => expect(getOverlayState().mcpSettings).toBe(false))
  } finally { act(() => screen.renderer.destroy()); resetOverlayState() }
})
it('commits entered command without echoing the saved launch fields and reuses the new revision', async () => {
  const rpc = vi.fn(async (method: string) => method === 'mcp.settings.get' ? settings : { ok: true, revision: 'b'.repeat(64) })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><McpSettingsOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 36 })
  try {
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('fixture') })
    await act(async () => { screen.mockInput.pressKey('TAB'); screen.mockInput.pressKey('TAB') })
    await screen.flush()
    await act(async () => screen.mockInput.pressKey('RETURN'))
    await screen.flush()
    await act(async () => screen.mockInput.typeText('bun'))
    await act(async () => screen.mockInput.pressKey('RETURN'))
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('(replacement entered)')
    await act(async () => screen.mockInput.pressKey('s', { ctrl: true }))
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('mcp.settings.save', { name: 'fixture', revision: settings.revision, changes: { command: 'bun' } }))
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Saved.') })
  } finally { act(() => screen.renderer.destroy()); resetOverlayState() }
})

it('creates a server from an empty configuration using the keyboard', async () => {
  const rpc = vi.fn(async (method: string) => method === 'mcp.settings.get' ? { ...settings, servers: [] } : { ok: true, revision: 'c'.repeat(64) })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><McpSettingsOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 36 })
  try {
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('F2  Add server') })
    await act(async () => screen.mockInput.pressKey('F2'))
    await screen.flush()
    await act(async () => screen.mockInput.typeText('new-server'))
    await act(async () => screen.mockInput.pressKey('RETURN'))
    await screen.flush()
    await act(async () => screen.mockInput.pressKey('RETURN'))
    await screen.flush()
    await act(async () => screen.mockInput.typeText('bun'))
    await act(async () => screen.mockInput.pressKey('RETURN'))
    await screen.flush()
    await act(async () => screen.mockInput.pressKey('s', { ctrl: true }))
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('mcp.settings.save', { name: 'new-server', revision: settings.revision, create: true, changes: { command: 'bun' } }))
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('1/1 · new-server') })
  } finally { act(() => screen.renderer.destroy()); resetOverlayState() }
})
