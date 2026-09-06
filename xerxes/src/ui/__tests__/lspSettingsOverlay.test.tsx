// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { expect, it, vi } from 'vitest'
import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { LspSettingsOverlay } from '../opentui/lspSettingsOverlay.js'
import { DARK_THEME } from '../theme.js'
import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
const row = { name: 'typescript', enabled: true, languageId: 'typescript', extensions: ['.ts'], timeoutMs: 30000, configuredFields: ['command', 'env'] }
const settings = { ok: true, revision: 'a'.repeat(64), servers: [row] }

it.each([[220, 65], [40, 18]])('keeps failed drafts and closes without losing the transcript at %ix%i', async (width, height) => {
  const rpc = vi.fn(async (method: string) => method === 'lsp.settings.get' ? settings : { ok: false, error: 'Settings changed' })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><LspSettingsOverlay t={DARK_THEME} /></GatewayProvider>, { width, height })
  try {
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('typescript') })
    await act(async () => screen.mockInput.pressKey('RETURN'))
    await act(async () => screen.mockInput.pressKey('s', { ctrl: true }))
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('lsp.settings.save', { name: 'typescript', revision: settings.revision, action: 'update', changes: { enabled: false } }))
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('false'); expect(screen.captureCharFrame()).not.toContain('Saving/loading') })
    patchOverlayState({ lspSettings: true }); await act(async () => screen.mockInput.pressKey('ESCAPE'))
    await vi.waitFor(() => expect(getOverlayState().lspSettings).toBe(false))
  } finally { act(() => screen.renderer.destroy()); resetOverlayState() }
})

it('confirms removal and uses the saved revision for later edits', async () => {
  const rpc = vi.fn(async (method: string, params: unknown) => method === 'lsp.settings.get' ? settings : { ...settings, revision: 'b'.repeat(64), servers: (params as { action: string }).action === 'remove' ? [] : [{ ...row, enabled: false }] })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><LspSettingsOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 36 })
  try {
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('typescript') })
    await act(async () => screen.mockInput.pressKey('RETURN'))
    await act(async () => screen.mockInput.pressKey('s', { ctrl: true }))
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Saved.') })
    await act(async () => screen.mockInput.pressKey('F4'))
    await act(async () => screen.mockInput.pressKey('n'))
    expect(rpc.mock.calls.filter(([method]) => method === 'lsp.settings.save')).toHaveLength(1)
    await act(async () => screen.mockInput.pressKey('F4'))
    await act(async () => screen.mockInput.pressKey('y'))
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('lsp.settings.save', { name: 'typescript', revision: 'b'.repeat(64), action: 'remove' }))
  } finally { act(() => screen.renderer.destroy()); resetOverlayState() }
})

it('creates a server by entering language, suffixes and command', async () => {
  const rpc = vi.fn(async (method: string) => method === 'lsp.settings.get' ? { ...settings, servers: [] } : settings)
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><LspSettingsOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 36 })
  const press = async (key: string) => { await act(async () => screen.mockInput.pressKey(key)); await screen.flush() }
  const enter = async (text: string) => { await act(async () => screen.mockInput.typeText(text)); await press('RETURN') }
  try {
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('F2 adds one') })
    await press('F2'); await enter('typescript')
    await press('RETURN'); await enter('typescript')
    await press('TAB'); await press('RETURN'); await enter('[".ts"]')
    await press('TAB'); await press('RETURN'); await enter('language-server')
    await act(async () => screen.mockInput.pressKey('s', { ctrl: true }))
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('lsp.settings.save', { name: 'typescript', revision: settings.revision, action: 'create', changes: { languageId: 'typescript', extensions: ['.ts'], command: 'language-server' } }))
  } finally { act(() => screen.renderer.destroy()); resetOverlayState() }
})
