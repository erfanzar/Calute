// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { expect, it, vi } from 'vitest'
import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { MonitorPolicy } from '../opentui/monitorPolicy.js'
import { DARK_THEME } from '../theme.js'
const monitor = { id: 'watch', terminalId: 'build', match: 'error', state: 'watching', expiresAt: 1000, reaction: 'queued', error: '', events: [], omittedEvents: 0,
  policy: { revision: 'guard', maxReactions: 3, maxDurationMs: 60000, maxTotalTokens: 100 } }
it.each([[220, 65], [40, 18]])('edits tokens and preserves draft after conflict at %ix%i', async (width, height) => {
  const rpc = vi.fn(async () => ({ ok: false, error: 'Reaction policy changed; refresh before editing' }))
  const closed = vi.fn(), saved = vi.fn()
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MonitorPolicy t={DARK_THEME} monitor={monitor} onClose={closed} onSaved={saved} /></GatewayProvider>, { width, height })
  try {
    await screen.flush()
    act(() => screen.mockInput.pressKey('TAB', { shift: true })); await screen.flush()
    act(() => screen.mockInput.pressKey('END')); await screen.flush()
    for (let i = 0; i < 3; i++) act(() => screen.mockInput.pressKey('BACKSPACE'))
    await screen.flush()
    await act(async () => screen.mockInput.typeText('250')); await screen.flush()
    act(() => screen.mockInput.pressKey('RETURN'))
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('monitor.update', { monitor_id: 'watch', revision: 'guard', max_reactions: 3, reaction_timeout_seconds: 60, max_total_tokens: 250 }))
    await screen.flush()
    expect(saved).not.toHaveBeenCalled()
    expect(screen.captureCharFrame()).toContain('250')
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).not.toContain('Saving…') })
    act(() => screen.mockInput.pressKey('ESCAPE')); await screen.flush()
    await vi.waitFor(() => expect(closed).toHaveBeenCalledTimes(1))
  } finally { act(() => screen.renderer.destroy()) }
})
