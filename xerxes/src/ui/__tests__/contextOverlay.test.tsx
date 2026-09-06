// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { afterEach, expect, it, vi } from 'vitest'
import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { getOverlayState, patchOverlayState, resetFlowOverlays, resetOverlayState } from '../app/overlayStore.js'
import { ContextOverlay } from '../opentui/contextOverlay.js'
import { DARK_THEME } from '../theme.js'
afterEach(() => resetOverlayState())
const page = { ok: true, generation: 'stable', note: 'Approximate tokens, not billing.', section: 'instructions', offset: 0, next_offset: null,
  sections: ['instructions', 'memory', 'conversation', 'tools', 'compaction'].map(id => ({ id, count: 1, available: true, estimated_tokens: 20, provenance: 'Latest assembly' })),
  entries: [{ index: 0, title: 'bootstrap', text: 'mandatory policy', truncated: false, estimated_tokens: 20 }] }
it.each([[220, 65], [40, 18]])('renders context and closes without replacing chat at %ix%i', async (width, height) => {
  patchOverlayState({ contextInspector: true }); resetFlowOverlays()
  expect(getOverlayState().contextInspector).toBe(true)
  const rpc = vi.fn(async () => page)
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ContextOverlay t={DARK_THEME} /></GatewayProvider>, { width, height })
  try {
    await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).toContain('mandatory policy')
    expect(screen.captureCharFrame()).toContain('~20')
    act(() => screen.mockInput.pressKey('ESCAPE')); await screen.flush()
    await vi.waitFor(() => expect(getOverlayState().contextInspector).toBe(false))
  } finally { act(() => screen.renderer.destroy()) }
})
it('changes sections and retains the last page when a generation becomes stale', async () => {
  const rpc = vi.fn(async (_method: string, params: Record<string, unknown>) => params.section === 'memory' ? { ok: false, error: 'Context changed; refresh the inspector' } : page)
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ContextOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 30 })
  try {
    await vi.waitFor(async () => {
      await screen.flush()
      expect(screen.captureCharFrame()).toContain('mandatory policy')
    })
    await act(async () => { screen.mockInput.pressKey('TAB') })
    await vi.waitFor(async () => {
      await screen.flush()
      expect(screen.captureCharFrame()).toContain('Context changed')
    })
    expect(rpc).toHaveBeenCalledWith('context.inspect', { section: 'memory', offset: 0, generation: 'stable' })
    expect(screen.captureCharFrame()).toContain('Context changed')
    expect(screen.captureCharFrame()).toContain('mandatory policy')
  } finally { act(() => screen.renderer.destroy()) }
})
it.each([[220, 65], [40, 18]])('opens compaction history by keyboard at %ix%i', async (width, height) => {
  const rpc = vi.fn(async (_method: string, params: Record<string, unknown>) => ({ ...page, section: params.section,
    entries: params.section === 'compaction' ? [{ index: 0, title: '2026-09-06 · compact', text: '1000 → 200 tokens', truncated: false, estimated_tokens: 0 }] : page.entries }))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ContextOverlay t={DARK_THEME} /></GatewayProvider>, { width, height })
  try {
    await screen.flush(); await screen.flush()
    act(() => screen.mockInput.pressArrow('left')); await screen.flush(); await screen.flush()
    expect(rpc).toHaveBeenCalledWith('context.inspect', { section: 'compaction', offset: 0, generation: 'stable' })
    expect(screen.captureCharFrame()).toContain('1000 → 200 tokens')
    expect(screen.captureCharFrame()).not.toContain('~0 tokens')
    act(() => screen.mockInput.pressKey('TAB')); await screen.flush(); await screen.flush()
    expect(rpc).toHaveBeenLastCalledWith('context.inspect', { section: 'instructions', offset: 0, generation: 'stable' })
  } finally { act(() => screen.renderer.destroy()) }
})
it('does not page a failed section using the retained previous section cursor', async () => {
  const rpc = vi.fn(async (_method: string, params: Record<string, unknown>) => params.section === 'memory'
    ? { ok: false, error: 'Context changed; refresh the inspector' }
    : { ...page, next_offset: 20 })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ContextOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 30 })
  try {
    await screen.flush(); await screen.flush()
    act(() => screen.mockInput.pressKey('TAB')); await screen.flush(); await screen.flush()
    const calls = rpc.mock.calls.length
    act(() => screen.mockInput.pressKey('n')); await screen.flush(); await screen.flush()
    expect(rpc).toHaveBeenCalledTimes(calls)
    expect(screen.captureCharFrame()).toContain('mandatory policy')
    act(() => screen.mockInput.pressKey('r')); await screen.flush(); await screen.flush()
    expect(rpc).toHaveBeenLastCalledWith('context.inspect', { section: 'memory', offset: 0 })
  } finally { act(() => screen.renderer.destroy()) }
})
it('pins a selected optional memory source and retains its page after a rejected mutation', async () => {
  let revision = 0
  const rpc = vi.fn(async (method: string, params: Record<string, unknown>) => {
    if (method === 'context.control') {
      if (params.action === 'exclude') return { ok: false, error: 'Wait for the active turn' }
      revision++
      return { ok: true, revision, applies_next_turn: true }
    }
    return { ...page, section: params.section, controls_revision: revision, generation: 'revision-' + revision,
      entries: params.section === 'memory' ? [{ ...page.entries[0]!, title: '[project] MEMORY.md', text: 'a durable fact',
        control: { scope: 'project', path: 'MEMORY.md', pinned: revision > 0, excluded: false } }] : page.entries }
  })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ContextOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 30 })
  try {
    await screen.flush(); await screen.flush()
    act(() => screen.mockInput.pressKey('TAB')); await screen.flush(); await screen.flush()
    act(() => screen.mockInput.pressKey('i')); await screen.flush(); await screen.flush()
    expect(rpc).toHaveBeenCalledWith('context.control', { action: 'pin', scope: 'project', path: 'MEMORY.md', revision: 0, generation: 'revision-0' })
    await vi.waitFor(() => expect(screen.captureCharFrame()).toContain('pinned'))
    expect(screen.captureCharFrame()).toContain('applies on the next turn')
    act(() => screen.mockInput.pressKey('x')); await screen.flush(); await screen.flush()
    await vi.waitFor(() => expect(screen.captureCharFrame()).toContain('Wait for the active turn'))
    expect(screen.captureCharFrame()).toContain('a durable fact')
  } finally { act(() => screen.renderer.destroy()) }
})

it('rejects a mismatched response without replacing the last valid page', async () => {
  const rpc = vi.fn(async (_method: string, params: Record<string, unknown>) => params.section === 'memory'
    ? { ...page, entries: [{ ...page.entries[0]!, text: 'wrong page content' }] } : page)
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ContextOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 30 })
  try {
    await screen.flush(); await screen.flush()
    act(() => screen.mockInput.pressKey('TAB')); await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).toContain('does not match')
    expect(screen.captureCharFrame()).toContain('mandatory policy')
    expect(screen.captureCharFrame()).not.toContain('wrong page content')
  } finally { act(() => screen.renderer.destroy()) }
})
