// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { expect, it, vi } from 'vitest'
import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { CustomAgentEditor } from '../opentui/customAgentEditor.js'
import { DARK_THEME } from '../theme.js'
it.each([[220, 65], [40, 18]])('edits actual markdown and preserves the draft on save failure at %sx%s', async (width, height) => {
  const content = '---\nname: reviewer\ndescription: Review code\n---\nFind defects.'
  const rpc = vi.fn(async (method: string) => method === 'agentPreset.projectList' ? { ok: true, agents: [{ id: 'reviewer', description: 'Review code' }] } : method === 'agentPreset.projectRead' ? { ok: true, content, revision: 'v1' } : { ok: false, error: 'Changed on disk' })
  const onClose = vi.fn()
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><CustomAgentEditor t={DARK_THEME} onClose={onClose} /></GatewayProvider>, { width, height })
  try {
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('reviewer')
    act(() => screen.mockInput.pressKey('RETURN'))
    await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Find defects.')
    act(() => screen.mockInput.pressKey('s', { ctrl: true }))
    await screen.flush(); await screen.flush()
    expect(rpc).toHaveBeenCalledWith('agentPreset.projectWrite', { id: 'reviewer', content, revision: 'v1' })
    expect(screen.captureCharFrame()).toContain('Changed on disk')
    expect(screen.captureCharFrame()).toContain('Find defects.')
    expect(onClose).not.toHaveBeenCalled()
  } finally { act(() => screen.renderer.destroy()) }
})
it('creates through the daemon and cancels edits without saving', async () => {
  const rpc = vi.fn(async (method: string) => method === 'agentPreset.projectList' ? { ok: true, agents: [] } : { ok: true })
  const onClose = vi.fn()
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><CustomAgentEditor t={DARK_THEME} onClose={onClose} /></GatewayProvider>, { width: 100, height: 28 })
  try {
    await screen.flush()
    act(() => screen.mockInput.pressKey('n'))
    await screen.flush()
    act(() => screen.mockInput.pressKey('ESCAPE'))
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('No custom agents.') })
    expect(rpc.mock.calls.some(([method]) => method === 'agentPreset.projectWrite')).toBe(false)
    expect(onClose).not.toHaveBeenCalled()
    act(() => screen.mockInput.pressKey('n'))
    await screen.flush()
    act(() => screen.mockInput.pressKey('s', { ctrl: true }))
    await screen.flush(); await screen.flush()
    expect(rpc).toHaveBeenCalledWith('agentPreset.projectWrite', { id: '', content: expect.stringContaining('name: new-agent'), revision: null })
    expect(screen.captureCharFrame()).toContain('No custom agents.')
  } finally { act(() => screen.renderer.destroy()) }
})
