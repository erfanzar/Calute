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
it('generates an editable draft without saving automatically', async () => {
  const content = '---\nname: jax-reviewer\ndescription: Review JAX\n---\nCheck array shapes.'
  const rpc = vi.fn(async (method: string) => method === 'agentPreset.projectList' ? { ok: true, agents: [] } : method === 'agentPreset.projectGenerate' ? { ok: true, id: 'jax-reviewer', content } : { ok: true })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><CustomAgentEditor t={DARK_THEME} onClose={() => {}} /></GatewayProvider>, { width: 110, height: 32 })
  try {
    await screen.flush()
    act(() => screen.mockInput.pressKey('g'))
    await screen.flush()
    await act(async () => screen.mockInput.typeText('Review JAX shapes'))
    act(() => screen.mockInput.pressKey('g', { ctrl: true }))
    await screen.flush(); await screen.flush()
    expect(rpc).toHaveBeenCalledWith('agentPreset.projectGenerate', { description: 'Review JAX shapes' })
    expect(screen.captureCharFrame()).toContain('Check array shapes.')
    expect(rpc.mock.calls.some(([method]) => method === 'agentPreset.projectWrite')).toBe(false)
    act(() => screen.mockInput.pressKey('s', { ctrl: true }))
    await screen.flush(); await screen.flush()
    expect(rpc).toHaveBeenCalledWith('agentPreset.projectWrite', { id: '', content, revision: null })
  } finally { act(() => screen.renderer.destroy()) }
})
it('keeps a generation description when the provider fails', async () => {
  const rpc = vi.fn(async (method: string) => method === 'agentPreset.projectList' ? { ok: true, agents: [] } : { ok: false, error: 'Provider unavailable' })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><CustomAgentEditor t={DARK_THEME} onClose={() => {}} /></GatewayProvider>, { width: 100, height: 28 })
  try {
    await screen.flush()
    act(() => screen.mockInput.pressKey('g'))
    await screen.flush()
    await act(async () => screen.mockInput.typeText('Check array shapes'))
    act(() => screen.mockInput.pressKey('g', { ctrl: true }))
    await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Provider unavailable')
    expect(screen.captureCharFrame()).toContain('Check array shapes')
  } finally { act(() => screen.renderer.destroy()) }
})
it.each([[120, 36], [44, 22]])('shows selected descriptions separately from the agent names at %sx%s', async (width, height) => {
  const agents = Array.from({ length: 15 }, (_, i) => ({ id: `specialist-${i}`, description: `Find concurrency defects in scheduler ${i}. Verify cancellation and provide focused regression tests.` }))
  const rpc = vi.fn(async () => ({ ok: true, agents }))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><CustomAgentEditor t={DARK_THEME} onClose={() => {}} /></GatewayProvider>, { width, height })
  try {
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('WHEN TO DELEGATE')
    expect(screen.captureCharFrame()).toContain('G Generate')
    for (let i = 0; i < 14; i++) await act(async () => screen.mockInput.pressKey('ARROW_DOWN'))
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('specialist-14')
    expect(screen.captureCharFrame()).toContain('15/15')
  } finally { act(() => screen.renderer.destroy()) }
})
