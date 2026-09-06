// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { afterEach, expect, it, vi } from 'vitest'
import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { getOverlayState, patchOverlayState, resetFlowOverlays, resetOverlayState } from '../app/overlayStore.js'
import { MonitorOverlay } from '../opentui/monitorOverlay.js'
import { listMonitors } from '../lib/monitors.js'
import { DARK_THEME } from '../theme.js'
const watch = { id: 'watch-1', terminalId: 'build', match: 'compile error', state: 'watching', expiresAt: 1000, reactionHealth: { state: 'queued', attempts: 1, maxReactions: 3 }, events: [{ text: 'compile error in app.ts' }] }
afterEach(() => resetOverlayState())
it.each([[220, 65], [110, 35], [60, 24], [40, 18]])('renders monitor evidence and controls at %ix%i', async (width, height) => {
  const rpc = vi.fn(async (method: string) => method === 'monitor.list' ? { ok: true, monitors: [watch] } : { ok: true, monitor: watch })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MonitorOverlay t={DARK_THEME} /></GatewayProvider>, { width, height })
  try {
    await screen.flush(); await screen.flush()
    const text = screen.captureCharFrame()
    expect(text).toContain('Monitors')
    expect(text).toContain('compile error')
    expect(text).toContain('S stop')
    expect(text).toContain('Esc close')
  } finally { act(() => screen.renderer.destroy()) }
})
it('stops only the selected monitor and preserves the user-opened overlay across completion', async () => {
  patchOverlayState({ monitors: true }); resetFlowOverlays()
  expect(getOverlayState().monitors).toBe(true)
  const rpc = vi.fn(async (method: string) => method === 'monitor.list' ? { ok: true, monitors: [watch] } : { ok: true, monitor: watch })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MonitorOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 30 })
  try {
    await screen.flush(); await screen.flush()
    act(() => screen.mockInput.pressKey('s'))
    await screen.flush()
    expect(rpc).toHaveBeenCalledWith('monitor.stop', { monitor_id: watch.id })
    act(() => screen.mockInput.pressKey('ESCAPE'))
    await vi.waitFor(() => expect(getOverlayState().monitors).toBe(false))
  } finally { act(() => screen.renderer.destroy()) }
})
it('keeps a stop failure visible rather than claiming the watch stopped', async () => {
  const rpc = vi.fn(async (method: string) => method === 'monitor.list' ? { ok: true, monitors: [watch] } : method === 'monitor.stop' ? { ok: false, error: 'Monitor no longer owned by this session' } : { ok: true, monitor: watch })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MonitorOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 30 })
  try {
    await screen.flush(); await screen.flush()
    act(() => screen.mockInput.pressKey('s'))
    await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Monitor no longer owned')
  } finally { act(() => screen.renderer.destroy()) }
})

it('shows delivery failure separately from a completed monitor and retains its evidence', async () => {
  const completed = { ...watch, state: 'source-ended', deliveryError: 'UI disconnected' }
  const rpc = vi.fn(async (method: string) => method === 'monitor.list' ? { ok: true, monitors: [completed] } : { ok: true, monitor: completed })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MonitorOverlay t={DARK_THEME} /></GatewayProvider>, { width: 220, height: 65 })
  try {
    await vi.waitFor(async () => {
      await screen.flush()
      const frame = screen.captureCharFrame()
      expect(frame).toContain('source-ended')
      expect(frame).toContain('Notification delivery: UI disconnected')
      expect(frame).toContain('compile error in app.ts')
    })
  } finally { act(() => screen.renderer.destroy()) }
})

it.each(['archived', 'interrupted', 'detached'])('renders %s watch evidence without counting it as attached', async state => {
  const stored = { ...watch, state, error: 'No live source is attached.', events: [{ text: 'retained command output' }] }
  const rpc = vi.fn(async (method: string) => method === 'monitor.list' ? { ok: true, monitors: [stored] } : { ok: true, monitor: stored })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MonitorOverlay t={DARK_THEME} /></GatewayProvider>, { width: 220, height: 65 })
  try {
    await screen.flush(); await screen.flush()
    const text = screen.captureCharFrame()
    expect(text).toContain('0 watching')
    expect(text).toContain(state)
    expect(text).toContain('retained command output')
  } finally { act(() => screen.renderer.destroy()) }
})

it.each([
  ['archived', 'cancel-reactions', 'S cancel reactions', true],
  ['archived', null, 'No cancellable work', false],
  ['detached', null, 'Owned by another daemon', false],
] as const)('offers only the applicable control for %s / %s', async (state, stopAction, label, sends) => {
  const stored = { ...watch, state, stopAction }
  const rpc = vi.fn(async (method: string) => method === 'monitor.list' ? { ok: true, monitors: [stored] } : { ok: true, monitor: stored })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MonitorOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 30 })
  try {
    await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).toContain(label)
    act(() => screen.mockInput.pressKey('s'))
    await screen.flush()
    expect(rpc.mock.calls.filter(args => args[0] === 'monitor.stop')).toHaveLength(sends ? 1 : 0)
  } finally { act(() => screen.renderer.destroy()) }
})

it('keeps reaction cancellation discoverable in a narrow terminal', async () => {
  const stored = { ...watch, state: 'archived', stopAction: 'cancel-reactions' }
  const rpc = vi.fn(async (method: string) => method === 'monitor.list' ? { ok: true, monitors: [stored] } : { ok: true, monitor: stored })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MonitorOverlay t={DARK_THEME} /></GatewayProvider>, { width: 40, height: 18 })
  try {
    await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).toContain('S cancel reactions')
    expect(screen.captureCharFrame()).toContain('N new')
  } finally { act(() => screen.renderer.destroy()) }
})

it('shows reaction token totals and blocked admission in the inspector', async () => {
  const budgeted = { ...watch, reactionHealth: { state: 'exhausted', attempts: 1, maxReactions: 3, tokenBudget: { maximum: 20, blocked: true }, usage: { inputTokens: 15, outputTokens: 5, complete: true } } }
  const rpc = vi.fn(async (method: string) => method === 'monitor.list' ? { ok: true, monitors: [budgeted] } : { ok: true, monitor: budgeted })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MonitorOverlay t={DARK_THEME} /></GatewayProvider>, { width: 220, height: 65 })
  try {
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('tokens 20/20') })
    expect(screen.captureCharFrame()).toContain('token admission blocked')
  } finally { act(() => screen.renderer.destroy()) }
})

it('renders a file source distinctly from terminal matching', async () => {
  const fileWatch = { id: 'file-1', terminalId: '', source: { kind: 'file', path: 'src/app.ts', workspace: '/repo' }, trigger: 'change', match: '', state: 'interrupted', expiresAt: 1000, events: [{ text: 'changed metadata' }] }
  const rpc = vi.fn(async (method: string) => method === 'monitor.list' ? { ok: true, monitors: [fileWatch] } : { ok: true, monitor: fileWatch })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MonitorOverlay t={DARK_THEME} /></GatewayProvider>, { width: 110, height: 35 })
  try {
    await screen.flush(); await screen.flush()
    const text = screen.captureCharFrame()
    expect(text).toContain('File changes')
    expect(text).toContain('src/app.ts')
    expect(text).toContain('metadata changes only')
    expect(text).toContain('interrupted')
  } finally { act(() => screen.renderer.destroy()) }
})

it('renders websocket source health and reconnect gap context', async () => {
  const websocket = { id: 'socket-1', terminalId: '', source: { kind: 'websocket', url: 'wss://events.example/ws' }, trigger: 'output', match: 'failure', state: 'interrupted', expiresAt: 1000, sourceStatus: 'Reconnected after an observation gap', events: [{ text: 'failure event' }] }
  const rpc = vi.fn(async (method: string) => method === 'monitor.list' ? { ok: true, monitors: [websocket] } : { ok: true, monitor: websocket })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MonitorOverlay t={DARK_THEME} /></GatewayProvider>, { width: 110, height: 35 })
  try {
    await screen.flush(); await screen.flush()
    const text = screen.captureCharFrame()
    expect(text).toContain('Websocket')
    expect(text).toContain('wss://events.example/ws')
    expect(text).toContain('Text-only server push')
    expect(text).toContain('observation gap')
  } finally { act(() => screen.renderer.destroy()) }
})

it.each([
  { trigger: 'change', terminalId: '', match: '' },
  { trigger: 'output', terminalId: '', match: '' , source: { kind: 'file', path: 'a.txt', workspace: '/repo' } },
  { trigger: 'change', terminalId: 'term-1', match: '', source: { kind: 'terminal', terminalId: 'term-1' } },
  { trigger: 'change', terminalId: '', match: '', source: { kind: 'file', path: '', workspace: '/repo' } },
  { trigger: 'completion', terminalId: '', match: 'failure', source: { kind: 'websocket', url: 'wss://events.example/ws' } },
  { trigger: 'output', terminalId: '', match: 'failure', source: { kind: 'websocket', url: '' } },
] as const)('rejects malformed monitor source %o', async monitor => {
  const rpc = vi.fn(async () => ({ ok: true, monitors: [ { id: 'bad', state: 'watching', expiresAt: 1000, ...monitor } ] }))
  await expect(listMonitors(rpc)).rejects.toThrow(/Invalid monitor source|File monitor source/)
})
