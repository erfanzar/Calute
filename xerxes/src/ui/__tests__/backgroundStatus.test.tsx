// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act, useState } from 'react'
import { afterEach, expect, it, vi } from 'vitest'
import { BackgroundStatus, parseBackgroundStatus } from '../opentui/backgroundStatus.js'
import { DEFAULT_THEME } from '../theme.js'
import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
const rpc = vi.fn()
vi.mock('../app/gatewayContext.js', () => ({ useOptionalGateway: () => gateway }))
const gateway = { rpc, gw: { request: rpc } }
afterEach(() => { vi.useRealTimers(); rpc.mockReset(); resetOverlayState() })

it('renders blue shell/watcher counts next to model settings and opens inspectors', async () => {
  rpc.mockResolvedValue({ ok: true, shells: 2, watchers: 1 })
  const setup = await testRender(<box flexDirection="row"><text>code mode · model · reasoning: medium</text><BackgroundStatus sessionId="chat-a" t={DEFAULT_THEME} /></box>, { width: 220, height: 5 })
  try {
    await act(async () => {}); await setup.flush()
    const line = setup.captureCharFrame().split('\n')[0]!
    expect(line).toContain('reasoning: medium · 2 shells running · 1 watcher active')
    await setup.mockMouse.click(line.indexOf('2 shells') + 1, 0); await setup.flush()
    expect(getOverlayState().terminals).toBe(true)
    await setup.mockMouse.click(line.indexOf('1 watcher') + 1, 0); await setup.flush()
    expect(getOverlayState().monitors).toBe(true)
    expect(rpc).toHaveBeenCalledWith('background.status', { session_id: 'chat-a' })
  } finally { act(() => setup.renderer.destroy()) }
})

it('clears finished work while idle and does not retain stale running counts after an error', async () => {
  rpc.mockResolvedValueOnce({ ok: true, shells: 1, watchers: 0 }).mockRejectedValueOnce(new Error('offline')).mockResolvedValue({ ok: true, shells: 0, watchers: 0 })
  const setup = await testRender(<BackgroundStatus sessionId="chat-a" t={DEFAULT_THEME} />, { width: 80, height: 5 })
  try {
    await act(async () => {}); await setup.flush()
    expect(setup.captureCharFrame()).toContain('1 shell running')
    await act(async () => { await Bun.sleep(2100) }); await setup.flush()
    expect(setup.captureCharFrame()).toContain('background status unavailable')
    expect(setup.captureCharFrame()).not.toContain('shell running')
    await act(async () => { await Bun.sleep(2100) }); await setup.flush()
    expect(setup.captureCharFrame().trim()).toBe('')
  } finally { act(() => setup.renderer.destroy()) }
})

it('rejects invalid counts instead of showing fabricated activity', () => {
  for (const value of [null, { ok: false }, { ok: true, shells: -1, watchers: 1 }, { ok: true, shells: 1, watchers: '2' }]) expect(() => parseBackgroundStatus(value)).toThrow()
})

it('ignores an old session reply arriving after switching chats', async () => {
  let resolveOld!: (value: unknown) => void
  rpc.mockReturnValueOnce(new Promise(resolve => { resolveOld = resolve })).mockResolvedValue({ ok: true, shells: 0, watchers: 0 })
  let switchSession!: (value: string) => void
  function Harness() {
    const [session, setSession] = useState('old')
    switchSession = setSession
    return <BackgroundStatus sessionId={session} t={DEFAULT_THEME} />
  }
  const setup = await testRender(<Harness />, { width: 60, height: 5 })
  try {
    await act(async () => {}); await setup.flush()
    await act(async () => { switchSession('new') }); await setup.flush()
    await act(async () => { resolveOld({ ok: true, shells: 9, watchers: 3 }) }); await setup.flush()
    expect(setup.captureCharFrame().trim()).toBe('')
    expect(rpc).toHaveBeenLastCalledWith('background.status', { session_id: 'new' })
  } finally { act(() => setup.renderer.destroy()) }
})

it('keeps both counters visible in a narrow composer', async () => {
  rpc.mockResolvedValue({ ok: true, shells: 2, watchers: 1 })
  const setup = await testRender(<box flexDirection="row" flexWrap="wrap"><text>code mode · model</text><BackgroundStatus sessionId="chat-a" t={DEFAULT_THEME} /></box>, { width: 40, height: 8 })
  try {
    await act(async () => {}); await setup.flush()
    const frame = setup.captureCharFrame()
    expect(frame).toContain('2 shells running')
    expect(frame).toContain('1 watcher active')
  } finally { act(() => setup.renderer.destroy()) }
})
