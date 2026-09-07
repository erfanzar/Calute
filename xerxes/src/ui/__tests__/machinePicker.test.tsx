// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { MachinePicker } from '../opentui/machinePicker.js'
import { DEFAULT_THEME } from '../theme.js'

describe('machine picker', () => {
  it('connects through the saved-machine RPC and keeps failures in the picker', async () => {
    const machine = { alias: 'gpu', target: 'host', workspacePath: '/work/repo' }
    const rpc = vi.fn(async (_method: string, args: { command: string }) => args.command === 'machine list' ? { ok: true, machines: [machine] } : { ok: true, machine })
    const connect = vi.fn(async () => { throw new Error('SSH unavailable') })
    const onCancel = vi.fn()
    const setup = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MachinePicker t={DEFAULT_THEME} onCancel={onCancel} connect={connect} /></GatewayProvider>, { width: 140, height: 40 })
    try {
      await act(async () => { await Bun.sleep(0) })
      await setup.flush()
      expect(setup.captureCharFrame()).toContain('gpu · host')
      await act(async () => { setup.renderer.keyInput.processParsedKey({ name: 'return', raw: '\r', sequence: '\r', ctrl: false, shift: false, meta: false, option: false, eventType: 'press', source: 'raw' }); await Bun.sleep(0) })
      await setup.flush()
      expect(rpc).toHaveBeenCalledWith('slash.exec', { command: 'machine connect gpu' })
      expect(connect).toHaveBeenCalledOnce()
      expect(setup.captureCharFrame()).toContain('SSH unavailable')
      expect(onCancel).not.toHaveBeenCalled()
    } finally { act(() => setup.renderer.destroy()) }
  })
})

it('shows setup guidance and closes with Escape in a narrow terminal', async () => {
  const onCancel = vi.fn()
  const setup = await testRender(<GatewayProvider value={{ rpc: async () => ({ ok: true, machines: [] }) } as unknown as GatewayServices}><MachinePicker t={DEFAULT_THEME} onCancel={onCancel} /></GatewayProvider>, { width: 50, height: 18 })
  try {
    await act(async () => { await Bun.sleep(0) })
    await setup.flush()
    expect(setup.captureCharFrame()).toContain('No remote workspaces')
    act(() => setup.renderer.keyInput.processParsedKey({ name: 'escape', raw: '\u001b', sequence: '\u001b', ctrl: false, shift: false, meta: false, option: false, eventType: 'press', source: 'raw' }))
    expect(onCancel).toHaveBeenCalledOnce()
  } finally { act(() => setup.renderer.destroy()) }
})
