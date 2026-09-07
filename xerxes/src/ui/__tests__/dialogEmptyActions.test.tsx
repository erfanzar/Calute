// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act, type ReactNode } from 'react'
import { expect, it } from 'vitest'
import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { ScheduleOverlay } from '../opentui/scheduleOverlay.js'
import { MonitorOverlay } from '../opentui/monitorOverlay.js'
import { CustomAgentEditor } from '../opentui/customAgentEditor.js'
import { McpSettingsOverlay } from '../opentui/mcpSettingsOverlay.js'
import { LspSettingsOverlay } from '../opentui/lspSettingsOverlay.js'
import { DARK_THEME as t } from '../theme.js'

const fixtures: [string, ReactNode, string, string, string][] = [
  ['schedules', <ScheduleOverlay t={t} />, 'No workspace schedules.', 'n', 'New schedule'],
  ['monitors', <MonitorOverlay t={t} />, 'No watches in this session.', 'n', 'New monitor'],
  ['agents', <CustomAgentEditor t={t} onClose={() => undefined} />, 'No custom agents.', 'n', 'new-agent'],
  ['mcp', <McpSettingsOverlay t={t} />, 'No MCP servers configured', 'F2', 'New server'],
  ['lsp', <LspSettingsOverlay t={t} />, 'No LSP servers configured', 'F2', 'New server'],
]

it.each(fixtures)('%s empty state exposes a working creation path and preserves cancellation', async (_name, component, empty, key, editor) => {
  const calls: string[] = []
  const rpc = async (method: string) => {
    calls.push(method)
    return { ok: true, revision: 'a'.repeat(64), jobs: [], monitors: [], agents: [], servers: [] }
  }
  for (const [width, height] of [[220, 65], [40, 18]]) {
    const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}>{component}</GatewayProvider>, { width, height })
    try {
      await act(async () => { await Bun.sleep(0) }); await screen.flush()
      expect(screen.captureCharFrame()).toContain(empty)
      const before = [...calls]
      await act(async () => { screen.mockInput.pressKey(key); await Bun.sleep(0) }); await screen.flush()
      expect(screen.captureCharFrame()).toContain(editor)
      await act(async () => { screen.mockInput.pressKey('ESCAPE'); await Bun.sleep(0) }); await screen.flush()
      expect(calls.slice(before.length).some(method => /(?:\.set|\.create|\.projectWrite)$/.test(method))).toBe(false)
    } finally { act(() => screen.renderer.destroy()) }
  }
})
