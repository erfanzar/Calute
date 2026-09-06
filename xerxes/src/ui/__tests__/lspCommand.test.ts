// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, it, vi } from 'vitest'
import { opsCommands } from '../app/slash/commands/ops.js'
import type { SlashRunCtx } from '../app/slash/types.js'

it('inspects LSP through the active session without replacing the conversation', async () => {
  const request = vi.fn(async () => ({ output: 'Language servers · fixture · idle' }))
  const sys = vi.fn(), page = vi.fn()
  const context = { sid: 'session', stale: () => false, guardedErr: vi.fn(), gateway: { gw: { request } }, transcript: { sys, page } } as unknown as SlashRunCtx
  const command = opsCommands.find(command => command.name === 'lsp')!
  command.run('status', context, 'lsp')
  await vi.waitFor(() => expect(request).toHaveBeenCalledWith('slash.exec', { command: 'lsp status', session_id: 'session' }))
  expect(page).not.toHaveBeenCalled()
  expect(command.help).toContain('status')
})

it('forwards host release through the active session without replacing chat', async () => {
  const request = vi.fn(async () => ({ output: 'Language server host released' }))
  const page = vi.fn()
  const context = { sid: 'selected', stale: () => false, guardedErr: vi.fn(), gateway: { gw: { request } }, transcript: { sys: vi.fn(), page } } as unknown as SlashRunCtx
  opsCommands.find(command => command.name === 'lsp')!.run('release typescript', context, 'lsp')
  await vi.waitFor(() => expect(request).toHaveBeenCalledWith('slash.exec', { command: 'lsp release typescript', session_id: 'selected' }))
  expect(page).not.toHaveBeenCalled()
})
