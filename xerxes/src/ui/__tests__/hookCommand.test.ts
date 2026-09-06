// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, it, vi } from 'vitest'
import { opsCommands } from '../app/slash/commands/ops.js'
import type { SlashRunCtx } from '../app/slash/types.js'

it('forwards hook inspection to the active daemon session and surfaces unavailable hosts', async () => {
  const request = vi.fn(async () => ({ warning: 'Shell hook inspection unavailable' }))
  const sys = vi.fn()
  const page = vi.fn()
  const context = { sid: 'session', stale: () => false, guardedErr: vi.fn(), gateway: { gw: { request } }, transcript: { sys, page } } as unknown as SlashRunCtx
  const command = opsCommands.find(command => command.name === 'hooks')!
  command.run('list', context, 'hooks')
  await vi.waitFor(() => expect(sys).toHaveBeenCalled())
  expect(request).toHaveBeenCalledWith('slash.exec', { command: 'hooks list', session_id: 'session' })
  expect(String(sys.mock.calls[0]?.[0])).toContain('Shell hook inspection unavailable')
  expect(page).not.toHaveBeenCalled()
})

it('forwards previews and preserves the transcript when the event is invalid', async () => {
  const request = vi.fn(async () => ({ output: 'error: Hook preview: unknown event' }))
  const sys = vi.fn(), page = vi.fn()
  const context = { sid: 'session', stale: () => false, guardedErr: vi.fn(), gateway: { gw: { request } }, transcript: { sys, page } } as unknown as SlashRunCtx
  const command = opsCommands.find(command => command.name === 'hooks')!
  command.run('preview typo ReadFile', context, 'hooks')
  await vi.waitFor(() => expect(sys).toHaveBeenCalled())
  expect(request).toHaveBeenCalledWith('slash.exec', { command: 'hooks preview typo ReadFile', session_id: 'session' })
  expect(page).not.toHaveBeenCalled()
  expect(command.help).toContain('preview')
})

it('forwards event-filtered failure inspection without replacing the transcript', async () => {
  const request = vi.fn(async () => ({ output: '' }))
  const sys = vi.fn(), page = vi.fn()
  const context = { sid: 'session', stale: () => false, guardedErr: vi.fn(), gateway: { gw: { request } }, transcript: { sys, page } } as unknown as SlashRunCtx
  const command = opsCommands.find(command => command.name === 'hooks')!
  command.run('failures PreToolUse', context, 'hooks')
  await vi.waitFor(() => expect(request).toHaveBeenCalledWith('slash.exec', { command: 'hooks failures PreToolUse', session_id: 'session' }))
  expect(page).not.toHaveBeenCalled()
  expect(command.help).toContain('failures')
})

it('forwards workspace review to the current session without replacing chat history', async () => {
  const request = vi.fn(async () => ({ output: '' }))
  const page = vi.fn()
  const context = { sid: 'session', stale: () => false, guardedErr: vi.fn(), gateway: { gw: { request } }, transcript: { sys: vi.fn(), page } } as unknown as SlashRunCtx
  const command = opsCommands.find(command => command.name === 'workspaces')!
  command.run('inspect retained-id', context, 'workspaces')
  await vi.waitFor(() => expect(request).toHaveBeenCalledWith('slash.exec', { command: 'workspaces inspect retained-id', session_id: 'session' }))
  expect(page).not.toHaveBeenCalled()
})
