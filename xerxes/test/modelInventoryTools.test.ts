// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { registerModelInventoryTool } from '../src/tools/modelInventoryTools.js'
import type { ToolCall } from '../src/types/toolCalls.js'
test('model inventory tool forwards session-scoped read-only discovery and cancellation', async () => {
  const registry = new ToolRegistry(), seen: unknown[] = []
  registerModelInventoryTool(registry, async (...args) => { seen.push(args); return { ok: true, entries: [] } })
  const call: ToolCall = { id: 'inventory', type: 'function', function: { name: 'list_available_models', arguments: { provider_profile: 'work', limit: 5 } } }
  await expect(registry.execute(call, { metadata: {} })).rejects.toThrow('session')
  expect(JSON.parse(await registry.execute(call, { sessionId: 'owner', metadata: {} }))).toEqual({ ok: true, entries: [] })
  expect(seen[0]).toMatchObject(['owner', { provider_profile: 'work', limit: 5 }, undefined])
  await expect(registry.execute(call, { sessionId: 'owner', metadata: {} }, AbortSignal.abort())).rejects.toThrow()
  expect(seen).toHaveLength(1)
})
