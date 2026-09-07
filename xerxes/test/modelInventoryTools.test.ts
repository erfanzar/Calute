// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { registerModelInventoryTool } from '../src/tools/modelInventoryTools.js'
import type { ToolCall } from '../src/types/toolCalls.js'
import { modelInventory } from '../src/runtime/modelInventory.js'

test('model-facing inventory recovers when a first-page call carries an empty revision', async () => {
  const registry = new ToolRegistry()
  registerModelInventoryTool(registry, (_session, params, signal) => modelInventory({
    profiles: () => [{ name: 'codex', provider: 'openai-codex', model: 'fixture', active: true }],
    discover: async () => ({ source: 'fixture', models: [{ id: 'fixture' }] }),
    reasoning: async () => ({ efforts: ['low', 'high'], source: 'provider_reported', shape: 'effort' }),
  }, params, signal))
  const result = JSON.parse(await registry.execute({ id: 'inventory', type: 'function', function: {
    name: 'list_available_models', arguments: { provider_profile: 'codex', query: '', revision: '', offset: 0, include_usage: true },
  } }, { sessionId: 'owner', metadata: {} }))
  expect(result.ok).toBe(true)
  expect(result.entries[0]).toMatchObject({ model: 'fixture', reasoning_efforts: ['low', 'high'] })
  expect(result.quota.status).toBe('unknown')
})
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
