// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { AgentDefinition } from '../agents/definitions.js'
import type { ToolRegistry } from '../executors/toolRegistry.js'
import type { JsonObject } from '../types/toolCalls.js'
export type ModelInventoryHost = (sessionId: string, params: JsonObject, signal?: AbortSignal) => Promise<unknown>
export function registerModelInventoryTool(registry: ToolRegistry, host: ModelInventoryHost): void {
  registry.register({ type: 'function', function: { name: 'list_available_models', description: 'Discover configured providers and actual model choices before delegating to an agent. Without provider_profile, lists configured profiles. With it, discovers models and runtime-supported reasoning levels, context and output capacities. Set include_usage with provider_profile to request profile-bound subscription usage; unavailable quota stays unknown. Use returned revision with subsequent page offsets; discovery does not change the conversation model.', parameters: { type: 'object', additionalProperties: false, properties: {
    include_usage: { type: 'boolean' },
    provider_profile: { type: 'string', maxLength: 512 }, query: { type: 'string', maxLength: 512 },
    offset: { type: 'integer', minimum: 0 }, limit: { type: 'integer', minimum: 1, maximum: 50 }, revision: { type: 'string', maxLength: 512 },
  } } } }, async (args, context, signal) => {
    if (!context.sessionId?.trim()) throw new Error('Model inventory requires a session')
    signal?.throwIfAborted()
    return JSON.stringify(await host(context.sessionId, args, signal))
  }, 'default', { concurrencySafe: true, destructive: false, openWorld: false, readOnly: true, maxResultBytes: 64000 })
}

/** Only widen built-in catalogs; user-defined allow/exclude rules still apply. */
export function addModelInventoryToBuiltinAgents(definitions: Map<string, AgentDefinition>): void {
  for (const [name, definition] of definitions) {
    if (definition.source === 'built-in') definitions.set(name, { ...definition, tools: [...new Set([...definition.tools, 'list_available_models'])] })
  }
}
