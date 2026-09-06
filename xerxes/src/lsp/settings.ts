// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { parseLspConfig, type LspServerConfig } from './config.js'
import type { LspManager } from './manager.js'
import type { LspSettingsStore } from './settingsStore.js'
import type { LspSettingsSnapshot } from './settingsStore.js'

export interface LspSettingsView {
  readonly revision: string
  readonly servers: readonly {
    readonly name: string
    readonly enabled: boolean
    readonly languageId: string
    readonly extensions: readonly string[]
    readonly timeoutMs: number
    readonly configuredFields: readonly string[]
  }[]
  readonly warnings: readonly string[]
}
const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value)

/** The editor can replace secret-bearing fields without reading their saved values. */
export function lspSettingsView(snapshot: LspSettingsSnapshot): LspSettingsView {
  return {
    revision: snapshot.revision,
    servers: snapshot.servers.map(server => ({
      name: server.name, enabled: server.enabled, languageId: server.languageId,
      extensions: [...server.extensions], timeoutMs: server.timeoutMs ?? 30_000,
      configuredFields: ['command', ...(server.args !== undefined ? ['args'] : []), ...(server.env !== undefined ? ['env'] : [])],
    })),
    warnings: [...snapshot.warnings ?? []],
  }
}

/** Pure preparation: validate the entire candidate before changing disk or live hosts. */
export function prepareLspSettingsEdit(snapshot: LspSettingsSnapshot, request: unknown): readonly LspServerConfig[] {
  if (!record(request) || Object.keys(request).some(key => !['name', 'revision', 'action', 'changes'].includes(key))) throw new Error('Invalid LSP settings request')
  if (typeof request.revision !== 'string' || request.revision !== snapshot.revision) throw new Error('LSP settings changed; reload before saving')
  if (typeof request.name !== 'string' || !request.name.trim() || request.name !== request.name.trim() || request.name.length > 128 || request.name.includes('\0')) throw new Error('LSP server name required')
  if (typeof request.action !== 'string' || !['create', 'update', 'remove'].includes(request.action)) throw new Error('LSP settings action must be create, update or remove')
  const existing = snapshot.servers.find(server => server.name === request.name)
  if (request.action === 'create' ? !!existing : !existing) throw new Error(request.action === 'create' ? 'LSP server name already exists' : 'LSP server is not configured')
  if (request.action === 'remove') {
    if (request.changes !== undefined) throw new Error('Remove does not accept field changes')
    return parseLspConfig({ servers: snapshot.servers.filter(server => server.name !== request.name) })
  }
  if (!record(request.changes) || Object.keys(request.changes).some(key => !['enabled', 'command', 'args', 'env', 'languageId', 'extensions', 'timeoutMs'].includes(key))) throw new Error('Invalid LSP settings fields; names cannot be changed')
  const candidate: Record<string, unknown> = { ...existing, name: request.name }
  for (const [key, value] of Object.entries(request.changes)) {
    if (value === null && ['args', 'env', 'timeoutMs'].includes(key)) delete candidate[key]
    else candidate[key] = value
  }
  return parseLspConfig({ servers: request.action === 'create' ? [...snapshot.servers, candidate] : snapshot.servers.map(server => server.name === request.name ? candidate : server) })
}

/** Persist and publish one validated edit, returning only masked client data. */
export async function saveLspSettings(manager: LspManager, store: LspSettingsStore, request: unknown, signal?: AbortSignal): Promise<LspSettingsView> {
  const previous = store.read()
  const servers = prepareLspSettingsEdit(previous, request)
  let saved: LspSettingsSnapshot | undefined
  await manager.reconfigure({ servers }, () => { saved = store.save({ servers }, previous.revision); return undefined }, signal)
  if (!saved) throw new Error('LSP settings were not committed')
  return lspSettingsView(saved)
}
