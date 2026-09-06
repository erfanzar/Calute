// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { LspHostOptions } from './host.js'

export interface LspServerConfig extends Omit<LspHostOptions, 'cwd' | 'connectionFactory'> {
  readonly name: string
  readonly extensions: readonly string[]
  readonly enabled: boolean
}
const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value)
const text = (value: unknown, limit: number): value is string => typeof value === 'string' && value.trim().length > 0 && value.length <= limit && !value.includes('\0')

/** User-owned executable configuration. Never load this from an untrusted workspace implicitly. */
export function parseLspConfig(value: unknown): readonly LspServerConfig[] {
  if (!record(value) || Object.keys(value).some(key => key !== 'servers') || !Array.isArray(value.servers) || value.servers.length > 32) throw new Error('LSP configuration requires a servers array with at most 32 entries')
  const names = new Set<string>()
  const extensions = new Set<string>()
  return value.servers.map((server): LspServerConfig => {
    if (!record(server) || Object.keys(server).some(key => !['name', 'command', 'args', 'env', 'languageId', 'extensions', 'enabled', 'timeoutMs'].includes(key))) throw new Error('Invalid LSP server fields')
    if (!text(server.name, 128) || names.has(server.name) || server.name !== server.name.trim()) throw new Error('LSP server names must be unique nonempty strings')
    names.add(server.name)
    if (!text(server.command, 4096) || !text(server.languageId, 128)) throw new Error('LSP server requires command and languageId')
    if (server.enabled !== undefined && typeof server.enabled !== 'boolean') throw new Error('LSP enabled must be boolean')
    const enabled = server.enabled !== false
    if (!Array.isArray(server.extensions) || !server.extensions.length || server.extensions.length > 64 || !server.extensions.every(ext => typeof ext === 'string' && /^\.[a-zA-Z0-9][a-zA-Z0-9._+-]{0,63}$/.test(ext))) throw new Error('LSP extensions must be nonempty dot-prefixed file suffixes')
    const suffixes = server.extensions as string[]
    for (const suffix of suffixes) {
      if (enabled && extensions.has(suffix)) throw new Error('Only one enabled LSP server may own each file suffix')
      if (enabled) extensions.add(suffix)
    }
    if (server.args !== undefined && (!Array.isArray(server.args) || server.args.length > 128 || !server.args.every(arg => typeof arg === 'string' && arg.length <= 8192 && !arg.includes('\0')))) throw new Error('LSP args must be a bounded string array')
    if (server.env !== undefined && (!record(server.env) || Object.keys(server.env).length > 128 || !Object.entries(server.env).every(([key, val]) => /^[A-Za-z_][A-Za-z0-9_]*$/.test(key) && typeof val === 'string' && val.length <= 8192 && !val.includes('\0')))) throw new Error('LSP env must contain valid environment variable strings')
    if (server.timeoutMs !== undefined && (!Number.isSafeInteger(server.timeoutMs) || (server.timeoutMs as number) < 1 || (server.timeoutMs as number) > 120_000)) throw new Error('LSP timeoutMs must be 1–120000')
    return { name: server.name, command: server.command, languageId: server.languageId, extensions: [...suffixes], enabled,
      ...(server.args !== undefined ? { args: [...server.args as string[]] } : {}),
      ...(server.env !== undefined ? { env: { ...server.env as Record<string, string> } } : {}),
      ...(server.timeoutMs !== undefined ? { timeoutMs: server.timeoutMs as number } : {}),
    }
  })
}
