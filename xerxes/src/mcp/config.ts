// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * `~/.xerxes/mcp.json` loader. One bad entry must not take down the daemon's
 * whole MCP setup: invalid entries become collected warnings (observable in
 * the daemon log), valid siblings still load, and a missing file simply means
 * no servers. Secrets stay in the file — the loader never logs entry contents.
 */

import { readMcpDocument } from "./document.js";

import type { MCPServerConfig, MCPTransport } from "./types.js";

export interface McpLoadedConfig {
  readonly servers: readonly MCPServerConfig[];
  readonly warnings: readonly string[];
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);

const text = (value: unknown): string => (typeof value === "string" ? value.trim() : "");

export function loadMcpConfig(path: string): McpLoadedConfig {
  let source: string | undefined;
  try { source = readMcpDocument(path); } catch {
    return { servers: [], warnings: [`mcp.json at ${path} could not be read; use a readable regular UTF-8 file of at most 1 MiB`] };
  }
  return source === undefined ? { servers: [], warnings: [] } : parseMcpConfigDocument(source, path);
}

/** Decode a captured document so settings revision checks use exactly these bytes. */
export function parseMcpConfigDocument(source: string, path: string): McpLoadedConfig {
  let parsed: unknown;
  try {
    parsed = JSON.parse(source);
  } catch {
    return {
      servers: [],
      // Parser errors may quote source text containing credentials.
      warnings: [`mcp.json at ${path} could not be read or is not valid JSON; check file access and JSON syntax`],
    };
  }
  const warnings: string[] = [];
  let entries: unknown[] = [];
  if (Array.isArray(parsed)) {
    entries = parsed;
  } else if (isRecord(parsed) && Array.isArray(parsed.servers)) {
    entries = parsed.servers;
  } else if (isRecord(parsed)) {
    // Loose map shape: { "name": { ...serverConfig } }.
    entries = Object.entries(parsed).map(([name, config]) => ({
      ...(isRecord(config) ? config : { __invalid: config }),
      name,
    }));
  } else {
    return { servers: [], warnings: [`mcp.json at ${path} is neither a server list nor a map`] };
  }
  const servers: MCPServerConfig[] = [];
  for (let index = 0; index < entries.length; index++) {
    const entry = entries[index];
    const label = isRecord(entry) && text(entry.name) ? `'${text(entry.name)}'` : `#${index + 1}`;
    if (!isRecord(entry)) {
      warnings.push(`mcp.json server ${label} is not an object — skipped`);
      continue;
    }
    const name = text(entry.name);
    if (!name) {
      warnings.push(`mcp.json server #${index + 1} has no name — skipped`);
      continue;
    }
    const result = parseMcpServerConfig(entry);
    if (!result.ok) warnings.push(`mcp.json server '${name.slice(0, 128).replace(/[\x00-\x1f\x7f]/g, '?')}': ${result.error} — skipped`);
    else servers.push(result.config);
  }
  return { servers, warnings };
}

/** Validate settings without executing a transport or including values in errors. */
export function parseMcpServerConfig(value: unknown): { ok: true; config: MCPServerConfig } | { ok: false; error: string } {
  const fail = (error: string) => ({ ok: false as const, error });
  if (!isRecord(value)) return fail('expected a server object');
  const name = text(value.name);
  if (!name || name.length > 128 || /[\x00-\x1f\x7f]/.test(name)) return fail('name must be 1–128 printable characters');
  const known = new Set(['name', 'command', 'args', 'env', 'transport', 'url', 'headers', 'enabled', 'allowPrivateNetwork', 'timeoutMs', 'protocolVersion', 'clientInfo', 'clientCapabilities']);
  if (Object.keys(value).some(key => !known.has(key))) return fail('unknown setting; check the documented MCP configuration fields');
  const config: {
    name: string; command?: string; args?: string[]; env?: Record<string, string>;
    transport?: MCPTransport; url?: string; headers?: Record<string, string>;
    enabled?: boolean; allowPrivateNetwork?: boolean; timeoutMs?: number;
    protocolVersion?: string; clientInfo?: { name: string; version: string }; clientCapabilities?: Record<string, unknown>;
  } = { name };
  for (const field of ['enabled', 'allowPrivateNetwork'] as const) {
    if (value[field] !== undefined) {
      if (typeof value[field] !== 'boolean') return fail(`${field} must be a boolean`);
      config[field] = value[field];
    }
  }
  for (const field of ['command', 'url', 'protocolVersion'] as const) {
    if (value[field] !== undefined) {
      if (typeof value[field] !== 'string' || !value[field].trim() || value[field].includes('\0')) return fail(`${field} must be a nonempty string without NUL characters`);
      config[field] = value[field];
    }
  }
  if (value.transport !== undefined) {
    if (value.transport !== 'stdio' && value.transport !== 'sse' && value.transport !== 'streamable_http') return fail('transport must be stdio, sse or streamable_http');
    config.transport = value.transport;
  }
  if ((config.transport ?? 'stdio') === 'stdio') {
    if (!config.command) return fail('stdio requires command; for a URL set transport to sse or streamable_http');
    if (config.url) return fail('stdio cannot also specify url');
  } else {
    if (!config.url || config.command) return fail('HTTP transport requires url and cannot specify command');
    try {
      const url = new URL(config.url);
      if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) return fail('url must be HTTP(S) without embedded credentials; use headers for authentication');
    } catch { return fail('url must be an absolute HTTP(S) endpoint'); }
  }
  if (value.args !== undefined) {
    if (!Array.isArray(value.args) || !value.args.every((arg): arg is string => typeof arg === 'string' && !arg.includes('\0'))) return fail('args must be an array of strings without NUL characters');
    config.args = [...value.args];
  }
  for (const field of ['env', 'headers'] as const) {
    const record = value[field];
    if (record === undefined) continue;
    if (!isRecord(record)) return fail(`${field} must be an object of string values`);
    const entries: Array<[string, string]> = [];
    for (const [key, item] of Object.entries(record)) {
      if (!key || key.includes('\0') || typeof item !== 'string' || item.includes('\0')) return fail(`${field} must contain nonempty keys and string values without NUL characters`);
      if (field === 'env' && key.includes('=')) return fail('env keys cannot contain equals signs');
      entries.push([key, item]);
    }
    config[field] = Object.fromEntries(entries);
    if (field === 'headers') {
      try { new Headers(config.headers); } catch { return fail('headers contain an invalid HTTP header name or value'); }
    }
  }
  if (value.timeoutMs !== undefined) {
    if (typeof value.timeoutMs !== 'number' || !Number.isFinite(value.timeoutMs) || value.timeoutMs <= 0 || value.timeoutMs > 2_147_483_647) return fail('timeoutMs must be positive and within the native timer range');
    config.timeoutMs = value.timeoutMs;
  }
  if (value.clientInfo !== undefined) {
    if (!isRecord(value.clientInfo) || !text(value.clientInfo.name) || !text(value.clientInfo.version)) return fail('clientInfo requires string name and version');
    config.clientInfo = { name: text(value.clientInfo.name), version: text(value.clientInfo.version) };
  }
  if (value.clientCapabilities !== undefined) {
    if (!isRecord(value.clientCapabilities)) return fail('clientCapabilities must be an object');
    config.clientCapabilities = { ...value.clientCapabilities };
  }
  return { ok: true, config };
}
