// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { loadMcpConfig, parseMcpServerConfig } from '../src/mcp/config.js'

describe('loadMcpConfig', () => {
  const cleanup: string[] = []
  afterEach(async () => {
    for (const dir of cleanup.splice(0)) await rm(dir, { recursive: true, force: true })
  })

  const tempConfig = async (content: string | null): Promise<string> => {
    const dir = await mkdtemp(join(tmpdir(), 'xerxes-mcp-config-'))
    cleanup.push(dir)
    const path = join(dir, 'mcp.json')
    if (content !== null) await writeFile(path, content)
    return path
  }

  test('a missing file means no servers and no warnings', async () => {
    const path = await tempConfig(null)
    expect(loadMcpConfig(path)).toEqual({ servers: [], warnings: [] })
  })

  test('the list shape retains disabled servers for health discovery', async () => {
    const path = await tempConfig(JSON.stringify({
      servers: [
        { name: 'filesystem', command: 'npx', args: ['-y', 'mcp-fs'] },
        { name: 'github', url: 'https://mcp.example.com', transport: 'streamable_http' },
        { name: 'off', command: 'noop', enabled: false },
      ],
    }))
    const config = loadMcpConfig(path)
    expect(config.servers.map(server => server.name)).toEqual(['filesystem', 'github', 'off'])
    expect(config.servers[2]?.enabled).toBe(false)
    expect(config.warnings).toEqual([])
  })

  test('the map shape is accepted with names folded in', async () => {
    const path = await tempConfig(JSON.stringify({
      filesystem: { command: 'npx' },
    }))
    const config = loadMcpConfig(path)
    expect(config.servers).toHaveLength(1)
    expect(config.servers[0]?.name).toBe('filesystem')
  })

  test('invalid entries become warnings; valid siblings still load', async () => {
    const path = await tempConfig(JSON.stringify({
      servers: [
        { command: 'npx' },
        { name: 'broken' },
        { name: 'good', command: 'noop' },
        'not an object',
      ],
    }))
    const config = loadMcpConfig(path)
    expect(config.servers.map(server => server.name)).toEqual(['good'])
    expect(config.warnings).toHaveLength(3)
    expect(config.warnings.some(warning => warning.includes("'broken'"))).toBe(true)
  })

  test('malformed JSON is one actionable warning, not a crash', async () => {
    const path = await tempConfig('{ nope')
    const config = loadMcpConfig(path)
    expect(config.servers).toEqual([])
    expect(config.warnings[0]).toContain('not valid JSON')
  })
})

describe('MCP configuration boundary', () => {
  test.each([
    { enabled: 'false' }, { allowPrivateNetwork: 'true' }, { args: [123] },
    { env: { TOKEN: 42 } }, { headers: { Authorization: ['secret'] } },
    { timeoutMs: 0 }, { timeoutMs: 2 ** 32 }, { transport: 'http' },
    { protocolVersion: 42 }, { clientInfo: { name: 'fixture', version: 42 } },
    { clientCapabilities: [] }, { enable: false }, { command: 'bad\0command' },
  ])('rejects malformed settings before transport construction: %j', fields => {
    expect(parseMcpServerConfig({ name: 'fixture', command: 'bun', ...fields }).ok).toBe(false)
  })
  test('requires an explicit compatible transport and validates endpoints without echoing credentials', () => {
    expect(parseMcpServerConfig({ name: 'fixture', url: 'https://example.com' }).ok).toBe(false)
    for (const url of ['file:///private', 'https://user:private-secret@example.com', 'not a URL']) {
      const result = parseMcpServerConfig({ name: 'fixture', transport: 'streamable_http', url })
      expect(result.ok).toBe(false)
      expect(JSON.stringify(result)).not.toContain('private-secret')
    }
    expect(parseMcpServerConfig({ name: ' fixture ', transport: 'streamable_http', url: 'http://127.0.0.1/mcp', allowPrivateNetwork: true, headers: { Authorization: 'Bearer private-secret' } }))
      .toMatchObject({ ok: true, config: { name: 'fixture', allowPrivateNetwork: true } })
  })
  test('malformed JSON diagnostics never quote the credential-bearing source', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'xm-invalid-'))
    try {
      const path = join(dir, 'mcp.json')
      await writeFile(path, '{"env":{"TOKEN":"private-parse-secret"}, missing-colon}')
      const result = loadMcpConfig(path)
      expect(result.servers).toEqual([])
      expect(result.warnings).toHaveLength(1)
      expect(result.warnings[0]).not.toContain('private-parse-secret')
    } finally { await rm(dir, { recursive: true, force: true }) }
  })
  test('invalid entries preserve valid siblings and disabled settings are validated', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'xm-siblings-'))
    try {
      const path = join(dir, 'mcp.json')
      await writeFile(path, JSON.stringify([
        { name: 'bad', command: 'bun', enabled: 'false', env: { TOKEN: 'private-value' } },
        { name: 'valid', command: 'bun', enabled: false },
      ]))
      const result = loadMcpConfig(path)
      expect(result.servers.map(server => server.name)).toEqual(['valid'])
      expect(result.warnings[0]).toContain('enabled must be a boolean')
      expect(result.warnings[0]).not.toContain('private-value')
    } finally { await rm(dir, { recursive: true, force: true }) }
  })
})
