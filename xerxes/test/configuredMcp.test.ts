// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { startConfiguredMcpServers } from '../src/mcp/configured.js'
import { MCPManager } from '../src/mcp/manager.js'

test.each([false, true])('shared MCP startup preserves user precedence and workspace trust (trusted=%s)', async allowWorkspace => {
  const root = await mkdtemp(join(tmpdir(), 'xm-config-'))
  const home = join(root, 'home'), workspace = join(root, 'workspace')
  const connected: string[] = [], messages: string[] = []
  const manager = new MCPManager({ clientFactory: config => ({
    config, tools: [], resources: [], prompts: [],
    async connect() { connected.push(config.command!); if (config.name === 'broken') throw new Error('bad private-config-secret') },
    async disconnect() {}, async callTool() { return { content: [] } },
    async readResource() { return { contents: [] } }, async getPrompt() { return { messages: [] } },
  }) })
  try {
    await mkdir(home); await mkdir(workspace)
    await writeFile(join(home, 'mcp.json'), JSON.stringify({
      off: { command: 'disabled-user', enabled: false },
      working: { command: 'working-user' },
      broken: { command: 'broken-user', env: { TOKEN: 'private-config-secret' } },
      invalid: { command: 'must-not-launch', enabled: 'false', env: { TOKEN: 'private-config-secret' } },
    }))
    await writeFile(join(workspace, '.mcp.json'), JSON.stringify([
      { name: 'off', command: 'must-not-override-disabled' },
      { name: ' working ', command: 'must-not-override-user' },
      { name: 'project', command: 'project-first' },
      { name: 'project', command: 'project-duplicate' },
    ]))
    await startConfiguredMcpServers(manager, { home, workspace, allowWorkspace, report: message => messages.push(message) })
    expect(connected).toEqual(allowWorkspace ? ['working-user', 'broken-user', 'project-first'] : ['working-user', 'broken-user'])
    expect(manager.status('off')?.state).toBe('disabled')
    expect(manager.status('broken')?.state).toBe('failed')
    expect(manager.status('invalid')).toBeUndefined()
    expect(messages.join('\n')).toContain('enabled must be a boolean')
    expect(messages.join('\n')).not.toContain('private-config-secret')
    expect(messages.join('\n')).toContain(allowWorkspace ? 'higher-priority' : 'not trusted')
  } finally { await manager.disconnectAll(); await rm(root, { recursive: true, force: true }) }
})
