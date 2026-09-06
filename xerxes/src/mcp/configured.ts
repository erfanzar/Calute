// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { loadMcpConfig } from './config.js'
import type { MCPManager } from './manager.js'

export interface ConfiguredMcpOptions {
  readonly home: string
  readonly workspace: string
  readonly allowWorkspace: boolean
  readonly report: (message: string) => void
  readonly onConnected?: (name: string) => void
}

/** Shared host configuration policy; project modules never override a user registration. */
export async function startConfiguredMcpServers(manager: MCPManager, options: ConfiguredMcpOptions): Promise<void> {
  const user = loadMcpConfig(join(options.home, 'mcp.json'))
  user.warnings.forEach(options.report)
  const servers = [...user.servers]
  const projectPath = join(options.workspace, '.mcp.json')
  if (existsSync(projectPath)) {
    if (!options.allowWorkspace) {
      options.report('project .mcp.json found but workspace config is not trusted — ignored (set XERXES_ALLOW_WORKSPACE_CONFIG=1 to enable)')
    } else {
      const project = loadMcpConfig(projectPath)
      project.warnings.forEach(options.report)
      const known = new Set(servers.map(server => server.name.trim()))
      for (const server of project.servers) {
        if (known.has(server.name.trim())) {
          options.report(`project server '${server.name}' ignored — a higher-priority registration with that name exists`)
          continue
        }
        known.add(server.name.trim())
        servers.push(server)
      }
    }
  }
  await Promise.all(servers.map(async server => {
    const connected = await manager.addServer(server)
    if (connected) options.onConnected?.(server.name)
    else if (server.enabled !== false) {
      options.report(`server '${server.name}' not connected${manager.lastFailure(server.name)?.error ? `: ${manager.lastFailure(server.name)!.error}` : ' (duplicate registration)'}`)
    }
  }))
}
