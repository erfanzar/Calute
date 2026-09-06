// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, mkdir, rm, realpath } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { registerCoreTools } from '../src/tools/index.js'
import { getActiveSession, runWithActiveSession } from '../src/runtime/sessionContext.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'
const call = (name: string, args: JsonObject): ToolCall => ({ id: crypto.randomUUID(), type: 'function', function: { name, arguments: args } })

test('shared core tools route concurrent file reads and commands to each active session workspace', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-tools-project-'))
  try {
    const a = join(root, 'a'), b = join(root, 'b')
    await Promise.all([mkdir(a), mkdir(b)])
    await Promise.all([Bun.write(join(a, 'marker.txt'), 'project A'), Bun.write(join(b, 'marker.txt'), 'project B')])
    const registry = new ToolRegistry()
    registerCoreTools(registry, { workspaceRoot: a, activeWorkspaceRoot: () => getActiveSession<{ cwd: string }>()?.cwd })
    const results = await Promise.all([a, b].map((cwd, index) => runWithActiveSession({ cwd }, async () => {
      const context = { sessionId: String(index), metadata: {} }
      const content = await registry.execute(call('ReadFile', { file_path: 'marker.txt' }), context)
      const command = JSON.parse(await registry.execute(call('exec_command', { cmd: process.execPath, args: ['-e', 'console.log(process.cwd())'] }), context))
      await expect(registry.execute(call('ReadFile', { file_path: join(index ? a : b, 'marker.txt') }), context)).rejects.toThrow('workspace root')
      return { content, cwd: command.stdout.trim() }
    })))
    expect(results[0]?.content).toContain('project A')
    expect(results[1]?.content).toContain('project B')
    expect(results.map(result => result.cwd)).toEqual([await realpath(a), await realpath(b)])
    expect(getActiveSession()).toBeUndefined()
  } finally { await rm(root, { recursive: true, force: true }) }
})
