// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises'
import { connect, type Socket } from 'node:net'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { mcpRuntimeToolName } from '../src/tools/mcpTools.js'

interface Frame { id?: number; result?: Record<string, unknown>; error?: unknown; params?: { type?: string; payload?: unknown } }
function bounded<T>(promise: Promise<T>, milliseconds = 10000): Promise<T> {
  let timer: ReturnType<typeof setTimeout>
  return Promise.race([promise, new Promise<never>((_, reject) => { timer = setTimeout(() => reject(new Error('fixture timeout')), milliseconds) })])
    .finally(() => clearTimeout(timer))
}

test.each(['daemon', 'oneshot', 'acp'] as const)('normal CLI %s publishes and executes configured MCP tools', async mode => {
  const root = await mkdtemp(join(tmpdir(), 'xm-'))
  const home = join(root, 'home'), socketPath = join(root, 's.sock'), marker = join(root, 'called')
  const toolName = mcpRuntimeToolName('fixture', 'echo')
  let sawSchema = false
  const provider = Bun.serve({ hostname: '127.0.0.1', port: 0, async fetch(request) {
    const body = await request.json() as { tools?: Array<{ function?: { name?: string } }>; messages?: Array<{ role?: string; content?: unknown }> }
    const hasSchema = body.tools?.some(tool => tool.function?.name === toolName) ?? false
    sawSchema ||= hasSchema
    const lastUser = body.messages?.findLastIndex(message => message.role === 'user') ?? -1
    const hasResult = body.messages?.some((message, index) => index > lastUser && message.role === 'tool')
    const chunk = hasSchema && !hasResult
      ? { choices: [{ delta: { tool_calls: [{ index: 0, id: 'mcp-call', type: 'function', function: { name: toolName, arguments: '{"value":"wired"}' } }] }, finish_reason: 'tool_calls' }] }
      : { choices: [{ delta: { content: 'MCP verified' }, finish_reason: 'stop' }] }
    return new Response(`data: ${JSON.stringify(chunk)}\n\ndata: [DONE]\n\n`, { headers: { 'content-type': 'text/event-stream' } })
  } })
  let child: ReturnType<typeof Bun.spawn> | undefined
  let socket: Socket | undefined
  let stderr = Promise.resolve('')
  try {
    await mkdir(join(home, 'daemon'), { recursive: true })
    await writeFile(join(home, 'daemon', 'config.json'), JSON.stringify({ project_directory: root, runtime: {
      model: 'gpt-4o', provider: 'openai', base_url: `${provider.url}v1`, api_key: 'fixture-key', permission_mode: 'accept-all',
    } }))
    const script = join(root, 'mcp.ts')
    await writeFile(script, `let buffer = ''; const decoder = new TextDecoder();
for await (const chunk of Bun.stdin.stream()) {
  buffer += decoder.decode(chunk, { stream: true });
  let end; while ((end = buffer.indexOf('\\n')) >= 0) {
    const line = buffer.slice(0, end); buffer = buffer.slice(end + 1); if (!line.trim()) continue;
    const request = JSON.parse(line); if (request.id === undefined) continue;
    let result = {};
    if (request.method === 'initialize') result = { protocolVersion: '2024-11-05', capabilities: { tools: {} }, serverInfo: { name: 'fixture', version: '1' } };
    if (request.method === 'tools/list') result = { tools: [{ name: 'echo', description: 'Fixture echo', inputSchema: { type: 'object', properties: { value: { type: 'string' } }, required: ['value'] } }] };
    if (request.method === 'tools/call') { await Bun.write(${JSON.stringify(marker)}, request.params.arguments.value); result = { content: [{ type: 'text', text: request.params.arguments.value }] }; }
    process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: request.id, result }) + '\\n');
  }
}`)
    await writeFile(join(home, 'mcp.json'), JSON.stringify({ fixture: { command: process.execPath, args: [script] } }))
    const args = mode === 'daemon' ? ['daemon', '--project-dir', root, '--socket', socketPath]
      : mode === 'acp' ? ['acp', '--project-dir', root] : ['Use the MCP fixture echo with value wired.']
    const subprocess = Bun.spawn([process.execPath, join(import.meta.dir, '../src/cli.ts'), ...args], {
      cwd: root, env: { ...process.env, XERXES_HOME: home, XERXES_DEFERRED_TOOL_LOADING: '0' }, stdin: 'pipe', stdout: 'pipe', stderr: 'pipe',
    })
    child = subprocess
    stderr = new Response(subprocess.stderr).text()
    if (mode === 'oneshot') {
      const output = new Response(subprocess.stdout).text()
      expect(await bounded(subprocess.exited)).toBe(0)
      expect(await output).toContain('MCP verified')
      expect(await Bun.file(marker).text()).toBe('wired')
      expect(sawSchema).toBe(true)
      return
    }
    if (mode === 'acp') {
      const responses = new Map<number, (frame: Frame) => void>()
      let nextId = 0
      const reading = (async () => {
        let buffer = ''; const decoder = new TextDecoder()
        const reader = subprocess.stdout.getReader()
        try { while (true) {
          const chunk = await reader.read()
          if (chunk.done) break
          buffer += decoder.decode(chunk.value, { stream: true }); let end
          while ((end = buffer.indexOf('\n')) >= 0) {
            const line = buffer.slice(0, end); buffer = buffer.slice(end + 1); if (!line) continue
            const frame = JSON.parse(line) as Frame
            if (frame.id !== undefined) { responses.get(frame.id)?.(frame); responses.delete(frame.id) }
          }
        } } finally { reader.releaseLock() }
      })()
      const rpc = (method: string, params: Record<string, unknown> = {}) => bounded(new Promise<Frame>(resolve => {
        const id = ++nextId; responses.set(id, resolve)
        subprocess.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n')
      }))
      const session = await rpc('open_session', { cwd: root })
      const result = await rpc('prompt', { session_id: session.result?.session_id, text: 'Use the MCP fixture echo with value wired.' })
      expect(result.error).toBeUndefined()
      expect(await Bun.file(marker).text()).toBe('wired')
      expect(sawSchema).toBe(true)
      await rpc('shutdown'); subprocess.stdin.end()
      expect(await bounded(subprocess.exited)).toBe(0)
      await reading
      return
    }
    for (let attempt = 0; attempt < 200; attempt++) {
      try { socket = await new Promise<Socket>((resolve, reject) => { const candidate = connect(socketPath); candidate.once('error', reject); candidate.once('connect', () => resolve(candidate)) }); break }
      catch { if (child.exitCode !== null) throw new Error(await stderr); await Bun.sleep(25) }
    }
    if (!socket) throw new Error('daemon did not open socket')
    let buffer = '', nextId = 0
    let turnEnded = false
    const errors: unknown[] = []
    const pending = new Map<number, (frame: Frame) => void>()
    socket.setEncoding('utf8')
    socket.on('data', chunk => { buffer += String(chunk); let end; while ((end = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, end); buffer = buffer.slice(end + 1); if (!line) continue;
      const frame = JSON.parse(line) as Frame; if (frame.params?.type === 'turn_end') turnEnded = true;
      if (frame.params?.type?.includes('error')) errors.push(frame.params.payload)
      if (frame.id !== undefined) { pending.get(frame.id)?.(frame); pending.delete(frame.id) }
    } })
    const rpc = (method: string, params: Record<string, unknown> = {}) => bounded(new Promise<Frame>(resolve => {
      const id = ++nextId; pending.set(id, resolve); socket!.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n')
    }))
    await rpc('initialize', { session_key: 'mcp-test', project_dir: root })
    let inventory: Frame = {}
    for (let attempt = 0; attempt < 100; attempt++) {
      inventory = await rpc('tool.inventory')
      if ((inventory.result?.tools as Array<{ name: string }> | undefined)?.some(tool => tool.name === toolName)) break
      await Bun.sleep(25)
    }
    expect(inventory.result?.tools).toEqual(expect.arrayContaining([expect.objectContaining({ name: toolName })]))
    const turn = await rpc('turn.submit', { text: 'Use the MCP fixture echo with value wired.' })
    expect(turn.error).toBeUndefined()
    for (let attempt = 0; attempt < 500 && !turnEnded; attempt++) await Bun.sleep(10)
    expect(turnEnded).toBe(true)
    if (!await Bun.file(marker).exists()) throw new Error(JSON.stringify({ sawSchema, errors, tool: (inventory.result?.tools as Array<{ name: string }>).find(tool => tool.name === toolName) }))
    expect(await Bun.file(marker).text()).toBe('wired')
    expect(sawSchema).toBe(true)
    expect((await rpc('slash', { command: '/mcp reconnect fixture' })).result?.ok).toBe(true)
    const refreshed = await rpc('tool.inventory')
    expect((refreshed.result?.tools as Array<{ name: string }>).filter(tool => tool.name === toolName)).toHaveLength(1)
    await rm(marker)
    const saved = await rpc('initialize', { session_key: 'mcp-after-reconnect', project_dir: root })
    turnEnded = false
    await rpc('turn.submit', { text: 'Use the reconnected MCP fixture.' })
    for (let attempt = 0; attempt < 500 && !turnEnded; attempt++) await Bun.sleep(10)
    expect(turnEnded).toBe(true)
    expect(await Bun.file(marker).text()).toBe('wired')
    expect(typeof saved.result?.session_id).toBe('string')
    socket.destroy(); subprocess.kill('SIGTERM'); await bounded(subprocess.exited)
    await rm(marker)
    const resumed = Bun.spawn([process.execPath, join(import.meta.dir, '../src/cli.ts'), '--resume', String(saved.result?.session_id), 'Use the MCP fixture after resuming.'], {
      cwd: root, env: { ...process.env, XERXES_HOME: home }, stdout: 'pipe', stderr: 'pipe',
    })
    child = resumed
    const resumedOutput = new Response(resumed.stdout).text()
    const resumedErrors = new Response(resumed.stderr).text()
    expect(await bounded(resumed.exited)).toBe(0)
    expect(await resumedOutput).toContain('MCP verified')
    expect(await resumedErrors).toBe('')
    expect(await Bun.file(marker).text()).toBe('wired')
  } finally {
    socket?.destroy(); if (child?.exitCode === null) child.kill('SIGTERM')
    if (child) { try { await bounded(child.exited) } catch { child.kill('SIGKILL'); await child.exited } }
    provider.stop(true); await rm(root, { recursive: true, force: true })
  }
}, 30000)
