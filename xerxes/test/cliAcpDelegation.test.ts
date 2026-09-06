// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { ProfileStore } from '../src/bridge/profiles.js'
import { AgentSettingsStore } from '../src/agents/settingsStore.js'

const CLI = join(import.meta.dir, '../src/cli.ts')

test('ACP CLI advertises the native delegation surface', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-bun-cli-acp-'))
  const home = join(root, 'home')
  const project = join(root, 'project')
  try {
    await Promise.all([
      mkdir(join(home, 'daemon'), { recursive: true }),
      mkdir(project, { recursive: true }),
    ])
    await writeFile(
      join(home, 'daemon', 'config.json'),
      JSON.stringify({
        runtime: {
          model: 'gpt-4o',
          provider: 'openai',
          base_url: 'http://127.0.0.1:1/v1',
          api_key: 'test-key',
          permission_mode: 'accept-all',
        },
      }),
      'utf8',
    )

    new AgentSettingsStore(join(home, 'daemon', 'agent-settings.sqlite')).save({
      smart: { model: 'gpt-5', reasoning_effort: 'high' },
    }, 0)

    const child = Bun.spawn([
      process.execPath,
      CLI,
      'acp',
      '--project-dir',
      project,
    ], {
      env: { ...process.env, XERXES_HOME: home },
      stdin: 'pipe',
      stderr: 'pipe',
      stdout: 'pipe',
    })
    child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'tools/list', params: {} })}\n`)
    child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id: 2, method: 'shutdown', params: {} })}\n`)
    child.stdin.end()

    const [stdout, stderr, exitCode] = await Promise.all([
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
      child.exited,
    ])
    expect(exitCode).toBe(0)
    expect(stderr).toBe('')
    const frames = stdout.trim().split('\n').map(line => JSON.parse(line) as Record<string, unknown>)
    const toolsFrame = frames.find(frame => frame.id === 1)
    const tools = Array.isArray(toolsFrame?.result) ? toolsFrame.result : []
    expect(tools).toEqual(expect.arrayContaining([
      expect.objectContaining({ function: expect.objectContaining({ name: 'AgentTool' }) }),
      expect.objectContaining({ function: expect.objectContaining({ name: 'SpawnAgents' }) }),
      expect.objectContaining({ function: expect.objectContaining({ name: 'AwaitAgents' }) }),
      expect.objectContaining({ function: expect.objectContaining({ name: 'SkillTool' }) }),
    ]))
    expect(tools).toEqual(expect.arrayContaining([
      expect.objectContaining({ function: expect.objectContaining({
        name: 'AgentTool',
        parameters: expect.objectContaining({ properties: expect.objectContaining({
          intelligence: expect.objectContaining({ enum: ['smart'] }),
        }) }),
      }) }),
    ]))
    expect(frames.find(frame => frame.id === 2)?.result).toEqual({ ok: true })
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('ACP CLI executes model discovery and returns its result to the model over an ACP session', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-acp-inventory-'))
  const home = join(root, 'home'), project = join(root, 'project')
  const requests: { messages?: { role: string; content?: string }[] }[] = []
  let catalogs = 0
  const endpoint = Bun.serve({ hostname: '127.0.0.1', port: 0, async fetch(request) {
    if (request.method === 'GET') { catalogs++; return Response.json({ data: [{ id: 'worker', context_length: 16000 }] }) }
    requests.push(await request.json() as { messages?: { role: string; content?: string }[] })
    const delta = requests.length === 1
      ? { tool_calls: [{ index: 0, id: 'inventory', function: { name: 'list_available_models', arguments: JSON.stringify({ provider_profile: 'fixture' }) } }] }
      : { content: 'ACP inventory ready' }
    return new Response(`data: ${JSON.stringify({ choices: [{ delta, finish_reason: requests.length === 1 ? 'tool_calls' : 'stop' }] })}\n\ndata: [DONE]\n\n`, { headers: { 'content-type': 'text/event-stream' } })
  } })
  let child: ReturnType<typeof Bun.spawn> | undefined
  try {
    await mkdir(join(home, 'daemon'), { recursive: true }); await mkdir(project, { recursive: true })
    await writeFile(join(home, 'daemon', 'config.json'), JSON.stringify({ runtime: { model: 'worker', provider: 'openai', base_url: `${endpoint.url}v1`, api_key: 'fixture-secret', permission_mode: 'accept-all' } }))
    const profiles = new ProfileStore(join(home, 'profiles.json'))
    profiles.save({ name: 'fixture', provider: 'openai', baseUrl: `${endpoint.url}v1`, apiKey: 'fixture-secret', model: 'worker' })
    const process = Bun.spawn([Bun.which('bun')!, CLI, 'acp', '--project-dir', project], { env: { ...globalThis.process.env, XERXES_HOME: home }, stdin: 'pipe', stdout: 'pipe', stderr: 'pipe' })
    child = process
    const stderr = new Response(process.stderr).text()
    const reader = process.stdout.getReader(), decoder = new TextDecoder()
    let buffered = ''
    const receive = async (id: number): Promise<{ result?: Record<string, unknown>; error?: unknown }> => {
      for (;;) {
        const newline = buffered.indexOf('\n')
        if (newline >= 0) {
          const line = buffered.slice(0, newline); buffered = buffered.slice(newline + 1)
          const frame = JSON.parse(line)
          if (frame.id === id) return frame
        } else {
          const chunk = await reader.read()
          if (chunk.done) throw new Error('ACP closed before response')
          buffered += decoder.decode(chunk.value, { stream: true })
        }
      }
    }
    process.stdin.write(JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'session/open', params: { cwd: project } }) + '\n')
    const session = await receive(1)
    expect(session.error).toBeUndefined()
    process.stdin.write(JSON.stringify({ jsonrpc: '2.0', id: 2, method: 'session/prompt', params: { session_id: session.result?.session_id, text: 'Discover worker models' } }) + '\n')
    expect((await receive(2)).result).toMatchObject({ ok: true, tool_calls_count: 1 })
    expect(catalogs).toBe(1)
    expect(requests).toHaveLength(2)
    const tool = requests[1]?.messages?.find(message => message.role === 'tool')
    expect(tool?.content).toContain('context_window')
    expect(tool?.content).toContain('worker')
    expect(tool?.content).not.toContain('fixture-secret')
    process.stdin.write(JSON.stringify({ jsonrpc: '2.0', id: 3, method: 'shutdown', params: {} }) + '\n')
    await receive(3); process.stdin.end()
    expect(await process.exited).toBe(0)
    expect(await stderr).toBe('')
    reader.releaseLock()
  } finally { child?.kill(); if (child) await child.exited; endpoint.stop(true); await rm(root, { recursive: true, force: true }) }
})
