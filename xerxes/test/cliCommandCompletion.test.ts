// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises'
import { connect, type Socket } from 'node:net'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

function bounded<T>(promise: Promise<T>, milliseconds = 10000): Promise<T> {
  let timer: ReturnType<typeof setTimeout>
  return Promise.race([promise, new Promise<never>((_, reject) => { timer = setTimeout(() => reject(new Error('fixture timeout')), milliseconds) })]).finally(() => clearTimeout(timer))
}

test('normal daemon wakes the owner after exec_command completion without a separate watch call', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xc-'))
  const home = join(root, 'home'), socketPath = join(root, 'd.sock')
  let spawned = false, reactions = 0
  let evidence = ''
  const provider = Bun.serve({ hostname: '127.0.0.1', port: 0, async fetch(request) {
    const body = await request.json() as { messages?: Array<{ role?: string; content?: unknown }> }
    const lastUser = body.messages?.findLast(message => message.role === 'user')
    const prompt = typeof lastUser?.content === 'string' ? lastUser.content : JSON.stringify(lastUser?.content)
    let chunk
    if (prompt?.includes('A monitor produced new evidence')) {
      reactions++; evidence = prompt
      chunk = { choices: [{ delta: { content: 'Background command completed.' }, finish_reason: 'stop' }] }
    } else if (!spawned) {
      spawned = true
      chunk = { choices: [{ delta: { tool_calls: [{ index: 0, id: 'command-call', type: 'function', function: {
        name: 'exec_command', arguments: JSON.stringify({ cmd: process.execPath, args: ['-e', 'await Bun.sleep(300); console.log("completion-proof")'], run_in_background: true, notify_on_completion: true }),
      } }] }, finish_reason: 'tool_calls' }] }
    } else chunk = { choices: [{ delta: { content: 'Command started.' }, finish_reason: 'stop' }] }
    return new Response(`data: ${JSON.stringify(chunk)}\n\ndata: [DONE]\n\n`, { headers: { 'content-type': 'text/event-stream' } })
  } })
  let child: ReturnType<typeof Bun.spawn> | undefined, socket: Socket | undefined
  let stderr = Promise.resolve('')
  try {
    await mkdir(join(home, 'daemon'), { recursive: true })
    await writeFile(join(home, 'daemon', 'config.json'), JSON.stringify({ project_directory: root, runtime: {
      model: 'gpt-4o', provider: 'openai', base_url: `${provider.url}v1`, api_key: 'fixture-key', permission_mode: 'accept-all',
    } }))
    const subprocess = Bun.spawn([process.execPath, join(import.meta.dir, '../src/cli.ts'), 'daemon', '--project-dir', root, '--socket', socketPath], {
      cwd: root, env: { ...process.env, XERXES_HOME: home, XERXES_DEFERRED_TOOL_LOADING: '0', XERXES_EDIT_DIAGNOSTICS: '0' }, stdin: 'ignore', stdout: 'ignore', stderr: 'pipe',
    })
    child = subprocess
    stderr = new Response(subprocess.stderr).text()
    for (let attempt = 0; attempt < 200; attempt++) {
      try { socket = await new Promise<Socket>((resolve, reject) => { const candidate = connect(socketPath); candidate.once('error', reject); candidate.once('connect', () => resolve(candidate)) }); break }
      catch { if (child.exitCode !== null) throw new Error(await stderr); await Bun.sleep(25) }
    }
    if (!socket) throw new Error('daemon did not open socket')
    let buffer = '', nextId = 0, ended = 0
    let completed!: () => void
    const completion = new Promise<void>(resolve => { completed = resolve })
    const pending = new Map<number, (frame: { result?: Record<string, unknown> }) => void>()
    socket.setEncoding('utf8')
    socket.on('data', chunk => { buffer += String(chunk); let end; while ((end = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, end); buffer = buffer.slice(end + 1); if (!line) continue
      const frame = JSON.parse(line)
      if (frame.params?.type === 'turn_end' && ++ended >= 2) completed()
      if (frame.id !== undefined) { pending.get(frame.id)?.(frame); pending.delete(frame.id) }
    } })
    const rpc = (method: string, params: Record<string, unknown> = {}) => bounded(new Promise<{ result?: Record<string, unknown> }>(resolve => {
      const id = ++nextId; pending.set(id, resolve); socket!.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n')
    }))
    await rpc('initialize', { session_key: 'completion-owner', project_dir: root })
    await rpc('turn.submit', { text: 'Start the command in the background and tell me when it finishes.' })
    await bounded(completion)
    expect(reactions).toBe(1)
    expect(evidence).toContain('completion-proof')
    expect(evidence).toContain('untrusted source data')
    const inspected = await rpc('context.inspect', { section: 'instructions' })
    expect(inspected.result?.entries).toEqual(expect.arrayContaining([expect.objectContaining({ title: 'bootstrap' })]))
    await rpc('shutdown')
    expect(await bounded(child.exited)).toBe(0)
  } finally {
    socket?.destroy()
    if (child && child.exitCode === null && child.signalCode === null) child.kill('SIGKILL')
    if (child) await bounded(child.exited)
    provider.stop(true)
    await rm(root, { recursive: true, force: true })
  }
}, 20000)
