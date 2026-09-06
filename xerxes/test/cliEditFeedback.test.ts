// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'


interface Frame { id?: number; result?: Record<string, unknown>; error?: unknown; params?: { type?: string; payload?: unknown } }
function bounded<T>(promise: Promise<T>, milliseconds = 10000): Promise<T> {
  let timer: ReturnType<typeof setTimeout>
  return Promise.race([promise, new Promise<never>((_, reject) => { timer = setTimeout(() => reject(new Error('fixture timeout')), milliseconds) })])
    .finally(() => clearTimeout(timer))
}

test.each(['text', 'json', 'stream-json', 'acp'] as const)('normal CLI %s delivers automatic semantic feedback after a write', async mode => {
  const root = await mkdtemp(join(tmpdir(), 'xm-'))
  const home = join(root, 'home'), marker = join(root, 'called')
  const toolName = 'LSPTool'
  let sawSchema = false
  const provider = Bun.serve({ hostname: '127.0.0.1', port: 0, async fetch(request) {
    const body = await request.json() as { tools?: Array<{ function?: { name?: string } }>; messages?: Array<{ role?: string; content?: unknown }> }
    const hasSchema = body.tools?.some(tool => tool.function?.name === toolName) ?? false
    sawSchema ||= hasSchema
    const lastUser = body.messages?.findLastIndex(message => message.role === 'user') ?? -1
    const hasResult = body.messages?.some((message, index) => index > lastUser && message.role === 'tool')
    const chunk = hasSchema && !hasResult
      ? { choices: [{ delta: { tool_calls: [{ index: 0, id: 'mcp-call', type: 'function', function: { name: 'WriteFile', arguments: '{"file_path":"created.ts","content":"broken source"}' } }] }, finish_reason: 'tool_calls' }] }
      : { choices: [{ delta: { content: 'LSP verified' }, finish_reason: 'stop' }] }
    return new Response(`data: ${JSON.stringify(chunk)}\n\ndata: [DONE]\n\n`, { headers: { 'content-type': 'text/event-stream' } })
  } })
  let child: ReturnType<typeof Bun.spawn> | undefined
  let stderr = Promise.resolve('')
  try {
    await mkdir(join(home, 'daemon'), { recursive: true })
    await writeFile(join(home, 'daemon', 'config.json'), JSON.stringify({ project_directory: root, runtime: {
      model: 'gpt-4o', provider: 'openai', base_url: `${provider.url}v1`, api_key: 'fixture-key', permission_mode: 'accept-all',
    } }))
    const script = join(root, 'lsp.ts')
    await writeFile(join(root, 'main.ts'), 'const wired = true')
    await writeFile(script, `import { LspMessageDecoder, encodeLspMessage } from ${JSON.stringify(join(import.meta.dir, '../src/lsp/framing.ts'))};
const decoder = new LspMessageDecoder(value => {
 const request = value;
 if (request.method === 'exit') process.exit(0);
 if (request.method === 'textDocument/didOpen' || request.method === 'textDocument/didChange') {
 require('node:fs').writeFileSync(${JSON.stringify(marker)}, 'wired');
 process.stdout.write(encodeLspMessage({jsonrpc:'2.0',method:'textDocument/publishDiagnostics',params:{...request.params.textDocument,diagnostics:[{message:'fixture semantic error',severity:1,range:{start:{line:0,character:0},end:{line:0,character:1}}}]}}));
}
 if (request.id === undefined) return;
 let result = null;
 if (request.method === 'initialize') result = { capabilities: { textDocumentSync: 1, hoverProvider: true } };
 if (request.method === 'textDocument/hover') { require('node:fs').writeFileSync(${JSON.stringify(marker)}, 'wired'); result = { contents: 'wired' }; }
 process.stdout.write(encodeLspMessage({ jsonrpc: '2.0', id: request.id, result }));
});
for await (const chunk of Bun.stdin.stream()) decoder.push(chunk);`)
    await writeFile(join(home, 'lsp.json'), JSON.stringify({ servers: [{ name: 'fixture', command: process.execPath, args: [script], languageId: 'typescript', extensions: ['.ts'] }] }))
    const args = mode === 'acp' ? ['acp', '--project-dir', root] : ['--output-format', mode, 'Create a source file.']
    const subprocess = Bun.spawn([process.execPath, join(import.meta.dir, '../src/cli.ts'), ...args], {
      cwd: root, env: { ...process.env, XERXES_HOME: home, XERXES_DEFERRED_TOOL_LOADING: '0' }, stdin: 'pipe', stdout: 'pipe', stderr: 'pipe',
    })
    child = subprocess
    stderr = new Response(subprocess.stderr).text()
    if (mode !== 'acp') {
      const output = new Response(subprocess.stdout).text()
      expect(await bounded(subprocess.exited)).toBe(0)
      const rendered = await output
      expect(rendered).toContain('fixture semantic error')
      if (mode === 'json') expect(JSON.parse(rendered).response).toContain('fixture semantic error')
      if (mode === 'stream-json') {
        const frames = rendered.trim().split('\n').map(line => JSON.parse(line))
        expect(frames.at(-1).type).toBe('result')
        expect(frames.some(frame => frame.type === 'text' && frame.text.includes('fixture semantic error'))).toBe(true)
      }
      expect(await Bun.file(marker).text()).toBe('wired')
      expect(sawSchema).toBe(true)
      return
    }
    if (mode === 'acp') {
      const events: string[] = []
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
            events.push(line)
            const frame = JSON.parse(line) as Frame
            if (frame.id !== undefined) { responses.get(frame.id)?.(frame); responses.delete(frame.id) }
          }
        } } finally { reader.releaseLock() }
      })()
      const rpc = (method: string, params: Record<string, unknown> = {}) => bounded(new Promise<Frame>(resolve => {
        const id = ++nextId; responses.set(id, resolve)
        subprocess.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n')
      }))
      const sessionRoot = join(root, 'acp-workspace')
      await mkdir(sessionRoot)
      const session = await rpc('open_session', { cwd: sessionRoot })
      const result = await rpc('prompt', { session_id: session.result?.session_id, text: 'Use the LSP fixture echo with value wired.' })
      expect(result.error).toBeUndefined()
      expect(events.join('\n')).toContain('fixture semantic error')
      expect(await Bun.file(join(sessionRoot, 'created.ts')).exists()).toBe(true)
      expect(await Bun.file(join(root, 'created.ts')).exists()).toBe(false)
      const diagnosticIndex = events.findIndex(event => event.includes('fixture semantic error'))
      expect(events.findIndex(event => event.includes('turn_end'))).toBeGreaterThan(diagnosticIndex)
      expect(await Bun.file(marker).text()).toBe('wired')
      expect(sawSchema).toBe(true)
      await rpc('shutdown'); subprocess.stdin.end()
      expect(await bounded(subprocess.exited)).toBe(0)
      await reading
      return
    }
  } finally {
    if (child?.exitCode === null) child.kill('SIGTERM')
    if (child) { try { await bounded(child.exited) } catch { child.kill('SIGKILL'); await child.exited } }
    provider.stop(true); await rm(root, { recursive: true, force: true })
  }
}, 30000)
