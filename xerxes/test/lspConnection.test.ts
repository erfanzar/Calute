// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, writeFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { LspConnection } from '../src/lsp/connection.js'

async function fixture() {
  const cwd = await mkdtemp(join(tmpdir(), 'xerxes-lsp-'))
  const script = join(cwd, 'server.ts')
  const framing = pathToFileURL(join(import.meta.dir, '../src/lsp/framing.ts')).href
  await writeFile(script, `import { LspMessageDecoder, encodeLspMessage } from ${JSON.stringify(framing)};
const send = value => Bun.stdout.write(encodeLspMessage(value));
const decoder = new LspMessageDecoder(message => {
  if (message.method === 'crash') process.exit(7);
  if (message.method === 'malformed') { Bun.stdout.write('Content-Length: -1\\r\\n\\r\\n'); return; }
  if (message.method === 'invalid-response') { send({jsonrpc:'2.0',id:message.id}); return; }
  if (message.method === 'hang') return;
  if (message.method === '$/cancelRequest') { send({jsonrpc:'2.0',method:'cancel-observed',params:message.params}); send({jsonrpc:'2.0',id:message.params.id,result:'late'}); return; }
  if (message.method === 'exit') process.exit(0);
  if ('id' in message) send({jsonrpc:'2.0',id:message.id,result:message.method === 'shutdown' ? null : message.params});
});
for await (const chunk of Bun.stdin.stream()) decoder.push(chunk);`)
  const connection = new LspConnection({ command: process.execPath, args: [script], cwd, timeoutMs: 1000 })
  return { cwd, script, connection, async close() { await connection.close(); await rm(cwd, { recursive: true, force: true }) } }
}

test('stdio LSP requests correlate concurrent replies and shut down the owned process', async () => {
  const f = await fixture()
  try {
    const results = await Promise.all(Array.from({ length: 20 }, (_, id) => f.connection.request('echo', { id, text: 'سلام 🌍' })))
    expect(results).toEqual(Array.from({ length: 20 }, (_, id) => ({ id, text: 'سلام 🌍' })))
    expect(f.connection.connected).toBe(true)
    await f.connection.close()
    expect(f.connection.connected).toBe(false)
    await expect(f.connection.request('echo')).rejects.toThrow('closed')
  } finally { await f.close() }
})

test('LSP cancellation and timeout settle promptly and ignore late responses', async () => {
  const f = await fixture()
  const events: string[] = []
  const connection = new LspConnection({ command: process.execPath, args: [f.script], cwd: f.cwd, timeoutMs: 1000, onNotification: method => events.push(method) })
  try {
    await connection.request('echo')
    const controller = new AbortController()
    const pending = connection.request('hang', {}, controller.signal)
    const result = pending.catch(error => error)
    await Bun.sleep(10)
    controller.abort('private-abort-reason')
    expect(await result).toMatchObject({ name: 'AbortError', message: 'LSP request cancelled' })
    await expect(connection.request('hang', {}, undefined, 25)).rejects.toThrow('timed out')
    expect(await connection.request('echo', { still: 'working' })).toEqual({ still: 'working' })
    expect(events).toContain('cancel-observed')
    await expect(connection.request('echo', {}, controller.signal)).rejects.toMatchObject({ name: 'AbortError' })
  } finally { await connection.close(); await f.close() }
})

test.each(['crash', 'malformed', 'invalid-response'])('LSP %s rejects every pending request without leaking source output', async action => {
  const f = await fixture()
  try {
    const waiting = f.connection.request('hang').catch(error => error)
    await expect(f.connection.request(action)).rejects.toThrow()
    expect(await waiting).toBeInstanceOf(Error)
    expect(f.connection.connected).toBe(false)
  } finally { await f.close() }
})

test('missing language server yields configuration guidance', () => {
  expect(() => new LspConnection({ command: '/not/a/language/server', cwd: tmpdir() })).toThrow('verify its executable')
})

test('LSP outbound admission is bounded and recovers after the queue drains', async () => {
  const f = await fixture()
  try {
    const writes = Array.from({ length: 130 }, () => f.connection.notify('notification', {}))
    const results = await Promise.allSettled(writes)
    expect(results.filter(result => result.status === 'rejected')).toHaveLength(2)
    expect(await f.connection.request('echo', 'available')).toBe('available')
  } finally { await f.close() }
})

test('closing a server that ignores shutdown is bounded and rejects its in-flight work', async () => {
  const f = await fixture()
  await writeFile(f.script, `process.on('SIGTERM', () => {}); for await (const chunk of Bun.stdin.stream()) {}`)
  const connection = new LspConnection({ command: process.execPath, args: [f.script], cwd: f.cwd })
  try {
    const pending = connection.request('hang').catch(error => error)
    const started = Date.now()
    const closing = connection.close()
    await expect(connection.request('echo')).rejects.toThrow('closing')
    await closing
    expect(Date.now() - started).toBeLessThan(2000)
    expect(await pending).toBeInstanceOf(Error)
    expect(connection.connected).toBe(false)
  } finally { await connection.close(); await f.close() }
})
