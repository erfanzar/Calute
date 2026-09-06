// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, writeFile, rm, symlink, realpath } from 'node:fs/promises'
import { pathToFileURL } from 'node:url'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { LspHost } from '../src/lsp/host.js'
import type { LspConnectionOptions } from '../src/lsp/connection.js'

async function fixture(capabilities: unknown = { textDocumentSync: 1, hoverProvider: true, definitionProvider: true, referencesProvider: true, documentSymbolProvider: true }) {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-lsp-host-'))
  await writeFile(join(root, 'source.ts'), 'const world = "🌍";\n')
  let notify: NonNullable<LspConnectionOptions['onNotification']> = () => {}
  const requests: { method: string; params: unknown }[] = [], notifications: { method: string; params: unknown }[] = []
  let closed = false
  let notificationHandler: ((method: string, params: unknown) => Promise<void>) | undefined
  let handler: ((method: string, params: unknown, signal?: AbortSignal) => Promise<unknown>) | undefined
  const options = { cwd: root, command: 'fake', languageId: 'typescript', connectionFactory: (options: LspConnectionOptions) => {
    notify = options.onNotification!
    return {
      get connected() { return !closed },
      async request(method: string, params: unknown, signal?: AbortSignal) {
        requests.push({ method, params })
        if (method === 'initialize') return { capabilities }
        return handler ? handler(method, params, signal) : { method, params }
      },
      async notify(method: string, params: unknown) { notifications.push({ method, params }); await notificationHandler?.(method, params) },
      async close() { closed = true },
    }
  } }
  return { root, options, requests, notifications, emit: (params: unknown) => notify('textDocument/publishDiagnostics', params),
    setNotificationHandler: (value: NonNullable<typeof notificationHandler>) => { notificationHandler = value },
    setHandler: (value: NonNullable<typeof handler>) => { handler = value }, closed: () => closed,
    cleanup: () => rm(root, { recursive: true, force: true }) }
}
const req = (action: string, filePath = 'source.ts') => ({ action, filePath, line: 0, character: 0 })

test('LSP host initializes a workspace and synchronizes changed documents before navigation', async () => {
  const f = await fixture()
  const host = await LspHost.start(f.options)
  try {
    expect(f.requests[0]).toMatchObject({ method: 'initialize', params: { capabilities: { general: { positionEncodings: ['utf-16'] } } } })
    expect(f.notifications[0]?.method).toBe('initialized')
    await host.execute(req('hover')); await host.execute(req('references'))
    expect(f.notifications.filter(row => row.method === 'textDocument/didOpen')).toHaveLength(1)
    await writeFile(join(f.root, 'source.ts'), 'updated\n')
    await host.execute(req('definition'))
    expect(f.notifications.at(-1)).toMatchObject({ method: 'textDocument/didChange', params: { textDocument: { version: 2 }, contentChanges: [{ text: 'updated\n' }] } })
    expect(f.requests.at(-1)?.method).toBe('textDocument/definition')
    await host.execute(req('symbols'))
    expect(f.requests.at(-1)).toMatchObject({ method: 'textDocument/documentSymbol' })
    expect((f.requests.at(-1)?.params as Record<string, unknown>).position).toBeUndefined()
    await expect(host.execute({ ...req('hover'), character: 999 })).rejects.toThrow('outside')
  } finally { await host.close(); await f.cleanup() }
})

test('LSP diagnostics are version-matched and malformed results are never fresh', async () => {
  const f = await fixture(), host = await LspHost.start(f.options)
  try {
    const first = await host.execute(req('diagnostics')) as { uri: string; version: number; fresh: boolean }
    expect(first.fresh).toBe(false)
    const diagnostic = { message: 'problem', range: { start: { line: 0, character: 0 }, end: { line: 0, character: 3 } } }
    f.emit({ uri: first.uri, diagnostics: [diagnostic] })
    expect(await host.execute(req('diagnostics'))).toMatchObject({ fresh: false })
    f.emit({ uri: first.uri, version: first.version, diagnostics: [diagnostic] })
    expect(await host.execute(req('diagnostics'))).toMatchObject({ fresh: true, diagnostics: [diagnostic] })
    await writeFile(join(f.root, 'source.ts'), 'new text')
    const second = await host.execute(req('diagnostics')) as typeof first
    expect(second.version).toBeGreaterThan(first.version)
    f.emit({ uri: first.uri, version: first.version, diagnostics: [diagnostic] })
    expect(await host.execute(req('diagnostics'))).toMatchObject({ fresh: false })
    f.emit({ uri: first.uri, version: second.version, diagnostics: [{ message: 'bad' }] })
    expect(await host.execute(req('diagnostics'))).toMatchObject({ fresh: false, reason: 'Malformed diagnostics response' })
  } finally { await host.close(); await f.cleanup() }
})

test('LSP scopes document reads and rejects unsupported actions and capabilities', async () => {
  const f = await fixture({ textDocumentSync: 2 }), host = await LspHost.start(f.options)
  const outside = await mkdtemp(join(tmpdir(), 'xerxes-lsp-outside-'))
  try {
    await writeFile(join(outside, 'private.ts'), 'private')
    await symlink(join(outside, 'private.ts'), join(f.root, 'escape.ts'))
    await expect(host.execute(req('diagnostics', 'escape.ts'))).rejects.toThrow('outside workspace')
    await expect(host.execute(req('diagnostics', '../private.ts'))).rejects.toThrow('escapes workspace')
    expect(f.notifications.filter(row => row.method === 'textDocument/didOpen')).toHaveLength(0)
    await expect(host.execute(req('constructor'))).rejects.toThrow('Unsupported')
    await expect(host.execute(req('hover'))).rejects.toThrow('does not support')
  } finally { await host.close(); await f.cleanup(); await rm(outside, { recursive: true, force: true }) }
})

test('LSP queued cancellation settles before preceding work and never dispatches the cancelled request', async () => {
  const f = await fixture(), host = await LspHost.start(f.options)
  const entered = Promise.withResolvers<void>(), release = Promise.withResolvers<void>()
  f.setHandler(async () => { entered.resolve(); await release.promise; return null })
  const first = host.execute(req('hover'))
  try {
    await entered.promise
    const controller = new AbortController()
    const pending = host.execute(req('definition'), controller.signal)
    const result = pending.catch(error => error)
    controller.abort('private reason')
    expect(await result).toMatchObject({ name: 'AbortError', message: 'LSP request cancelled' })
    release.resolve(); await first
    await host.execute(req('symbols'))
    expect(f.requests.some(row => row.method === 'textDocument/definition')).toBe(false)
  } finally { release.resolve(); await first; await host.close(); await f.cleanup() }
})

test.each([null, { positionEncoding: 'utf-8', textDocumentSync: 1 }, { textDocumentSync: 0 }])('invalid initialization closes its connection: %j', async capabilities => {
  const f = await fixture(capabilities)
  try { await expect(LspHost.start(f.options)).rejects.toThrow(); expect(f.closed()).toBe(true) }
  finally { await f.cleanup() }
})

test('navigation refuses results after the underlying file changes during a request', async () => {
  const f = await fixture(), host = await LspHost.start(f.options)
  f.setHandler(async () => { await writeFile(join(f.root, 'source.ts'), 'changed externally'); return { contents: 'old document result' } })
  try { await expect(host.execute(req('hover'))).rejects.toThrow('changed during navigation') }
  finally { await host.close(); await f.cleanup() }
})

test('document eviction closes the old document and never reuses its diagnostic version', async () => {
  const f = await fixture(), host = await LspHost.start(f.options)
  try {
    const first = await host.execute(req('diagnostics')) as { uri: string; version: number }
    for (let i = 0; i < 32; i++) {
      await writeFile(join(f.root, `${i}.ts`), 'content')
      await host.execute(req('diagnostics', `${i}.ts`))
    }
    expect(f.notifications.some(row => row.method === 'textDocument/didClose' && (row.params as { textDocument: { uri: string } }).textDocument.uri === first.uri)).toBe(true)
    const reopened = await host.execute(req('diagnostics')) as { version: number }
    expect(reopened.version).toBeGreaterThan(first.version)
    f.emit({ uri: first.uri, version: first.version, diagnostics: [] })
    expect(await host.execute(req('diagnostics'))).toMatchObject({ fresh: false })
  } finally { await host.close(); await f.cleanup() }
})

test('initialization cancellation never launches an already-cancelled host', async () => {
  const f = await fixture()
  let launched = false
  const controller = new AbortController(); controller.abort()
  try {
    await expect(LspHost.start({ ...f.options, connectionFactory: options => { launched = true; return f.options.connectionFactory(options) } }, controller.signal)).rejects.toMatchObject({ name: 'AbortError' })
    expect(launched).toBe(false)
  } finally { await f.cleanup() }
})

test('native host initializes and receives versioned diagnostics over a real Bun stdio server', async () => {
  const f = await fixture()
  const script = join(f.root, 'server.ts')
  await writeFile(script, `import { LspMessageDecoder, encodeLspMessage } from ${JSON.stringify(pathToFileURL(join(import.meta.dir, '../src/lsp/framing.ts')).href)};
let root;
const send = value => Bun.stdout.write(encodeLspMessage(value));
const decoder = new LspMessageDecoder(message => {
  if (message.method === 'initialize') { root = message.params.rootUri; send({jsonrpc:'2.0',id:message.id,result:{capabilities:{textDocumentSync:1,hoverProvider:true}}}); }
  else if (message.method === 'textDocument/didOpen' || message.method === 'textDocument/didChange') {
    const doc=message.params.textDocument;
    send({jsonrpc:'2.0',method:'textDocument/publishDiagnostics',params:{uri:doc.uri,version:doc.version,diagnostics:[]}});
  } else if (message.method === 'textDocument/hover') send({jsonrpc:'2.0',id:message.id,result:{contents:'hover',root}});
  else if (message.method === 'shutdown') send({jsonrpc:'2.0',id:message.id,result:null});
  else if (message.method === 'exit') process.exit(0);
});
for await (const chunk of Bun.stdin.stream()) decoder.push(chunk);`)
  const host = await LspHost.start({ command: process.execPath, args: [script], cwd: f.root, languageId: 'typescript' })
  try {
    expect(await host.execute(req('hover'))).toMatchObject({ contents: 'hover', root: pathToFileURL(await realpath(f.root)).href })
    expect(await host.execute(req('diagnostics'))).toMatchObject({ fresh: true, version: 1, diagnostics: [] })
    await writeFile(join(f.root, 'source.ts'), 'changed')
    await host.execute(req('hover'))
    expect(await host.execute(req('diagnostics'))).toMatchObject({ fresh: true, version: 2, diagnostics: [] })
  } finally { await host.close(); await f.cleanup() }
})

test('diagnostics waits for matching publication but leaves timeout results unconfirmed', async () => {
  const f = await fixture(), host = await LspHost.start(f.options)
  let publish!: () => void
  f.setNotificationHandler(async (method, params) => {
    if (method === 'textDocument/didOpen') {
      const doc = (params as { textDocument: { uri: string; version: number } }).textDocument
      publish = () => f.emit({ ...doc, diagnostics: [] })
      setTimeout(() => { f.emit({ ...doc, version: doc.version - 1, diagnostics: [] }); setTimeout(publish, 10) }, 10)
    }
  })
  try {
    expect(await host.execute({ ...req('diagnostics'), diagnosticsWaitMs: 1000 })).toMatchObject({ fresh: true, diagnostics: [] })
    await writeFile(join(f.root, 'source.ts'), 'updated')
    expect(await host.execute({ ...req('diagnostics'), diagnosticsWaitMs: 10 })).toMatchObject({ fresh: false })
    await expect(host.execute({ ...req('diagnostics'), diagnosticsWaitMs: 5001 })).rejects.toThrow('0–5000')
  } finally { await host.close(); await f.cleanup() }
})

test('diagnostics rejects source changes while awaiting publication', async () => {
  const f = await fixture(), host = await LspHost.start(f.options)
  f.setNotificationHandler(async (method, params) => {
    if (method !== 'textDocument/didOpen') return
    await writeFile(join(f.root, 'source.ts'), 'edited outside the host')
    f.emit({ ...(params as { textDocument: object }).textDocument, diagnostics: [] })
  })
  try { await expect(host.execute({ ...req('diagnostics'), diagnosticsWaitMs: 1000 })).rejects.toThrow('changed during diagnostics') }
  finally { await host.close(); await f.cleanup() }
})

test.each(['cancel', 'close'] as const)('diagnostics wait releases promptly on %s', async kind => {
  const f = await fixture(), host = await LspHost.start(f.options)
  let opened!: () => void; const ready = new Promise<void>(resolve => { opened = resolve })
  f.setNotificationHandler(async method => { if (method === 'textDocument/didOpen') opened() })
  const controller = new AbortController()
  const work = host.execute({ ...req('diagnostics'), diagnosticsWaitMs: 5000 }, controller.signal).catch(error => error)
  try {
    await ready
    await Bun.sleep(10)
    const start = Date.now()
    if (kind === 'cancel') controller.abort(); else await host.close()
    expect(await work).toBeInstanceOf(Error)
    expect(Date.now() - start).toBeLessThan(1000)
  } finally { await host.close(); await f.cleanup() }
})

test.skipIf(process.platform === 'win32')('named pipes are rejected without waiting for a writer', async () => {
  const f = await fixture(), host = await LspHost.start(f.options)
  try {
    const result = Bun.spawnSync(['mkfifo', join(f.root, 'pipe.ts')])
    expect(result.exitCode).toBe(0)
    const start = Date.now()
    await expect(host.execute(req('diagnostics', 'pipe.ts'))).rejects.toThrow('regular file')
    expect(Date.now() - start).toBeLessThan(1000)
  } finally { await host.close(); await f.cleanup() }
})
