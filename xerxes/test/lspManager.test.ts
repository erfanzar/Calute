// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, expect, test } from 'bun:test'
import { mkdtemp, mkdir, rm, symlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { parseLspConfig } from '../src/lsp/config.js'
import { LspManager } from '../src/lsp/manager.js'

const roots: string[] = []
const managers: LspManager[] = []
afterEach(async () => { await Promise.all(managers.splice(0).map(manager => manager.close().catch(() => {}))); await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))) })
async function workspace() { const root = await mkdtemp(join(tmpdir(), 'xerxes-lsp-manager-')); roots.push(root); return root }
const server = { name: 'typescript', command: 'configured-server', languageId: 'typescript', extensions: ['.ts'] }
const request = { action: 'hover', filePath: 'main.ts', line: 0, character: 0 }

test('configuration rejects invalid and ambiguous executable settings without echoing secrets', () => {
  for (const value of [null, {}, { servers: [server, server] }, { servers: [{ ...server, args: [1] }] }, { servers: [{ ...server, env: { KEY: 1 } }] }, { servers: [{ ...server, enabled: 'false' }] }, { servers: [{ ...server, extensions: ['ts'] }] }, { servers: [{ ...server, timeoutMs: 0 }] }, { servers: [{ ...server, secret: 'sensitive-value' }] }, { servers: [server, { ...server, name: 'other' }] }]) {
    expect(() => parseLspConfig(value)).toThrow()
    try { parseLspConfig(value) } catch (error) { expect(String(error)).not.toContain('sensitive-value') }
  }
  expect(parseLspConfig({ servers: [server, { ...server, name: 'disabled', enabled: false }] })).toHaveLength(2)
})

test('reuses canonical workspace hosts, isolates workspaces, and chooses the longest suffix', async () => {
  const root = await workspace(); const second = join(root, 'second'); await mkdir(second)
  const alias = join(root, 'alias'); await symlink(second, alias)
  const launches: string[] = []; let closes = 0
  const manager = new LspManager({ servers: [server, { ...server, name: 'declarations', command: 'declaration-server', extensions: ['.d.ts'] }] }, async options => {
    launches.push(`${options.command}:${options.cwd}:${options.languageId}`)
    return { execute: async req => req.filePath, close: async () => { closes++ } }
  }); managers.push(manager)
  await Promise.all([manager.forWorkspace(root).execute(request), manager.forWorkspace(root).execute(request)])
  expect(launches).toHaveLength(1)
  await manager.forWorkspace(second).execute(request); await manager.forWorkspace(alias).execute(request)
  expect(launches).toHaveLength(2)
  await manager.forWorkspace(root).execute({ ...request, filePath: 'types.d.ts' })
  expect(launches).toHaveLength(3); expect(launches[2]).toStartWith('declaration-server:')
  await expect(manager.forWorkspace(root).execute({ ...request, filePath: '../escape.ts' })).rejects.toThrow('escapes')
  await expect(manager.forWorkspace(root).execute({ ...request, filePath: 'main.rs' })).rejects.toThrow('No enabled')
  await manager.close(); expect(closes).toBe(3)
  await expect(manager.forWorkspace(root).execute(request)).rejects.toThrow('closed')
})

test('cancelled caller does not cancel shared initialization or another caller', async () => {
  const root = await workspace(); let release!: () => void; const gate = new Promise<void>(resolve => { release = resolve })
  let launched!: () => void; const ready = new Promise<void>(resolve => { launched = resolve }); let startupSignal!: AbortSignal; let calls = 0
  const manager = new LspManager({ servers: [server] }, async (_options, signal) => { startupSignal = signal; launched(); await gate; return { execute: async () => { calls++; return 'ok' }, close: async () => {} } }); managers.push(manager)
  const controller = new AbortController()
  const first = manager.forWorkspace(root).execute(request, controller.signal)
  const assertion = first.catch(error => error)
  await ready; const second = manager.forWorkspace(root).execute(request)
  controller.abort(); expect(String(await assertion)).toContain('cancelled'); expect(startupSignal.aborted).toBe(false)
  release(); expect(await second).toBe('ok'); expect(calls).toBe(1)
})

test('failed initialization can retry; pre-cancelled requests never launch', async () => {
  const root = await workspace(); let launches = 0
  const manager = new LspManager({ servers: [server] }, async () => { launches++; if (launches === 1) throw new Error('initialization failed'); return { execute: async () => 'ok', close: async () => {} } }); managers.push(manager)
  await expect(manager.forWorkspace(root).execute(request, AbortSignal.abort())).rejects.toThrow('cancelled')
  expect(launches).toBe(0)
  await expect(manager.forWorkspace(root).execute(request)).rejects.toThrow('initialization failed')
  expect(await manager.forWorkspace(root).execute(request)).toBe('ok'); expect(launches).toBe(2)
})

test('shutdown aborts initialization and closes a late host without executing requests', async () => {
  const root = await workspace(); let release!: () => void; let launched!: () => void
  const gate = new Promise<void>(resolve => { release = resolve }); const ready = new Promise<void>(resolve => { launched = resolve })
  let startupSignal!: AbortSignal; let closed = false; let calls = 0
  const manager = new LspManager({ servers: [server] }, async (_options, signal) => { startupSignal = signal; launched(); await gate; return { execute: async () => { calls++ }, close: async () => { closed = true } } }); managers.push(manager)
  const work = manager.forWorkspace(root).execute(request); const assertion = work.catch(error => error)
  await ready; const closing = manager.close(); expect(startupSignal.aborted).toBe(true)
  release(); await closing; expect(String(await assertion)).toContain('closed'); expect(closed).toBe(true); expect(calls).toBe(0)
})

test('bounds distinct workspace hosts and reports shutdown failures', async () => {
  const root = await workspace(); let launches = 0; let closes = 0
  const manager = new LspManager({ servers: [server] }, async () => { launches++; return { execute: async () => 'ok', close: async () => { closes++; if (closes === 1) throw new Error('shutdown failed') } } }); managers.push(manager)
  for (let index = 0; index < 33; index++) {
    const child = join(root, String(index)); await mkdir(child)
    if (index === 32) await expect(manager.forWorkspace(child).execute(request)).rejects.toThrow('limit')
    else await manager.forWorkspace(child).execute(request)
  }
  expect(launches).toBe(32)
  await expect(manager.close()).rejects.toThrow('Failed to close')
  expect(closes).toBe(32)
})

test('health reflects lazy startup, failures, retry, disconnect and shutdown without launch secrets', async () => {
  const root = await workspace(); const other = await workspace()
  let release!: () => void; let launched!: () => void
  const gate = new Promise<void>(resolve => { release = resolve }); const ready = new Promise<void>(resolve => { launched = resolve })
  let connected = true; let attempts = 0
  const manager = new LspManager({ servers: [{ ...server, command: 'secret-command', env: { SECRET: 'secret-value' } }, { ...server, name: 'off', enabled: false }] }, async () => {
    if (++attempts === 1) { launched(); await gate; throw new Error('secret-value') }
    return { get connected() { return connected }, execute: async () => 'ok', close: async () => { connected = false } }
  }); managers.push(manager)
  expect((await manager.health(root)).map(row => row.state)).toEqual(['idle', 'disabled'])
  expect(attempts).toBe(0)
  const work = manager.forWorkspace(root).execute(request).catch(error => error)
  await ready; expect((await manager.health(root))[0]?.state).toBe('starting')
  expect((await manager.health(other))[0]?.state).toBe('idle')
  release(); await work
  expect((await manager.health(root))[0]?.state).toBe('failed')
  expect(JSON.stringify(await manager.health(root))).not.toContain('secret')
  expect(await manager.forWorkspace(root).execute(request)).toBe('ok')
  expect((await manager.health(root))[0]?.state).toBe('ready')
  connected = false
  expect((await manager.health(root))[0]?.state).toBe('failed')
  await manager.close()
  expect((await manager.health(root)).every(row => row.state === 'closed')).toBe(true)
})

test('release is workspace-scoped and blocks replacement until cleanup finishes', async () => {
  const root = await workspace(), other = await workspace()
  let finish!: () => void; const gate = new Promise<void>(resolve => { finish = resolve })
  let closes = 0, starts = 0
  const manager = new LspManager({ servers: [server] }, async () => { starts++; return { execute: async () => 'ok', close: async () => { closes++; await gate } } }); managers.push(manager)
  await manager.forWorkspace(root).execute(request); await manager.forWorkspace(other).execute(request)
  const releasing = manager.release(root, server.name)
  while ((await manager.health(root))[0]?.state !== 'stopping') await Bun.sleep(1)
  const duplicate = manager.release(root, server.name)
  await expect(manager.forWorkspace(root).execute(request)).rejects.toThrow('stopping')
  expect(await manager.forWorkspace(other).execute(request)).toBe('ok')
  expect(starts).toBe(2)
  finish(); await Promise.all([releasing, duplicate]); expect(closes).toBe(1)
  expect((await manager.health(root))[0]?.state).toBe('idle')
  await manager.forWorkspace(root).execute(request); expect(starts).toBe(3)
})

test('failed release remains observable and prevents a second server until cleanup succeeds', async () => {
  const root = await workspace(); let closes = 0, starts = 0
  const manager = new LspManager({ servers: [server] }, async () => { starts++; return { execute: async () => 'ok', close: async () => { if (++closes === 1) throw new Error('secret-close-error') } } }); managers.push(manager)
  await manager.forWorkspace(root).execute(request)
  await expect(manager.release(root, server.name)).rejects.toThrow('cleanup failed')
  expect((await manager.health(root))[0]).toMatchObject({ state: 'failed', detail: 'Language server cleanup failed; retry release before reconnecting.' })
  await expect(manager.forWorkspace(root).execute(request)).rejects.toThrow('cleanup')
  expect(starts).toBe(1)
  await manager.release(root, server.name)
  await manager.forWorkspace(root).execute(request); expect(starts).toBe(2)
  await expect(manager.release(root, 'unknown')).rejects.toThrow('Unknown')
})

test('release during initialization closes a late host and prevents its request from executing', async () => {
  const root = await workspace(); let ready!: () => void, finish!: () => void
  const launched = new Promise<void>(resolve => { ready = resolve }), gate = new Promise<void>(resolve => { finish = resolve })
  let calls = 0, closes = 0
  const manager = new LspManager({ servers: [server] }, async () => { ready(); await gate; return { execute: async () => { calls++ }, close: async () => { closes++ } } }); managers.push(manager)
  const work = manager.forWorkspace(root).execute(request).catch(error => error)
  await launched; const releasing = manager.release(root, server.name)
  while ((await manager.health(root))[0]?.state !== 'stopping') await Bun.sleep(1)
  finish(); await releasing
  expect(await work).toBeInstanceOf(Error); expect(calls).toBe(0); expect(closes).toBe(1)
})

test('reconfiguration closes changed hosts before commit and retains unchanged hosts', async () => {
  const root = await workspace(); const events: string[] = []
  const other = { ...server, name: 'rust', extensions: ['.rs'] }
  const manager = new LspManager({ servers: [server, other] }, async options => { events.push('start:' + options.command); return { execute: async () => options.command, close: async () => { events.push('close:' + options.command) } } }); managers.push(manager)
  await manager.forWorkspace(root).execute(request)
  await manager.forWorkspace(root).execute({ ...request, filePath: 'main.rs' })
  await manager.reconfigure({ servers: [{ ...server, command: 'new-server' }, other] }, () => { events.push('commit'); return undefined })
  expect(events).toEqual(['start:configured-server', 'start:configured-server', 'close:configured-server', 'commit'])
  expect(await manager.forWorkspace(root).execute(request)).toBe('new-server')
  await manager.forWorkspace(root).execute({ ...request, filePath: 'main.rs' })
  expect(events.filter(event => event.startsWith('start'))).toHaveLength(3)
})

test('failed persistence and cancellation retain prior configuration without replaying work', async () => {
  const root = await workspace()
  const manager = new LspManager({ servers: [server] }, async options => ({ execute: async () => options.command, close: async () => {} })); managers.push(manager)
  await manager.forWorkspace(root).execute(request)
  await expect(manager.reconfigure({ servers: [{ ...server, command: 'new' }] }, () => { throw new Error('stale revision') })).rejects.toThrow('stale revision')
  expect(await manager.forWorkspace(root).execute(request)).toBe('configured-server')
  let committed = false
  await expect(manager.reconfigure({ servers: [] }, () => { committed = true; return undefined }, AbortSignal.abort())).rejects.toThrow('cancelled')
  expect(committed).toBe(false); expect(manager.configured).toBe(true)
})

test('settings updates reject concurrent edits and new requests while cleanup is pending', async () => {
  const root = await workspace(); let finish!: () => void; const gate = new Promise<void>(resolve => { finish = resolve })
  const manager = new LspManager({ servers: [server] }, async () => ({ execute: async () => 'ok', close: async () => { await gate } })); managers.push(manager)
  await manager.forWorkspace(root).execute(request)
  const changing = manager.reconfigure({ servers: [] }, () => undefined)
  await expect(manager.reconfigure({ servers: [] }, () => undefined)).rejects.toThrow('already in progress')
  await expect(manager.forWorkspace(root).execute(request)).rejects.toThrow('update in progress')
  finish(); await changing; expect(manager.configured).toBe(false)
})

test('cleanup failure prevents committing a new configuration', async () => {
  const root = await workspace(); let committed = false; let fail = true
  const manager = new LspManager({ servers: [server] }, async () => ({ execute: async () => 'ok', close: async () => { if (fail) throw new Error('cleanup failed') } })); managers.push(manager)
  await manager.forWorkspace(root).execute(request)
  try {
    await expect(manager.reconfigure({ servers: [] }, () => { committed = true; return undefined })).rejects.toThrow('not saved')
    expect(committed).toBe(false); expect(manager.configured).toBe(true)
    expect((await manager.health(root))[0]?.state).toBe('failed')
  } finally { fail = false }
})
