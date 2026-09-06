// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { pathToFileURL } from 'node:url'
import { PluginRegistry } from '../src/extensions/plugins.js'

interface FixtureModule {
  readonly entered: { promise: Promise<void> }
  readonly gate: { resolve(): void }
  late(): void
}
async function fixture(fail: boolean) {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-plugin-transaction-'))
  const path = join(directory, 'plugin.mjs')
  await writeFile(path, `
export const entered = Promise.withResolvers()
export const gate = Promise.withResolvers()
let registry
export function late() { registry.registerTool('late-mutation', () => null, undefined, 'pending') }
export async function register(value) {
  registry = value
  registry.registerPlugin({ name: 'pending' })
  registry.registerTool('pending-tool', () => registry.getTool('host-later')?.(), undefined, 'pending')
  registry.registerHook('on_turn_end', () => 'pending', undefined, 'pending')
  registry.registerChannel('pending-channel', { kind: 'test' }, undefined, 'pending')
  registry.registerProvider('pending-provider', { createClient() { throw new Error('not invoked') } }, undefined, 'pending')
  entered.resolve()
  await gate.promise
  ${fail ? "throw new Error('registration failed')" : ''}
}
`)
  const module = await import(pathToFileURL(path).href) as FixtureModule
  return { directory, module, close: () => rm(directory, { recursive: true, force: true }) }
}

test('pending and failed registrations never publish capabilities or roll back concurrent host changes', async () => {
  const f = await fixture(true), registry = new PluginRegistry()
  const loading = registry.discover(f.directory)
  try {
    await f.module.entered.promise
    expect(registry.pluginNames).toEqual([])
    expect(registry.getTool('pending-tool')).toBeUndefined()
    expect(registry.getChannel('pending-channel')).toBeUndefined()
    expect(registry.getProvider('pending-provider')).toBeUndefined()
    expect(registry.getHooks('on_turn_end')).toEqual([])
    registry.registerPlugin({ name: 'host' })
    registry.registerTool('host-later', () => 'host', undefined, 'host')
    const hook = () => 'host'
    registry.registerHook('on_turn_end', hook, undefined, 'host')
    f.module.gate.resolve()
    expect(await loading).toEqual([])
    expect(registry.pluginNames).toEqual(['host'])
    expect(registry.getTool('host-later')?.()).toBe('host')
    expect(registry.getHooks('on_turn_end')).toEqual([hook])
    expect(registry.loadErrors[0]).toContain('registration failed')
    expect(() => f.module.late()).toThrow('registration is closed')
  } finally { f.module.gate.resolve(); await loading; await f.close() }
})

test('successful registration publishes together and retained readers follow the live registry', async () => {
  const f = await fixture(false), registry = new PluginRegistry()
  const loading = registry.discover(f.directory)
  try {
    await f.module.entered.promise
    expect(registry.inventory()).toEqual([])
    f.module.gate.resolve()
    expect(await loading).toEqual(['pending'])
    expect(registry.getChannel('pending-channel')).toEqual({ kind: 'test' })
    expect(registry.getProvider('pending-provider')).toBeDefined()
    expect(registry.getHooks('on_turn_end')).toHaveLength(1)
    registry.registerTool('host-later', () => 'added after commit')
    expect(registry.getTool('pending-tool')?.()).toBe('added after commit')
    expect(() => f.module.late()).toThrow('registration is closed')
    expect(registry.getTool('late-mutation')).toBeUndefined()
    expect(await registry.discover(f.directory)).toEqual([])
  } finally { f.module.gate.resolve(); await loading; await f.close() }
})

test('live conflicts detected at commit reject the entire staged registration', async () => {
  const f = await fixture(false), registry = new PluginRegistry()
  const loading = registry.discover(f.directory)
  try {
    await f.module.entered.promise
    registry.registerTool('pending-tool', () => 'host winner')
    f.module.gate.resolve()
    expect(await loading).toEqual([])
    expect(registry.getTool('pending-tool')?.()).toBe('host winner')
    expect(registry.getPlugin('pending')).toBeUndefined()
    expect(registry.getChannel('pending-channel')).toBeUndefined()
    expect(registry.getHooks('on_turn_end')).toEqual([])
    expect(registry.loadErrors[0]).toContain('conflicts')
  } finally { f.module.gate.resolve(); await loading; await f.close() }
})
