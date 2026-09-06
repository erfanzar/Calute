// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { lspSettingsView, prepareLspSettingsEdit } from '../src/lsp/settings.js'
import { parseLspConfig } from '../src/lsp/config.js'
const snapshot = { revision: 'revision', servers: parseLspConfig({ servers: [{ name: 'ts', command: 'secret-command', args: ['secret-arg'], env: { TOKEN: 'secret-value' }, languageId: 'typescript', extensions: ['.ts'] }] }) }

test('settings view exposes edit metadata without launch arguments or credentials', () => {
  const view = lspSettingsView(snapshot)
  expect(JSON.stringify(view)).not.toContain('secret')
  expect(view.servers[0]).toMatchObject({ name: 'ts', enabled: true, configuredFields: ['command', 'args', 'env'], timeoutMs: 30000 })
  expect(view.servers[0]?.extensions).not.toBe(snapshot.servers[0]?.extensions)
})

test('partial updates retain saved fields; explicit null clears optional fields without mutating the snapshot', () => {
  const changed = prepareLspSettingsEdit(snapshot, { name: 'ts', revision: 'revision', action: 'update', changes: { enabled: false } })
  expect(changed[0]).toMatchObject({ enabled: false, command: 'secret-command', args: ['secret-arg'], env: { TOKEN: 'secret-value' } })
  const cleared = prepareLspSettingsEdit(snapshot, { name: 'ts', revision: 'revision', action: 'update', changes: { args: null, env: null, timeoutMs: null } })
  expect(cleared[0]?.args).toBeUndefined(); expect(cleared[0]?.env).toBeUndefined()
  expect(snapshot.servers[0]?.enabled).toBe(true); expect(snapshot.servers[0]?.env).toEqual({ TOKEN: 'secret-value' })
})

test('create and remove validate full-document suffix ownership and explicit target identity', () => {
  const changes = { command: 'server', languageId: 'rust', extensions: ['.rs'] }
  const created = prepareLspSettingsEdit(snapshot, { name: 'rust', revision: 'revision', action: 'create', changes })
  expect(created).toHaveLength(2)
  expect(prepareLspSettingsEdit(snapshot, { name: 'ts', revision: 'revision', action: 'remove' })).toEqual([])
  expect(() => prepareLspSettingsEdit(snapshot, { name: 'rust', revision: 'revision', action: 'create', changes: { ...changes, extensions: ['.ts'] } })).toThrow('suffix')
  for (const request of [
    { name: 'ts', revision: 'old', action: 'update', changes: {} },
    { name: 'ts', revision: 'revision', action: 'create', changes },
    { name: 'missing', revision: 'revision', action: 'remove' },
    { name: 'ts', revision: 'revision', action: 'remove', changes: {} },
    { name: 'ts', revision: 'revision', action: 'update', changes: { name: 'renamed' } },
    { name: 'ts', revision: 'revision', action: 'update', changes: { command: null } },
  ]) expect(() => prepareLspSettingsEdit(snapshot, request)).toThrow()
})
