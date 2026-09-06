// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { LspManager } from '../src/lsp/manager.js'
import { LspSettingsStore } from '../src/lsp/settingsStore.js'
import { saveLspSettings } from '../src/lsp/settings.js'

test('settings apply commits disk and live configuration together and returns masked data', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lsp-apply-'))
  const store = new LspSettingsStore(join(root, 'lsp.json'))
  const manager = new LspManager({ servers: [] }, async () => { throw new Error('Saving must not eagerly start a server') })
  try {
    const saved = await saveLspSettings(manager, store, { action: 'create', revision: store.read().revision, name: 'ts', changes: { command: 'secret-command', env: { TOKEN: 'secret-token' }, languageId: 'typescript', extensions: ['.ts'] } })
    expect(JSON.stringify(saved)).not.toContain('secret')
    expect(manager.configured).toBe(true)
    expect(store.read().servers[0]?.env).toEqual({ TOKEN: 'secret-token' })
    expect((await manager.health(root))[0]?.state).toBe('idle')
    await expect(saveLspSettings(manager, store, { action: 'remove', revision: 'stale', name: 'ts' })).rejects.toThrow('changed')
    expect(manager.configured).toBe(true)
    await saveLspSettings(manager, store, { action: 'remove', revision: saved.revision, name: 'ts' })
    expect(manager.configured).toBe(false); expect(store.read().servers).toEqual([])
  } finally { await manager.close(); await rm(root, { recursive: true, force: true }) }
})
