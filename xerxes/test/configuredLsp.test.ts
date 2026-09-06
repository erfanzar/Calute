// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, rm, writeFile, symlink } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { loadConfiguredLsp } from '../src/lsp/configured.js'

test('user LSP configuration is explicit, bounded, validated, and does not echo invalid contents', async () => {
  const home = await mkdtemp(join(tmpdir(), 'xerxes-lsp-config-'))
  const path = join(home, 'lsp.json')
  try {
    const missing = loadConfiguredLsp(home); expect(missing.configured).toBe(false); await missing.close()
    await writeFile(join(home, '.lsp.json'), JSON.stringify({ servers: [{ command: 'do-not-launch' }] }))
    const workspaceOnly = loadConfiguredLsp(home); expect(workspaceOnly.configured).toBe(false); await workspaceOnly.close()
    await writeFile(path, '{sensitive-value')
    try { loadConfiguredLsp(home); throw new Error('expected rejection') } catch (error) { expect(String(error)).toContain('valid UTF-8 JSON'); expect(String(error)).not.toContain('sensitive-value') }
    await writeFile(path, Buffer.from([0xff])); expect(() => loadConfiguredLsp(home)).toThrow('UTF-8')
    await writeFile(path, ' '.repeat(1024 * 1024 + 1)); expect(() => loadConfiguredLsp(home)).toThrow('1 MiB')
    await writeFile(path, JSON.stringify({ servers: [{ name: 'fixture', command: 'not-launched-until-requested', languageId: 'typescript', extensions: ['.ts'] }] }))
    const configured = loadConfiguredLsp(home); expect(configured.configured).toBe(true); await configured.close()
    await rm(path); await symlink(join(home, '.lsp.json'), path)
    expect(() => loadConfiguredLsp(home)).toThrow('regular file')
  } finally { await rm(home, { recursive: true, force: true }) }
})
