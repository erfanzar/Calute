// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, rm, realpath } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { pathToFileURL } from 'node:url'
import { EditFeedback } from '../src/runtime/editFeedback.js'
import { EditDiagnostics } from '../src/runtime/editDiagnostics.js'
import { LspHost } from '../src/lsp/host.js'

/** Explicit local opt-in; never downloads or installs a language server. */
const command = process.env.XERXES_TEST_CLANGD?.trim()
test.skipIf(!command)('installed clangd supports navigation and versioned edit diagnostics', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-real-lsp-'))
  const path = join(root, 'main.cpp')
  let host: LspHost | undefined
  try {
    const source = 'int square(int x) { return x * x; }\nint main() { return square(4); }\n'
    await Bun.write(path, source)
    host = await LspHost.start({ command: command!, args: ['--background-index=false', '--clang-tidy=false'], cwd: root, languageId: 'cpp', timeoutMs: 10000 })
    const request = { filePath: 'main.cpp', line: 1, character: 21 }
    const hover = await host.execute({ ...request, action: 'hover' })
    expect(JSON.stringify(hover)).toContain('square')
    const uri = pathToFileURL(await realpath(path)).href
    expect(await host.execute({ ...request, action: 'definition' })).toEqual(expect.arrayContaining([expect.objectContaining({ uri, range: expect.objectContaining({ start: { line: 0, character: 4 } }) })]))
    expect(await host.execute({ ...request, action: 'references' })).toEqual(expect.arrayContaining([expect.objectContaining({ uri, range: expect.objectContaining({ start: { line: 1, character: 20 } }) })]))
    expect(await host.execute({ ...request, action: 'symbols' })).toEqual(expect.arrayContaining([expect.objectContaining({ name: 'square' }), expect.objectContaining({ name: 'main' })]))
    expect(await host.execute({ ...request, action: 'diagnostics', diagnosticsWaitMs: 5000 })).toMatchObject({ version: 1, fresh: true, diagnostics: [] })
    await Bun.write(path, 'int main() { return missing_symbol; }\n')
    expect(await host.execute({ ...request, action: 'diagnostics', diagnosticsWaitMs: 5000 })).toMatchObject({ version: 2, fresh: true, diagnostics: expect.arrayContaining([expect.objectContaining({ severity: 1, message: expect.stringContaining('missing_symbol') })]) })
    await Bun.write(path, source)
    expect(await host.execute({ ...request, action: 'diagnostics', diagnosticsWaitMs: 5000 })).toMatchObject({ version: 3, fresh: true, diagnostics: [] })
    const feedback = new EditFeedback(root, new EditDiagnostics(root, { fileExists: () => false }), host)
    await feedback.wrap({ execute: async () => { await Bun.write(path, 'int main() { return new_missing_symbol; }\n'); return 'written' } }).execute({ id: 'real-write', type: 'function', function: { name: 'WriteFile', arguments: { file_path: 'main.cpp', content: 'unused fixture argument' } } }, { metadata: {} })
    const report = await feedback.report()
    expect(report).toContain('1 new problem(s)')
    expect(report).toContain('new_missing_symbol')
    expect(report).not.toContain('unconfirmed')
    await host.close()
    expect(host.connected).toBe(false)
    await expect(host.execute({ ...request, action: 'hover' })).rejects.toThrow('unavailable')
  } finally { await host?.close(); await rm(root, { recursive: true, force: true }) }
}, 30000)
