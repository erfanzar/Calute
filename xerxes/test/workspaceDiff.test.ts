// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, rm, symlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { collectGitDiff } from '../src/workspace/gitDiff.js'

test('untracked files are navigable diff sections including empty files and safe symlinks', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'xr-diff-'))
  try {
    await Bun.spawn(['git', 'init', '-q', dir]).exited
    await Bun.write(join(dir, 'new file.ts'), 'export const answer = 42\n')
    await Bun.write(join(dir, 'empty.txt'), '')
    await symlink('/etc/passwd', join(dir, 'external-link'))
    const result = await collectGitDiff({ cwd: dir, includeUntracked: true })
    expect(result.kind).toBe('ok')
    if (result.kind !== 'ok') throw new Error('missing diff')
    expect(result.diff.lines.filter(line => line.kind === 'file').map(line => line.text)).toEqual(expect.arrayContaining(['new file.ts', 'empty.txt', 'external-link']))
    expect(result.diff.lines.some(line => line.text === '+export const answer = 42')).toBe(true)
    expect(result.diff.lines.some(line => line.text === 'Empty untracked file')).toBe(true)
    expect(result.diff.lines.map(line => line.text).join('\n')).not.toContain('root:')
    const bounded = await collectGitDiff({ cwd: dir, includeUntracked: true, maxLines: 4 })
    expect(bounded.kind === 'ok' && bounded.diff.truncated).toBe(true)
  } finally { await rm(dir, { recursive: true, force: true }) }
})
