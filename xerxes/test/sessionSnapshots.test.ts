// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, utimesSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { SnapshotManager } from '../src/session/snapshots.js'

function workspaceFixture(prefix: string): { directory: string; shadow: string; workspace: string } {
  const directory = mkdtempSync(join(tmpdir(), prefix))
  const workspace = join(directory, 'workspace')
  mkdirSync(workspace)
  return { directory, shadow: join(directory, 'shadow'), workspace }
}

test('snapshots exclude their own storage when it lives inside the workspace', async () => {
  const { directory, workspace } = workspaceFixture('xerxes-nested-shadow-')
  try {
    const shadow = join(workspace, 'private-state')
    const manager = new SnapshotManager(workspace, { shadowRoot: shadow })
    writeFileSync(join(workspace, 'a.txt'), 'original')
    const first = await manager.snapshot('first')
    writeFileSync(join(shadow, 'must-not-capture.txt'), 'private state')
    writeFileSync(join(workspace, 'a.txt'), 'changed')
    await manager.snapshot('second')
    await manager.rollback(first.id)
    expect(readFileSync(join(shadow, 'must-not-capture.txt'), 'utf8')).toBe('private state')
    expect(readFileSync(join(workspace, 'a.txt'), 'utf8')).toBe('original')
  } finally { rmSync(directory, { recursive: true, force: true }) }
})

test('snapshot records carry the session and turn they precede', async () => {
  if (!Bun.which('git')) return
  const { directory, shadow, workspace } = workspaceFixture('xerxes-snapshot-link-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    writeFileSync(join(workspace, 'a.txt'), 'turn zero', 'utf8')
    await snapshots.snapshot('turn-0', { sessionId: 'a1b2c3d4', turnIndex: 0 })
    writeFileSync(join(workspace, 'a.txt'), 'turn one', 'utf8')
    const second = await snapshots.snapshot('turn-1', { sessionId: 'a1b2c3d4', turnIndex: 1 })
    await snapshots.snapshot('turn-0', { sessionId: 'ffffffff', turnIndex: 0 })
    const manual = await snapshots.snapshot('manual')

    expect(manual.sessionId).toBeUndefined()
    expect(manual.turnIndex).toBeUndefined()
    expect(snapshots.listForSession('a1b2c3d4').map(record => record.turnIndex)).toEqual([0, 1])
    expect(snapshots.getForTurn('a1b2c3d4', 1)?.id).toBe(second.id)
    expect(snapshots.getForTurn('a1b2c3d4', 9)).toBeUndefined()
    expect(snapshots.getForTurn('', 0)).toBeUndefined()

    // "Take me back to before turn 1" restores the tree as the user saw it then.
    writeFileSync(join(workspace, 'a.txt'), 'agent damage', 'utf8')
    await snapshots.rollback(second.id)
    expect(readFileSync(join(workspace, 'a.txt'), 'utf8')).toBe('turn one')
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('concurrent managers serialize snapshots through one shadow repository', async () => {
  if (!Bun.which('git')) return
  const { directory, shadow, workspace } = workspaceFixture('xerxes-snapshot-concurrent-')
  try {
    writeFileSync(join(workspace, 'a.txt'), 'shared content', 'utf8')
    const managers = Array.from({ length: 8 }, () => new SnapshotManager(workspace, { shadowRoot: shadow }))

    const records = await Promise.all(managers.map((manager, index) => manager.snapshot(`parallel-${index}`)))

    expect(new SnapshotManager(workspace, { shadowRoot: shadow }).list()).toHaveLength(records.length)
    expect(new Set(records.map(record => record.id)).size).toBe(records.length)
    expect(new Set(records.map(record => record.commitSha)).size).toBe(records.length)
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('separate processes serialize first snapshot initialization', async () => {
  const realGit = Bun.which('git')
  if (!realGit) return
  const { directory, shadow, workspace } = workspaceFixture('xerxes-snapshot-process-race-')
  const bin = join(directory, 'bin')
  const gate = join(directory, 'init-gate')
  mkdirSync(bin)
  mkdirSync(gate)
  try {
    writeFileSync(join(workspace, 'a.txt'), 'shared content', 'utf8')
    const gitWrapper = join(bin, 'git')
    writeFileSync(gitWrapper, `#!${process.execPath}\n` + String.raw`
import { closeSync, openSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
const args = process.argv.slice(2)
if (args[0] === 'init') {
  closeSync(openSync(join(process.env.SNAPSHOT_INIT_GATE!, String(process.pid)), 'w'))
  const deadline = Date.now() + 250
  while (readdirSync(process.env.SNAPSHOT_INIT_GATE!).length < 2 && Date.now() < deadline) {
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 2)
  }
}
const child = Bun.spawnSync([process.env.REAL_GIT!, ...args], { cwd: process.cwd(), env: process.env, stdout: 'inherit', stderr: 'inherit' })
process.exit(child.exitCode)
`, 'utf8')
    chmodSync(gitWrapper, 0o755)
    const worker = join(directory, 'worker.ts')
    writeFileSync(worker, `
import { SnapshotManager } from ${JSON.stringify(join(import.meta.dir, '../src/session/snapshots.ts'))}
const manager = new SnapshotManager(process.argv[2]!, { shadowRoot: process.argv[3]! })
await manager.snapshot(process.argv[4]!)
`, 'utf8')
    const env = {
      ...process.env,
      PATH: `${bin}:${process.env.PATH ?? ''}`,
      REAL_GIT: realGit,
      SNAPSHOT_INIT_GATE: gate,
    }
    const children = ['process-one', 'process-two'].map(label => Bun.spawn(
      [process.execPath, worker, workspace, shadow, label],
      { env, stdout: 'pipe', stderr: 'pipe' },
    ))
    const results = await Promise.all(children.map(async child => ({
      exitCode: await child.exited,
      stderr: await new Response(child.stderr).text(),
    })))

    expect(results).toEqual([
      { exitCode: 0, stderr: '' },
      { exitCode: 0, stderr: '' },
    ])
    expect(new SnapshotManager(workspace, { shadowRoot: shadow }).list()).toHaveLength(2)
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('an abandoned stale repository lock is recovered', async () => {
  if (!Bun.which('git')) return
  const { directory, shadow, workspace } = workspaceFixture('xerxes-snapshot-stale-lock-')
  try {
    writeFileSync(join(workspace, 'a.txt'), 'content', 'utf8')
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    mkdirSync(shadow, { recursive: true })
    const lock = `${snapshots.shadowDirectory}.lock`
    writeFileSync(lock, '999999999\n', 'utf8')
    const stale = new Date(Date.now() - 120_000)
    utimesSync(lock, stale, stale)

    const record = await snapshots.snapshot('after-stale-lock')

    expect(record.label).toBe('after-stale-lock')
    expect(existsSync(lock)).toBeFalse()
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('reset waits behind an in-flight repository operation', async () => {
  if (!Bun.which('git')) return
  const { directory, shadow, workspace } = workspaceFixture('xerxes-snapshot-reset-race-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    writeFileSync(join(workspace, 'a.txt'), 'content', 'utf8')
    const snapshotting = snapshots.snapshot('in-flight')
    const resetting = snapshots.reset()

    await Promise.all([snapshotting, resetting])

    expect(snapshots.list()).toEqual([])
    expect(existsSync(snapshots.shadowDirectory)).toBeFalse()
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('a retried turn resolves to its newest capture', async () => {
  if (!Bun.which('git')) return
  const { directory, shadow, workspace } = workspaceFixture('xerxes-snapshot-retry-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    writeFileSync(join(workspace, 'a.txt'), 'first attempt', 'utf8')
    await snapshots.snapshot('turn-3', { sessionId: 'a1b2c3d4', turnIndex: 3 })
    writeFileSync(join(workspace, 'a.txt'), 'second attempt', 'utf8')
    const retry = await snapshots.snapshot('turn-3', { sessionId: 'a1b2c3d4', turnIndex: 3 })

    expect(snapshots.getForTurn('a1b2c3d4', 3)?.id).toBe(retry.id)
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('records written before the session link stay readable and roll back', async () => {
  if (!Bun.which('git')) return
  const { directory, shadow, workspace } = workspaceFixture('xerxes-snapshot-legacy-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    writeFileSync(join(workspace, 'a.txt'), 'legacy content', 'utf8')
    const record = await snapshots.snapshot('legacy')

    // Rewrite the log in the five-field shape earlier versions produced.
    const recordsPath = join(snapshots.shadowDirectory, '_records.txt')
    writeFileSync(
      recordsPath,
      `${[record.id, record.label, record.commitSha, record.createdAt, record.workspaceDir].join('\t')}\n`,
      'utf8',
    )

    const [restored] = snapshots.list()
    expect(restored?.id).toBe(record.id)
    expect(restored?.sessionId).toBeUndefined()
    expect(restored?.turnIndex).toBeUndefined()

    writeFileSync(join(workspace, 'a.txt'), 'changed', 'utf8')
    await snapshots.rollback(record.id)
    expect(readFileSync(join(workspace, 'a.txt'), 'utf8')).toBe('legacy content')
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('pruning keeps the session link on the rewritten records', async () => {
  if (!Bun.which('git')) return
  const { directory, shadow, workspace } = workspaceFixture('xerxes-snapshot-prune-link-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    writeFileSync(join(workspace, 'a.txt'), 'one', 'utf8')
    await snapshots.snapshot('turn-0', { sessionId: 'a1b2c3d4', turnIndex: 0 })
    writeFileSync(join(workspace, 'a.txt'), 'two', 'utf8')
    const second = await snapshots.snapshot('turn-1', { sessionId: 'a1b2c3d4', turnIndex: 1 })

    expect(await snapshots.prune({ keep: 1 })).toBe(1)
    const retained = snapshots.list()
    expect(retained).toHaveLength(1)
    expect(retained[0]).toMatchObject({ id: second.id, sessionId: 'a1b2c3d4', turnIndex: 1 })
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('single-file restore leaves unrelated edits alone and can be undone', async () => {
  if (!Bun.which('git')) return
  const { directory, shadow, workspace } = workspaceFixture('xerxes-snapshot-restore-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    mkdirSync(join(workspace, 'src'))
    writeFileSync(join(workspace, 'src', 'damaged.txt'), 'good version', 'utf8')
    writeFileSync(join(workspace, 'keep.txt'), 'original', 'utf8')
    const snapshot = await snapshots.snapshot('base', { sessionId: 'a1b2c3d4', turnIndex: 0 })

    writeFileSync(join(workspace, 'src', 'damaged.txt'), 'agent damage', 'utf8')
    writeFileSync(join(workspace, 'keep.txt'), 'deliberate later edit', 'utf8')
    writeFileSync(join(workspace, 'added.txt'), 'new work', 'utf8')

    const restored = await snapshots.restoreFile(snapshot.id, 'src/damaged.txt')

    expect(restored.path).toBe('src/damaged.txt')
    expect(readFileSync(join(workspace, 'src', 'damaged.txt'), 'utf8')).toBe('good version')
    // A single-file restore is not a rollback: everything else survives.
    expect(readFileSync(join(workspace, 'keep.txt'), 'utf8')).toBe('deliberate later edit')
    expect(existsSync(join(workspace, 'added.txt'))).toBe(true)

    // The pre-restore capture makes a mistaken restore reversible.
    await snapshots.rollback(restored.previous.id)
    expect(readFileSync(join(workspace, 'src', 'damaged.txt'), 'utf8')).toBe('agent damage')
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('single-file restore refuses untracked files and paths outside the workspace', async () => {
  if (!Bun.which('git')) return
  const { directory, shadow, workspace } = workspaceFixture('xerxes-snapshot-restore-guard-')
  try {
    writeFileSync(join(directory, 'outside.txt'), 'do not touch', 'utf8')
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    writeFileSync(join(workspace, 'a.txt'), 'content', 'utf8')
    const snapshot = await snapshots.snapshot('base')

    await expect(snapshots.restoreFile(snapshot.id, '../outside.txt')).rejects.toThrow(
      /escapes the snapshot workspace/,
    )
    await expect(snapshots.restoreFile(snapshot.id, 'never-existed.txt')).rejects.toThrow(
      /does not track never-existed.txt/,
    )
    await expect(snapshots.restoreFile('missing-ref', 'a.txt')).rejects.toThrow(/snapshot not found/)
    await expect(snapshots.restoreFile(snapshot.id, '   ')).rejects.toThrow(/file path is required/)
    // A refused restore never takes a pre-restore capture either.
    expect(snapshots.list()).toHaveLength(1)
    expect(readFileSync(join(directory, 'outside.txt'), 'utf8')).toBe('do not touch')
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('rollback preserves ignored files absent from its backup even when ignore rules change', async () => {
  const { directory, shadow, workspace } = workspaceFixture('xerxes-rollback-ignored-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    writeFileSync(join(workspace, 'tracked.txt'), 'before')
    const target = await snapshots.snapshot('target')
    writeFileSync(join(workspace, 'tracked.txt'), 'edited')
    writeFileSync(join(workspace, '.gitignore'), 'local-data/\nprivate.log\n')
    mkdirSync(join(workspace, 'local-data'))
    writeFileSync(join(workspace, 'local-data', 'unsaved.txt'), 'irreplaceable')
    writeFileSync(join(workspace, 'private.log'), 'ignored evidence')
    writeFileSync(join(workspace, 'new[1].txt'), 'backed up')
    writeFileSync(join(workspace, 'new1.txt'), 'also backed up')
    await snapshots.rollback(target.id)
    expect(readFileSync(join(workspace, 'tracked.txt'), 'utf8')).toBe('before')
    expect(readFileSync(join(workspace, 'local-data', 'unsaved.txt'), 'utf8')).toBe('irreplaceable')
    expect(readFileSync(join(workspace, 'private.log'), 'utf8')).toBe('ignored evidence')
    expect(existsSync(join(workspace, 'new[1].txt'))).toBe(false)
    expect(existsSync(join(workspace, 'new1.txt'))).toBe(false)
    const backup = snapshots.list().find(record => record.label === `pre-rollback:${target.id}`)!
    await snapshots.rollback(backup.id)
    expect(readFileSync(join(workspace, 'new[1].txt'), 'utf8')).toBe('backed up')
    expect(readFileSync(join(workspace, 'private.log'), 'utf8')).toBe('ignored evidence')
  } finally { rmSync(directory, { recursive: true, force: true }) }
})

test('snapshot preview includes new files while preserving workspace and shadow index', async () => {
  const { directory, shadow, workspace } = workspaceFixture('xerxes-snapshot-preview-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    writeFileSync(join(workspace, 'a.txt'), 'before\n')
    writeFileSync(join(workspace, 'deleted.txt'), 'restore me\n')
    writeFileSync(join(workspace, '.gitignore'), '*.log\n')
    for (const args of [['init', '--quiet'], ['add', 'a.txt']]) {
      const result = Bun.spawnSync(['git', ...args], { cwd: workspace })
      expect(result.exitCode).toBe(0)
    }
    const realIndex = readFileSync(join(workspace, '.git', 'index'))
    const record = await snapshots.snapshot('target')
    const index = join(snapshots.shadowDirectory, '.git', 'index')
    const before = readFileSync(index)
    writeFileSync(join(workspace, 'a.txt'), 'after\n')
    writeFileSync(join(workspace, 'new.txt'), 'remove me\n')
    writeFileSync(join(workspace, 'ignored.log'), 'not captured')
    rmSync(join(workspace, 'deleted.txt'))
    const diff = await snapshots.diff(record.id, true)
    expect(diff).toContain('-after')
    expect(diff).toContain('+before')
    expect(diff).toContain('-remove me')
    expect(diff).toContain('+restore me')
    expect(diff).not.toContain('ignored.log')
    expect(readFileSync(index)).toEqual(before)
    expect(readFileSync(join(workspace, '.git', 'index'))).toEqual(realIndex)
    expect(readFileSync(join(workspace, 'a.txt'), 'utf8')).toBe('after\n')
    expect(snapshots.list()).toHaveLength(1)
    await expect(snapshots.diff('missing-snapshot')).rejects.toThrow('not found')
  } finally { rmSync(directory, { recursive: true, force: true }) }
})

 test('oversized snapshot previews fail explicitly and allow a later preview', async () => {
  const { directory, shadow, workspace } = workspaceFixture('xerxes-preview-limit-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    writeFileSync(join(workspace, 'a.txt'), 'before\n')
    const record = await snapshots.snapshot('target')
    writeFileSync(join(workspace, 'a.txt'), 'a'.repeat(3 * 1024 * 1024))
    await expect(snapshots.diff(record.id)).rejects.toThrow('output exceeded')
    writeFileSync(join(workspace, 'a.txt'), 'small\n')
    expect(await snapshots.diff(record.id)).toContain('+small')
    expect(snapshots.list()).toHaveLength(1)
  } finally { rmSync(directory, { recursive: true, force: true }) }
})

test.each(['edit', 'create', 'delete'] as const)('reviewed restore refuses a concurrent %s and accepts a fresh preview', async change => {
  const { directory, shadow, workspace } = workspaceFixture('xerxes-preview-revision-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    const path = join(workspace, 'a.txt')
    writeFileSync(path, 'target\n')
    const target = await snapshots.snapshot('target')
    writeFileSync(path, 'reviewed\n')
    const preview = await snapshots.preview(target.id)
    if (change === 'edit') writeFileSync(path, 'concurrent\n')
    if (change === 'create') writeFileSync(join(workspace, 'new.txt'), 'concurrent\n')
    if (change === 'delete') rmSync(path)
    await expect(snapshots.rollback(target.id, preview.revision)).rejects.toThrow('preview is stale')
    if (change === 'delete') expect(existsSync(path)).toBe(false)
    else expect(readFileSync(path, 'utf8')).toBe(change === 'edit' ? 'concurrent\n' : 'reviewed\n')
    if (change === 'create') expect(readFileSync(join(workspace, 'new.txt'), 'utf8')).toBe('concurrent\n')
    const refreshed = await snapshots.preview(target.id)
    await snapshots.rollback(target.id, refreshed.revision)
    expect(readFileSync(path, 'utf8')).toBe('target\n')
  } finally { rmSync(directory, { recursive: true, force: true }) }
})

test('restore revisions cannot be reused for another snapshot target', async () => {
  const { directory, shadow, workspace } = workspaceFixture('xerxes-preview-target-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    const path = join(workspace, 'a.txt')
    writeFileSync(path, 'first')
    const first = await snapshots.snapshot('first')
    writeFileSync(path, 'second')
    const second = await snapshots.snapshot('second')
    const preview = await snapshots.preview(first.id)
    await expect(snapshots.rollback(second.id, preview.revision)).rejects.toThrow('another target')
    expect(readFileSync(path, 'utf8')).toBe('second')
  } finally { rmSync(directory, { recursive: true, force: true }) }
})

test.each(['directory', 'file-parent', 'symlink-parent'] as const)('restore preflight preserves earlier files when a later target has a %s obstruction', async obstruction => {
  const { directory, shadow, workspace } = workspaceFixture('xerxes-restore-preflight-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    writeFileSync(join(workspace, 'a.txt'), 'target')
    mkdirSync(join(workspace, 'z'))
    writeFileSync(join(workspace, 'z', 'last.txt'), 'target last')
    const target = await snapshots.snapshot('target')
    writeFileSync(join(workspace, 'a.txt'), 'keep my edits')
    rmSync(join(workspace, 'z'), { recursive: true })
    if (obstruction === 'directory') {
      mkdirSync(join(workspace, 'z', 'last.txt'), { recursive: true })
      writeFileSync(join(workspace, 'z', 'last.txt', 'ignored.log'), 'unbacked data')
    } else if (obstruction === 'file-parent') {
      writeFileSync(join(workspace, 'z'), 'blocking file')
    } else {
      mkdirSync(join(directory, 'outside'))
      writeFileSync(join(directory, 'outside', 'last.txt'), 'outside data')
      symlinkSync(join(directory, 'outside'), join(workspace, 'z'))
    }
    await expect(snapshots.rollback(target.id)).rejects.toThrow('Restore refused')
    expect(readFileSync(join(workspace, 'a.txt'), 'utf8')).toBe('keep my edits')
    await expect(snapshots.restoreFile(target.id, 'z/last.txt')).rejects.toThrow('Restore refused')
    expect(snapshots.list()).toHaveLength(1)
    if (obstruction === 'symlink-parent') expect(readFileSync(join(directory, 'outside', 'last.txt'), 'utf8')).toBe('outside data')
    if (obstruction === 'directory') expect(readFileSync(join(workspace, 'z', 'last.txt', 'ignored.log'), 'utf8')).toBe('unbacked data')
  } finally { rmSync(directory, { recursive: true, force: true }) }
})

test('file-scoped preview restores a literal filename while preserving later unrelated edits', async () => {
  const { directory, shadow, workspace } = workspaceFixture('xerxes-file-revision-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    const filename = ' a[1].txt '
    writeFileSync(join(workspace, filename), 'target')
    writeFileSync(join(workspace, 'other.txt'), 'original other')
    const target = await snapshots.snapshot('target')
    writeFileSync(join(workspace, filename), 'reviewed')
    const preview = await snapshots.preview(target.id, true, filename)
    expect(preview.files).toContain(filename)
    expect(preview.action).toBe('restore')
    writeFileSync(join(workspace, 'other.txt'), 'new unrelated edit')
    await snapshots.restoreFile(target.id, filename, preview.revision)
    expect(readFileSync(join(workspace, filename), 'utf8')).toBe('target')
    expect(readFileSync(join(workspace, 'other.txt'), 'utf8')).toBe('new unrelated edit')
    writeFileSync(join(workspace, filename), 'concurrent')
    await expect(snapshots.restoreFile(target.id, filename, preview.revision)).rejects.toThrow('stale')
    expect(readFileSync(join(workspace, filename), 'utf8')).toBe('concurrent')
    await expect(snapshots.preview(target.id, true, '../outside')).rejects.toThrow('escapes')
  } finally { rmSync(directory, { recursive: true, force: true }) }
})

test('guarded file removal is explicit, backed up, and leaves unrelated files intact', async () => {
  const { directory, shadow, workspace } = workspaceFixture('xerxes-file-remove-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    writeFileSync(join(workspace, 'other.txt'), 'target')
    const target = await snapshots.snapshot('target')
    writeFileSync(join(workspace, 'new.txt'), 'new file')
    const preview = await snapshots.preview(target.id, true, 'new.txt')
    expect(preview.action).toBe('remove')
    expect(preview.diff).toContain('-new file')
    writeFileSync(join(workspace, 'other.txt'), 'unrelated')
    const removed = await snapshots.restoreFile(target.id, 'new.txt', preview.revision)
    expect(existsSync(join(workspace, 'new.txt'))).toBe(false)
    expect(readFileSync(join(workspace, 'other.txt'), 'utf8')).toBe('unrelated')
    await snapshots.restoreFile(removed.previous.id, 'new.txt')
    expect(readFileSync(join(workspace, 'new.txt'), 'utf8')).toBe('new file')
    await expect(snapshots.preview(target.id, true, 'absent.txt')).rejects.toThrow('absent')
  } finally { rmSync(directory, { recursive: true, force: true }) }
})

test('restore journals precede writes, persist failure details, and pin recovery backups during pruning', async () => {
  const { directory, shadow, workspace } = workspaceFixture('xerxes-restore-journal-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    writeFileSync(join(workspace, 'a.txt'), 'target')
    const target = await snapshots.snapshot('target')
    writeFileSync(join(workspace, 'a.txt'), 'original work')
    const git = snapshots as unknown as { runGitUnlocked(args: readonly string[], index?: string, max?: number): Promise<string> }
    const original = git.runGitUnlocked.bind(snapshots)
    git.runGitUnlocked = async (args, index, max) => {
      if (args[0] === 'checkout-index') {
        const fresh = new SnapshotManager(workspace, { shadowRoot: shadow })
        const attempts = await fresh.restoreAttempts()
        expect(attempts.at(-1)?.phase).toBe('prepared')
        expect(readFileSync(join(workspace, 'a.txt'), 'utf8')).toBe('original work')
        writeFileSync(join(workspace, 'a.txt'), 'partially restored')
        throw new Error('injected checkout failure')
      }
      return original(args, index, max)
    }
    await expect(snapshots.rollback(target.id)).rejects.toThrow('workspace may be partially changed')
    const fresh = new SnapshotManager(workspace, { shadowRoot: shadow })
    const attempt = (await fresh.restoreAttempts()).at(-1)!
    expect(attempt.phase).toBe('failed')
    expect(attempt.error).toContain('injected checkout failure')
    expect(attempt.targetId).toBe(target.id)
    await fresh.prune({ keep: 0 })
    expect(fresh.get(attempt.backupId)).toBeDefined()
    expect(fresh.get(target.id)).toBeDefined()
    writeFileSync(join(fresh.shadowDirectory, '_restore-attempts.json'), JSON.stringify(Array.from({ length: 128 }, (_, i) => ({ ...attempt, id: i === 0 ? attempt.id : `pending-${i}` }))))
    await fresh.rollback(attempt.backupId)
    expect(readFileSync(join(workspace, 'a.txt'), 'utf8')).toBe('original work')
    expect((await fresh.restoreAttempts()).at(-1)?.phase).toBe('completed')
    expect((await fresh.restoreAttempts()).find(row => row.id === attempt.id)?.phase).toBe('recovered')
  } finally { rmSync(directory, { recursive: true, force: true }) }
})

test('corrupt restore recovery metadata refuses mutation instead of hiding evidence', async () => {
  const { directory, shadow, workspace } = workspaceFixture('xerxes-journal-corrupt-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    writeFileSync(join(workspace, 'a.txt'), 'target')
    const target = await snapshots.snapshot('target')
    writeFileSync(join(workspace, 'a.txt'), 'keep current work')
    writeFileSync(join(snapshots.shadowDirectory, '_restore-attempts.json'), '[{"phase":"invented"}]')
    await expect(snapshots.rollback(target.id)).rejects.toThrow('Invalid restore recovery record')
    expect(readFileSync(join(workspace, 'a.txt'), 'utf8')).toBe('keep current work')
    await expect(snapshots.restoreAttempts()).rejects.toThrow('Invalid restore recovery record')
  } finally { rmSync(directory, { recursive: true, force: true }) }
})

test('a killed restore owner leaves recoverable prepared evidence and releases its dead-owner lock promptly', async () => {
  const { directory, shadow, workspace } = workspaceFixture('xerxes-restore-crash-')
  const worker = join(directory, 'restore-owner.ts')
  writeFileSync(worker, `
    import { SnapshotManager } from ${JSON.stringify(join(import.meta.dir, '../src/session/snapshots.ts'))};
    import { writeFileSync } from 'node:fs';
    import { join } from 'node:path';
    const workspace = process.argv[2]!;
    const manager = new SnapshotManager(workspace, { shadowRoot: process.argv[3]! });
    writeFileSync(join(workspace, 'a.txt'), 'target');
    const target = await manager.snapshot('target');
    writeFileSync(join(workspace, 'a.txt'), 'original work');
    const original = manager.runGitUnlocked.bind(manager);
    manager.runGitUnlocked = async (args, index, max) => {
      if (args[0] === 'checkout-index') {
        writeFileSync(join(workspace, 'a.txt'), 'partially restored');
        console.log(JSON.stringify((await manager.restoreAttempts()).at(-1)));
        Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 10000);
      }
      return original(args, index, max);
    };
    await manager.rollback(target.id);
  `)
  const child = Bun.spawn([process.execPath, worker, workspace, shadow], { stdout: 'pipe', stderr: 'pipe', timeout: 12000 })
  try {
    const reader = child.stdout.getReader()
    let text = ''
    while (!text.includes('\n')) {
      const chunk = await reader.read()
      if (chunk.done) throw new Error(`Restore child exited before checkpoint: ${await new Response(child.stderr).text()}`)
      text += new TextDecoder().decode(chunk.value)
    }
    reader.releaseLock()
    const checkpoint = JSON.parse(text.split('\n')[0]!) as { id: string; backupId: string }
    child.kill('SIGKILL')
    await child.exited
    const fresh = new SnapshotManager(workspace, { shadowRoot: shadow })
    expect((await fresh.restoreAttempts()).find(row => row.id === checkpoint.id)?.phase).toBe('prepared')
    expect(readFileSync(join(workspace, 'a.txt'), 'utf8')).toBe('partially restored')
    await fresh.rollback(checkpoint.backupId)
    expect(readFileSync(join(workspace, 'a.txt'), 'utf8')).toBe('original work')
    expect((await fresh.restoreAttempts()).find(row => row.id === checkpoint.id)?.phase).toBe('recovered')
  } finally {
    child.kill('SIGKILL'); await child.exited
    rmSync(directory, { recursive: true, force: true })
  }
}, 15000)

test('failed recovery retries retain the original backup without growing unresolved journal records', async () => {
  const { directory, shadow, workspace } = workspaceFixture('xerxes-recovery-retries-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    writeFileSync(join(workspace, 'a.txt'), 'original')
    const backup = await snapshots.snapshot('backup')
    writeFileSync(join(workspace, 'a.txt'), 'partial')
    const anchor = { id: 'anchor', targetId: 'original-target', backupId: backup.id, path: null, phase: 'failed' as const, updatedAt: new Date().toISOString() }
    writeFileSync(join(snapshots.shadowDirectory, '_restore-attempts.json'), JSON.stringify([anchor]))
    const git = snapshots as unknown as { runGitUnlocked(args: readonly string[], index?: string, max?: number): Promise<string> }
    const original = git.runGitUnlocked.bind(snapshots)
    let failure = 0
    git.runGitUnlocked = async (args, index, max) => { if (args[0] === 'checkout-index') { writeFileSync(join(workspace, 'a.txt'), `unknown partial write ${++failure}`); throw new Error('retry failure') } return original(args, index, max) }
    for (let retry = 0; retry < 4; retry++) {
      await expect(snapshots.rollback(backup.id)).rejects.toThrow('retry failure')
      const attempts = await snapshots.restoreAttempts()
      expect(attempts).toHaveLength(2)
      expect(attempts[0]).toEqual(anchor)
    }
    git.runGitUnlocked = original
    await snapshots.rollback(backup.id)
    expect(readFileSync(join(workspace, 'a.txt'), 'utf8')).toBe('original')
    expect((await snapshots.restoreAttempts()).filter(row => row.phase === 'failed' || row.phase === 'prepared')).toHaveLength(0)
  } finally { rmSync(directory, { recursive: true, force: true }) }
})

test.each([false, true])('failed checkout reverses known target files and preserves concurrent content (conflict=%s)', async conflict => {
  const { directory, shadow, workspace } = workspaceFixture('xerxes-auto-reversal-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    writeFileSync(join(workspace, 'a.txt'), 'target A')
    writeFileSync(join(workspace, 'b.txt'), 'target B')
    writeFileSync(join(workspace, 'added.txt'), 'target addition')
    const target = await snapshots.snapshot('target')
    writeFileSync(join(workspace, 'a.txt'), 'original A')
    writeFileSync(join(workspace, 'b.txt'), 'original B')
    rmSync(join(workspace, 'added.txt'))
    writeFileSync(join(workspace, 'removed.txt'), 'original removed')
    const git = snapshots as unknown as { runGitUnlocked(args: readonly string[], index?: string, max?: number): Promise<string> }
    const original = git.runGitUnlocked.bind(snapshots)
    git.runGitUnlocked = async (args, index, max) => {
      const result = await original(args, index, max)
      if (args[0] === 'checkout-index') {
        rmSync(join(workspace, 'removed.txt'))
        if (conflict) writeFileSync(join(workspace, 'b.txt'), 'concurrent work')
        throw new Error('injected failure after target writes')
      }
      return result
    }
    await expect(snapshots.rollback(target.id)).rejects.toThrow(conflict ? 'need review' : 'restored automatically')
    expect(readFileSync(join(workspace, 'a.txt'), 'utf8')).toBe('original A')
    expect(readFileSync(join(workspace, 'b.txt'), 'utf8')).toBe(conflict ? 'concurrent work' : 'original B')
    expect(existsSync(join(workspace, 'added.txt'))).toBe(false)
    expect(readFileSync(join(workspace, 'removed.txt'), 'utf8')).toBe('original removed')
    expect((await snapshots.restoreAttempts()).at(-1)?.phase).toBe(conflict ? 'failed' : 'reverted')
  } finally { rmSync(directory, { recursive: true, force: true }) }
})

test('failed selected-file removal restores its backed-up leaf automatically', async () => {
  const { directory, shadow, workspace } = workspaceFixture('xerxes-remove-reversal-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    writeFileSync(join(workspace, 'other.txt'), 'unrelated')
    const target = await snapshots.snapshot('target')
    writeFileSync(join(workspace, 'new.txt'), 'recover this file')
    const preview = await snapshots.preview(target.id, true, 'new.txt')
    const git = snapshots as unknown as { runGitUnlocked(args: readonly string[], index?: string, max?: number): Promise<string> }
    const original = git.runGitUnlocked.bind(snapshots)
    git.runGitUnlocked = async (args, index, max) => {
      if (args[0] === 'update-index' && args[1] === '--force-remove') throw new Error('injected index write failure')
      return original(args, index, max)
    }
    await expect(snapshots.restoreFile(target.id, 'new.txt', preview.revision)).rejects.toThrow('restored automatically')
    expect(readFileSync(join(workspace, 'new.txt'), 'utf8')).toBe('recover this file')
    expect(readFileSync(join(workspace, 'other.txt'), 'utf8')).toBe('unrelated')
    expect((await snapshots.restoreAttempts()).at(-1)?.phase).toBe('reverted')
  } finally { rmSync(directory, { recursive: true, force: true }) }
})
