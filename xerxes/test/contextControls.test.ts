// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { readContextControls, updateContextControls } from '../src/context/controls.js'

const pin = (revision: number, scope = 'project', path = 'notes.md', content = 'remember this') => ({
  action: 'pin', revision, scope, path, content,
})
const pathPatch = (action: string, revision: number, scope = 'project', path = 'notes.md') => ({
  action, revision, scope, path,
})

test('missing metadata controls read as an empty versioned value', () => {
  expect(readContextControls({})).toEqual({ version: 1, revision: 0, pins: [], excluded: [] })
})

test('pin, exclude, and include resolve conflicts and increment revisions immutably', () => {
  const empty = readContextControls({})
  const excluded = updateContextControls(empty, pathPatch('exclude', 0))
  const pinned = updateContextControls(excluded, pin(1, 'project', 'notes.md', 'new content'))
  const included = updateContextControls(pinned, pathPatch('include', 2))

  expect(excluded).toMatchObject({ revision: 1, pins: [], excluded: [{ scope: 'project', path: 'notes.md' }] })
  expect(pinned).toMatchObject({ revision: 2, pins: [{ content: 'new content' }], excluded: [] })
  expect(included).toMatchObject({ revision: 3, pins: [{ content: 'new content' }], excluded: [] })
  expect(empty).toEqual({ version: 1, revision: 0, pins: [], excluded: [] })
})

test('unpin and include are idempotent removals while pin replaces content', () => {
  const first = updateContextControls(readContextControls({}), pin(0, 'global', 'guide.md', 'one'))
  const replaced = updateContextControls(first, pin(1, 'global', 'guide.md', 'two'))
  const unpinned = updateContextControls(replaced, pathPatch('unpin', 2, 'global', 'guide.md'))
  const unchanged = updateContextControls(unpinned, pathPatch('unpin', 3, 'global', 'guide.md'))

  expect(replaced).toMatchObject({ revision: 2, pins: [{ scope: 'global', path: 'guide.md', content: 'two' }] })
  expect(unpinned).toMatchObject({ revision: 3, pins: [] })
  expect(unchanged).toEqual({ version: 1, revision: 4, pins: [], excluded: [] })
})

test('rejects malformed stored controls, stale patches, unknown fields, and invalid paths', () => {
  expect(() => readContextControls({ context_controls: null })).toThrow(/Invalid metadata\.context_controls/)
  expect(() => readContextControls({ context_controls: { version: 2, revision: 0, pins: [], excluded: [] } })).toThrow('must be 1')
  expect(() => readContextControls({ context_controls: { version: 1, revision: 0, pins: [{ scope: 'project', path: 'a.md', content: 'x', extra: true }], excluded: [] } })).toThrow('unknown field')

  const controls = readContextControls({})
  expect(() => updateContextControls(controls, pin(1))).toThrow('revision mismatch')
  expect(() => updateContextControls(controls, { ...pin(0), arbitrary: true })).toThrow('unknown field')
  for (const path of ['../escape.md', '/absolute.md', 'dir\\file.md', 'notes.txt', 'dir//file.md', 'dir/./file.md', 'bad\nname.md']) {
    expect(() => updateContextControls(controls, pin(0, 'project', path))).toThrow(/Invalid context control patch\.path/)
  }
  expect(() => updateContextControls(controls, pin(0, 'team', 'notes.md'))).toThrow(/scope.*global or project/)
})

test('enforces per-pin, total, count, and exclusion limits', () => {
  const oversized = 'é'.repeat(4_001)
  expect(() => updateContextControls(readContextControls({}), pin(0, 'project', 'large.md', oversized))).toThrow('8000-byte')

  let controls = readContextControls({})
  for (let index = 0; index < 16; index += 1) {
    controls = updateContextControls(controls, pin(controls.revision, 'project', `pin-${index}.md`, 'x'))
  }
  expect(() => updateContextControls(controls, pin(controls.revision, 'project', 'seventeenth.md', 'x'))).toThrow('16-pin')

  const total = readContextControls({})
  const largeContent = 'x'.repeat(8_000)
  let totalControls = total
  for (let index = 0; index < 4; index += 1) {
    totalControls = updateContextControls(totalControls, pin(totalControls.revision, 'project', `total-${index}.md`, largeContent))
  }
  expect(() => updateContextControls(totalControls, pin(totalControls.revision, 'project', 'total-five.md', 'x'))).toThrow('32000-byte total')

  const excluded = readContextControls({
    context_controls: {
      version: 1,
      revision: 0,
      pins: [],
      excluded: Array.from({ length: 128 }, (_, index) => ({ scope: 'project', path: `excluded-${index}.md` })),
    },
  })
  expect(() => updateContextControls(excluded, pathPatch('exclude', 0, 'project', 'one-more.md'))).toThrow('128-exclusion')
})

test('returned controls and updates do not retain mutable input arrays or arbitrary properties', () => {
  const stored = {
    version: 1,
    revision: 4,
    pins: [{ scope: 'project', path: 'notes.md', content: 'hello' }],
    excluded: [],
  }
  const metadata = { context_controls: stored }
  const controls = readContextControls(metadata)
  const next = updateContextControls(controls, pin(4, 'global', 'other.md', 'world'))

  stored.pins[0]!.content = 'changed externally'
  stored.pins.push({ scope: 'project', path: 'another.md', content: 'external' })
  expect(controls.pins).toEqual([{ scope: 'project', path: 'notes.md', content: 'hello' }])
  expect(next).toMatchObject({ revision: 5, pins: [{ path: 'notes.md' }, { path: 'other.md' }] })
  expect(Object.keys(next)).toEqual(['version', 'revision', 'pins', 'excluded'])
  expect(() => (controls.pins as unknown as { push(value: unknown): number }).push({})).toThrow()
})
