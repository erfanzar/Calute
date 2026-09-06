// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { inspectSessionContext } from '../src/context/inspection.js'

test('inspection separates assembled memory from instructions and labels missing scaffold', () => {
  const session = { id: 'owner', model: 'test', messages: [] }
  expect(inspectSessionContext(session).sections.find(row => row.id === 'memory')?.available).toBe(false)
  const configured = { ...session, requestScaffold: { capturedAt: 100, systemSegments: [{ name: 'bootstrap', text: 'mandatory policy' }, { name: 'memory', text: 'retrieved fact' }], toolSchemas: [{ name: 'tool' }] } }
  const memory = inspectSessionContext(configured, { section: 'memory' })
  expect(memory.entries.map(entry => entry.text)).toEqual(['retrieved fact'])
  expect(memory.sections.find(row => row.id === 'instructions')?.count).toBe(1)
  expect(memory.note).toContain('not billing')
  expect(memory.captured_at).toBe(100)
})

test('persisted controls remain visible without scaffold and invalidate stale source pages', () => {
  const session = { id: 'owner', model: 'test', messages: [], metadata: { context_controls: {
    version: 1, revision: 1, pins: [{ scope: 'project', path: 'MEMORY.md', content: 'saved snapshot' }],
    excluded: [{ scope: 'global', path: 'USER.md' }],
  } } }
  const page = inspectSessionContext(session, { section: 'memory' })
  expect(page.entries).toMatchObject([
    { text: 'saved snapshot', control: { pinned: true, excluded: false } },
    { control: { pinned: false, excluded: true } },
  ])
  expect(page.sections.find(row => row.id === 'memory')).toMatchObject({ available: true, estimated_tokens: 0 })
  expect(page.captured_at).toBeNull()
  session.metadata.context_controls.revision++
  expect(() => inspectSessionContext(session, { generation: page.generation })).toThrow('Context changed')
})

test('pages are bounded and stale generations reject equal-length transcript edits', () => {
  const session = { id: 'owner', model: 'test', messages: Array.from({ length: 25 }, (_, index) => ({ role: 'user', content: `${index} ` + 'x'.repeat(9000) })) }
  const first = inspectSessionContext(session, { section: 'conversation' })
  expect(first.entries).toHaveLength(20)
  expect(first.entries[0]?.text.length).toBe(8000)
  expect(first.entries[0]?.truncated).toBe(true)
  const next = inspectSessionContext(session, { section: 'conversation', offset: first.next_offset, generation: first.generation })
  expect(next.entries).toHaveLength(5)
  expect(next.next_offset).toBeNull()
  session.messages[0]!.content = 'changed'
  expect(() => inspectSessionContext(session, { generation: first.generation })).toThrow('Context changed')
  expect(() => inspectSessionContext(session, { offset: -1 })).toThrow('Invalid context page')
  expect(() => inspectSessionContext(session, { section: 'hidden' })).toThrow('Invalid context page')
})

test('tools inspection exposes retained evidence without a scaffold and keeps the transcript intact', () => {
  const session = { id: 'owner', model: 'test', messages: [
    { role: 'user', content: 'Check the build' },
    { role: 'tool', name: 'ExecCommand', tool_call_id: 'build', content: 'Build failed: missing dependency' },
  ] }
  const before = JSON.stringify(session)
  const tools = inspectSessionContext(session, { section: 'tools' })
  expect(tools.sections.find(row => row.id === 'tools')).toMatchObject({ available: true, count: 1, provenance: 'Retained tool results and latest assembled schemas' })
  expect(tools.entries[0]?.title).toBe('Tool result · message 2 · ExecCommand')
  expect(tools.entries[0]?.text).toContain('Build failed: missing dependency')
  expect(tools.note).toContain('must not be summed')
  expect(inspectSessionContext(session, { section: 'conversation' }).entries).toHaveLength(2)
  expect(JSON.stringify(session)).toBe(before)
  session.messages[1]!.content = 'Build passed'
  expect(() => inspectSessionContext(session, { section: 'tools', generation: tools.generation })).toThrow('Context changed')
})

test('compaction history retains bounded validated metadata across reloads and invalidates pages', async () => {
  const { recordCompaction, compactionHistory } = await import('../src/context/compactionHistory.js')
  const stamp = { compacted_at: '2026-09-06T00:00:00.000Z', reason: 'compact', messages_summarized: 8, tokens_before: 1000, tokens_after: 200, archive_path: '/recorded/not-verified.jsonl' }
  const metadata: Record<string, unknown> = { last_compaction: stamp }
  expect(compactionHistory(metadata)).toEqual([stamp])
  for (let index = 1; index <= 101; index++) recordCompaction(metadata, { ...stamp, compacted_at: new Date(Date.parse(stamp.compacted_at) + index * 1000).toISOString() })
  const session = { id: 'owner', model: 'test', messages: [], metadata: JSON.parse(JSON.stringify(metadata)) }
  const page = inspectSessionContext(session, { section: 'compaction' })
  expect(page.sections.find(entry => entry.id === 'compaction')).toMatchObject({ count: 100, available: true, estimated_tokens: 0 })
  expect(page.entries).toHaveLength(20)
  expect(page.entries[0]?.text).toContain('00:01:41')
  expect(page.entries[0]?.estimated_tokens).toBe(0)
  expect(page.note).toContain('not verified files')
  recordCompaction(session.metadata, { ...stamp, reason: 'auto-compact' })
  expect(() => inspectSessionContext(session, { generation: page.generation })).toThrow('Context changed')
  expect(compactionHistory({ compaction_history: [null, { ...stamp, tokens_after: -1 }, stamp] })).toEqual([stamp])
  expect(() => recordCompaction(metadata, { ...stamp, tokens_after: NaN })).toThrow('Invalid')
})
