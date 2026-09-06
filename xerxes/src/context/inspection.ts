// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { DaemonSession } from '../daemon/runtime.js'
import { compactionHistory } from './compactionHistory.js'
import { readContextControls } from './controls.js'
import { estimateContextTokens } from './windowUsage.js'

export const CONTEXT_SECTIONS = ['instructions', 'memory', 'conversation', 'tools', 'compaction'] as const
export type ContextSection = typeof CONTEXT_SECTIONS[number]

/** A local inspection snapshot, never a claim to reproduce the exact live wire request. */
export function inspectSessionContext(session: Pick<DaemonSession, 'id' | 'model' | 'messages' | 'requestScaffold'> & { metadata?: Readonly<Record<string, unknown>> }, params: Readonly<Record<string, unknown>> = {}) {
  const section = params.section ?? 'instructions'
  const offset = params.offset ?? 0
  if (!CONTEXT_SECTIONS.includes(section as ContextSection) || typeof offset !== 'number' || !Number.isSafeInteger(offset) || offset < 0) throw new Error('Invalid context page')
  const scaffold = session.requestScaffold
  const segments = scaffold?.systemSegments ?? (scaffold?.systemPrompt ? [{ name: 'assembled instructions (layer provenance unavailable)', text: scaffold.systemPrompt }] : [])
  const memoryNames = new Set(['memory', 'self_memory'])
  const history = compactionHistory(session.metadata).reverse()
  const controls = readContextControls(session.metadata ?? {})
  const sources = new Map<string, { scope: string; path: string; content: string }>()
  for (const source of scaffold?.memorySources ?? []) sources.set(source.scope + ':' + source.path, source)
  for (const source of controls.pins) sources.set(source.scope + ':' + source.path, source)
  for (const source of controls.excluded) sources.set(source.scope + ':' + source.path, { ...source, content: 'Excluded from optional memory retrieval on the next turn.' })
  const memoryEntries = [...sources.values()].map(source => ({
    title: `[${source.scope}] ${source.path}`, render: () => source.content,
    value: { role: 'system', content: source.content },
    control: { scope: source.scope, path: source.path,
      pinned: controls.pins.some(item => item.scope === source.scope && item.path === source.path),
      excluded: controls.excluded.some(item => item.scope === source.scope && item.path === source.path) },
  }))
  const toolResults = session.messages.flatMap((message, index) => message.role === 'tool'
    ? [{ title: `Tool result · message ${index + 1}${typeof message.name === 'string' ? ' · ' + message.name : ''}`, render: () => JSON.stringify(message, null, 2), value: message }]
    : [])
  let retainedTurn = 0
  const conversation = session.messages.map((message, index) => {
    if (message.role === 'user') retainedTurn++
    return { title: `${index + 1} · ${message.role}${message.role === 'user' ? ' · retained turn ' + retainedTurn : ''}`, render: () => JSON.stringify(message, null, 2), value: message }
  })
  const groups = {
    compaction: history.map(entry => ({ title: entry.compacted_at + ' · ' + entry.reason, render: () => JSON.stringify(entry, null, 2), value: { role: 'system', content: JSON.stringify(entry) } })),
    instructions: segments.filter(segment => !memoryNames.has(segment.name)).map(segment => ({ title: segment.name, render: () => segment.text, value: { role: 'system', content: segment.text } })),
    memory: [...memoryEntries, ...segments.filter(segment => memoryNames.has(segment.name)).map(segment => ({ title: segment.name + (memoryEntries.length ? ' · assembled reference' : ''), render: () => segment.text, value: { role: 'system', content: segment.text } }))],
    conversation,
    tools: [...toolResults, ...(scaffold?.toolSchemas ?? []).map((tool, index) => ({ title: `Tool schema ${index + 1}`, render: () => JSON.stringify(tool, null, 2), value: { role: 'system', content: JSON.stringify(tool) } }))],
  }
  const fingerprint = new Bun.CryptoHasher('sha256').update(session.id + ':' + session.model + ':' + (scaffold?.capturedAt ?? 0))
  fingerprint.update(JSON.stringify(controls))
  for (const id of CONTEXT_SECTIONS) for (const entry of groups[id]) fingerprint.update(JSON.stringify([id, entry.title, entry.value]))
  const generation = fingerprint.digest('hex')
  if (params.generation !== undefined && params.generation !== generation) throw new Error('Context changed; refresh the inspector')
  const selected = groups[section as ContextSection]
  const count = (value: Readonly<Record<string, unknown>>) => estimateContextTokens([value], { model: session.model })
  return {
    ok: true, session_id: session.id, model: session.model, generation, controls_revision: controls.revision,
    captured_at: scaffold?.capturedAt ?? null,
    note: 'Approximate token contributions, not billing. Instructions, memory and schemas are the latest assembled scaffold; conversation is the currently retained transcript. Tools includes retained tool results followed by schemas. Tool results also appear in conversation, so section estimates overlap and must not be summed. In-flight provider messages may differ. Compaction shows up to 100 retained successful events (legacy sessions may have only the last stamp); archive paths are recorded locations, not verified files. History is not sent to the model. Optional source snapshots can be pinned or excluded for the next turn. Source previews overlap the assembled memory reference and are excluded from section token totals. Pinned snapshots persist; exclusions affect automatic recall, not explicit memory-read tools. Retrieval scores are not exposed.',
    sections: CONTEXT_SECTIONS.map(id => ({ id, count: groups[id].length,
      available: id === 'compaction' || id === 'conversation' || (id === 'tools' && toolResults.length > 0) || (id === 'memory' ? sources.size > 0 || scaffold?.systemSegments !== undefined : scaffold !== undefined),
      estimated_tokens: id === 'compaction' ? 0 : groups[id].reduce((total, entry) => total + ('control' in entry ? 0 : count(entry.value)), 0),
      provenance: id === 'compaction' ? 'Persisted compaction metadata' : id === 'conversation' ? 'Retained session transcript' : id === 'tools' ? 'Retained tool results and latest assembled schemas' : 'Latest daemon context assembly',
    })),
    section, offset, next_offset: offset + 20 < selected.length ? offset + 20 : null,
    entries: selected.slice(offset, offset + 20).map((entry, index) => { const text = entry.render(); return { index: offset + index, title: entry.title,
      text: text.slice(0, 8000), truncated: text.length > 8000, estimated_tokens: section === 'compaction' ? 0 : count(entry.value),
      ...('control' in entry ? { control: entry.control } : {}) } }),
  }
}
