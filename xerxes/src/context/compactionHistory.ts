// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { CompactionStamp } from '../daemon/compactionRunner.js'

const LIMIT = 100
function stamp(value: unknown): CompactionStamp | undefined {
  if (!value || typeof value !== 'object') return undefined
  const row = value as Record<string, unknown>
  if (typeof row.compacted_at !== 'string' || !Number.isFinite(Date.parse(row.compacted_at)) || typeof row.reason !== 'string') return undefined
  for (const key of ['messages_summarized', 'tokens_before', 'tokens_after']) {
    const value = row[key]
    if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) return undefined
  }
  return {
    compacted_at: row.compacted_at, reason: row.reason.slice(0, 256),
    messages_summarized: row.messages_summarized as number,
    tokens_before: row.tokens_before as number, tokens_after: row.tokens_after as number,
    ...(typeof row.archive_path === 'string' ? { archive_path: row.archive_path.slice(0, 4096) } : {}),
    ...(typeof row.archive_error === 'string' ? { archive_error: row.archive_error.slice(0, 4096) } : {}),
  }
}

/** Old sessions expose their last stamp without inventing earlier events. */
export function compactionHistory(metadata: Readonly<Record<string, unknown>> = {}): CompactionStamp[] {
  const history = Array.isArray(metadata.compaction_history)
    ? metadata.compaction_history.slice(-LIMIT).map(stamp).filter((entry): entry is CompactionStamp => entry !== undefined) : []
  const last = stamp(metadata.last_compaction)
  if (last && !history.some(entry => JSON.stringify(entry) === JSON.stringify(last))) history.push(last)
  return history.slice(-LIMIT)
}

export function recordCompaction(metadata: Record<string, unknown>, value: CompactionStamp): void {
  const validated = stamp(value)
  if (!validated) throw new Error('Invalid compaction history record')
  metadata.compaction_history = [...compactionHistory(metadata), validated].slice(-LIMIT)
  metadata.last_compaction = value
}
