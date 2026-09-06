// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export interface TerminalOutputCursor { readonly streamId: string; readonly offset: number }
export interface TerminalOutputPage {
  readonly text: string
  readonly cursor: TerminalOutputCursor
  readonly droppedChars: number
  readonly hasMore: boolean
  readonly running: boolean
}

/** Offsets count JavaScript UTF-16 code units, never bytes or terminal cells. */
export function terminalOutputPage(streamId: string, tail: string, total: number, running: boolean,
  cursor?: TerminalOutputCursor, limit = 20_000): TerminalOutputPage {
  if (!Number.isSafeInteger(limit) || limit < 1 || limit > 200_000) throw new Error('Invalid output page limit')
  if (cursor && (cursor.streamId !== streamId || !Number.isSafeInteger(cursor.offset) || cursor.offset < 0 || cursor.offset > total)) {
    throw new Error('Invalid output cursor; refresh this terminal without a cursor')
  }
  const requested = cursor?.offset ?? 0
  const retainedFrom = total - tail.length
  const from = Math.max(requested, retainedFrom)
  const end = Math.min(total, from + limit)
  return { text: tail.slice(from - retainedFrom, end - retainedFrom), cursor: { streamId, offset: end },
    droppedChars: Math.max(0, retainedFrom - requested), hasMore: end < total, running }
}

export function parseTerminalOutputCursor(value: unknown): TerminalOutputCursor | undefined {
  if (value === undefined) return undefined
  if (!value || typeof value !== 'object') throw new Error('Invalid output cursor')
  const cursor = value as Record<string, unknown>
  if (typeof cursor.streamId !== 'string' || !cursor.streamId || cursor.streamId.length > 8192 ||
      typeof cursor.offset !== 'number' || !Number.isSafeInteger(cursor.offset) || cursor.offset < 0) throw new Error('Invalid output cursor')
  return { streamId: cursor.streamId, offset: cursor.offset }
}
