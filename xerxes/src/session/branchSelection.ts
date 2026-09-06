// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** The message prefix that can be used as the history of a historical branch. */
export interface BranchTurnSelection {
  /** Number of transcript messages through the selected completed turn. */
  readonly messageCount: number
  /** Number of retained user turns in that prefix. */
  readonly turnCount: number
}

/**
 * Select a completed, retained user turn as the end of a historical branch.
 *
 * Selection is based on message roles and tool-call IDs. It does not inspect
 * assistant text, mutate messages, or construct a replacement transcript.
 * Messages after the next user turn are outside the selected prefix and are
 * deliberately not validated, since this helper cannot reconstruct history
 * that was compacted away.
 */
export function selectBranchTurn(
  messages: readonly Readonly<Record<string, unknown>>[],
  turn: number,
): BranchTurnSelection {
  if (!Number.isSafeInteger(turn) || turn < 1) {
    throw new RangeError('turn must be a positive 1-based integer')
  }

  const userTurnStarts: number[] = []
  for (let index = 0; index < messages.length; index += 1) {
    if (messages[index]?.role === 'user') userTurnStarts.push(index)
  }
  if (turn > userTurnStarts.length) {
    throw new RangeError(`retained user turn ${turn} is out of range`)
  }

  const selectedStart = userTurnStarts[turn - 1]
  if (selectedStart === undefined) {
    throw new RangeError(`retained user turn ${turn} is out of range`)
  }
  const messageCount = userTurnStarts[turn] ?? messages.length
  validatePrefix(messages, selectedStart, messageCount, turn)
  return { messageCount, turnCount: turn }
}

function validatePrefix(
  messages: readonly Readonly<Record<string, unknown>>[],
  selectedStart: number,
  messageCount: number,
  selectedTurn: number,
): void {
  let currentTurn = 0
  let sawAssistant = false
  let sawFinalAssistant = false
  let pendingToolCalls = new Set<string>()

  const finishTurn = (): void => {
    if (currentTurn === 0) return
    if (pendingToolCalls.size > 0 || !sawAssistant || !sawFinalAssistant) {
      throw incompleteTurn(selectedTurn, currentTurn)
    }
    sawAssistant = false
    sawFinalAssistant = false
    pendingToolCalls = new Set<string>()
  }

  for (let index = 0; index < messageCount; index += 1) {
    const message = messages[index]
    if (message === undefined) {
      throw new Error(`message ${index} is missing`)
    }
    const role = message.role

    if (role === 'user') {
      finishTurn()
      currentTurn += 1
      continue
    }

    // System messages are transcript preamble. They are retained as part of
    // the prefix and do not make a user turn complete or incomplete.
    if (role === 'system') continue

    if (role === 'assistant') {
      if (currentTurn === 0) {
        throw new Error(`assistant message ${index} precedes the first user turn`)
      }
      if (pendingToolCalls.size > 0) {
        throw incompleteTurn(selectedTurn, currentTurn)
      }
      const toolCallIds = assistantToolCallIds(message, index, selectedTurn, currentTurn)
      sawAssistant = true
      if (toolCallIds.length === 0) {
        sawFinalAssistant = true
      } else {
        sawFinalAssistant = false
        pendingToolCalls = new Set(toolCallIds)
      }
      continue
    }

    if (role === 'tool') {
      if (currentTurn === 0) {
        throw new Error(`tool message ${index} precedes the first user turn`)
      }
      const toolCallId = message.tool_call_id
      if (typeof toolCallId !== 'string' || toolCallId.trim() === '' || !pendingToolCalls.delete(toolCallId)) {
        throw incompleteTurn(selectedTurn, currentTurn)
      }
      continue
    }

    throw new Error(`message ${index} has unsupported role`)
  }

  // The selected prefix always contains the selected user turn. A malformed
  // preamble or an earlier incomplete turn is rejected before the branch can
  // copy it, and the selected turn is checked by the same rule.
  if (currentTurn === 0 || selectedStart >= messageCount) {
    throw new RangeError(`retained user turn ${selectedTurn} is out of range`)
  }
  finishTurn()
}

function assistantToolCallIds(
  message: Readonly<Record<string, unknown>>,
  index: number,
  selectedTurn: number,
  currentTurn: number,
): string[] {
  const rawToolCalls = message.tool_calls
  if (rawToolCalls === undefined) return []
  if (!Array.isArray(rawToolCalls)) {
    throw incompleteTurn(selectedTurn, currentTurn, `assistant message ${index} has malformed tool_calls`)
  }

  const ids: string[] = []
  const seen = new Set<string>()
  for (const rawToolCall of rawToolCalls) {
    if (!isRecord(rawToolCall) || typeof rawToolCall.id !== 'string' || rawToolCall.id.trim() === '') {
      throw incompleteTurn(selectedTurn, currentTurn, `assistant message ${index} has a tool call without an ID`)
    }
    if (seen.has(rawToolCall.id)) {
      throw incompleteTurn(selectedTurn, currentTurn, `assistant message ${index} repeats a tool call ID`)
    }
    seen.add(rawToolCall.id)
    ids.push(rawToolCall.id)
  }
  return ids
}

function incompleteTurn(selectedTurn: number, currentTurn: number, detail = 'tool activity or the final response is incomplete'): Error {
  return new Error(`retained user turn ${selectedTurn} is incomplete (turn ${currentTurn}: ${detail})`)
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
