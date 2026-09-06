// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { selectBranchTurn } from '../src/session/branchSelection.js'

type Message = Readonly<Record<string, unknown>>

const assistant = (content: string): Message => ({ role: 'assistant', content })
const user = (content: string): Message => ({ role: 'user', content })

test('selects the completed retained turn and excludes later user turns', () => {
  const messages: readonly Message[] = [
    { role: 'system', content: 'preamble' },
    user('first'),
    assistant('first answer'),
    user('second'),
    assistant('second answer'),
    user('later'),
    assistant('later answer'),
  ]

  expect(selectBranchTurn(messages, 2)).toEqual({ messageCount: 5, turnCount: 2 })
})

test('counts a structurally paired tool round and requires its final assistant response', () => {
  const messages: readonly Message[] = [
    user('inspect this'),
    {
      role: 'assistant',
      content: 'I will inspect it.',
      tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'ReadFile', arguments: '{}' } }],
    },
    { role: 'tool', tool_call_id: 'call-1', content: 'file contents' },
    assistant('The file is valid.'),
    user('follow up'),
    assistant('follow-up answer'),
  ]

  expect(selectBranchTurn(messages, 1)).toEqual({ messageCount: 4, turnCount: 1 })
})

test('uses structural tool fields instead of assistant content text', () => {
  const messages: readonly Message[] = [
    user('call a tool'),
    assistant('The words tool_call and tool_result here are ordinary text.'),
  ]
  const before = structuredClone(messages)

  expect(selectBranchTurn(messages, 1)).toEqual({ messageCount: 2, turnCount: 1 })
  expect(messages).toEqual(before)
})

test('rejects invalid and out-of-range turn positions', () => {
  const messages: readonly Message[] = [user('one'), assistant('done')]

  expect(() => selectBranchTurn(messages, 0)).toThrow('positive 1-based integer')
  expect(() => selectBranchTurn(messages, 1.5)).toThrow('positive 1-based integer')
  expect(() => selectBranchTurn(messages, Number.MAX_SAFE_INTEGER + 1)).toThrow('positive 1-based integer')
  expect(() => selectBranchTurn(messages, 2)).toThrow('out of range')
})

test('rejects incomplete selected turns and dangling tool activity', () => {
  const missingResult: readonly Message[] = [
    user('inspect'),
    { role: 'assistant', content: '', tool_calls: [{ id: 'call-1' }] },
  ]
  const orphanResult: readonly Message[] = [
    user('inspect'),
    assistant('done'),
    { role: 'tool', tool_call_id: 'call-1', content: 'orphan' },
  ]
  const noFinalResponse: readonly Message[] = [
    user('inspect'),
    { role: 'assistant', content: '', tool_calls: [{ id: 'call-1' }] },
    { role: 'tool', tool_call_id: 'call-1', content: 'result' },
  ]

  expect(() => selectBranchTurn(missingResult, 1)).toThrow('incomplete')
  expect(() => selectBranchTurn(orphanResult, 1)).toThrow('incomplete')
  expect(() => selectBranchTurn(noFinalResponse, 1)).toThrow('incomplete')
})

test('does not treat an earlier assistant response as final after a later tool round', () => {
  const messages: readonly Message[] = [
    user('inspect again'),
    assistant('initial response'),
    { role: 'assistant', content: '', tool_calls: [{ id: 'call-1' }] },
    { role: 'tool', tool_call_id: 'call-1', content: 'result' },
  ]

  expect(() => selectBranchTurn(messages, 1)).toThrow('incomplete')
})

test('rejects malformed or mismatched tool pairs before the selected cutoff', () => {
  const duplicateResult: readonly Message[] = [
    user('inspect'),
    { role: 'assistant', content: '', tool_calls: [{ id: 'call-1' }] },
    { role: 'tool', tool_call_id: 'call-1', content: 'first' },
    { role: 'tool', tool_call_id: 'call-1', content: 'duplicate' },
    assistant('done'),
  ]
  const malformedCall: readonly Message[] = [
    user('inspect'),
    { role: 'assistant', content: '', tool_calls: [{ type: 'function' }] },
    assistant('done'),
  ]

  expect(() => selectBranchTurn(duplicateResult, 1)).toThrow('incomplete')
  expect(() => selectBranchTurn(malformedCall, 1)).toThrow('incomplete')
})

test('does not inspect messages after the next user turn', () => {
  const messages: readonly Message[] = [
    user('complete this'),
    assistant('done'),
    user('unfinished later turn'),
    { role: 'assistant', content: '', tool_calls: [{ id: 'later-call' }] },
  ]

  expect(selectBranchTurn(messages, 1)).toEqual({ messageCount: 2, turnCount: 1 })
})
