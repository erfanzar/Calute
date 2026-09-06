// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { findGoalEvidenceExecution, successfulGoalEvidence } from '../src/runtime/goalEvidence.js'

test('ambiguous IDs and failed or unfinished executions cannot be cited as successful evidence', () => {
  const record = { toolCallId: 'call', name: 'ExecCommand', permitted: true, result: '{"exit_code":0}' }
  expect(findGoalEvidenceExecution([record, record], 'call')).toBeUndefined()
  expect(successfulGoalEvidence(record, 'another-call')).toBe(false)
  expect(successfulGoalEvidence(findGoalEvidenceExecution([record], 'call'), 'call')).toBe(true)
  for (const result of ['{"exit_code":2}', '{"ok":false}', '{"running":true}', '{"status":"cancelled"}', '{"isError":true}', '{"session_id":10,"exit_code":null}', 'Tool execution failed: disk unavailable']) {
    expect(successfulGoalEvidence({ ...record, result }, 'call')).toBe(false)
  }
  expect(successfulGoalEvidence({ ...record, permitted: false }, 'call')).toBe(false)
  expect(successfulGoalEvidence({ ...record, name: 'get_goal' }, 'call')).toBe(false)
})
