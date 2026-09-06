// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Checks execution outcome, not whether the model's relevance claim is true. */
export function findGoalEvidenceExecution(records: readonly unknown[], toolCallId: string): unknown {
  const matches = records.filter(record => {
    if (!record || typeof record !== 'object' || Array.isArray(record)) return false
    const row = record as Record<string, unknown>
    return (row.toolCallId ?? row.tool_call_id) === toolCallId
  })
  return matches.length === 1 ? matches[0] : undefined
}

export function successfulGoalEvidence(record: unknown, toolCallId: string): boolean {
  if (!record || typeof record !== 'object' || Array.isArray(record)) return false
  const row = record as Record<string, unknown>
  if ((row.toolCallId ?? row.tool_call_id) !== toolCallId || row.permitted !== true || typeof row.result !== 'string') return false
  if (['get_goal', 'create_goal', 'update_goal'].includes(String(row.name))) return false
  if (/^(Tool execution failed:|Cancelled before execution|Denied by permission)/i.test(row.result.trim())) return false
  let output: unknown
  try { output = JSON.parse(row.result) } catch { return true }
  if (!output || typeof output !== 'object' || Array.isArray(output)) return true
  const result = output as Record<string, unknown>
  if (result.ok === false || result.success === false || result.passed === false || result.isError === true || result.is_error === true
    || result.timedOut === true || result.timed_out === true || result.running === true || result.error) return false
  if (['running', 'pending', 'queued', 'cancelled', 'canceled', 'failed', 'error', 'denied', 'timeout'].includes(String(result.status).toLowerCase())) return false
  const exitCode = result.exit_code ?? result.exitCode
  if (typeof exitCode === 'number') return exitCode === 0
  if (result.session_id !== undefined || result.sessionId !== undefined) return false
  return true
}
