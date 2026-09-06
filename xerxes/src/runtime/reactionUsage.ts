// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { ReactionUsage } from './reactionMailbox.js'

/** Preserve measured usage even when execution rejects after spending tokens. */
export class ReactionExecutionError extends Error {
  constructor(cause: unknown, readonly usage: ReactionUsage) {
    super(cause instanceof Error ? cause.message : String(cause), { cause })
    this.name = 'ReactionExecutionError'
  }
}

/** Counts only children started during this reaction, deduplicating cumulative events. */
export class ReactionChildUsage {
  private readonly children = new Map<string, { input: number; output: number }>()
  private uncertain = false

  observe(payload: Record<string, unknown>): void {
    if (typeof payload.agent_id !== 'string') { this.uncertain = true; return }
    const event = payload.event as { type?: unknown } | undefined
    const key = `${payload.agent_id}:${String(payload.task_index ?? '')}`
    if (event?.type === 'turn_begin' && !this.children.has(key)) {
      if (this.children.size >= 4096) { this.uncertain = true; return }
      this.children.set(key, { input: 0, output: 0 })
    }
    const child = this.children.get(key)
    // Old background agents can share the session event stream. They are not
    // attributable to this reaction merely because they emit while it runs.
    if (!child) return
    for (const [field, wire] of [['input', 'input_tokens'], ['output', 'output_tokens']] as const) {
      const value = payload[wire]
      if (value === undefined) continue
      if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) { this.uncertain = true; continue }
      child[field] = Math.max(child[field], value)
    }
  }

  addTo(parent: ReactionUsage): ReactionUsage {
    let inputTokens = parent.inputTokens
    let outputTokens = parent.outputTokens
    for (const child of this.children.values()) {
      inputTokens = Math.min(Number.MAX_SAFE_INTEGER, inputTokens + child.input)
      outputTokens = Math.min(Number.MAX_SAFE_INTEGER, outputTokens + child.output)
    }
    // Current subagent events do not carry a trustworthy usage-complete bit.
    // Never label partial child counters as a complete billable total.
    return { inputTokens, outputTokens, complete: parent.complete && !this.uncertain && this.children.size === 0 }
  }
}
