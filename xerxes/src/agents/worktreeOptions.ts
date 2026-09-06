// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { ValidationError } from '../core/errors.js'

export class WorktreeSetupError extends Error {
  constructor(message: string, readonly worktree: { path: string; branch: string }, cause: unknown) {
    super(message, { cause })
    this.name = 'WorktreeSetupError'
  }
}

/** Validate an explicit Git revision before it reaches a subprocess or snapshot. */
export function parseWorktreeRef(value: unknown): string | undefined {
  if (value === undefined) return undefined
  if (typeof value !== 'string' || !value.trim() || value.length > 1024 || /[\x00-\x1f\x7f]/.test(value) || value.trim().startsWith('-')) {
    throw new ValidationError('worktree_ref', 'must be a nonempty Git revision of at most 1024 characters, without control characters or a leading dash', value)
  }
  return value.trim()
}

export function parseWorktreeSource(value: unknown): 'working-tree' | undefined {
  if (value === undefined) return undefined
  if (value === 'working-tree') return value
  throw new ValidationError('worktree_source', 'must be working-tree when specified', value)
}
