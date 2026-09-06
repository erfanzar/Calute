// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SpawnedAgentSnapshot } from '../operators/subagents.js'
import type { SubagentRetryRequest } from './runtime.js'

export interface OwnedSubagentRetryRequest {
  readonly options: {
    readonly message?: string
    readonly sourceAgentId: string
  }
  readonly task: string
}

/** Bind a wire retry request to its live session owner before invoking a host. */
export function resolveSubagentRetryRequest(
  request: SubagentRetryRequest,
  readSession: (sessionKey: string) => { readonly id: string } | undefined,
): OwnedSubagentRetryRequest {
  const sessionKey = request.sessionKey?.trim()
  if (!sessionKey) throw new Error('subagent retry requires an owning session')
  const session = readSession(sessionKey)
  if (!session?.id?.trim()) throw new Error('subagent retry session no longer exists')
  const task = request.task.trim()
  if (!task) throw new Error('subagent retry requires a task id or stable name')
  return {
    options: {
      sourceAgentId: session.id,
      ...(request.message === undefined ? {} : { message: request.message }),
    },
    task,
  }
}

/** Resolve a retry target only when it belongs to the requesting session. */
export function resolveOwnedSubagentRetry(
  snapshots: readonly Pick<SpawnedAgentSnapshot, 'id' | 'name' | 'sourceAgentId'>[],
  task: string | undefined,
  sourceAgentId: string | undefined,
): Pick<SpawnedAgentSnapshot, 'id' | 'name' | 'sourceAgentId'> {
  const owner = sourceAgentId?.trim()
  if (!owner) throw new Error('subagent retry requires an owning session')
  const target = task?.trim()
  if (!target) throw new Error('subagent retry requires a task id or stable name')

  const exactIds = snapshots.filter(snapshot => snapshot.id === target)
  if (exactIds.length) {
    if (exactIds.some(snapshot => snapshot.sourceAgentId !== owner)) {
      throw new Error('subagent retry target belongs to another session')
    }
    if (exactIds.length > 1) throw new Error('subagent retry task id is ambiguous; use a unique task id')
    return exactIds[0]!
  }

  const sameOwnerNames = snapshots.filter(snapshot => (
    snapshot.sourceAgentId === owner && snapshot.name === target
  ))
  if (sameOwnerNames.length === 1) return sameOwnerNames[0]!
  if (sameOwnerNames.length > 1) throw new Error('subagent retry name is ambiguous; use the task id')
  throw new Error('subagent retry task was not found in the owning session')
}
