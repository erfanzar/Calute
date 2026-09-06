// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { GatewayRpc } from '../app/interfaces.js'

export interface WorkspaceRecord { id: string; taskId: string; path: string; branch: string; error?: string }
export interface WorkspaceReview extends WorkspaceRecord { setup?: string; reviewId?: string; base: string; head: string; snapshotTree?: string; status: string; diff: string }
const object = (value: unknown): Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
function record(value: unknown): WorkspaceRecord {
  const row = object(value)
  if (typeof row.id !== 'string' || typeof row.path !== 'string' || typeof row.branch !== 'string') throw new Error('Invalid workspace record')
  return { id: row.id, path: row.path, branch: row.branch, taskId: typeof row.taskId === 'string' ? row.taskId : 'Unavailable record', ...(typeof row.error === 'string' ? { error: row.error } : {}) }
}
export async function listWorkspaces(rpc: GatewayRpc, after?: string) {
  const result = object(await rpc('workspace.list', after ? { after } : {}))
  const inventory = object(result.inventory)
  if (result.ok !== true || !Array.isArray(inventory.records)) throw new Error(typeof result.error === 'string' ? result.error : 'Workspace inventory unavailable')
  return { records: inventory.records.map(record), next: typeof inventory.next === 'string' ? inventory.next : undefined }
}
export async function inspectWorkspace(rpc: GatewayRpc, id: string): Promise<WorkspaceReview> {
  const result = object(await rpc('workspace.inspect', { workspace_id: id }))
  if (result.ok !== true) throw new Error(typeof result.error === 'string' ? result.error : 'Workspace review unavailable')
  const detail = object(result.review), row = record(detail)
  if (row.id !== id || typeof detail.base !== 'string' || typeof detail.head !== 'string' || typeof detail.diff !== 'string' || typeof detail.status !== 'string') throw new Error('Invalid workspace review')
  return { ...row, ...(typeof detail.setup === 'string' ? { setup: detail.setup } : {}), ...(typeof detail.reviewId === 'string' ? { reviewId: detail.reviewId } : {}), base: detail.base, head: detail.head, diff: detail.diff, status: detail.status, ...(typeof detail.snapshotTree === 'string' ? { snapshotTree: detail.snapshotTree } : {}) }
}

export interface WorkspaceApplyCheck { message: string; destination: string; destinationState?: string }
export async function checkWorkspaceApply(rpc: GatewayRpc, review: WorkspaceReview): Promise<WorkspaceApplyCheck> {
  if (!review.reviewId) throw new Error('This daemon cannot bind integration checks to a review; update it first')
  const response = object(await rpc('workspace.checkApply', { workspace_id: review.id, review_id: review.reviewId }))
  const check = object(response.check)
  if (response.ok !== true || check.reviewId !== review.reviewId || typeof check.canApply !== 'boolean' || typeof check.destination !== 'string') throw new Error(typeof response.error === 'string' ? response.error : 'Invalid integration check')
  return { destination: check.destination, ...(check.canApply && typeof check.destinationState === 'string' && /^[a-f0-9]{64}$/.test(check.destinationState) ? { destinationState: check.destinationState } : {}), message: check.canApply ? `Patch applies cleanly to ${check.destination} at check time. No files changed.` : `Cannot apply: ${typeof check.error === 'string' ? check.error : 'integration check failed'}` }
}

export async function applyWorkspaceReview(rpc: GatewayRpc, review: WorkspaceReview, check: WorkspaceApplyCheck): Promise<string> {
  if (!review.reviewId || !check.destinationState) throw new Error('Check the current review before applying')
  const response = object(await rpc('workspace.apply', { workspace_id: review.id, review_id: review.reviewId, destination_state: check.destinationState, confirm: true }))
  const result = object(response.integration)
  if (response.ok !== true || result.status !== 'applied' || result.destination !== check.destination || typeof result.backupPath !== 'string') throw new Error(typeof response.error === 'string' ? response.error : 'Apply result unavailable; inspect the destination before retrying')
  return `Applied to ${result.destination}. Backup: ${result.backupPath}`
}

export interface WorkspaceIntegration { id: string; backupPath: string; destination: string; status: string; paths: string[]; error: string }
export async function inspectWorkspaceIntegration(rpc: GatewayRpc, id: string) {
  const response = object(await rpc('workspace.integration.inspect', { integration_id: id }))
  const inspection = object(response.inspection)
  if (response.ok !== true || inspection.id !== id || !Array.isArray(inspection.files)) throw new Error(typeof response.error === 'string' ? response.error : 'Recovery inspection unavailable')
  return inspection.files.map(value => {
    const file = object(value)
    if (typeof file.path !== 'string' || typeof file.reason !== 'string' || !['preserve', 'restore', 'unchanged', 'conflict'].includes(String(file.action))) throw new Error('Invalid recovery file state')
    return { path: file.path, action: String(file.action), reason: file.reason }
  })
}
export async function listWorkspaceIntegrations(rpc: GatewayRpc, after?: string) {
  const response = object(await rpc('workspace.integrations', after ? { after } : {}))
  const inventory = object(response.inventory)
  if (response.ok !== true || !Array.isArray(inventory.records)) throw new Error(typeof response.error === 'string' ? response.error : 'Integration inventory unavailable')
  const records = inventory.records.map(value => {
    const row = object(value)
    if (typeof row.id !== 'string' || typeof row.backupPath !== 'string') throw new Error('Invalid integration record')
    return { id: row.id, backupPath: row.backupPath, destination: typeof row.destination === 'string' ? row.destination : '', status: typeof row.status === 'string' ? row.status : 'unavailable', error: typeof row.error === 'string' ? row.error : '', paths: Array.isArray(row.paths) ? row.paths.filter((path): path is string => typeof path === 'string') : [] }
  })
  return { records, next: typeof inventory.next === 'string' ? inventory.next : undefined }
}
export async function recoverWorkspaceIntegration(rpc: GatewayRpc, id: string) {
  const response = object(await rpc('workspace.recover', { integration_id: id, confirm: true }))
  const recovery = object(response.recovery)
  if (response.ok !== true || recovery.id !== id || !['abandoned', 'applied', 'rolled-back', 'needs-recovery'].includes(String(recovery.status)) || !Array.isArray(recovery.conflicts)) throw new Error(typeof response.error === 'string' ? response.error : 'Recovery result unavailable; inspect before retrying')
  if (recovery.status === 'abandoned') return 'Incomplete preparation abandoned. No destination files changed.'
  if (recovery.status === 'applied') return 'Completed apply preserved; leftover lock released.'
  return recovery.status === 'rolled-back' ? 'Original files restored.' : `Concurrent edits preserved. Still needs recovery: ${recovery.conflicts.join(', ')}`
}
