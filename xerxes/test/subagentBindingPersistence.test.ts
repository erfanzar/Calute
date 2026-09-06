// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'
import {
  replacePersistedSubagentSnapshots,
  SUBAGENT_SNAPSHOT_METADATA_KEY,
} from '../src/agents/subagentPersistence.js'
import { recoverSubagentSnapshots } from '../src/daemon/subagentCoordinator.js'
import { SubAgentTask } from '../src/agents/subagentManager.js'
import type { SpawnedAgentSnapshot } from '../src/operators/subagents.js'

const sourceId = 'parent-session'

function snapshot(modelCallBindings?: SpawnedAgentSnapshot['modelCallBindings']): SpawnedAgentSnapshot {
  return {
    agentId: 'coder',
    closed: true,
    createdAt: '2026-01-01T00:00:00.000Z',
    id: 'child-1',
    model: 'test-model',
    ...(modelCallBindings === undefined ? {} : { modelCallBindings }),
    name: 'child',
    promptProfile: 'coder',
    queueSize: 0,
    sourceAgentId: sourceId,
    status: 'completed',
    title: 'child',
    updatedAt: '2026-01-01T00:00:01.000Z',
  }
}

function snapshotWithWorkspace(workspace?: string): SpawnedAgentSnapshot {
  return {
    ...snapshot(),
    ...(workspace === undefined ? {} : { workspace }),
  }
}

function record(bindings: unknown): Readonly<Record<string, unknown>> {
  return {
    id: 'child-1',
    name: 'child',
    source_agent_id: sourceId,
    status: 'completed',
    created_at: '2026-01-01T00:00:00.000Z',
    updated_at: '2026-01-01T00:00:01.000Z',
    model_call_bindings: bindings,
  }
}

describe('subagent model-call binding persistence', () => {
  test('writes snake_case bindings and recovers goal ownership', () => {
    const metadata: Record<string, unknown> = {}
    replacePersistedSubagentSnapshots(metadata, [snapshot([{ kind: 'goal', sessionId: sourceId, goalId: 'goal-1' }])])
    const wires = metadata[SUBAGENT_SNAPSHOT_METADATA_KEY] as readonly Record<string, unknown>[]
    expect(wires).toHaveLength(1)
    const wire = wires[0]!
    expect(wire.model_call_bindings).toEqual([{ kind: 'goal', session_id: sourceId, goal_id: 'goal-1' }])
    expect(recoverSubagentSnapshots([], sourceId, [wire])[0]?.modelCallBindings).toEqual([
      { kind: 'goal', sessionId: sourceId, goalId: 'goal-1' },
    ])
  })

  test('persists and recovers the absolute source workspace', () => {
    const metadata: Record<string, unknown> = {}
    replacePersistedSubagentSnapshots(metadata, [snapshotWithWorkspace('/workspace/project')])
    const wires = metadata[SUBAGENT_SNAPSHOT_METADATA_KEY] as readonly Record<string, unknown>[]
    expect(wires[0]?.workspace).toBe('/workspace/project')
    expect(recoverSubagentSnapshots([], sourceId, wires)[0]?.workspace).toBe('/workspace/project')
  })

  test('persists and recovers a provider route fingerprint', () => {
    const route = 'a'.repeat(64)
    const metadata: Record<string, unknown> = {}
    replacePersistedSubagentSnapshots(metadata, [{ ...snapshot(), providerRoute: route }])
    const wires = metadata[SUBAGENT_SNAPSHOT_METADATA_KEY] as readonly Record<string, unknown>[]
    expect(wires[0]?.provider_route).toBe(route)
    expect(recoverSubagentSnapshots([], sourceId, wires)[0]?.providerRoute).toBe(route)
  })

  test('keeps malformed present provider routes as an explicit invalid sentinel', () => {
    const recovered = recoverSubagentSnapshots([], sourceId, [{
      ...record([]),
      provider_route: 'not-a-sha256-route',
    }])
    expect(recovered[0]?.providerRoute).toBe('')
  })

  test('retains provider selectors and an invalid route on the task record', () => {
    const task = new SubAgentTask({
      providerProfile: 'work-profile',
      reasoningEffort: 'high',
      providerRoute: '',
      id: 'child-1',
      name: 'child',
      prompt: 'work',
    })
    expect(task.providerProfile).toBe('work-profile')
    expect(task.reasoningEffort).toBe('high')
    expect(task.providerRoute).toBe('')
  })

  test('preserves prior workspace when the persisted field is absent', () => {
    const prior = snapshotWithWorkspace('/workspace/project')
    const priorWire: Record<string, unknown> = { ...prior, source_agent_id: sourceId }
    delete priorWire.source_agent_id
    const next = {
      id: prior.id,
      name: prior.name,
      source_agent_id: sourceId,
      status: prior.status,
      created_at: prior.createdAt,
      updated_at: '2026-01-01T00:00:02.000Z',
    }
    expect(recoverSubagentSnapshots([], sourceId, [priorWire, next])[0]?.workspace).toBe('/workspace/project')
  })

  test('keeps malformed present workspace as an explicit invalid sentinel', () => {
    const recovered = recoverSubagentSnapshots([], sourceId, [{
      ...record([]),
      workspace: 'relative/project',
    }])
    expect(recovered[0]?.workspace).toBe('')
  })

  test('retains explicit empty bindings and previous bindings when absent', () => {
    expect(recoverSubagentSnapshots([], sourceId, [record([])])[0]?.modelCallBindings).toEqual([])
    const prior = recoverSubagentSnapshots([], sourceId, [record([{ kind: 'goal', session_id: sourceId, goal_id: 'goal-1' }])])[0]
    const withoutField = { ...record(undefined) }
    delete withoutField.model_call_bindings
    expect(recoverSubagentSnapshots([], sourceId, [prior as unknown as Readonly<Record<string, unknown>>, withoutField])[0]?.modelCallBindings).toEqual([
      { kind: 'goal', sessionId: sourceId, goalId: 'goal-1' },
    ])
  })

  test('recovers camelCase bindings from transcript/tool wire observations', () => {
    const camelRecord: Record<string, unknown> = {
      ...record(undefined),
      modelCallBindings: [{ kind: 'goal', sessionId: sourceId, goalId: 'goal-2' }],
    }
    delete camelRecord.model_call_bindings
    const recovered = recoverSubagentSnapshots([], sourceId, [camelRecord])
    expect(recovered[0]?.modelCallBindings).toEqual([
      { kind: 'goal', sessionId: sourceId, goalId: 'goal-2' },
    ])
  })

  test('fail-closed recovery preserves an unrecoverable marker for malformed values', () => {
    const malformed = [
      null,
      [{ kind: 'future' }],
      [{ kind: 'goal', session_id: sourceId, goal_id: 'goal-1' }, { kind: 'unrecoverable' }],
      [{ kind: 'goal', session_id: '', goal_id: 'goal-1' }],
      Array.from({ length: 65 }, () => ({ kind: 'unrecoverable' })),
    ]
    for (const bindings of malformed) {
      expect(recoverSubagentSnapshots([], sourceId, [record(bindings)])[0]?.modelCallBindings).toEqual([
        { kind: 'unrecoverable' },
      ])
    }
  })
})
