// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { ToolRegistry } from '../executors/toolRegistry.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'

export type ScheduleToolHost = (sessionId: string, action: string, params: JsonObject, signal?: AbortSignal) => Promise<unknown>

/** The daemon owns persistence, scope, validation and execution; tools are an input adapter. */
export function registerScheduleTools(registry: ToolRegistry, host: ScheduleToolHost): void {
  const definitions: ToolDefinition[] = [
    { type: 'function', function: { name: 'list_schedules', description: 'List durable schedules in this session’s workspace, including revision, next run and execution state. Jobs execute only while their owning daemon is running. Use /schedules for interactive management.', parameters: { type: 'object', additionalProperties: false, properties: {} } } },
    { type: 'function', function: { name: 'manage_schedule', description: 'Create, inspect, edit, pause, resume, cancel or run a workspace schedule. Create/edit requires prompt, paused and exactly one timing field. Edit requires the latest revision from list/inspect and an idle job. Cron uses the explicit timezone (UTC by default); missed occurrences run once by default, or skip after the configured lateness allowance. Pause stops future occurrences; cancel requests cancellation of current work. Run executes once under shared capacity and lease limits. Only create or activate recurring work when directly requested by the user. New jobs should be paused for review unless activation was requested.', parameters: { type: 'object', additionalProperties: false, properties: {
      action: { type: 'string', enum: ['inspect', 'create', 'update', 'pause', 'resume', 'cancel', 'run', 'complete'] },
      stop_condition: { type: ['string', 'null'], maxLength: 4000, description: 'Condition to check on each session follow-up. Null removes it; omission preserves it on edit. Complete only with current evidence.' },
      evidence: { type: 'string', maxLength: 8000, description: 'Required for complete: what this attempt checked and why the configured condition is met. Only the executing follow-up may complete itself. This records a model report and stops future wakes.' },
      schedule_id: { type: 'string' }, revision: { type: 'string' },
      prompt: { type: 'string', minLength: 1, maxLength: 32000 }, paused: { type: 'boolean' },
      schedule: { type: 'string', description: 'Five-field cron expression.' },
      timezone: { type: 'string', description: 'IANA timezone for recurring cron, default UTC. Spring gaps are skipped; fall repetitions both run.' },
      at: { type: 'string', description: 'Future one-shot ISO timestamp including timezone.' },
      interval_seconds: { type: 'integer', minimum: 1, maximum: 86400 },
      timeout_seconds: { type: 'integer', minimum: 1, maximum: 3600 },
      max_retries: { type: 'integer', minimum: 0, maximum: 10 },
      target: { type: 'string', enum: ['session', 'independent'], description: 'session binds a bounded follow-up to this conversation; requires max_runs and expires_at. Omission preserves the existing target on edit; new jobs default to independent.' },
      expires_at: { type: ['string', 'null'], description: 'ISO timestamp with timezone after which no new attempt may start. Null removes expiry. Already running attempts retain their execution timeout.' },
      max_runs: { type: ['integer', 'null'], minimum: 1, maximum: 10000, description: 'Lifetime execution attempts, including manual runs, retries, failures and cancellation. Null removes the limit. Editing does not reset attempts already used.' },
      max_model_calls: { type: ['integer', 'null'], minimum: 1, maximum: 10000, description: 'Logical model calls per execution attempt, shared by the parent, descendants and auxiliary completions. Null removes the limit. This is not a token or network-request cap.' },
      max_total_tokens: { type: ['integer', 'null'], minimum: 1, maximum: Number.MAX_SAFE_INTEGER, description: 'Lifetime measured-token admission threshold across attempts, retries, descendants and auxiliary calls, including cached input. Blocks new calls when reached or historical usage is incomplete. Calls already in flight may overshoot; this is not a hard billing cap. Null removes the limit without clearing usage.' },
      deliver: { type: 'string', description: 'Configured channel name, or none for archive only. Only set external delivery when the user explicitly requests that destination.' },
      recipient: { type: 'string', maxLength: 512, description: 'Channel recipient or room ID. Required for external delivery; empty for archive only.' },
      missed_run_policy: { type: 'string', enum: ['coalesce', 'skip'], description: 'Run once after missed occurrences (default), or skip occurrences beyond the lateness allowance. Missed one-shots pause for review. Overlapping executions are always forbidden.' },
      misfire_grace_seconds: { type: 'integer', minimum: 1, maximum: 86400, description: 'Lateness allowance for skip policy; default 300 seconds.' },
    }, required: ['action'] } } },
  ]
  for (const definition of definitions) registry.register(definition, async (args, context, signal) => {
    if (!context.sessionId?.trim()) throw new Error('Schedule tools require an authenticated session context')
    signal?.throwIfAborted()
    const action = definition.function.name === 'list_schedules' ? 'list' : args.action
    if (typeof action !== 'string') throw new Error('Schedule action is required')
    if (['create', 'update', 'resume', 'run'].includes(action) && context.metadata.goal_turn_human !== true) {
      throw new Error('Creating or activating scheduled work requires a direct user turn')
    }
    const result = await host(context.sessionId, action, args, signal)
    return JSON.stringify(result)
  }, 'default', { concurrencySafe: false, destructive: false, openWorld: true,
    readOnly: definition.function.name === 'list_schedules', maxResultBytes: 64_000 })
}
