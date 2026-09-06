// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { realpath } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { Scheduler, type ScheduledTrigger } from '../runtime/scheduler.js'
import { CronJob, JobStore, nextFireAt } from './jobs.js'

function converted(trigger: ScheduledTrigger, id: string, projectRoot: string): CronJob {
  const timing = trigger.schedule
  if (trigger.payload.dependencies.length) throw new Error('Dependent trigger tasks require explicit workflow migration')
  if (timing.kind !== 'interval' && timing.kind !== 'cron') throw new Error('Event/webhook source migration is not available')
  let schedule = ''
  if (timing.kind === 'cron') {
    // Legacy evaluates restricted DOM and DOW with AND, unlike POSIX OR.
    if (timing.day && timing.day !== '*' && timing.dayOfWeek && timing.dayOfWeek !== '*') throw new Error('Combined day constraints require explicit cadence conversion')
    schedule = `${timing.minute ?? '*'} ${timing.hour ?? '*'} ${timing.day ?? '*'} * ${timing.dayOfWeek ?? '*'}`
    nextFireAt(schedule)
  }
  return new CronJob({ id, projectRoot, prompt: trigger.payload.objective, schedule, paused: true,
    ...(timing.kind === 'interval' ? { intervalSeconds: timing.intervalSeconds } : {}),
    metadata: { migrated_trigger_id: trigger.id, migrated_owner: trigger.owner } })
}

export async function previewScheduleMigration(source: Scheduler, projectRoot: string): Promise<{ id: string; owner: string; objective: string; enabled: boolean; destination: string | null; supported: boolean; reason: string | null }[]> {
  return [...(await source.load()).triggers.values()].map(trigger => {
    let reason: string | null = null
    try { converted(trigger, 'preview', projectRoot) } catch (error) { reason = error instanceof Error ? error.message : String(error) }
    return { id: trigger.id, owner: trigger.owner, objective: trigger.payload.objective, enabled: trigger.enabled, destination: trigger.migratedTo ?? null, supported: reason === null, reason }
  })
}
/** Migrate one selected time trigger. Imported jobs require explicit resume. */
export async function migrateScheduledTrigger(source: Scheduler, target: JobStore, triggerId: string, projectRoot: string): Promise<string> {
  const trigger = (await source.load()).triggers.get(triggerId)
  if (!trigger) throw new Error('Unknown trigger')
  const sourcePath = join(await realpath(dirname(source.logPath)), 'scheduler.jsonl')
  const id = `legacy-${Bun.hash(`${sourcePath}:${triggerId}`).toString(16)}`
  const project = await realpath(projectRoot)
  const destination = `${await realpath(target.path)}#${id}#${project}`
  // Reject unsupported conversion before fencing the source.
  converted(trigger, id, project)
  await source.migrateTrigger(triggerId, destination, async current => {
    const job = converted(current, id, project)
    const existing = target.get(id)
    if (existing) {
      if (existing.metadata.migration_destination !== destination) throw new Error('Migration destination conflicts with an existing job')
      return
    }
    job.metadata.migration_destination = destination
    target.add(job)
  })
  return id
}
