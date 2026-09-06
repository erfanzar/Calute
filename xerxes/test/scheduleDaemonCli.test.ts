// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, readFile, readdir, rm, realpath } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import process from 'node:process'

import { InMemoryDaemonRuntime, type DaemonEvent, type TurnRunner } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { JobStore } from '../src/cron/jobs.js'

const rootDir = resolve(import.meta.dir, '..')

function spawnCli(args: readonly string[]): {
  readonly process: ReturnType<typeof Bun.spawn>
  readonly stdout: Promise<string>
  readonly stderr: Promise<string>
} {
  const child = Bun.spawn({
    cmd: [process.execPath, 'src/cli.ts', ...args],
    cwd: rootDir,
    stdout: 'pipe',
    stderr: 'pipe',
  })
  return {
    process: child,
    stdout: new Response(child.stdout).text(),
    stderr: new Response(child.stderr).text(),
  }
}

async function waitFor<T>(read: () => T | Promise<T>, predicate: (value: T) => boolean, timeoutMs = 4_500): Promise<T> {
  const deadline = Date.now() + timeoutMs
  let value = await read()
  while (!predicate(value)) {
    if (Date.now() >= deadline) throw new Error(`condition was not met before ${timeoutMs}ms: ${JSON.stringify(value)}`)
    await Bun.sleep(50)
    value = await read()
  }
  return value
}

function jobIdFromStore(store: JobStore): string {
  const jobs = store.listJobs()
  if (jobs.length !== 1) throw new Error(`expected one scheduled job, found ${jobs.length}`)
  return jobs[0]!.id
}

async function invokeCli(args: readonly string[]): Promise<{ readonly exit: number; readonly stdout: string; readonly stderr: string }> {
  const child = spawnCli(args)
  return { exit: await child.process.exited, stdout: await child.stdout, stderr: await child.stderr }
}

test('schedule CLI uses the daemon CronScheduler for automatic runs and durable pause/resume', async () => {
  const directory = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-schedule-daemon-cli-')))
  const project = join(directory, 'project')
  const otherProject = join(directory, 'other-project')
  await Bun.write(join(project, 'marker.txt'), 'project')
  await Bun.write(join(otherProject, 'marker.txt'), 'other')
  const socketPath = join(directory, 'daemon.sock')
  const jobsPath = join(directory, 'jobs.json')
  const leasePath = join(directory, 'scheduler.lease')
  const archiveDirectory = join(directory, 'archive')
  const calls: string[] = []
  const runner: TurnRunner = {
    async *run(session, text): AsyncGenerator<DaemonEvent> {
      calls.push(`${session.cwd}:${text}`)
      yield { type: 'text_part', payload: { text: 'scheduled fixture result' } }
    },
  }
  const runtime = new InMemoryDaemonRuntime(runner, {
    currentProjectDirectory: project,
    sessionDirectory: join(directory, 'sessions'),
  })
  const store = new JobStore(jobsPath, { projectRoot: project })
  const server = new DaemonServer({
    socketPath,
    projectDirectory: project,
    runtime,
    autoTitle: false,
    cronStoreFactory: () => store,
    cronLeasePath: leasePath,
    cronArchiveDirectory: archiveDirectory,
    cronPollInterval: 25,
  })
  await server.start()
  try {
    const create = await invokeCli([
      'schedule', 'create',
      '--schedule', 'interval:1',
      '--objective', 'fixture automatic run',
      '--project-dir', project,
      '--socket', socketPath,
      '--timezone', 'UTC',
    ])
    expect(create.exit).toBe(0)
    expect(create.stderr).toBe('')

    const persisted = new JobStore(jobsPath, { projectRoot: project })
    const id = jobIdFromStore(persisted)
    const created = persisted.get(id)
    expect(created).toMatchObject({ projectRoot: project, timezone: 'UTC', paused: false })

    await waitFor(() => store.get(id)?.lastRunAt, value => Boolean(value))
    expect(calls).toHaveLength(1)
    expect(calls[0]).toContain('fixture automatic run')
    const archived = await readdir(join(archiveDirectory, id))
    expect(archived.some(name => name.endsWith('.md'))).toBe(true)
    const archive = archived.find(name => name.endsWith('.md'))
    expect(archive).toBeDefined()
    expect(await readFile(join(archiveDirectory, id, archive!), 'utf8')).toContain('scheduled fixture result')

    const pause = await invokeCli(['schedule', 'disable', '--id', id, '--project-dir', project, '--socket', socketPath])
    expect(pause.exit).toBe(0)
    expect(store.get(id)?.paused).toBe(true)
    const callsBeforePause = calls.length
    await Bun.sleep(1_200)
    expect(calls).toHaveLength(callsBeforePause)

    const manual = await invokeCli(['schedule', 'fire', '--id', id, '--project-dir', project, '--socket', socketPath])
    expect(manual.exit).toBe(0)
    await waitFor(() => store.get(id)?.runsStarted ?? 0, runs => runs >= 2)
    expect(calls).toHaveLength(callsBeforePause + 1)

    await server.stop()
    const restarted = new DaemonServer({
      socketPath,
      projectDirectory: project,
      runtime,
      autoTitle: false,
      cronStoreFactory: () => new JobStore(jobsPath, { projectRoot: project }),
      cronLeasePath: leasePath,
      cronArchiveDirectory: archiveDirectory,
      cronPollInterval: 25,
    })
    await restarted.start()
    try {
      expect(new JobStore(jobsPath, { projectRoot: project }).get(id)?.paused).toBe(true)
      const resume = await invokeCli(['schedule', 'enable', '--id', id, '--project-dir', project, '--socket', socketPath])
      expect(resume.exit).toBe(0)
      await waitFor(() => new JobStore(jobsPath, { projectRoot: project }).get(id)?.runsStarted ?? 0, runs => runs >= 3)

      const foreignRemove = await invokeCli(['schedule', 'remove', '--id', id, '--project-dir', otherProject, '--socket', socketPath])
      expect(foreignRemove.exit).not.toBe(0)
      expect(`${foreignRemove.stdout}\n${foreignRemove.stderr}`).toMatch(/workspace|project|not found/i)
      expect(new JobStore(jobsPath, { projectRoot: project }).get(id)).toBeDefined()
    } finally {
      await restarted.stop()
    }
  } finally {
    await server.stop()
    await rm(directory, { recursive: true, force: true })
  }
}, 20_000)
