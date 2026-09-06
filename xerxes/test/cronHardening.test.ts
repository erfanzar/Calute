// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import {
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
} from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  archiveOutput,
  CronJob,
  CronScheduler,
  JobStore,
  nextFireAt,
} from '../src/cron/index.js'

function temporaryDirectory(): string {
  return mkdtempSync(join(tmpdir(), 'xerxes-cron-hardening-'))
}

function removeDirectory(path: string): void {
  rmSync(path, { recursive: true, force: true })
}

test('scheduler preserves the local timezone across repeated fall-back executions', async () => {
  const directory = temporaryDirectory()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(new CronJob({ id: 'fold', prompt: 'work', schedule: '30 1 * * *', timezone: 'America/New_York', nextRunAt: '2026-11-01T05:30:00Z' }))
    const scheduler = new CronScheduler(store, () => 'done')
    expect(await scheduler.tick(new Date('2026-11-01T05:30:00Z'))).toEqual(['fold'])
    expect(new JobStore(store.path).get('fold')?.nextRunAt).toBe('2026-11-01T06:30:00.000Z')
    expect(await scheduler.tick(new Date('2026-11-01T06:30:00Z'))).toEqual(['fold'])
    expect(store.get('fold')?.nextRunAt).toBe('2026-11-02T06:30:00.000Z')
  } finally { removeDirectory(directory) }
})

function silenceConsole(): { errors: unknown[][]; restore: () => void } {
  const errors: unknown[][] = []
  const previousError = console.error
  const previousWarn = console.warn
  console.error = (...args: unknown[]) => {
    errors.push(args)
  }
  console.warn = () => {}
  return {
    errors,
    restore: () => {
      console.error = previousError
      console.warn = previousWarn
    },
  }
}

test('nextFireAt finds Feb-29 schedules across non-leap years', () => {
  // 2026 and 2027 are not leap years; the next valid fire is 2028-02-29.
  expect(
    nextFireAt('0 9 29 2 *', new Date('2026-06-01T00:00:00.000Z')).toISOString(),
  ).toBe('2028-02-29T09:00:00.000Z')
  expect(
    nextFireAt('0 9 29 2 *', new Date('2027-03-01T00:00:00.000Z')).toISOString(),
  ).toBe('2028-02-29T09:00:00.000Z')
  // Inside a leap year the upcoming Feb-29 is used when it is still ahead.
  expect(
    nextFireAt('0 9 29 2 *', new Date('2028-02-01T00:00:00.000Z')).toISOString(),
  ).toBe('2028-02-29T09:00:00.000Z')
})

test('nextFireAt ORs restricted day-of-month and day-of-week fields', () => {
  // POSIX semantics: with both fields restricted, either match fires.
  // From Friday 2026-05-15, "0 9 1 * 1" fires Monday 2026-05-18, not only
  // when the 1st is a Monday (2026-06-01).
  expect(
    nextFireAt('0 9 1 * 1', new Date('2026-05-15T12:00:00.000Z')).toISOString(),
  ).toBe('2026-05-18T09:00:00.000Z')
  // With only one day field restricted, that field alone gates the match.
  expect(
    nextFireAt('0 9 * * 1', new Date('2026-05-15T12:00:00.000Z')).toISOString(),
  ).toBe('2026-05-18T09:00:00.000Z')
  expect(
    nextFireAt('0 9 1 * *', new Date('2026-05-15T12:00:00.000Z')).toISOString(),
  ).toBe('2026-06-01T09:00:00.000Z')
})

test('nextFireAt accepts 7 as a Sunday day-of-week alias', () => {
  // 2026-05-16 is a Saturday; the next Sunday is 2026-05-17.
  const fromSeven = nextFireAt('0 9 * * 7', new Date('2026-05-16T12:00:00.000Z'))
  expect(fromSeven.toISOString()).toBe('2026-05-17T09:00:00.000Z')
  expect(fromSeven.getUTCDay()).toBe(0)
  // Ranges including 7 map onto Sunday too.
  expect(
    nextFireAt('0 9 * * 5-7', new Date('2026-05-16T12:00:00.000Z')).toISOString(),
  ).toBe('2026-05-17T09:00:00.000Z')
})

test('a failed one-shot is retained, retried, then paused after bounded retries', async () => {
  const directory = temporaryDirectory()
  const { restore } = silenceConsole()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(
      new CronJob({
        id: 'flaky-once',
        prompt: 'p',
        oneshot: true,
        nextRunAt: '2026-05-15T12:00:00.000Z',
      }),
    )
    let failures = 0
    const scheduler = new CronScheduler(
      store,
      () => {
        failures += 1
        throw new Error('boom')
      },
      { maxOneShotRetries: 2, oneShotRetryBaseMs: 1_000 },
    )
    const now = new Date('2026-05-15T12:00:00.000Z')

    // First failure: retained with retry_count 1 and a 1s backoff.
    expect(await scheduler.tick(now)).toEqual([])
    let job = store.get('flaky-once')
    expect(job).toBeDefined()
    expect(job?.paused).toBe(false)
    expect(job?.nextRunAt).toBe('2026-05-15T12:00:01.000Z')
    expect(job?.metadata).toMatchObject({
      last_error: 'boom',
      retry_count: 1,
    })

    // Second failure (retry 2): backoff doubles to 2s.
    expect(
      await scheduler.tick(new Date('2026-05-15T12:00:01.500Z')),
    ).toEqual([])
    job = store.get('flaky-once')
    expect(job?.nextRunAt).toBe('2026-05-15T12:00:03.000Z')
    expect(job?.metadata.retry_count).toBe(2)

    // Third failure exceeds the retry budget: the job is paused, not deleted.
    expect(
      await scheduler.tick(new Date('2026-05-15T12:00:04.000Z')),
    ).toEqual([])
    job = store.get('flaky-once')
    expect(job).toBeDefined()
    expect(job?.paused).toBe(true)
    expect(failures).toBe(3)
  } finally {
    restore()
    removeDirectory(directory)
  }
})

test('a retried one-shot that eventually succeeds is removed', async () => {
  const directory = temporaryDirectory()
  const { restore } = silenceConsole()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(
      new CronJob({
        id: 'recovers',
        prompt: 'p',
        oneshot: true,
        nextRunAt: '2026-05-15T12:00:00.000Z',
      }),
    )
    let attempts = 0
    const scheduler = new CronScheduler(
      store,
      () => {
        attempts += 1
        if (attempts === 1) throw new Error('transient')
        return 'done'
      },
      { oneShotRetryBaseMs: 1_000 },
    )

    expect(await scheduler.tick(new Date('2026-05-15T12:00:00.000Z'))).toEqual([])
    expect(store.get('recovers')).toBeDefined()
    expect(
      await scheduler.tick(new Date('2026-05-15T12:00:01.500Z')),
    ).toEqual(['recovers'])
    expect(store.get('recovers')).toBeUndefined()
  } finally {
    restore()
    removeDirectory(directory)
  }
})

test('a recurring job without a schedule is paused instead of hot-looping', async () => {
  const directory = temporaryDirectory()
  const { restore } = silenceConsole()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(
      new CronJob({
        id: 'scheduleless',
        prompt: 'p',
        schedule: '',
        nextRunAt: '2026-05-15T12:00:00.000Z',
      }),
    )
    const runs: string[] = []
    const scheduler = new CronScheduler(store, (job) => {
      runs.push(job.id)
      return ''
    })

    expect(await scheduler.tick(new Date('2026-05-15T12:00:00.000Z'))).toEqual([])
    const job = store.get('scheduleless')
    expect(job?.paused).toBe(true)
    expect(job?.nextRunAt).toBeUndefined()
    // Later polls stay inert: no run, and no repeated rescheduling work.
    expect(await scheduler.tick(new Date('2026-05-15T12:01:00.000Z'))).toEqual([])
    expect(runs).toEqual([])
  } finally {
    restore()
    removeDirectory(directory)
  }
})

test('a hung job times out without starving other due jobs or future ticks', async () => {
  const directory = temporaryDirectory()
  const { errors, restore } = silenceConsole()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(
      new CronJob({
        id: 'hung',
        prompt: 'p',
        schedule: '* * * * *',
        nextRunAt: '2026-05-15T12:00:00.000Z',
      }),
    )
    store.add(
      new CronJob({
        id: 'quick',
        prompt: 'p',
        schedule: '* * * * *',
        nextRunAt: '2026-05-15T12:00:00.000Z',
      }),
    )
    const scheduler = new CronScheduler(
      store,
      (job) =>
        job.id === 'hung'
          ? new Promise<string>(() => {})
          : `ran:${job.id}`,
      { jobTimeout: 25 },
    )
    const now = new Date('2026-05-15T12:00:00.000Z')

    const ran = await scheduler.tick(now)
    expect(ran).toEqual(['quick'])
    // The hung job was reported and rescheduled instead of wedging the queue.
    expect(errors.some((entry) => entry[0] === 'CronScheduler job hung failed')).toBe(true)
    expect(store.get('hung')?.nextRunAt).toBe('2026-05-15T12:01:00.000Z')
    expect(store.get('hung')?.metadata.last_error).toContain('timed out')
    // A subsequent tick is not blocked by the previous hung run.
    expect(
      await scheduler.tick(new Date('2026-05-15T12:01:00.000Z')),
    ).toEqual(['quick'])
  } finally {
    restore()
    removeDirectory(directory)
  }
})

test('archiveOutput never overwrites same-second runs and prunes to the retention cap', async () => {
  const directory = temporaryDirectory()
  try {
    const instant = new Date('2026-05-15T12:00:00.000Z')
    const first = await archiveOutput(directory, 'job1', 'first', instant)
    const second = await archiveOutput(directory, 'job1', 'second', instant)
    expect(first).not.toBe(second)
    expect(readFileSync(first, 'utf8')).toBe('first')
    expect(readFileSync(second, 'utf8')).toBe('second')

    for (let index = 0; index < 12; index += 1) {
      await archiveOutput(
        directory,
        'job1',
        `extra:${index}`,
        new Date(instant.getTime() + (index + 1) * 1_000),
        { retention: 10 },
      )
    }
    const remaining = readdirSync(join(directory, 'job1')).sort()
    expect(remaining).toHaveLength(10)
    // The oldest archives were pruned; the newest are retained.
    const contents = remaining.map((name) =>
      readFileSync(join(directory, 'job1', name), 'utf8'),
    )
    expect(contents).toContain('extra:11')
    expect(contents).not.toContain('first')
    expect(contents).not.toContain('second')
  } finally {
    removeDirectory(directory)
  }
})

test('job store saves atomically without leaving temporary files behind', () => {
  const directory = temporaryDirectory()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(new CronJob({ id: 'atomic', prompt: 'p' }))
    store.update('atomic', { paused: true })
    store.remove('atomic')
    store.add(new CronJob({ id: 'kept', prompt: 'p' }))

    expect(readdirSync(directory).sort()).toEqual(['jobs.json', 'jobs.json.writer.sqlite'])
    const persisted = JSON.parse(
      readFileSync(join(directory, 'jobs.json'), 'utf8'),
    ) as Array<Record<string, unknown>>
    expect(persisted).toHaveLength(1)
    expect(persisted[0]?.id).toBe('kept')
    expect(store.get('kept')?.prompt).toBe('p')
  } finally {
    removeDirectory(directory)
  }
})

test('timeout aborts the runner and retains ownership until it settles', async () => {
  const directory = temporaryDirectory()
  const { restore } = silenceConsole()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(new CronJob({ id: 'owned', prompt: 'p', schedule: '* * * * *', nextRunAt: '2026-05-15T12:00:00.000Z' }))
    let calls = 0
    let signal: AbortSignal | undefined
    let finish!: (value: string) => void
    const scheduler = new CronScheduler(store, (_job, cancellation) => {
      calls++
      signal = cancellation
      return new Promise<string>((resolve) => { finish = resolve })
    }, { jobTimeout: 10 })
    expect(await scheduler.tick(new Date('2026-05-15T12:00:00Z'))).toEqual([])
    expect(signal?.aborted).toBe(true)
    expect(await scheduler.tick(new Date('2026-05-15T12:01:00Z'))).toEqual([])
    expect(calls).toBe(1)
    finish('late success')
    await Bun.sleep(0)
    expect(store.get('owned')?.metadata.last_error).toContain('timed out')
    await scheduler.tick(new Date('2026-05-15T12:02:00Z'))
    expect(calls).toBe(2)
    finish('cleanup')
  } finally { restore(); removeDirectory(directory) }
})

test('stop cancels an active manually ticked runner', async () => {
  const directory = temporaryDirectory()
  const { restore } = silenceConsole()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(new CronJob({ id: 'stop', prompt: 'p', oneshot: true, nextRunAt: '2026-05-15T12:00:00Z' }))
    let started!: () => void
    const ready = new Promise<void>((resolve) => { started = resolve })
    const scheduler = new CronScheduler(store, (_job, signal) => new Promise<string>((_resolve, reject) => {
      signal.addEventListener('abort', () => reject(signal.reason), { once: true })
      started()
    }), { jobTimeout: 0 })
    const tick = scheduler.tick(new Date('2026-05-15T12:00:00Z'))
    await ready
    scheduler.stop()
    expect(await tick).toEqual([])
    expect(store.get('stop')?.metadata.last_error).toContain('scheduler stopped')
  } finally { restore(); removeDirectory(directory) }
})

test('manual jobs share ownership, project limits, and global limits with scheduled jobs', async () => {
  const directory = temporaryDirectory()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    const first = new CronJob({ id: 'first', prompt: 'p', projectRoot: '/one' })
    const sameProject = new CronJob({ id: 'same', prompt: 'p', projectRoot: '/one' })
    const other = new CronJob({ id: 'other', prompt: 'p', projectRoot: '/two' })
    const third = new CronJob({ id: 'third', prompt: 'p', projectRoot: '/three' })
    const pending = Promise.withResolvers<string>()
    const scheduler = new CronScheduler(store, () => 'automatic', { maxConcurrentJobs: 2, maxConcurrentJobsPerProject: 1 })
    const running = scheduler.runNow(first, () => pending.promise)
    await expect(scheduler.runNow(first, async () => 'duplicate')).rejects.toThrow('already running')
    await expect(scheduler.runNow(sameProject, async () => 'same')).rejects.toThrow('concurrency limit')
    const second = scheduler.runNow(other, () => pending.promise)
    await expect(scheduler.runNow(third, async () => 'third')).rejects.toThrow('concurrency limit')
    expect(scheduler.activeCount).toBe(2)
    let idle = false
    const drained = scheduler.waitForIdle().then(() => { idle = true })
    await Bun.sleep(0)
    expect(idle).toBe(false)
    pending.resolve('done')
    await Promise.all([running, second, drained])
    expect(scheduler.activeCount).toBe(0)
    expect(await scheduler.runNow(sameProject, async () => 'allowed')).toBe('allowed')
  } finally { removeDirectory(directory) }
})

test('manual runs cannot bypass the scheduler lease', async () => {
  const directory = temporaryDirectory()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    const scheduler = new CronScheduler(store, () => 'out', { holdsLease: () => false })
    await expect(scheduler.runNow(new CronJob({ id: 'job', prompt: 'p' }), async () => 'out')).rejects.toThrow('another daemon')
  } finally { removeDirectory(directory) }
})

test('deferred scheduled jobs preserve their due time until capacity is available', async () => {
  const directory = temporaryDirectory()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    for (const id of ['a', 'b', 'c']) store.add(new CronJob({ id, prompt: 'p', oneshot: true, nextRunAt: '2026-05-15T12:00:00Z' }))
    const scheduler = new CronScheduler(store, job => job.id, { maxConcurrentJobs: 1 })
    const now = new Date('2026-05-15T12:00:00Z')
    expect(await scheduler.tick(now)).toEqual(['a'])
    expect(store.get('b')?.nextRunAt).toBe('2026-05-15T12:00:00Z')
    expect(await scheduler.tick(now)).toEqual(['b'])
    expect(await scheduler.tick(now)).toEqual(['c'])
  } finally { removeDirectory(directory) }
})

test('operator cancellation stays visible and retains admission until cleanup', async () => {
  const directory = temporaryDirectory()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    const scheduler = new CronScheduler(store, () => '', { jobTimeout: 0 })
    const job = new CronJob({ id: 'cancel', prompt: 'work' })
    let finish!: () => void
    let signal!: AbortSignal
    const running = scheduler.runNow(job, async received => {
      signal = received
      await new Promise<void>(resolve => { finish = resolve })
      received.throwIfAborted()
    })
    await Promise.resolve()
    expect(scheduler.state(job.id)).toBe('running')
    expect(scheduler.cancel(job.id)).toBe(true)
    expect(signal.aborted).toBe(true)
    expect(scheduler.state(job.id)).toBe('cancelling')
    await expect(scheduler.runNow(job, async () => '')).rejects.toThrow('already running')
    finish()
    await expect(running).rejects.toThrow('cancelled by operator')
    await scheduler.waitForIdle()
    expect(scheduler.state(job.id)).toBe('idle')
    expect(scheduler.cancel(job.id)).toBe(false)
  } finally { removeDirectory(directory) }
})

test('one-shot delivery failure stays inspectable without rerunning its successful prompt', async () => {
  const directory = temporaryDirectory()
  const logs = silenceConsole()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(new CronJob({ id: 'once', prompt: 'work', oneshot: true, nextRunAt: '2026-05-15T12:00:00Z' }))
    let executions = 0
    const scheduler = new CronScheduler(store, () => { executions++; return 'finished output' }, { onComplete: () => { throw new Error('destination offline') } })
    await scheduler.tick(new Date('2026-05-15T12:00:00Z'))
    const persisted = new JobStore(store.path).get('once')
    expect(persisted?.paused).toBe(true)
    expect(persisted?.metadata.delivery_state).toBe('failed')
    expect(persisted?.metadata.delivery_error).toContain('destination offline')
    await scheduler.tick(new Date('2026-05-16T12:00:00Z'))
    expect(executions).toBe(1)
  } finally { logs.restore(); removeDirectory(directory) }
})

test('persisted per-job timeout and zero retry override scheduler defaults', async () => {
  const directory = temporaryDirectory()
  const logs = silenceConsole()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(new CronJob({ id: 'limited', prompt: 'work', oneshot: true, nextRunAt: '2026-05-15T12:00:00Z', timeoutMs: 5, maxRetries: 0 }))
    const persisted = new JobStore(store.path).get('limited')!
    expect(persisted.timeoutMs).toBe(5)
    expect(persisted.maxRetries).toBe(0)
    let aborted = false
    const scheduler = new CronScheduler(store, async (_job, signal) => {
      await new Promise<void>(resolve => signal.addEventListener('abort', () => { aborted = true; resolve() }, { once: true }))
      signal.throwIfAborted()
      return ''
    }, { jobTimeout: 10000, maxOneShotRetries: 10 })
    expect(await scheduler.tick(new Date('2026-05-15T12:00:00Z'))).toEqual([])
    await scheduler.waitForIdle()
    expect(aborted).toBe(true)
    expect(store.get('limited')?.paused).toBe(true)
    expect(store.get('limited')?.metadata.retry_count).toBe(1)
    expect(() => store.update('limited', { timeoutMs: -1 })).toThrow()
    expect(store.get('limited')?.timeoutMs).toBe(5)
  } finally { logs.restore(); removeDirectory(directory) }
})

test('interval jobs persist and coalesce missed ticks without becoming one-shot jobs', async () => {
  const directory = temporaryDirectory()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(new CronJob({ id: 'interval', prompt: 'work', intervalSeconds: 30 }))
    const scheduler = new CronScheduler(store, () => 'done')
    const start = new Date('2026-05-15T12:00:00Z')
    expect(await scheduler.tick(start)).toEqual([])
    expect(store.get('interval')?.nextRunAt).toBe('2026-05-15T12:00:30.000Z')
    expect(await scheduler.tick(new Date('2026-05-15T12:00:29Z'))).toEqual([])
    expect(await scheduler.tick(new Date('2026-05-15T12:02:00Z'))).toEqual(['interval'])
    expect(new JobStore(store.path).get('interval')?.nextRunAt).toBe('2026-05-15T12:02:30.000Z')
    expect(() => store.update('interval', { schedule: '* * * * *' })).toThrow()
    expect(store.get('interval')?.intervalSeconds).toBe(30)
    expect(store.update('interval', { intervalSeconds: null, schedule: '* * * * *' })?.intervalSeconds).toBeUndefined()
  } finally { removeDirectory(directory) }
})
