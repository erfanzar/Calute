// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { SessionOperationQueue } from '../src/runtime/sessionOperationQueue.js'

const tick = () => new Promise<void>(resolve => queueMicrotask(resolve))

test('queues per key, gives queued human work precedence, and runs different keys in parallel', async () => {
  const queue = new SessionOperationQueue()
  const order: string[] = []
  const first = queue.run('one', async () => { order.push('one-background'); await tick(); return 'background' }, 'background')
  await tick()
  const human = queue.run('one', async () => { order.push('one-human'); return 'human' }, 'human')
  const other = queue.run('two', async () => { order.push('two-human'); return 'other' })
  expect(await Promise.all([first, human, other])).toEqual(['background', 'human', 'other'])
  expect(order.filter(value => value.startsWith('one-'))).toEqual(['one-background', 'one-human'])
  expect(order).toContain('two-human')
})

test('same-tick priority selection happens before the first operation starts', async () => {
  const queue = new SessionOperationQueue()
  const order: string[] = []
  const background = queue.run('session', async () => { order.push('background'); return 1 }, 'background')
  const human = queue.run('session', async () => { order.push('human'); return 2 })
  expect(await Promise.all([background, human])).toEqual([1, 2])
  expect(order).toEqual(['human', 'background'])
})

test('running work is nonpreemptive and rejections release the key', async () => {
  const queue = new SessionOperationQueue()
  let release!: () => void
  const gate = new Promise<void>(resolve => { release = resolve })
  const running = queue.run('session', async () => { await gate; return 'first' })
  const next = queue.run('session', async () => 'second')
  expect(queue.has('session')).toBe(true)
  release()
  await expect(running).resolves.toBe('first')
  await expect(next).resolves.toBe('second')
  await expect(queue.run('failure', async () => { throw new Error('operation failed') })).rejects.toThrow('operation failed')
  await expect(queue.run('failure', async () => 'recovered')).resolves.toBe('recovered')
})

test('close rejects queued work, leaves active work alone, and drain waits for active work', async () => {
  const queue = new SessionOperationQueue()
  let release!: () => void
  const gate = new Promise<void>(resolve => { release = resolve })
  const active = queue.run('session', async () => { await gate; return 'active' })
  await tick()
  const queued = queue.run('session', async () => 'queued')
  const draining = queue.drain()
  queue.close()
  await expect(queued).rejects.toThrow('queue closed')
  expect(queue.has('session')).toBe(true)
  let drained = false
  void draining.then(() => { drained = true })
  await tick()
  expect(drained).toBe(false)
  release()
  await expect(active).resolves.toBe('active')
  await draining
  expect(drained).toBe(true)
  await expect(queue.run('later', async () => 'nope')).rejects.toThrow('no new session work')
})

test('human pending state excludes a running task and clears after admission', async () => {
  const queue = new SessionOperationQueue()
  let release!: () => void
  const gate = new Promise<void>(resolve => { release = resolve })
  const active = queue.run('session', async () => { await gate; return undefined })
  await tick()
  expect(queue.hasHumanPending('session')).toBe(false)
  const queued = queue.run('session', async () => undefined)
  expect(queue.hasHumanPending('session')).toBe(true)
  release()
  await active
  await queued
  expect(queue.hasHumanPending('session')).toBe(false)
})
