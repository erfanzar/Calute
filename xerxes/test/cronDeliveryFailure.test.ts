// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { DeliveryError, routeOutput } from '../src/cron/delivery.js'
test('delivery errors preserve the archive path and missing senders cannot appear successful', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'cron-delivery-'))
  try {
    for (const sender of [undefined, async () => { throw new Error('offline') }]) {
      try {
        await routeOutput({ platform: 'slack', recipient: 'room' }, 'finished result', { archiveDirectory: directory, jobId: 'job', ...(sender ? { sender } : {}) })
        throw new Error('Expected delivery failure')
      } catch (error) {
        expect(error).toBeInstanceOf(DeliveryError)
        expect(await readFile((error as DeliveryError).archivePath, 'utf8')).toBe('finished result')
      }
    }
  } finally { await rm(directory, { recursive: true, force: true }) }
})
