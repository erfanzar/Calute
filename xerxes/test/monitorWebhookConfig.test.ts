// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { daemonMonitorWebhookHub } from '../src/daemon/monitorWebhooks.js'
import type { DaemonConfig } from '../src/daemon/config.js'

const secret = 'webhook-config-test-only-secret-123456789'
const config = (value?: unknown): DaemonConfig => ({ channels: {}, control: {}, maxConcurrentTurns: 1, projectDirectory: '/tmp', runtime: value === undefined ? {} : { monitor_webhooks: value }, workspace: {} })

test('webhook configuration is opt-in and discovery never returns secrets', async () => {
  expect(daemonMonitorWebhookHub(config(), {})).toBeUndefined()
  const hub = daemonMonitorWebhookHub(config({ port: 0, sources: [{ name: 'build', secret_env: 'BUILD_WEBHOOK_SECRET' }] }), { BUILD_WEBHOOK_SECRET: secret })!
  expect(hub.url).toBeUndefined()
  expect(hub.list()).toEqual([{ name: 'build' }])
  expect(JSON.stringify(hub.list())).not.toContain(secret)
  await hub.stop()
})

test('webhook configuration rejects missing secrets and invalid host settings without exposing secrets', () => {
  for (const value of [null, {}, { port: -1, sources: [] }, { host: 'a\nb', sources: [] }, { sources: [{ name: '../bad', secret_env: 'KEY' }] }, { sources: [{ name: 'build', secret: secret }] }, { sources: [{ name: 'build', secret_env: 'KEY' }] }]) {
    expect(() => daemonMonitorWebhookHub(config(value), {})).toThrow()
  }
  expect(() => daemonMonitorWebhookHub(config({ sources: [{ name: 'build', secret_env: 'KEY' }] }), { KEY: 'short' })).toThrow(/32 secret bytes/)
  expect(() => daemonMonitorWebhookHub(config({ sources: [{ name: 'build', secret_env: 'KEY' }, { name: 'build', secret_env: 'KEY' }] }), { KEY: secret })).toThrow()
})
