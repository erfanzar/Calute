// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createHmac } from 'node:crypto'
import { WebhookMonitorHub } from '../src/runtime/webhookMonitorSource.js'
const secret = 'webhook-bounds-test-secret-1234567890'
const url = 'http://127.0.0.1/monitors/build'
function headers(body: string, id: string): HeadersInit {
  const stamp = String(Math.floor(Date.now() / 1000))
  return { 'x-xerxes-timestamp': stamp, 'x-xerxes-delivery-id': id, 'x-xerxes-signature': 'sha256=' + createHmac('sha256', secret).update(`${stamp}.${id}.${body}`).digest('hex') }
}
function request(id: string): Request { return new Request(url, { method: 'POST', body: 'ok', headers: headers('ok', id) }) }

test('full replay cache refuses admission without forgetting admitted IDs', async () => {
  const hub = new WebhookMonitorHub({ sources: [{ name: 'build', secret }] })
  let delivered = 0
  await hub.open('build', () => { delivered++ }, () => {})
  try {
    for (let i = 0; i < 4096; i++) expect((await hub.handle(request(String(i)))).status).toBe(202)
    expect((await hub.handle(request('overflow'))).status).toBe(429)
    expect((await hub.handle(request('0'))).status).toBe(200)
    expect(delivered).toBe(4096)
  } finally { await hub.stop() }
})

test('concurrent reads are capped and shutdown cancels every pending stream', async () => {
  const hub = new WebhookMonitorHub({ sources: [{ name: 'build', secret }] })
  await hub.open('build', () => {}, () => {})
  let cancelled = 0
  const pending: Promise<Response>[] = []
  try {
    for (let i = 0; i < 32; i++) {
      const body = new ReadableStream<Uint8Array>({ cancel() { cancelled++ } })
      pending.push(hub.handle(new Request(url, { method: 'POST', body, headers: headers('', String(i)) })))
    }
    expect((await hub.handle(request('overflow'))).status).toBe(429)
    await hub.stop()
    expect((await Promise.all(pending)).map(response => response.status)).toEqual(Array(32).fill(503))
    expect(cancelled).toBe(32)
  } finally { await hub.stop() }
})

test('body deadline cancels a stalled stream rather than admitting a truncated message', async () => {
  const hub = new WebhookMonitorHub({ sources: [{ name: 'build', secret }] })
  let delivered = 0, cancelled = false
  await hub.open('build', () => { delivered++ }, () => {})
  try {
    const body = new ReadableStream<Uint8Array>({ cancel() { cancelled = true } })
    const response = await hub.handle(new Request(url, { method: 'POST', body, headers: headers('', 'slow') }))
    expect(response.status).toBe(408)
    expect(cancelled).toBe(true)
    expect(delivered).toBe(0)
  } finally { await hub.stop() }
}, { timeout: 10000 })

test('pre-aborted requests never publish even an empty signed body', async () => {
  const hub = new WebhookMonitorHub({ sources: [{ name: 'build', secret }] })
  let delivered = false
  await hub.open('build', () => { delivered = true }, () => {})
  try {
    const response = await hub.handle(new Request(url, { method: 'POST', headers: headers('', 'abort'), signal: AbortSignal.abort() }))
    expect(response.status).toBe(400)
    expect(delivered).toBe(false)
  } finally { await hub.stop() }
})
