// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHmac } from 'node:crypto'
import { expect, test } from 'bun:test'
import { WebhookMonitorHub } from '../src/runtime/webhookMonitorSource.js'

const secret = '0123456789abcdef0123456789abcdef'

function signed(name: string, body: string | Uint8Array, delivery = 'delivery-1', timestamp = Math.floor(Date.now() / 1_000)): Request {
  const bytes = typeof body === 'string' ? new TextEncoder().encode(body) : body
  const signature = createHmac('sha256', secret).update(`${timestamp}.${delivery}.`).update(bytes).digest('hex')
  return new Request(`http://127.0.0.1/monitors/${name}`, {
    method: 'POST', body: bytes as unknown as BodyInit,
    headers: { 'x-xerxes-timestamp': String(timestamp), 'x-xerxes-delivery-id': delivery, 'x-xerxes-signature': `sha256=${signature}` },
  })
}

test('lists configured names and delivers authenticated events with replay protection', async () => {
  const hub = new WebhookMonitorHub({ sources: [{ name: 'alerts', secret }] })
  const events: { text: string; identity: string }[] = []
  const watcher = await hub.open('alerts', event => events.push(event), () => undefined)
  const first = await hub.handle(signed('alerts', 'hello'))
  const duplicate = await hub.handle(signed('alerts', 'hello'))
  expect(hub.list()).toEqual([{ name: 'alerts' }])
  expect(first.status).toBe(202)
  expect(duplicate.status).toBe(200)
  expect(await duplicate.json()).toEqual({ accepted: true, duplicate: true })
  expect(events).toEqual([{ text: 'hello', identity: 'delivery-1' }])
  watcher.close()
  expect((await hub.handle(signed('alerts', 'later'))).status).toBe(410)
  await hub.stop()
})

test('rejects malformed, unauthorized, stale, oversized, and invalid UTF-8 requests', async () => {
  const hub = new WebhookMonitorHub({ sources: [{ name: 'alerts', secret }] })
  await hub.open('alerts', () => undefined, () => undefined)
  expect((await hub.handle(new Request('http://127.0.0.1/monitors/missing', { method: 'POST' }))).status).toBe(404)
  expect((await hub.handle(new Request('http://127.0.0.1/monitors/alerts', { method: 'GET' }))).status).toBe(405)
  expect((await hub.handle(signed('alerts', 'body', 'stale', Math.floor(Date.now() / 1_000) - 301))).status).toBe(401)
  const bad = signed('alerts', 'body', 'bad')
  bad.headers.set('x-xerxes-signature', 'sha256=' + '0'.repeat(64))
  expect((await hub.handle(bad)).status).toBe(401)
  expect((await hub.handle(signed('alerts', new Uint8Array([0xc3, 0x28]), 'utf8'))).status).toBe(400)
  expect((await hub.handle(signed('alerts', new Uint8Array(65 * 1024), 'large'))).status).toBe(413)
  await hub.stop()
})

test('isolates sources, supports abort unsubscribe, and notifies active watchers on stop', async () => {
  const hub = new WebhookMonitorHub({ sources: [{ name: 'one', secret }, { name: 'two', secret }] })
  const first: string[] = []
  const errors: unknown[] = []
  const controller = new AbortController()
  await hub.open('one', event => first.push(event.text), error => errors.push(error), controller.signal)
  await hub.open('two', () => { throw new Error('wrong source') }, () => undefined)
  controller.abort()
  expect((await hub.handle(signed('one', 'ignored', 'one-id'))).status).toBe(410)
  expect((await hub.handle(signed('two', 'ok', 'two-id'))).status).toBe(202)
  const watcher = await hub.open('one', () => undefined, error => errors.push(error))
  await hub.stop()
  watcher.close()
  expect(errors.some(error => String(error).includes('source closed'))).toBe(true)
})

test('starts a bounded local listener and exposes its URL', async () => {
  const hub = new WebhookMonitorHub({ sources: [{ name: 'alerts', secret }] })
  await hub.open('alerts', () => undefined, () => undefined)
  expect(hub.url).toBeUndefined()
  hub.start()
  expect(hub.url?.hostname).toBe('127.0.0.1')
  const outgoing = signed('alerts', 'live')
  const liveRequest = new Request(new URL('/monitors/alerts', hub.url), outgoing)
  expect((await fetch(liveRequest)).status).toBe(202)
  await hub.stop()
  expect(hub.url).toBeUndefined()
})

test('cancels a pending streaming read when stopped', async () => {
  const hub = new WebhookMonitorHub({ sources: [{ name: 'alerts', secret }] })
  await hub.open('alerts', () => undefined, () => undefined)
  let cancelled = false
  const body = new ReadableStream<Uint8Array>({ cancel() { cancelled = true } })
  const auth = signed('alerts', '')
  const request = new Request('http://127.0.0.1/monitors/alerts', { method: 'POST', body, duplex: 'half', headers: auth.headers } as RequestInit)
  const pending = hub.handle(request)
  await hub.stop()
  expect((await pending).status).toBe(503)
  expect(cancelled).toBe(true)
})

test('cancels oversized streaming bodies and reports callback failures', async () => {
  const hub = new WebhookMonitorHub({ sources: [{ name: 'alerts', secret }] })
  const chunks = [new Uint8Array(64 * 1024), new Uint8Array(1)]
  let cancelled = false
  const body = new ReadableStream<Uint8Array>({
    pull(controller) { const chunk = chunks.shift(); if (chunk) controller.enqueue(chunk) },
    cancel() { cancelled = true },
  })
  await hub.open('alerts', () => undefined, () => undefined)
  const streamTimestamp = Math.floor(Date.now() / 1_000)
  const streamDelivery = 'stream-large'
  const streamBody = new Uint8Array(65 * 1024)
  const streamSignature = createHmac('sha256', secret).update(`${streamTimestamp}.${streamDelivery}.`).update(streamBody).digest('hex')
  const oversized = new Request('http://127.0.0.1/monitors/alerts', { method: 'POST', body, duplex: 'half', headers: {
    'x-xerxes-timestamp': String(streamTimestamp), 'x-xerxes-delivery-id': streamDelivery, 'x-xerxes-signature': `sha256=${streamSignature}`,
  } } as RequestInit)
  expect((await hub.handle(oversized)).status).toBe(413)
  expect(cancelled).toBe(true)

  const errors: unknown[] = []
  let second: { close: () => void } | undefined
  await hub.open('alerts', () => { second?.close(); throw new Error('callback failed') }, error => errors.push(error))
  second = await hub.open('alerts', () => errors.push('should not receive'), () => undefined)
  const delivered = await hub.handle(signed('alerts', 'callback', 'callback-error'))
  expect(delivered.status).toBe(202)
  expect(String(errors[0])).toContain('callback failed')
  expect(errors).not.toContain('should not receive')
  await hub.stop()
})

test('admits concurrent duplicate deliveries once and enforces source count bounds', async () => {
  const hub = new WebhookMonitorHub({ sources: [{ name: 'alerts', secret }] })
  const events: string[] = []
  await hub.open('alerts', event => events.push(event.text), () => undefined)
  const responses = await Promise.all([hub.handle(signed('alerts', 'same', 'concurrent')), hub.handle(signed('alerts', 'same', 'concurrent'))])
  expect(responses.map(response => response.status).sort()).toEqual([200, 202])
  expect(events).toEqual(['same'])
  expect(() => new WebhookMonitorHub({ sources: [] })).toThrow()
  expect(() => new WebhookMonitorHub({ sources: Array.from({ length: 65 }, (_, index) => ({ name: `s${index}`, secret })) })).toThrow()
  await hub.stop()
})
