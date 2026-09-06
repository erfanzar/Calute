// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHmac, timingSafeEqual } from 'node:crypto'

const MAX_BODY_BYTES = 64 * 1024
const BODY_TIMEOUT_MS = 5_000
const MAX_ACTIVE_REQUESTS = 32
const REPLAY_WINDOW_SECONDS = 300
const MAX_REPLAY_IDS = 4_096
const NAME_PATTERN = /^[a-zA-Z0-9_-]{1,64}$/
const DELIVERY_PATTERN = /^[A-Za-z0-9_-]{1,128}$/

export interface WebhookMonitorSource {
  list(): readonly { readonly name: string }[]
  open(
    name: string,
    onEvent: (event: { readonly text: string; readonly identity: string }) => void,
    onError: (error: unknown) => void,
    signal?: AbortSignal,
  ): Promise<{ readonly name: string; readonly close: () => void }>
}

interface SourceConfig { readonly name: string; readonly secret: string }
interface Subscriber {
  readonly onEvent: (event: { readonly text: string; readonly identity: string }) => void
  readonly onError: (error: unknown) => void
  readonly cleanup: () => void
}

export class WebhookMonitorHub implements WebhookMonitorSource {
  private readonly sourceConfigs: readonly SourceConfig[]
  private readonly sourceByName = new Map<string, SourceConfig>()
  private readonly subscribers = new Map<string, Set<Subscriber>>()
  private readonly replay = new Map<string, Map<string, number>>()
  private readonly host: string
  private readonly port: number
  private server: Bun.Server<unknown> | undefined
  private activeRequests = 0
  private readonly pendingRequests = new Set<AbortController>()
  private stopping = false

  constructor(options: { sources: readonly { name: string; secret: string }[]; host?: string; port?: number }) {
    if (options.sources.length < 1 || options.sources.length > 64) throw new Error('Webhook monitor sources must contain between 1 and 64 sources')
    const configs: SourceConfig[] = []
    for (const source of options.sources) {
      if (!NAME_PATTERN.test(source.name)) throw new Error(`Invalid webhook monitor name: ${source.name}`)
      if (this.sourceByName.has(source.name)) throw new Error(`Duplicate webhook monitor name: ${source.name}`)
      if (new TextEncoder().encode(source.secret).byteLength < 32) throw new Error(`Webhook monitor secret is too short: ${source.name}`)
      const config = Object.freeze({ name: source.name, secret: source.secret })
      configs.push(config)
      this.sourceByName.set(config.name, config)
      this.subscribers.set(config.name, new Set())
      this.replay.set(config.name, new Map())
    }
    this.sourceConfigs = Object.freeze(configs)
    this.host = options.host ?? '127.0.0.1'
    this.port = options.port ?? 0
    if (!Number.isInteger(this.port) || this.port < 0 || this.port > 65_535) throw new Error('Webhook monitor port must be an integer from 0 to 65535')
  }

  list(): readonly { readonly name: string }[] {
    return this.sourceConfigs.map(source => Object.freeze({ name: source.name }))
  }

  start(): void {
    if (this.server) return
    this.stopping = false
    this.server = Bun.serve<unknown>({
      hostname: this.host,
      port: this.port,
      maxRequestBodySize: MAX_BODY_BYTES,
      fetch: request => this.handle(request),
    })
  }

  async stop(): Promise<void> {
    if (this.stopping) return
    this.stopping = true
    for (const controller of this.pendingRequests) controller.abort(new Error('Webhook monitor hub stopped'))
    this.pendingRequests.clear()
    for (const [name, watchers] of this.subscribers) {
      const snapshot = [...watchers]
      watchers.clear()
      this.subscribers.set(name, new Set())
      // Notify a snapshot: clearing the live set above prevents any in-flight
      // request from dispatching to a subscriber after shutdown.
      for (const watcher of snapshot) {
        watcher.cleanup()
        safeError(watcher.onError, new Error(`Webhook monitor source closed: ${name}`))
      }
    }
    const server = this.server
    this.server = undefined
    if (server) await server.stop(true)
  }

  get url(): URL | undefined { return this.server?.url }

  async open(name: string, onEvent: (event: { readonly text: string; readonly identity: string }) => void, onError: (error: unknown) => void, signal?: AbortSignal): Promise<{ readonly name: string; readonly close: () => void }> {
    if (signal?.aborted) throw signal.reason ?? new Error('Webhook monitor was aborted before opening')
    if (this.stopping) throw new Error('Webhook monitor hub is stopped')
    const source = this.sourceByName.get(name)
    if (!source) throw new Error(`Unknown webhook monitor source: ${name}`)
    const watchers = this.subscribers.get(name)
    if (!watchers) throw new Error(`Unknown webhook monitor source: ${name}`)
    let cleanup = (): void => undefined
    const watcher: Subscriber = { onEvent, onError, cleanup: () => cleanup() }
    watchers.add(watcher)
    let closed = false
    const close = (): void => {
      if (closed) return
      closed = true
      watchers.delete(watcher)
      signal?.removeEventListener('abort', abort)
    }
    const abort = (): void => close()
    cleanup = (): void => signal?.removeEventListener('abort', abort)
    signal?.addEventListener('abort', abort, { once: true })
    if (signal?.aborted) close()
    return Object.freeze({ name: source.name, close })
  }

  async handle(request: Request): Promise<Response> {
    if (request.signal.aborted) return jsonResponse({ error: 'request cancelled' }, 400)
    if (request.method !== 'POST') return jsonResponse({ error: 'method not allowed' }, 405)
    const route = parseRoute(request.url)
    if (route === undefined) return jsonResponse({ error: 'malformed monitor path' }, 400)
    const source = this.sourceByName.get(route)
    if (!source) return jsonResponse({ error: 'unknown monitor' }, 404)
    const watchers = this.subscribers.get(route)
    if (!watchers || watchers.size === 0) return jsonResponse({ error: 'monitor has no active watcher' }, 410)
    if (this.stopping) return jsonResponse({ error: 'monitor hub is stopped' }, 503)
    if (this.activeRequests >= MAX_ACTIVE_REQUESTS) return jsonResponse({ error: 'too many active requests' }, 429)
    const encoding = request.headers.get('content-encoding')
    if (encoding && encoding.trim().toLowerCase() !== 'identity') return jsonResponse({ error: 'content compression is not supported' }, 400)
    const length = request.headers.get('content-length')
    if (length !== null && (!/^\d+$/.test(length) || !Number.isSafeInteger(Number(length)))) return jsonResponse({ error: 'invalid content length' }, 400)
    if (length !== null && Number(length) > MAX_BODY_BYTES) return jsonResponse({ error: 'request body too large' }, 413)

    this.activeRequests += 1
    const controller = new AbortController()
    this.pendingRequests.add(controller)
    try {
      const timestampHeader = request.headers.get('x-xerxes-timestamp')
      const deliveryId = request.headers.get('x-xerxes-delivery-id')
      const signature = request.headers.get('x-xerxes-signature')
      if (!timestampHeader || !deliveryId || !signature) return jsonResponse({ error: 'authentication headers required' }, 401)
      if (!/^[-]?\d+$/.test(timestampHeader) || !Number.isSafeInteger(Number(timestampHeader))) return jsonResponse({ error: 'invalid timestamp' }, 400)
      const timestamp = Number(timestampHeader)
      if (Math.abs(Math.floor(Date.now() / 1_000) - timestamp) > REPLAY_WINDOW_SECONDS) return jsonResponse({ error: 'stale timestamp' }, 401)
      if (!DELIVERY_PATTERN.test(deliveryId)) return jsonResponse({ error: 'invalid delivery id' }, 400)
      if (!/^sha256=[0-9a-f]{64}$/i.test(signature)) return jsonResponse({ error: 'malformed signature' }, 400)
      const body = await readBody(request, controller.signal)
      const expected = createHmac('sha256', source.secret).update(`${timestampHeader}.${deliveryId}.`).update(body).digest()
      const supplied = Buffer.from(signature.slice(7), 'hex')
      if (supplied.byteLength !== expected.byteLength || !timingSafeEqual(supplied, expected)) return jsonResponse({ error: 'invalid signature' }, 401)
      const cache = this.replay.get(route)
      if (!cache) return jsonResponse({ error: 'unknown monitor' }, 404)
      const now = Math.floor(Date.now() / 1_000)
      if (Math.abs(now - timestamp) > REPLAY_WINDOW_SECONDS) return jsonResponse({ error: 'stale timestamp' }, 401)
      let text: string
      try { text = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(body) } catch { return jsonResponse({ error: 'body is not valid UTF-8' }, 400) }
      if (this.stopping) return jsonResponse({ error: 'monitor hub stopped' }, 503)
      if (this.subscribers.get(route) !== watchers || watchers.size === 0) return jsonResponse({ error: 'monitor has no active watcher' }, 410)
      purgeReplay(cache, now)
      if (cache.has(deliveryId)) return jsonResponse({ accepted: true, duplicate: true }, 200)
      if (cache.size >= MAX_REPLAY_IDS) return jsonResponse({ error: 'replay cache at capacity' }, 429)
      cache.set(deliveryId, timestamp)
      const event = { text, identity: deliveryId }
      for (const watcher of [...watchers]) {
        if (this.stopping || this.subscribers.get(route) !== watchers) break
        if (!watchers.has(watcher)) continue
        try { watcher.onEvent(event) } catch (error) { safeError(watcher.onError, error) }
      }
      return jsonResponse({ accepted: true, duplicate: false }, 202)
    } catch (error) {
      if (controller.signal.aborted || this.stopping) return jsonResponse({ error: 'monitor hub stopped' }, 503)
      if (error instanceof BodyLimitError) return jsonResponse({ error: error.message }, 413)
      if (error instanceof BodyTimeoutError) return jsonResponse({ error: error.message }, 408)
      return jsonResponse({ error: 'malformed request' }, 400)
    } finally {
      this.pendingRequests.delete(controller)
      this.activeRequests -= 1
    }
  }
}

class BodyLimitError extends Error {}
class BodyTimeoutError extends Error {}
class BodyCancelledError extends Error {}

async function readBody(request: Request, signal: AbortSignal): Promise<Uint8Array> {
  if (!request.body) return new Uint8Array()
  const reader = request.body.getReader()
  const bodyBuffer = new Uint8Array(MAX_BODY_BYTES)
  let size = 0
  let reads = 0
  const startedAt = Date.now()
  const timeout = Symbol('body timeout')
  const abort = Symbol('body aborted')
  let timer: ReturnType<typeof setTimeout> | undefined
  let resolveCancellation: ((value: symbol) => void) | undefined
  const cancellation = new Promise<symbol>(resolve => { resolveCancellation = resolve })
  const cancel = (kind: symbol): void => {
    if (cancelled !== undefined) return
    cancelled = kind
    resolveCancellation?.(kind)
    void reader.cancel().catch(() => undefined)
  }
  let cancelled: symbol | undefined
  const onAbort = (): void => cancel(abort)
  const onTimeout = (): void => cancel(timeout)
  signal.addEventListener('abort', onAbort, { once: true })
  request.signal.addEventListener('abort', onAbort, { once: true })
  if (signal.aborted) onAbort()
  else if (request.signal.aborted) onAbort()
  timer = setTimeout(onTimeout, BODY_TIMEOUT_MS)
  try {
    while (true) {
      if (++reads > 65_537 || Date.now() - startedAt >= BODY_TIMEOUT_MS) { cancel(timeout); throw new BodyTimeoutError('request body read timed out') }
      const read = reader.read().then(result => ({ kind: 'read' as const, result }), error => ({ kind: 'error' as const, error }))
      const result = await Promise.race([read, cancellation])
      if (cancelled !== undefined) {
        if (cancelled === timeout) throw new BodyTimeoutError('request body read timed out')
        throw new BodyCancelledError('request body read cancelled')
      }
      if (typeof result === 'symbol') {
        if (result === timeout) throw new BodyTimeoutError('request body read timed out')
        if (result === abort) throw new BodyCancelledError('request body read cancelled')
        throw new BodyCancelledError('request body read cancelled')
      }
      if (result.kind === 'error') throw result.error
      if (result.result.done) break
      size += result.result.value.byteLength
      if (size > MAX_BODY_BYTES) { cancel(abort); throw new BodyLimitError('request body too large') }
      if (result.result.value.byteLength > 0) bodyBuffer.set(result.result.value, size - result.result.value.byteLength)
    }
  } finally {
    if (timer !== undefined) clearTimeout(timer)
    signal.removeEventListener('abort', onAbort)
    request.signal.removeEventListener('abort', onAbort)
    try { reader.releaseLock() } catch {}
  }
  return bodyBuffer.slice(0, size)
}

function purgeReplay(cache: Map<string, number>, now: number): void {
  for (const [id, timestamp] of cache) if (Math.abs(now - timestamp) > REPLAY_WINDOW_SECONDS) cache.delete(id)
}

function parseRoute(value: string): string | undefined {
  let pathname: string
  try { pathname = new URL(value).pathname } catch { return undefined }
  const parts = pathname.split('/')
  if (parts.length !== 3 || parts[1] !== 'monitors' || !parts[2]) return undefined
  try { return decodeURIComponent(parts[2]!) }
  catch { return undefined }
}

function jsonResponse(value: unknown, status: number): Response { return Response.json(value, { status }) }
function safeError(callback: (error: unknown) => void, error: unknown): void { try { callback(error) } catch {} }
