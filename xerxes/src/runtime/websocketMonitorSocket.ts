// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash, randomBytes } from 'node:crypto'
import { createConnection, type Socket } from 'node:net'
import { connect as connectTls, type TLSSocket } from 'node:tls'
import { WebSocketMonitorDecoder, WebSocketMonitorProtocolError } from './websocketMonitorFrames.js'

const MAX_HEADER_BYTES = 16 * 1024
const MAX_PENDING_BYTES = 16 * 1024
const MAX_CONTROL_PAYLOAD = 125

export interface MonitorSocket {
  onopen?: () => void
  onmessage?: (event: { data: string }) => void
  onerror?: (error: Error) => void
  onclose?: (event: { code: number; reason: string }) => void
  close(): void
}

export function createBoundedMonitorSocket(address: string): MonitorSocket {
  const parsed = parseAddress(address)
  let socket: Socket | TLSSocket | undefined
  let decoder: WebSocketMonitorDecoder | undefined
  let closed = false
  let opened = false
  let terminal = false
  let header = new Uint8Array(0)
  let pending: Uint8Array[] = []
  let pendingBytes = 0

  let closeTimer: ReturnType<typeof setTimeout> | undefined
  const clearCloseTimer = (): void => {
    if (closeTimer !== undefined) { clearTimeout(closeTimer); closeTimer = undefined }
  }
  const release = (): void => {
    clearCloseTimer()
    decoder?.close()
    decoder = undefined
    header = new Uint8Array(0)
    pending = []
    pendingBytes = 0
  }
  const fail = (error: unknown): void => {
    if (terminal || closed) return
    terminal = true
    const failure = error instanceof Error ? error : new Error(String(error))
    const active = socket
    socket = undefined
    try { active?.destroy() } catch { /* best effort */ }
    release()
    try { client.onerror?.(failure) } catch { /* user callback is isolated */ }
  }
  const sendCloseAndDestroy = (active: Socket | TLSSocket | undefined, frame?: Uint8Array): void => {
    if (!active) return
    try {
      if (frame) active.end(frame)
      else active.end()
    } catch { try { active.destroy() } catch { /* best effort */ } }
    closeTimer = setTimeout(() => { try { active.destroy() } catch { /* best effort */ } }, 250)
  }
  const finish = (code = 1000, reason = ''): void => {
    if (terminal) return
    terminal = true
    closed = true
    const active = socket
    release()
    try {
      if (active && opened && code !== 1006 && code !== 1005) {
        const reasonBytes = new TextEncoder().encode(reason).slice(0, MAX_CONTROL_PAYLOAD - 2)
        const payload = new Uint8Array(2 + reasonBytes.byteLength)
        payload[0] = code >>> 8; payload[1] = code & 0xff; payload.set(reasonBytes, 2)
        sendCloseAndDestroy(active, maskedFrame(0x8, payload))
      } else sendCloseAndDestroy(active)
    } catch { /* best effort */ }
    socket = undefined
    try { client.onclose?.({ code, reason }) } catch { /* user callback is isolated */ }
  }
  const flush = (): void => {
    if (!socket || closed) return
    while (pending.length) {
      const item = pending[0]!
      if (socket.writableLength + item.byteLength > MAX_PENDING_BYTES) return
      let writable: boolean
      try { writable = socket.write(item) } catch (error) { fail(error); return }
      pending.shift()
      pendingBytes -= item.byteLength
      if (!writable) return
    }
  }
  const sendControl = (opcode: number, payload: Uint8Array): void => {
    if (payload.byteLength > MAX_CONTROL_PAYLOAD || closed) { fail(new WebSocketMonitorProtocolError('WebSocket control payload exceeds 125 bytes')); return }
    const frame = maskedFrame(opcode, payload)
    if (pendingBytes + (socket?.writableLength ?? 0) + frame.byteLength > MAX_PENDING_BYTES) { fail(new WebSocketMonitorProtocolError('WebSocket monitor outbound buffer is full')); return }
    pending.push(frame); pendingBytes += frame.byteLength; flush()
  }
  const decoderCallbacks = {
    onText: (text: string) => { try { client.onmessage?.({ data: text }) } catch (error) { fail(error) } },
    onPing: (payload: Uint8Array) => sendControl(0xA, payload),
    onClose: (code: number, reason: string) => finish(code, reason),
  }
  const client: MonitorSocket = { close: () => {
    if (closed || terminal) return
    closed = true
    pending = []
    pendingBytes = 0
    const active = socket
    release()
    if (active && opened) sendCloseAndDestroy(active, maskedFrame(0x8, new Uint8Array([0x03, 0xE8])))
    else { try { active?.destroy() } catch { /* best effort */ } }
    socket = undefined
  } }
  decoder = new WebSocketMonitorDecoder(decoderCallbacks)

  const protocolError = (message: string): void => fail(new WebSocketMonitorProtocolError(message))
  const receive = (data: Uint8Array): void => {
    if (closed || terminal) return
    try {
      if (!opened) {
        const priorBytes = header.byteLength
        const capacity = MAX_HEADER_BYTES - priorBytes
        const prefix = data.subarray(0, Math.min(data.byteLength, capacity))
        const joined = new Uint8Array(priorBytes + prefix.byteLength)
        joined.set(header); joined.set(prefix, priorBytes); header = joined
        const end = findHeaderEnd(header)
        if (end < 0) {
          if (header.byteLength >= MAX_HEADER_BYTES) return protocolError('WebSocket handshake headers exceed 16KiB')
          return
        }
        const text = new TextDecoder().decode(header.slice(0, end))
        validateHandshake(text, expectedAccept)
        opened = true
        client.onopen?.()
        if (closed || terminal) return
        const consumed = Math.max(0, end + 4 - priorBytes)
        const rest = data.subarray(consumed); header = new Uint8Array(0)
        if (rest.byteLength) decoder!.push(rest)
        return
      }
      decoder!.push(data)
    } catch (error) { fail(error) }
  }
  const key = randomBytes(16).toString('base64')
  const expectedAccept = createHash('sha1').update(key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').digest('base64')
  const request = `GET ${parsed.path} HTTP/1.1\r\nHost: ${parsed.hostHeader}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: ${key}\r\nSec-WebSocket-Version: 13\r\n\r\n`
  const active = parsed.tls
    ? connectTls({ host: parsed.hostname, port: parsed.port, servername: parsed.hostname, rejectUnauthorized: true })
    : createConnection({ host: parsed.hostname, port: parsed.port })
  socket = active
  const connected = (): void => {
      if (closed || terminal) { active.destroy(); return }
      const bytes = new TextEncoder().encode(request)
      if (bytes.byteLength > MAX_PENDING_BYTES) { fail(new Error('WebSocket handshake request exceeds outbound buffer')); return }
      pending.push(bytes); pendingBytes += bytes.byteLength; flush()
    }
  active.once(parsed.tls ? 'secureConnect' : 'connect', connected)
  active.on('data', (data: Uint8Array) => receive(data instanceof Uint8Array ? data : new Uint8Array(data)))
  active.on('drain', flush)
  active.on('end', () => { if (!opened) fail(new Error('WebSocket monitor closed during handshake')); else finish(1006, 'Connection closed') })
  active.on('close', () => { clearCloseTimer(); if (!terminal && !closed) { if (!opened) fail(new Error('WebSocket monitor closed during handshake')); else finish(1006, 'Connection closed') } })
  active.on('error', (error: Error) => fail(error))
  return client
}

function maskedFrame(opcode: number, payload: Uint8Array): Uint8Array {
  const key = randomBytes(4), frame = new Uint8Array(6 + payload.byteLength)
  frame[0] = 0x80 | opcode; frame[1] = 0x80 | payload.byteLength; frame.set(key, 2)
  for (let i = 0; i < payload.byteLength; i++) frame[6 + i] = payload[i]! ^ key[i % 4]!
  return frame
}
function findHeaderEnd(value: Uint8Array): number {
  for (let i = 3; i < value.byteLength; i++) if (value[i - 3] === 13 && value[i - 2] === 10 && value[i - 1] === 13 && value[i] === 10) return i - 3
  return -1
}
function validateHandshake(raw: string, expectedAccept: string): void {
  const lines = raw.split('\r\n'); if (!/^HTTP\/1\.1 101(?:\s|$)/.test(lines[0] ?? '')) throw new WebSocketMonitorProtocolError('Invalid WebSocket handshake status')
  const headers = new Map<string, string>()
  const critical = new Set(['upgrade', 'connection', 'sec-websocket-accept', 'sec-websocket-extensions', 'sec-websocket-protocol'])
  for (const line of lines.slice(1)) { const separator = line.indexOf(':'); if (separator <= 0) throw new WebSocketMonitorProtocolError('Malformed WebSocket handshake header'); const name = line.slice(0, separator).trim().toLowerCase(); const value = line.slice(separator + 1).trim(); if (critical.has(name) && headers.has(name)) throw new WebSocketMonitorProtocolError(`Duplicate WebSocket handshake header: ${name}`); headers.set(name, headers.has(name) ? `${headers.get(name)}, ${value}` : value) }
  if (headers.get('upgrade')?.toLowerCase() !== 'websocket' || !headers.get('connection')?.toLowerCase().split(',').map(item => item.trim()).includes('upgrade')) throw new WebSocketMonitorProtocolError('Invalid WebSocket upgrade headers')
  if (headers.get('sec-websocket-accept') !== expectedAccept) throw new WebSocketMonitorProtocolError('Invalid WebSocket handshake accept')
  if (headers.has('sec-websocket-extensions') || headers.has('sec-websocket-protocol')) throw new WebSocketMonitorProtocolError('Unsolicited WebSocket handshake extension or subprotocol')
}
function parseAddress(address: string): { hostname: string; port: number; path: string; hostHeader: string; tls: boolean } {
  let url: URL
  try { url = new URL(address) } catch { throw new Error('Invalid WebSocket monitor address') }
  if (url.protocol !== 'ws:' && url.protocol !== 'wss:') throw new Error('WebSocket monitor address must use ws:// or wss://')
  if (url.username || url.password || url.search || url.hash) throw new Error('WebSocket monitor address must not contain credentials, query or fragment')
  const port = url.port ? Number(url.port) : url.protocol === 'wss:' ? 443 : 80
  if (!Number.isSafeInteger(port) || port < 1 || port > 65535) throw new Error('Invalid WebSocket monitor port')
  return { hostname: url.hostname.replace(/^\[|\]$/g, ''), port, path: url.pathname || '/', hostHeader: url.host, tls: url.protocol === 'wss:' }
}
