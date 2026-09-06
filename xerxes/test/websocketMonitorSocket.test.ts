// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createHash } from 'node:crypto'
import { createServer, type Socket, type Server } from 'node:net'
import { once } from 'node:events'
import { createBoundedMonitorSocket } from '../src/runtime/websocketMonitorSocket.js'
import { WebSocketMonitorProtocolError } from '../src/runtime/websocketMonitorFrames.js'

async function rawServer(handler: (socket: Socket, request: string) => void): Promise<{ server: Server; url: string }> {
  const server = createServer(socket => {
    let buffer = Buffer.alloc(0)
    socket.on('data', chunk => {
      buffer = Buffer.concat([buffer, Buffer.from(chunk)])
      const end = buffer.indexOf('\r\n\r\n')
      if (end >= 0) { const request = buffer.subarray(0, end + 4).toString('utf8'); buffer = buffer.subarray(end + 4); handler(socket, request) }
    })
  })
  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  const address = server.address()
  if (!address || typeof address === 'string') throw new Error('raw test server did not bind')
  return { server, url: `ws://127.0.0.1:${address.port}/events` }
}

function accept(request: string): string {
  const key = request.match(/^Sec-WebSocket-Key:\s*(.+)$/im)?.[1]?.trim()
  if (!key) throw new Error('missing websocket key')
  return createHash('sha1').update(`${key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11`).digest('base64')
}

function handshake(request: string, extra = ''): string {
  return `HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept(request)}\r\n${extra}\r\n`
}

function frame(opcode: number, payload: string): Buffer {
  const body = Buffer.from(payload)
  return Buffer.concat([Buffer.from([0x80 | opcode, body.length]), body])
}

async function waitFor<T>(read: () => T | undefined, timeout = 2_000): Promise<T> {
  const deadline = Date.now() + timeout
  for (;;) { const value = read(); if (value !== undefined) return value; if (Date.now() >= deadline) throw new Error('timed out'); await Bun.sleep(10) }
}

test('bounded socket interoperates with a local RFC6455 server and answers ping', async () => {
  let pong = ''
  const { server, url } = await rawServer((socket, request) => {
    socket.write(handshake(request))
    socket.write(frame(1, 'hello'))
    socket.write(frame(9, 'ping'))
    socket.on('data', data => {
      const bytes = Buffer.from(data as Uint8Array)
      if ((bytes[0] ?? 0) & 0x0f) {
        const length = (bytes[1] ?? 0) & 0x7f
        const mask = bytes.subarray(2, 6)
        const payload = bytes.subarray(6, 6 + length)
        pong = Buffer.from(payload.map((value, index) => value ^ (mask[index % 4] ?? 0))).toString()
      }
    })
  })
  const events: string[] = [], errors: Error[] = [], closes: number[] = []
  const client = createBoundedMonitorSocket(url)
  client.onopen = () => undefined
  client.onmessage = event => events.push(event.data)
  client.onerror = error => errors.push(error)
  client.onclose = event => closes.push(event.code)
  try {
    await waitFor(() => events[0])
    await waitFor(() => pong || undefined)
    expect(events).toEqual(['hello'])
    expect(pong).toBe('ping')
    expect(errors).toHaveLength(0)
  } finally { client.close(); server.close(); await once(server, 'close') }
})

test('bounded socket rejects malformed handshake and oversized input as protocol errors', async () => {
  const cases: Array<{ response: (request: string) => string; frame?: Buffer }> = [
    { response: () => 'HTTP/1.1 200 Nope\r\nContent-Length: 0\r\n\r\n' },
    { response: () => `HTTP/1.1 101 Switching Protocols\r\n${'X-Fill: abcdefghijklmnopqrstuvwxyz\r\n'.repeat(500)}\r\n` },
    { response: request => handshake(request), frame: Buffer.from([0x81, 0x7f, 0, 0, 0, 0, 0, 1, 0, 1]) },
  ]
  for (const [caseIndex, item] of cases.entries()) {
    const { server, url } = await rawServer((socket, request) => {
      socket.write(item.response(request))
      if (item.frame) socket.write(item.frame)
    })
    const errors: Error[] = []
    const client = createBoundedMonitorSocket(url)
    client.onerror = error => errors.push(error)
    try {
      const error = await waitFor(() => errors[0]).catch(error => { throw new Error(`case ${caseIndex}: ${String(error)}`) })
      expect(error).toBeInstanceOf(WebSocketMonitorProtocolError)
    } finally { client.close(); server.close(); await once(server, 'close') }
  }
})
