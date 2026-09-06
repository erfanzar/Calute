// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { createHash } from 'node:crypto'
import { nativeWebSocketMonitorSource } from '../src/runtime/websocketMonitorSource.js'
import { RunHistory } from '../src/runtime/runHistory.js'
import { TerminalMonitors } from '../src/runtime/terminalMonitors.js'
import { TerminalRegistry } from '../src/runtime/terminalRegistry.js'

async function until(predicate: () => boolean, timeout = 2000): Promise<void> {
  const deadline = Date.now() + timeout
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error('WebSocket test condition timed out')
    await Bun.sleep(10)
  }
}

test('aborting a stalled upgrade releases the pending connection without a late monitor', async () => {
  let accepted = 0, closed = 0
  const sockets = new Set<Bun.Socket>()
  const server = Bun.listen({ hostname: '127.0.0.1', port: 0, socket: {
    open(socket) { accepted++; sockets.add(socket) }, data() {},
    close(socket) { closed++; sockets.delete(socket) },
  } })
  const controller = new AbortController(), errors: unknown[] = [], events: unknown[] = []
  try {
    const pending = nativeWebSocketMonitorSource.open(`ws://127.0.0.1:${server.port}/events`, event => events.push(event), () => {}, error => errors.push(error), controller.signal)
    const rejected = pending.catch(error => error)
    await until(() => accepted === 1)
    controller.abort(new Error('cancel stalled upgrade'))
    expect(String(await rejected)).toContain('cancel stalled upgrade')
    await until(() => closed === 1)
    expect(errors).toEqual([])
    expect(events).toEqual([])
  } finally { controller.abort(); for (const socket of sockets) socket.terminate(); server.stop(true) }
})

test('oversized declared frames fail a live watch immediately without body or reconnect', async () => {
  let connections = 0, peer: Bun.Socket<{ header: string }> | undefined
  const sockets = new Set<Bun.Socket<{ header: string }>>()
  const server = Bun.listen<{ header: string }>({ hostname: '127.0.0.1', port: 0, socket: {
    open(socket) { socket.data = { header: '' }; peer = socket; sockets.add(socket); connections++ },
    data(socket, bytes) {
      socket.data.header += bytes.toString()
      if (!socket.data.header.includes('\r\n\r\n')) return
      const key = /^Sec-WebSocket-Key:\s*(.+)$/im.exec(socket.data.header)?.[1]?.trim()
      if (!key) return
      const accept = createHash('sha1').update(key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').digest('base64')
      socket.write(`HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`)
      socket.data.header = ''
    }, close(socket) { sockets.delete(socket) },
  } })
  const history = new RunHistory(':memory:'), errors: unknown[] = []
  const monitors = new TerminalMonitors(new TerminalRegistry(), history, undefined, error => errors.push(error), undefined, undefined, { source: nativeWebSocketMonitorSource, resolveWorkspace: () => '/repo' })
  try {
    const watch = await monitors.startWebSocket('owner', { url: `ws://127.0.0.1:${server.port}/feed`, match: 'error' })
    const header = Buffer.alloc(10)
    header[0] = 0x81; header[1] = 127; header.writeBigUInt64BE(1_073_741_824n, 2)
    peer!.write(header)
    await until(() => monitors.inspect('owner', watch.id).state === 'failed')
    expect(errors).toHaveLength(1)
    expect(monitors.inspect('owner', watch.id).events).toEqual([])
    expect(history.inspect('owner', watch.id)?.state).toBe('failed')
    await until(() => sockets.size === 0)
    await Bun.sleep(350)
    expect(connections).toBe(1)
  } finally { monitors.close(); history.close(); for (const socket of sockets) socket.terminate(); server.stop(true) }
})
