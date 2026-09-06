// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { WebSocketMonitorDecoder, WebSocketMonitorProtocolError } from '../src/runtime/websocketMonitorFrames.js'

function frame(opcode: number, payload: Uint8Array, fin = true): Uint8Array {
  const length = payload.byteLength
  const header = length < 126 ? new Uint8Array(2) : length <= 65_535 ? new Uint8Array(4) : new Uint8Array(10)
  header[0] = (fin ? 0x80 : 0) | opcode
  if (length < 126) header[1] = length
  else if (length <= 65_535) { header[1] = 126; header[2] = length >>> 8; header[3] = length & 0xff }
  else { header[1] = 127; let value = BigInt(length); for (let index = 9; index >= 2; index -= 1) { header[index] = Number(value & 0xffn); value >>= 8n } }
  const result = new Uint8Array(header.byteLength + length)
  result.set(header)
  result.set(payload, header.byteLength)
  return result
}

function text(value: string): Uint8Array { return new TextEncoder().encode(value) }
function concat(...parts: Uint8Array[]): Uint8Array {
  const result = new Uint8Array(parts.reduce((sum, part) => sum + part.byteLength, 0))
  let offset = 0
  for (const part of parts) { result.set(part, offset); offset += part.byteLength }
  return result
}

test('decoder handles every byte split and coalesced text frames at the 64KiB boundary', () => {
  const messages: string[] = []
  const decoder = new WebSocketMonitorDecoder({ onText: value => messages.push(value), onPing: () => undefined, onClose: () => undefined })
  const encoded = frame(1, new Uint8Array(65_536).fill(97))
  for (const byte of encoded) decoder.push(new Uint8Array([byte]))
  expect(messages).toEqual(['a'.repeat(65_536)])
  decoder.push(concat(frame(1, text('one')), frame(1, text('two'))))
  expect(messages.slice(-2)).toEqual(['one', 'two'])
})

test('decoder rejects huge advertised lengths before buffering payload and bounds fragmented aggregate', () => {
  const decoder = new WebSocketMonitorDecoder({ onText: () => undefined, onPing: () => undefined, onClose: () => undefined })
  const huge = new Uint8Array([0x81, 0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01])
  expect(() => decoder.push(huge)).toThrow(WebSocketMonitorProtocolError)
  expect(decoder.bufferedBytes).toBe(0)
  const fragmented = new WebSocketMonitorDecoder({ onText: () => undefined, onPing: () => undefined, onClose: () => undefined }, 5)
  fragmented.push(frame(1, text('abc'), false))
  expect(fragmented.bufferedBytes).toBe(3)
  expect(() => fragmented.push(frame(0, text('cdef')))).toThrow(/exceeds/)
  expect(() => new WebSocketMonitorDecoder({ onText: () => undefined, onPing: () => undefined, onClose: () => undefined }).push(new Uint8Array([0x81, 0x7e, 0x00, 0x01, 0x61]))).toThrow(/minimal/)
})

test('decoder validates UTF8, masking, RSV, opcodes, and control framing', () => {
  const make = () => new WebSocketMonitorDecoder({ onText: () => undefined, onPing: () => undefined, onClose: () => undefined })
  expect(() => make().push(new Uint8Array([0x81, 0x01, 0xff]))).toThrow(/UTF-8/)
  expect(() => make().push(new Uint8Array([0x81, 0x81, 0, 0, 0, 0, 0]))).toThrow(/Masked/)
  expect(() => make().push(new Uint8Array([0xc1, 0]))).toThrow(/extensions/)
  expect(() => make().push(new Uint8Array([0x82, 0]))).toThrow(/Binary/)
  expect(() => make().push(new Uint8Array([0x09, 0]))).toThrow(/final/)
})

test('decoder permits ping interleaved in fragmented text and validates close reason', () => {
  const messages: string[] = []
  const pings: Uint8Array[] = []
  const closes: Array<[number, string]> = []
  const decoder = new WebSocketMonitorDecoder({ onText: value => messages.push(value), onPing: value => pings.push(value), onClose: (code, reason) => closes.push([code, reason]) })
  decoder.push(concat(frame(1, text('hel'), false), frame(9, text('x')), frame(0, text('lo'))))
  expect(messages).toEqual(['hello'])
  expect(new TextDecoder().decode(pings[0])).toBe('x')
  decoder.push(frame(8, concat(new Uint8Array([0x03, 0xe8]), text('bye'))))
  expect(closes).toEqual([[1000, 'bye']])
})

test('decoder preserves fragmented UTF8 and BOM and releases partial messages on close', () => {
  const messages: string[] = []
  const decoder = new WebSocketMonitorDecoder({ onText: value => messages.push(value), onPing: () => {}, onClose: () => {} })
  const value = text('\uFEFF🙂')
  decoder.push(concat(frame(1, value.subarray(0, 4), false), frame(0, value.subarray(4)), frame(1, text(''))))
  expect(messages).toEqual(['\uFEFF🙂', ''])
  decoder.push(frame(1, text('partial'), false))
  decoder.push(frame(8, new Uint8Array()))
  expect(decoder.bufferedBytes).toBe(0)
})

test('decoder rejects invalid close payloads and fragmentation sequences', () => {
  const make = () => new WebSocketMonitorDecoder({ onText: () => {}, onPing: () => {}, onClose: () => {} })
  for (const payload of [new Uint8Array([1]), new Uint8Array([3, 237]), new Uint8Array([3, 232, 255])]) {
    expect(() => make().push(frame(8, payload))).toThrow(WebSocketMonitorProtocolError)
  }
  expect(() => make().push(frame(0, text('orphan')))).toThrow(/continuation/)
  expect(() => make().push(concat(frame(1, text('first'), false), frame(1, text('nested'))))).toThrow(/Nested/)
  expect(() => make().push(new Uint8Array([0x81, 127, 128, 0, 0, 0, 0, 0, 0, 0]))).toThrow(/MSB/)
})

test('closing inside delivery stops coalesced subsequent frames', () => {
  const messages: string[] = []
  const decoder = new WebSocketMonitorDecoder({ onText: value => { messages.push(value); decoder.close() }, onPing: () => {}, onClose: () => {} })
  decoder.push(concat(frame(1, text('one')), frame(1, text('two'))))
  expect(messages).toEqual(['one'])
  expect(decoder.bufferedBytes).toBe(0)
})
