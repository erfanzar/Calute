// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { encodeLspMessage, LspMessageDecoder } from '../src/lsp/framing.js'
const bytes = (value: string) => new TextEncoder().encode(value)

test('LSP framing counts UTF-8 bytes and supports every split boundary', () => {
  const message = { jsonrpc: '2.0', id: 7, result: 'سلام 🌍' }
  const frame = encodeLspMessage(message)
  expect(new TextDecoder().decode(frame)).toStartWith(`Content-Length: ${bytes(JSON.stringify(message)).length}\r\n`)
  for (let split = 0; split <= frame.length; split++) {
    const received: unknown[] = []
    const decoder = new LspMessageDecoder(value => received.push(value))
    decoder.push(frame.subarray(0, split)); decoder.push(frame.subarray(split)); decoder.end()
    expect(received).toEqual([message])
    expect(() => decoder.push(frame)).toThrow('closed')
  }
})

test('LSP framing streams consecutive messages without treating body newlines as framing', () => {
  const frames = [encodeLspMessage({ result: '\r\n\r\n' }), encodeLspMessage({ id: 2, result: null })]
  const combined = new Uint8Array(frames[0]!.length + frames[1]!.length)
  combined.set(frames[0]!); combined.set(frames[1]!, frames[0]!.length)
  const received: unknown[] = []
  const decoder = new LspMessageDecoder(value => received.push(value))
  for (const byte of combined) decoder.push(new Uint8Array([byte]))
  decoder.end()
  expect(received).toEqual([{ result: '\r\n\r\n' }, { id: 2, result: null }])
})

test.each([
  'Content-Length: -1\r\n\r\n', 'Content-Length: 0\r\n\r\n',
  'Content-Length: 999999999999999999\r\n\r\n',
  'Content-Length: 2\r\nContent-Length: 2\r\n\r\n{}',
  'Other: 2\r\n\r\n', 'Content-Length 2\r\n\r\n',
  'Content-Length: 2\r\nContent-Type: application/vscode-jsonrpc; charset=latin1\r\n\r\n{}',
  'Content-Length: 2\r\nContent-Type: x\r\nContent-Type: x\r\n\r\n{}',
  'Content-Length: 2\r\nOther: é\r\n\r\n{}',
  'Content-Length: 2\r\nOther: \0\r\n\r\n{}',
])('rejects malformed headers and never resumes after failure: %s', frame => {
  const decoder = new LspMessageDecoder(() => { throw new Error('must not deliver') })
  expect(() => decoder.push(bytes(frame))).toThrow()
  expect(() => decoder.push(encodeLspMessage({}))).toThrow('closed')
})

test('bounds headers and body allocation and does not expose invalid source text', () => {
  expect(() => new LspMessageDecoder(() => {}).push(bytes('x'.repeat(8193)))).toThrow('header exceeds')
  expect(() => new LspMessageDecoder(() => {}, 4).push(bytes('Content-Length: 5\r\n\r\n'))).toThrow('byte limit')
  const decoder = new LspMessageDecoder(() => {})
  expect(() => decoder.push(bytes('Content-Length: 16\r\n\r\nprivate-secret!!'))).toThrow('not valid UTF-8 JSON')
})

test('rejects invalid UTF-8 and incomplete EOF, accepting supported charset aliases', () => {
  for (const value of ['utf-8', 'utf8']) {
    const received: unknown[] = []
    const decoder = new LspMessageDecoder(value => received.push(value))
    decoder.push(bytes(`Content-Length: 2\r\nContent-Type: application/vscode-jsonrpc; charset=${value}\r\n\r\n{}`)); decoder.end()
    expect(received).toEqual([{}])
  }
  const decoder = new LspMessageDecoder(() => {})
  decoder.push(bytes('Content-Length: 1\r\n\r\n'))
  expect(() => decoder.push(new Uint8Array([255]))).toThrow('UTF-8')
  for (const partial of ['Content-Len', 'Content-Length: 3\r\n\r\n{}']) {
    const truncated = new LspMessageDecoder(() => {})
    truncated.push(bytes(partial))
    expect(() => truncated.end()).toThrow('incomplete')
  }
})
