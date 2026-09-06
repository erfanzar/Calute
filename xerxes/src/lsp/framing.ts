// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

const HEADER_LIMIT = 8192
export const LSP_MESSAGE_LIMIT = 4 * 1024 * 1024
const decoder = new TextDecoder('utf-8', { fatal: true })

/** Incremental LSP Content-Length framing. A protocol failure is terminal: trying
 * to guess the next header could turn arbitrary server output into an RPC frame.
 * Each frame is decoded independently so UTF-8 splits across reads are safe.
 */
export class LspMessageDecoder {
  private header: number[] = []
  private body: Uint8Array | undefined
  private bodyOffset = 0
  private failed = false
  constructor(private readonly receive: (message: unknown) => void, private readonly maxBytes = LSP_MESSAGE_LIMIT) {
    if (!Number.isSafeInteger(maxBytes) || maxBytes < 1 || maxBytes > LSP_MESSAGE_LIMIT) throw new RangeError('Invalid LSP message limit')
  }

  push(chunk: Uint8Array): void {
    if (this.failed) throw new Error('LSP decoder is closed')
    try {
      let offset = 0
      while (offset < chunk.length) {
        if (!this.body) {
          this.header.push(chunk[offset++]!)
          if (this.header.length > HEADER_LIMIT) throw new Error('LSP header exceeds 8192 bytes')
          const end = this.header.length
          if (end < 4 || this.header[end - 4] !== 13 || this.header[end - 3] !== 10 || this.header[end - 2] !== 13 || this.header[end - 1] !== 10) continue
          const length = this.contentLength()
          this.header = []
          this.body = new Uint8Array(length)
          this.bodyOffset = 0
        }
        const count = Math.min(chunk.length - offset, this.body.length - this.bodyOffset)
        this.body.set(chunk.subarray(offset, offset + count), this.bodyOffset)
        offset += count
        this.bodyOffset += count
        if (this.bodyOffset !== this.body.length) continue
        let message: unknown
        try { message = JSON.parse(decoder.decode(this.body)) } catch { throw new Error('LSP body is not valid UTF-8 JSON') }
        this.body = undefined
        this.bodyOffset = 0
        this.receive(message)
      }
    } catch (error) {
      this.failed = true; this.header = []; this.body = undefined; this.bodyOffset = 0
      throw error
    }
  }

  end(): void {
    if (this.failed) throw new Error('LSP decoder is closed')
    if (this.header.length || this.body) {
      this.failed = true; this.header = []; this.body = undefined
      throw new Error('Language server closed with an incomplete LSP message')
    }
    this.failed = true
  }

  private contentLength(): number {
    if (this.header.some(byte => byte > 126 || (byte < 32 && ![9, 10, 13].includes(byte)))) throw new Error('LSP headers must use printable ASCII')
    const header = String.fromCharCode(...this.header).slice(0, -4)
    let length: number | undefined
    let contentType = false
    for (const line of header.split('\r\n')) {
      const match = /^([A-Za-z0-9-]+):[ \t]*(.*)$/.exec(line)
      if (!match) throw new Error('Malformed LSP header')
      const name = match[1]!.toLowerCase(), value = match[2]!.trim()
      if (name === 'content-length') {
        if (length !== undefined || !/^[0-9]+$/.test(value)) throw new Error('Invalid or duplicate LSP Content-Length')
        length = Number(value)
        if (!Number.isSafeInteger(length) || length < 1 || length > this.maxBytes) throw new Error('LSP message exceeds the configured byte limit or is empty')
      } else if (name === 'content-type') {
        if (contentType) throw new Error('Duplicate LSP Content-Type')
        contentType = true
        const charset = /(?:^|;)\s*charset\s*=\s*"?([^;"\s]+)"?/i.exec(value)?.[1]?.toLowerCase()
        if (charset && charset !== 'utf-8' && charset !== 'utf8') throw new Error('LSP supports UTF-8 content only')
      }
    }
    if (length === undefined) throw new Error('Missing LSP Content-Length')
    return length
  }
}

export function encodeLspMessage(message: unknown): Uint8Array {
  const json = JSON.stringify(message)
  if (typeof json !== 'string') throw new Error('LSP message cannot be serialized')
  const body = new TextEncoder().encode(json)
  if (body.length > LSP_MESSAGE_LIMIT) throw new Error('LSP message exceeds the byte limit')
  const header = new TextEncoder().encode(`Content-Length: ${body.length}\r\n\r\n`)
  const frame = new Uint8Array(header.length + body.length)
  frame.set(header); frame.set(body, header.length)
  return frame
}
