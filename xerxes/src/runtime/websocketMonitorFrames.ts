// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export class WebSocketMonitorProtocolError extends Error {
  override readonly name = 'WebSocketMonitorProtocolError'
}

interface DecoderCallbacks {
  readonly onText: (text: string) => void
  readonly onPing: (payload: Uint8Array) => void
  readonly onClose: (code: number, reason: string) => void
}

const TEXT_DECODER = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true })
const MAX_ALLOWED_MESSAGE_BYTES = 16 * 1024 * 1024

export class WebSocketMonitorDecoder {
  private readonly callbacks: DecoderCallbacks
  private readonly maxMessageBytes: number
  private readonly header = new Uint8Array(14)
  private headerBytes = 0
  private headerLength = 0
  private headerReady = false
  private payloadRemaining = 0
  private payloadRead = 0
  private payloadKind: 'text' | 'control' | undefined
  private payloadOpcode = 0
  private payloadBuffer: Uint8Array | undefined
  private messageBuffer: Uint8Array | undefined
  private messageLength = 0
  private fragmentedOpcode: 1 | undefined
  private stopped = false

  constructor(callbacks: DecoderCallbacks, maxMessageBytes = 65_536) {
    if (!Number.isSafeInteger(maxMessageBytes) || maxMessageBytes < 0 || maxMessageBytes > MAX_ALLOWED_MESSAGE_BYTES) throw new RangeError(`maxMessageBytes must be between 0 and ${MAX_ALLOWED_MESSAGE_BYTES}`)
    this.callbacks = callbacks
    this.maxMessageBytes = maxMessageBytes
  }

  get bufferedBytes(): number {
    return this.messageLength + this.headerBytes + this.payloadRead
  }

  push(chunk: Uint8Array): void {
    if (this.stopped) return
    let offset = 0
    try {
      while (offset < chunk.byteLength && !this.stopped) {
        if (!this.headerReady && this.headerBytes < 2) {
          const count = Math.min(2 - this.headerBytes, chunk.byteLength - offset)
          this.header.set(chunk.subarray(offset, offset + count), this.headerBytes)
          this.headerBytes += count
          offset += count
          if (this.headerBytes < 2) continue
          this.beginHeader()
        }
        if (!this.headerReady && this.headerBytes < this.headerLength) {
          const count = Math.min(this.headerLength - this.headerBytes, chunk.byteLength - offset)
          this.header.set(chunk.subarray(offset, offset + count), this.headerBytes)
          this.headerBytes += count
          offset += count
          if (this.headerBytes < this.headerLength) continue
        }
        if (!this.headerReady) {
          this.finishHeader()
          this.headerReady = true
        }
        if (this.payloadRemaining > 0) {
          const count = Math.min(this.payloadRemaining, chunk.byteLength - offset)
          this.copyPayload(chunk.subarray(offset, offset + count))
          this.payloadRemaining -= count
          this.payloadRead += count
          offset += count
          if (this.payloadRemaining > 0) continue
        }
        this.finishFrame()
      }
    } catch (error) {
      this.stopped = true
      this.headerBytes = 0
      this.headerLength = 0
      this.headerReady = false
      this.payloadRemaining = 0
      this.payloadRead = 0
      this.payloadKind = undefined
      this.messageBuffer = undefined
      this.messageLength = 0
      this.payloadBuffer = undefined
      if (error instanceof WebSocketMonitorProtocolError) throw error
      throw protocolError(String(error))
    }
  }

  close(): void {
    this.stopped = true
    this.headerBytes = 0
    this.headerLength = 0
    this.headerReady = false
    this.payloadRemaining = 0
    this.payloadRead = 0
    this.payloadBuffer = undefined
    this.messageBuffer = undefined
    this.messageLength = 0
    this.fragmentedOpcode = undefined
  }

  private beginHeader(): void {
    const first = this.header[0] ?? 0
    const second = this.header[1] ?? 0
    if ((first & 0x70) !== 0) throw protocolError('WebSocket extensions are not supported')
    if ((second & 0x80) !== 0) throw protocolError('Masked server frames are invalid')
    const fin = (first & 0x80) !== 0
    const opcode = first & 0x0f
    const length = second & 0x7f
    if (![0, 1, 2, 8, 9, 10].includes(opcode)) throw protocolError(`Invalid WebSocket opcode: ${opcode}`)
    const control = opcode >= 8
    if (control && (!fin || length > 125)) throw protocolError('Control frames must be final and at most 125 bytes')
    this.payloadOpcode = opcode
    this.headerLength = length < 126 ? 2 : length === 126 ? 4 : 10
  }

  private finishHeader(): void {
    const first = this.header[0] ?? 0
    const fin = (first & 0x80) !== 0
    const opcode = this.payloadOpcode
    let length: number
    const encoded = this.header[1] ?? 0
    if (encoded < 126) length = encoded
    else if (encoded === 126) {
      length = ((this.header[2] ?? 0) << 8) | (this.header[3] ?? 0)
      if (length < 126) throw protocolError('Non-minimal WebSocket payload length encoding')
    }
    else {
      if ((this.header[2] ?? 0) & 0x80) throw protocolError('WebSocket 64-bit payload length has its MSB set')
      let value = 0n
      for (let index = 2; index < 10; index += 1) value = (value << 8n) | BigInt(this.header[index] ?? 0)
      if (value < 65_536n) throw protocolError('Non-minimal WebSocket payload length encoding')
      if (value > BigInt(this.maxMessageBytes)) throw protocolError('WebSocket payload exceeds configured message limit')
      length = Number(value)
    }
    if (opcode >= 8 && length > 125) throw protocolError('Control frame payload exceeds 125 bytes')
    if (opcode === 2) throw protocolError('Binary WebSocket frames are not supported')
    if (opcode === 0 && this.fragmentedOpcode === undefined) throw protocolError('Unexpected WebSocket continuation frame')
    if (opcode === 1 && this.fragmentedOpcode !== undefined) throw protocolError('Nested fragmented WebSocket text message')
    if (opcode === 1 && !fin) this.fragmentedOpcode = 1
    if (opcode === 1 || opcode === 0) {
      if (length > this.maxMessageBytes - this.messageLength) throw protocolError('Fragmented WebSocket text message exceeds configured limit')
      this.payloadKind = 'text'
      this.ensureMessageBuffer()
    } else if (opcode >= 8) {
      this.payloadKind = 'control'
      this.payloadBuffer = new Uint8Array(length)
    } else this.payloadKind = undefined
    this.payloadRemaining = length
    this.payloadRead = 0
  }

  private ensureMessageBuffer(): void {
    if (this.messageBuffer === undefined) this.messageBuffer = new Uint8Array(this.maxMessageBytes)
  }

  private copyPayload(bytes: Uint8Array): void {
    if (this.payloadKind === 'control') this.payloadBuffer?.set(bytes, this.payloadRead)
    else if (this.payloadKind === 'text') this.messageBuffer?.set(bytes, this.messageLength + this.payloadRead)
  }

  private finishFrame(): void {
    const opcode = this.payloadOpcode
    const fin = ((this.header[0] ?? 0) & 0x80) !== 0
    if (this.payloadKind === 'text') {
      this.messageLength += this.payloadRead
      if (fin) {
        const text = decodeUtf8(this.messageBuffer?.subarray(0, this.messageLength) ?? new Uint8Array())
        this.messageLength = 0
        this.fragmentedOpcode = undefined
        try { this.callbacks.onText(text) } catch (error) { throw protocolError(`Text callback failed: ${String(error)}`) }
      }
    } else if (opcode === 9) {
      try { this.callbacks.onPing(this.payloadBuffer ?? new Uint8Array()) } catch (error) { throw protocolError(`Ping callback failed: ${String(error)}`) }
    } else if (opcode === 8) {
      const payload = this.payloadBuffer ?? new Uint8Array()
      let code = 1005
      let reason = ''
      if (payload.byteLength === 1) throw protocolError('WebSocket close payload cannot contain one byte')
      if (payload.byteLength >= 2) {
        code = ((payload[0] ?? 0) << 8) | (payload[1] ?? 0)
        if (!isValidCloseCode(code)) throw protocolError(`Invalid WebSocket close code: ${code}`)
        reason = decodeUtf8(payload.subarray(2))
      }
      this.close()
      try { this.callbacks.onClose(code, reason) } catch (error) { throw protocolError(`Close callback failed: ${String(error)}`) }
    }
    this.headerBytes = 0
    this.headerLength = 0
    this.headerReady = false
    this.payloadRemaining = 0
    this.payloadRead = 0
    this.payloadBuffer = undefined
    this.payloadKind = undefined
  }
}

function decodeUtf8(bytes: Uint8Array): string {
  try { return TEXT_DECODER.decode(bytes) } catch { throw protocolError('Invalid UTF-8 in WebSocket text or close reason') }
}

function isValidCloseCode(code: number): boolean {
  return (code >= 1000 && code <= 1014 && code !== 1004 && code !== 1005 && code !== 1006) || (code >= 3000 && code <= 4999)
}

function protocolError(message: string): WebSocketMonitorProtocolError {
  return new WebSocketMonitorProtocolError(message)
}
