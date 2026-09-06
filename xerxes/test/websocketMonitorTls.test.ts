// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { fileURLToPath } from 'node:url'

const xerxesDirectory = fileURLToPath(new URL('..', import.meta.url))
const certificatePath = fileURLToPath(new URL('./fixtures/websocket-localhost-cert.pem', import.meta.url))
const privateKeyPath = fileURLToPath(new URL('./fixtures/websocket-localhost-key.pem', import.meta.url))

test('wss monitor rejects a self-signed certificate even when NODE_TLS_REJECT_UNAUTHORIZED is disabled', async () => {
  const server = Bun.serve<undefined>({
    port: 0,
    tls: {
      cert: await Bun.file(certificatePath).text(),
      key: await Bun.file(privateKeyPath).text(),
    },
    fetch(_request, server) {
      if (server.upgrade(_request, { data: undefined })) return
      return new Response('websocket only', { status: 426 })
    },
    websocket: { open() {}, message() {} },
  })

  const url = `wss://127.0.0.1:${server.port}/events`
  const childScript = `
    const { nativeWebSocketMonitorSource } = await import('./src/runtime/websocketMonitorSource.ts')
    try {
      const connection = await nativeWebSocketMonitorSource.open(process.env.XERXES_TEST_WSS_URL ?? '', () => {}, () => {}, () => {})
      connection.close()
      console.log('accepted')
    } catch (error) {
      console.log('rejected:' + String(error))
    }
  `

  let child: Bun.Subprocess<'ignore', 'pipe', 'pipe'> | undefined
  let killTimer: ReturnType<typeof setTimeout> | undefined
  try {
    child = Bun.spawn([process.execPath, '-e', childScript], {
      cwd: xerxesDirectory,
      env: { ...process.env, NODE_TLS_REJECT_UNAUTHORIZED: '0', XERXES_TEST_WSS_URL: url },
      stdin: 'ignore',
      stdout: 'pipe',
      stderr: 'pipe',
    })
    killTimer = setTimeout(() => { try { child?.kill() } catch {} }, 5_000)
    const [output, errorOutput, exitCode] = await Promise.all([
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
      child.exited,
    ])
    expect(exitCode).toBe(0)
    expect(`${output}\n${errorOutput}`).toMatch(/rejected:/)
    expect(`${output}\n${errorOutput}`).toMatch(/certificate|TLS|verify|self-signed/i)
    expect(output).not.toContain('accepted')
  } finally {
    if (killTimer !== undefined) clearTimeout(killTimer)
    try { child?.kill() } catch {}
    server.stop(true)
  }
}, { timeout: 10_000 })
