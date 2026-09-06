// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { MCPServerConfig } from './types.js'

/** Parameters for the MCP connection retry schedule. Delays are measured in seconds. */
export interface ReconnectPolicyOptions {
  readonly baseSeconds?: number
  readonly factor?: number
  readonly maxAttempts?: number
  readonly maxSeconds?: number
}

/** Exponential-backoff policy used when an MCP server disconnects. */
export class ReconnectPolicy {
  readonly baseSeconds: number
  readonly factor: number
  readonly maxAttempts: number
  readonly maxSeconds: number

  constructor(options: ReconnectPolicyOptions = {}) {
    this.maxAttempts = positiveInteger(options.maxAttempts ?? 5, 'maxAttempts')
    this.baseSeconds = positiveNumber(options.baseSeconds ?? 1, 'baseSeconds')
    this.factor = positiveNumber(options.factor ?? 2, 'factor')
    this.maxSeconds = positiveNumber(options.maxSeconds ?? 60, 'maxSeconds')
  }

  /** Return the delay after a one-based failed connection attempt. */
  delayForAttempt(attempt: number): number {
    const normalizedAttempt = Math.max(1, Math.floor(attempt))
    return Math.min(this.maxSeconds, this.baseSeconds * this.factor ** (normalizedAttempt - 1))
  }
}

/** Raised after the final MCP connection retry fails. Its message is always credential-scrubbed. */
export class MCPReconnectError extends Error {
  readonly attempts: number

  constructor(message: string, attempts: number) {
    super(scrubCredentials(message))
    this.name = new.target.name
    this.attempts = attempts
  }
}

export interface ReconnectWithBackoffOptions {
  /** Stop future attempts and interrupt backoff; an active connect still settles normally. */
  readonly signal?: AbortSignal
  /** Called after each failed attempt, before the next delay. */
  readonly onError?: (attempt: number, error: unknown) => void | Promise<void>
  readonly policy?: ReconnectPolicy | ReconnectPolicyOptions
  /** Injectable sleep implementation. It receives seconds, not milliseconds. */
  readonly sleep?: (seconds: number) => void | Promise<void>
}

/**
 * Retry a connection operation with exponential backoff.
 *
 * The retry hook receives the original error so hosts can apply their own
 * observability policy. The thrown terminal error only exposes a redacted
 * message so credentials from subprocess or HTTP diagnostics cannot escape
 * through the normal lifecycle API.
 */
export async function reconnectWithBackoff<T>(
  connect: () => T | Promise<T>,
  options: ReconnectWithBackoffOptions = {},
): Promise<T> {
  const policy = options.policy instanceof ReconnectPolicy
    ? options.policy
    : new ReconnectPolicy(options.policy)
  let lastError: unknown

  for (let attempt = 1; attempt <= policy.maxAttempts; attempt += 1) {
    throwIfCancelled(options.signal)
    try {
      return await connect()
    } catch (error) {
      throwIfCancelled(options.signal)
      lastError = error
      await options.onError?.(attempt, error)
      throwIfCancelled(options.signal)
      if (attempt >= policy.maxAttempts) {
        break
      }
      await waitForRetry(policy.delayForAttempt(attempt), options)
    }
  }

  throw new MCPReconnectError(errorMessage(lastError), policy.maxAttempts)
}

/**
 * Replace common API key, bearer token, and password fragments with a safe
 * marker. Callers can also pass literal secret values (for example configured
 * env or header values) that are redacted verbatim wherever they appear.
 */
export function scrubCredentials(text: string, secrets: readonly string[] = []): string {
  let scrubbed = text
    .replace(/\b(password)\b\s*(?::|=|\s)\s*['"]?([^\s'";,]+)/gi, '$1=[redacted]')
    .replace(/\b(api[_-]?key)\b\s*(?::|=|\s)\s*['"]?([A-Za-z0-9._-]{8,})/gi, '$1=[redacted]')
    .replace(/\b(token)\b\s*(?::|=|\s)\s*['"]?([A-Za-z0-9._-]{16,})/gi, '$1=[redacted]')
    .replace(/\b(authorization\s*:\s*bearer)\s+([A-Za-z0-9._-]+)/gi, '$1=[redacted]')
    .replace(/\bsk-[A-Za-z0-9_-]{16,}\b/g, '[redacted]')
  for (const secret of secrets) {
    if (secret) {
      scrubbed = scrubbed.replaceAll(secret, '[redacted]')
    }
  }
  return scrubbed
}

/**
 * Collect configured env and header values so lifecycle and subprocess
 * diagnostics can redact them verbatim. Values shorter than four characters
 * are skipped because they are too likely to collide with ordinary text.
 */
export function mcpConfigSecrets(config: MCPServerConfig): readonly string[] {
  const secrets: string[] = []
  for (const record of [config.env, config.headers]) {
    for (const value of Object.values(record ?? {})) {
      if (value.length >= 4 && !secrets.includes(value)) {
        secrets.push(value)
      }
    }
  }
  return secrets
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message
  }
  return String(error)
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 1) {
    throw new RangeError(name + ' must be an integer of at least 1')
  }
  return value
}

function positiveNumber(value: number, name: string): number {
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError(name + ' must be positive')
  }
  return value
}

function throwIfCancelled(signal?: AbortSignal): void {
  // Do not expose arbitrary host-provided abort reasons through diagnostics.
  if (signal?.aborted) throw new DOMException('MCP reconnect cancelled', 'AbortError')
}

function waitForRetry(seconds: number, options: ReconnectWithBackoffOptions): Promise<void> {
  return new Promise((resolve, reject) => {
    const signal = options.signal
    let timer: ReturnType<typeof setTimeout> | undefined
    const finish = (error?: unknown) => {
      if (timer !== undefined) clearTimeout(timer)
      signal?.removeEventListener('abort', abort)
      if (error !== undefined) reject(error)
      else resolve()
    }
    const abort = () => finish(new DOMException('MCP reconnect cancelled', 'AbortError'))
    signal?.addEventListener('abort', abort, { once: true })
    if (signal?.aborted) { abort(); return }
    if (options.sleep) {
      // Observe custom sleepers even if cancellation wins; hosts own their timers.
      Promise.resolve().then(() => options.sleep!(seconds)).then(() => finish(), finish)
    } else {
      timer = setTimeout(() => finish(), seconds * 1_000)
    }
  })
}
