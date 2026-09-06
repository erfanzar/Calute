// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export type ContextControlScope = 'global' | 'project'

export interface ContextControlPin {
  readonly scope: ContextControlScope
  readonly path: string
  readonly content: string
}

export interface ContextControlExclusion {
  readonly scope: ContextControlScope
  readonly path: string
}

export interface ContextControls {
  readonly version: 1
  readonly revision: number
  readonly pins: readonly ContextControlPin[]
  readonly excluded: readonly ContextControlExclusion[]
}

type ControlPatchAction = 'pin' | 'unpin' | 'exclude' | 'include'

interface ParsedPatch {
  readonly action: ControlPatchAction
  readonly revision: number
  readonly scope: ContextControlScope
  readonly path: string
  readonly content?: string
}

const MAX_PINS = 16
const MAX_PIN_BYTES = 8_000
const MAX_TOTAL_PIN_BYTES = 32_000
const MAX_EXCLUSIONS = 128

/** Read validated per-session context controls from metadata. */
export function readContextControls(metadata: Readonly<Record<string, unknown>>): ContextControls {
  if (!hasOwn(metadata, 'context_controls')) return emptyControls()
  return parseControls(metadata.context_controls, 'metadata.context_controls')
}

/** Apply one optimistic-concurrency context-control patch without mutating the current value. */
export function updateContextControls(
  current: ContextControls,
  patch: Readonly<Record<string, unknown>>,
): ContextControls {
  const base = parseControls(current, 'current context controls')
  const parsed = parsePatch(patch)
  if (parsed.revision !== base.revision) {
    throw new Error(
      `Context controls revision mismatch: expected ${base.revision}, received ${parsed.revision}`,
    )
  }
  if (base.revision >= Number.MAX_SAFE_INTEGER) {
    throw new Error('Context controls revision cannot be incremented safely')
  }

  const pins = base.pins.map(pin => ({ ...pin }))
  const excluded = base.excluded.map(entry => ({ ...entry }))
  const matchingPin = pins.findIndex(pin => samePath(pin, parsed))
  const matchingExclusion = excluded.findIndex(entry => samePath(entry, parsed))

  switch (parsed.action) {
    case 'pin': {
      const replacement: ContextControlPin = {
        scope: parsed.scope,
        path: parsed.path,
        content: parsed.content!,
      }
      if (matchingPin === -1) pins.push(replacement)
      else pins[matchingPin] = replacement
      if (matchingExclusion !== -1) excluded.splice(matchingExclusion, 1)
      break
    }
    case 'unpin':
      if (matchingPin !== -1) pins.splice(matchingPin, 1)
      break
    case 'exclude':
      if (matchingPin !== -1) pins.splice(matchingPin, 1)
      if (matchingExclusion === -1) excluded.push({ scope: parsed.scope, path: parsed.path })
      break
    case 'include':
      if (matchingExclusion !== -1) excluded.splice(matchingExclusion, 1)
      break
  }

  validateLimits(pins, excluded, 'updated context controls')
  return makeControls(base.revision + 1, pins, excluded)
}

function parseControls(value: unknown, label: string): ContextControls {
  if (!isRecord(value)) throw invalid(label, 'must be an object')
  assertKnownKeys(value, ['version', 'revision', 'pins', 'excluded'], label)
  if (value.version !== 1) throw invalid(`${label}.version`, 'must be 1')
  const revision = parseRevision(value.revision, `${label}.revision`)
  if (!Array.isArray(value.pins)) throw invalid(`${label}.pins`, 'must be an array')
  if (!Array.isArray(value.excluded)) throw invalid(`${label}.excluded`, 'must be an array')

  const pins: ContextControlPin[] = value.pins.map((entry, index) => parsePin(entry, `${label}.pins[${index}]`))
  const excluded: ContextControlExclusion[] = value.excluded.map((entry, index) => parseExclusion(entry, `${label}.excluded[${index}]`))
  validateUnique(pins, `${label}.pins`)
  validateUnique(excluded, `${label}.excluded`)
  if (pins.some(pin => excluded.some(entry => samePath(pin, entry)))) {
    throw invalid(label, 'cannot contain both a pin and exclusion for the same scope/path')
  }
  validateLimits(pins, excluded, label)
  return makeControls(revision, pins, excluded)
}

function parsePin(value: unknown, label: string): ContextControlPin {
  if (!isRecord(value)) throw invalid(label, 'must be an object')
  assertKnownKeys(value, ['scope', 'path', 'content'], label)
  const scope = parseScope(value.scope, `${label}.scope`)
  const path = normalizePath(value.path, `${label}.path`)
  if (typeof value.content !== 'string') throw invalid(`${label}.content`, 'must be a string')
  return { scope, path, content: value.content }
}

function parseExclusion(value: unknown, label: string): ContextControlExclusion {
  if (!isRecord(value)) throw invalid(label, 'must be an object')
  assertKnownKeys(value, ['scope', 'path'], label)
  return {
    scope: parseScope(value.scope, `${label}.scope`),
    path: normalizePath(value.path, `${label}.path`),
  }
}

function parsePatch(value: Readonly<Record<string, unknown>>): ParsedPatch {
  if (!isRecord(value)) throw invalid('context control patch', 'must be an object')
  const action = value.action
  if (action !== 'pin' && action !== 'unpin' && action !== 'exclude' && action !== 'include') {
    throw invalid('context control patch.action', 'must be pin, unpin, exclude, or include')
  }
  const allowed = action === 'pin'
    ? ['action', 'revision', 'scope', 'path', 'content']
    : ['action', 'revision', 'scope', 'path']
  assertKnownKeys(value, allowed, 'context control patch')
  const revision = parseRevision(value.revision, 'context control patch.revision')
  const parsed: ParsedPatch = {
    action,
    revision,
    scope: parseScope(value.scope, 'context control patch.scope'),
    path: normalizePath(value.path, 'context control patch.path'),
    ...(action === 'pin' ? { content: parseContent(value.content, 'context control patch.content') } : {}),
  }
  return parsed
}

function parseRevision(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) {
    throw invalid(label, 'must be a non-negative safe integer')
  }
  return value
}

function parseContent(value: unknown, label: string): string {
  if (typeof value !== 'string') throw invalid(label, 'must be a string')
  if (utf8Bytes(value) > MAX_PIN_BYTES) {
    throw new Error(`${label} exceeds the ${MAX_PIN_BYTES}-byte limit`)
  }
  return value
}

function parseScope(value: unknown, label: string): ContextControlScope {
  if (value === 'global' || value === 'project') return value
  throw invalid(label, 'must be global or project')
}

function normalizePath(value: unknown, label: string): string {
  if (typeof value !== 'string' || value.length === 0 || utf8Bytes(value) > 1024) throw invalid(label, 'must be a non-empty relative .md path of at most 1024 UTF-8 bytes')
  if (/[\\\u0000-\u001f\u007f-\u009f]/u.test(value)) {
    throw invalid(label, 'must not contain backslashes or control characters')
  }
  if (value.startsWith('/') || /^[A-Za-z]:/.test(value)) {
    throw invalid(label, 'must be relative')
  }
  if (!value.endsWith('.md')) throw invalid(label, 'must end with .md')
  const segments = value.split('/')
  if (segments.some(segment => segment === '' || segment === '.' || segment === '..')) {
    throw invalid(label, 'must be normalized and must not traverse directories')
  }
  return value
}

function validateLimits(
  pins: readonly ContextControlPin[],
  excluded: readonly ContextControlExclusion[],
  label: string,
): void {
  if (pins.length > MAX_PINS) throw new Error(`${label} exceeds the ${MAX_PINS}-pin limit`)
  if (excluded.length > MAX_EXCLUSIONS) throw new Error(`${label} exceeds the ${MAX_EXCLUSIONS}-exclusion limit`)
  const totalBytes = pins.reduce((total, pin) => total + utf8Bytes(pin.content), 0)
  if (pins.some(pin => utf8Bytes(pin.content) > MAX_PIN_BYTES)) {
    throw new Error(`${label} contains a pin exceeding the ${MAX_PIN_BYTES}-byte limit`)
  }
  if (totalBytes > MAX_TOTAL_PIN_BYTES) {
    throw new Error(`${label} exceeds the ${MAX_TOTAL_PIN_BYTES}-byte total pin limit`)
  }
}

function validateUnique(entries: readonly (ContextControlPin | ContextControlExclusion)[], label: string): void {
  const seen = new Set<string>()
  for (const entry of entries) {
    const key = controlKey(entry.scope, entry.path)
    if (seen.has(key)) throw invalid(label, `contains duplicate ${entry.scope}:${entry.path}`)
    seen.add(key)
  }
}

function makeControls(
  revision: number,
  pins: readonly ContextControlPin[],
  excluded: readonly ContextControlExclusion[],
): ContextControls {
  return Object.freeze({
    version: 1 as const,
    revision,
    pins: Object.freeze(pins.map(pin => Object.freeze({ ...pin }))),
    excluded: Object.freeze(excluded.map(entry => Object.freeze({ ...entry }))),
  })
}

function emptyControls(): ContextControls {
  return makeControls(0, [], [])
}

function samePath(left: { readonly scope: ContextControlScope; readonly path: string }, right: { readonly scope: ContextControlScope; readonly path: string }): boolean {
  return left.scope === right.scope && left.path === right.path
}

function controlKey(scope: ContextControlScope, path: string): string {
  return `${scope}\u0000${path}`
}

function utf8Bytes(value: string): number {
  return new TextEncoder().encode(value).byteLength
}

function assertKnownKeys(value: Readonly<Record<string, unknown>>, keys: readonly string[], label: string): void {
  const allowed = new Set(keys)
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) throw invalid(label, `contains unknown field ${JSON.stringify(key)}`)
  }
}

function invalid(label: string, detail: string): Error {
  return new Error(`Invalid ${label}: ${detail}`)
}

function hasOwn(value: Readonly<Record<string, unknown>>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key)
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
