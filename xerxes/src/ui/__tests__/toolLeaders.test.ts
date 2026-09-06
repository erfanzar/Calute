// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'
import { isQuietToolName } from '../opentui/messageLine.js'

describe('quiet read-only calls', () => {
  it('tints display names of read-only tools faint', () => {
    for (const name of ['Read File', 'Glob', 'Grep', 'List', 'View']) {
      expect(isQuietToolName(name)).toBe(true)
    }
  })

  it('leaves mutating and network calls at full outcome colour', () => {
    for (const name of ['Bash', 'Edit', 'Write', 'WebFetch', 'Browser Click']) {
      expect(isQuietToolName(name)).toBe(false)
    }
  })
})
