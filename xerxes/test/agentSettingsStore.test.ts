// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { Database } from 'bun:sqlite'
import { expect, test } from 'bun:test'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { AgentSettingsStore } from '../src/agents/settingsStore.js'
test('agent settings persist and reject stale edits without overwriting newer settings', () => {
  const directory = mkdtempSync(join(tmpdir(), 'agent-settings-'))
  try {
    const store = new AgentSettingsStore(join(directory, 'settings.sqlite'))
    expect(store.read()).toEqual({ revision: 0 })
    store.save({ smart: { model: 'deep', provider_profile: 'work', reasoning_effort: 'high' } }, 0)
    const reopened = new AgentSettingsStore(store.path)
    expect(reopened.read()).toMatchObject({ revision: 1, settings: { smart: { model: 'deep' } } })
    expect(() => reopened.save({ light: 'fast' }, 0)).toThrow('changed')
    expect(reopened.read().settings?.smart).toBeDefined()
    expect(() => reopened.save({ smart: { model: '' } }, 1)).toThrow()
    expect(reopened.read().revision).toBe(1)
  } finally { rmSync(directory, { recursive: true, force: true }) }
})

test('routing notes are scoped, revision guarded, and clearing retains its revision', () => {
  const directory = mkdtempSync(join(tmpdir(), 'routing-notes-'))
  try {
    const path = join(directory, 'settings.sqlite')
    const store = new AgentSettingsStore(path)
    const note = store.saveRoutingNote('provider', 'model', 'Use for difficult debugging', 0)
    expect(note.revision).toBe(1)
    expect(new AgentSettingsStore(path).routingNotes()).toEqual([note])
    expect(() => store.saveRoutingNote('provider', 'model', 'Stale', 0)).toThrow('changed')
    store.saveRoutingNote('provider', '', 'Prefer for routine work', 0)
    store.saveRoutingNote('provider', 'model', '', 1)
    expect(() => store.saveRoutingNote('provider', 'model', 'Stale', 0)).toThrow('changed')
    expect(store.routingNotes()).toHaveLength(2)
    expect(() => store.saveRoutingNote('provider', 'model', 'x'.repeat(2001), 2)).toThrow('Invalid')
  } finally { rmSync(directory, { recursive: true, force: true }) }
})

test('persisted routing note corruption fails explicitly and cannot be overwritten as a fresh note', () => {
  const directory = mkdtempSync(join(tmpdir(), 'corrupt-routing-notes-'))
  try {
    const store = new AgentSettingsStore(join(directory, 'settings.sqlite'))
    store.saveRoutingNote('profile', 'model', 'valid', 0)
    const db = new Database(store.path)
    try {
      for (const revision of [0, -1, 1.5, 'invalid']) {
        db.query('UPDATE routing_notes SET revision=?').run(revision)
        expect(() => store.routingNotes()).toThrow('persisted settings revision')
        expect(() => store.saveRoutingNote('profile', 'model', 'replacement', 0)).toThrow('persisted settings revision')
      }
      db.query('UPDATE routing_notes SET revision=1,note=?').run('x'.repeat(2001))
      expect(() => store.routingNotes()).toThrow('persisted routing note')
      expect(() => store.saveRoutingNote('profile', 'model', 'replacement', 1)).toThrow('persisted routing note')
    } finally { db.close() }
  } finally { rmSync(directory, { recursive: true, force: true }) }
})

test('settings revisions cannot overflow or treat a corrupted revision as an empty store', () => {
  const directory = mkdtempSync(join(tmpdir(), 'corrupt-settings-'))
  try {
    const store = new AgentSettingsStore(join(directory, 'settings.sqlite'))
    store.save({ light: 'small' }, 0)
    const db = new Database(store.path)
    try {
      db.query('UPDATE agent_settings SET revision=?').run(Number.MAX_SAFE_INTEGER)
      expect(store.read().revision).toBe(Number.MAX_SAFE_INTEGER)
      expect(() => store.save({ light: 'other' }, Number.MAX_SAFE_INTEGER)).toThrow('Invalid settings revision')
      db.query('UPDATE agent_settings SET revision=0').run()
      expect(() => store.read()).toThrow('persisted settings revision')
      expect(() => store.save({ light: 'other' }, 0)).toThrow('persisted settings revision')
    } finally { db.close() }
  } finally { rmSync(directory, { recursive: true, force: true }) }
})

test('routing-note reads reject an oversized persisted collection', () => {
  const directory = mkdtempSync(join(tmpdir(), 'routing-note-count-'))
  try {
    const store = new AgentSettingsStore(join(directory, 'settings.sqlite'))
    store.routingNotes()
    const db = new Database(store.path)
    try {
      db.transaction(() => { for (let i = 0; i < 501; i++) db.query('INSERT INTO routing_notes VALUES(?,?,?,?)').run('profile', `model-${i}`, '', 1) })()
      expect(() => store.routingNotes()).toThrow('limit exceeded')
    } finally { db.close() }
  } finally { rmSync(directory, { recursive: true, force: true }) }
})
