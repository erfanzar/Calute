// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { Database } from 'bun:sqlite'
import { mkdirSync } from 'node:fs'
import { dirname } from 'node:path'
import { parseAgentIntelligenceConfig, type AgentIntelligenceConfig } from './intelligence.js'
export interface RoutingNote { provider_profile: string; model: string; note: string; revision: number }

function checkedRevision(value: unknown): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1) throw new Error('Invalid persisted settings revision')
  return value
}
function checkedRoutingNote(row: Record<string, unknown>): RoutingNote {
  const { provider_profile: profile, model, note } = row
  if (typeof profile !== 'string' || !profile || profile.length > 512 || profile.trim() !== profile
    || typeof model !== 'string' || model.length > 512 || model.trim() !== model
    || typeof note !== 'string' || note.length > 2000 || note.trim() !== note) throw new Error('Invalid persisted routing note')
  return { provider_profile: profile, model, note, revision: checkedRevision(row.revision) }
}

/** Atomic settings shared by daemon instances; callers own provider validation. */
export class AgentSettingsStore {
  constructor(readonly path: string) {}
  private open(): Database {
    mkdirSync(dirname(this.path), { recursive: true, mode: 0o700 })
    const db = new Database(this.path)
    db.exec('PRAGMA busy_timeout=1000; CREATE TABLE IF NOT EXISTS agent_settings (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL, settings TEXT NOT NULL)')
    db.exec('CREATE TABLE IF NOT EXISTS routing_notes (provider_profile TEXT NOT NULL, model TEXT NOT NULL, note TEXT NOT NULL, revision INTEGER NOT NULL, PRIMARY KEY(provider_profile,model))')
    return db
  }
  read(): { revision: number; settings?: AgentIntelligenceConfig } {
    const db = this.open()
    try {
      const row = db.query<{ revision: number; settings: string }, []>('SELECT revision,settings FROM agent_settings WHERE id=1').get()
      return row ? { revision: checkedRevision(row.revision), settings: parseAgentIntelligenceConfig(JSON.parse(row.settings)) } : { revision: 0 }
    } finally { db.close() }
  }
  routingNotes(): RoutingNote[] {
    const db = this.open()
    try {
      const rows = db.query<Record<string, unknown>, []>('SELECT provider_profile,model,note,revision FROM routing_notes ORDER BY provider_profile,model LIMIT 501').all()
      if (rows.length > 500) throw new Error('Persisted routing note limit exceeded')
      return rows.map(checkedRoutingNote)
    }
    finally { db.close() }
  }
  saveRoutingNote(profile: string, model: string, note: string, revision: number): RoutingNote {
    if (typeof profile !== 'string' || !profile.trim() || profile.length > 512 || typeof model !== 'string' || model.length > 512
      || typeof note !== 'string' || note.length > 2000 || !Number.isSafeInteger(revision) || revision < 0 || revision === Number.MAX_SAFE_INTEGER) throw new Error('Invalid routing note')
    profile = profile.trim(); model = model.trim(); note = note.trim()
    const db = this.open()
    try {
      return db.transaction(() => {
        const stored = db.query<Record<string, unknown>, [string, string]>('SELECT provider_profile,model,note,revision FROM routing_notes WHERE provider_profile=? AND model=?').get(profile, model)
        const previous = stored ? checkedRoutingNote(stored) : undefined
        if ((previous?.revision ?? 0) !== revision) throw new Error('Routing note changed; reload before saving')
        if (!previous && db.query<{ count: number }, []>('SELECT COUNT(*) AS count FROM routing_notes').get()!.count >= 500) throw new Error('Routing note limit reached')
        const result = { provider_profile: profile, model, note, revision: revision + 1 }
        // Retain an empty tombstone so clearing a note cannot revive a stale editor.
        db.query('INSERT INTO routing_notes VALUES(?,?,?,?) ON CONFLICT(provider_profile,model) DO UPDATE SET note=excluded.note,revision=excluded.revision')
          .run(profile, model, note, result.revision)
        return result
      }).immediate()
    } finally { db.close() }
  }
  save(value: unknown, revision: number): { revision: number; settings: AgentIntelligenceConfig } {
    const settings = parseAgentIntelligenceConfig(value)
    if (!Number.isSafeInteger(revision) || revision < 0 || revision === Number.MAX_SAFE_INTEGER) throw new Error('Invalid settings revision')
    const db = this.open()
    try {
      return db.transaction(() => {
        const previous = db.query<{ revision: unknown }, []>('SELECT revision FROM agent_settings WHERE id=1').get()
        const old = previous ? checkedRevision(previous.revision) : 0
        if (old !== revision) throw new Error('Agent settings changed; reload before saving')
        db.query('INSERT INTO agent_settings VALUES(1,?,?) ON CONFLICT(id) DO UPDATE SET revision=excluded.revision,settings=excluded.settings').run(old + 1, JSON.stringify(settings))
        return { revision: old + 1, settings }
      }).immediate()
    } finally { db.close() }
  }
}
