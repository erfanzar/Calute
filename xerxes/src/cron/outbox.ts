// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { Database } from 'bun:sqlite'
import { chmodSync, mkdirSync } from 'node:fs'
import { dirname } from 'node:path'
import { processIsAlive } from '../core/processLiveness.js'
import type { DeliverySender, DeliveryTarget } from './delivery.js'

export interface OutboxDelivery {
  id: string; job: string; platform: string; recipient: string; content: string;
  archive: string; state: 'pending' | 'sending' | 'sent' | 'uncertain'; error: string | null; attempts: number
}
/** Durable delivery state; never retries external effects implicitly. */
export class DeliveryOutbox {
  private readonly pid: number
  constructor(readonly path: string, options: { pid?: number; isAlive?: (pid: number) => boolean } = {}) {
    this.pid = options.pid ?? process.pid
    mkdirSync(dirname(path), { recursive: true, mode: 0o700 })
    this.withDb(db => db.exec(`CREATE TABLE IF NOT EXISTS deliveries (
      id TEXT PRIMARY KEY, job TEXT NOT NULL, platform TEXT NOT NULL,
      recipient TEXT NOT NULL, content TEXT NOT NULL, archive TEXT NOT NULL,
      state TEXT NOT NULL, error TEXT, attempts INTEGER NOT NULL DEFAULT 0,
      created INTEGER NOT NULL
    )`))
    this.withDb(db => db.transaction(() => {
      const columns = new Set(db.query<{ name: string }, []>('PRAGMA table_info(deliveries)').all().map(row => row.name))
      if (!columns.has('executor_pid')) db.exec('ALTER TABLE deliveries ADD COLUMN executor_pid INTEGER')
      if (!columns.has('claim_id')) db.exec('ALTER TABLE deliveries ADD COLUMN claim_id TEXT')
      const isAlive = options.isAlive ?? processIsAlive
      for (const row of db.query<{ id: string; executor_pid: number | null }, []>("SELECT id,executor_pid FROM deliveries WHERE state='sending'").all()) {
        if (row.executor_pid !== null && Number.isSafeInteger(row.executor_pid) && row.executor_pid > 0 && !isAlive(row.executor_pid)) {
          db.query("UPDATE deliveries SET state='uncertain', error='Sender process exited; delivery outcome is unknown' WHERE id=? AND state='sending'").run(row.id)
        }
      }
    }).immediate())
    chmodSync(path, 0o600)
  }
  enqueue(job: string, target: DeliveryTarget, content: string, archive: string): string {
    if (!job || !target.platform || new TextEncoder().encode(content).length > 1_000_000) throw new Error('Invalid delivery or output exceeds 1 MB outbox limit')
    return this.withDb(db => db.transaction(() => {
      const count = db.query<{ count: number }, []>("SELECT count(*) AS count FROM deliveries WHERE state != 'sent'").get()!.count
      if (count >= 128) throw new Error('Delivery outbox is full; resolve pending deliveries before adding more')
      const id = crypto.randomUUID()
      db.query('INSERT INTO deliveries(id,job,platform,recipient,content,archive,state,created) VALUES (?,?,?,?,?,?,?,?)').run(id, job, target.platform, target.recipient ?? '', content, archive, 'pending', Date.now())
      return id
    }).immediate())
  }
  inspect(job: string, id: string): OutboxDelivery | null {
    return this.withDb(db => db.query<OutboxDelivery, [string, string]>('SELECT * FROM deliveries WHERE job=? AND id=?').get(job, id))
  }
  list(job: string): Omit<OutboxDelivery, 'content'>[] {
    return this.withDb(db => db.query<Omit<OutboxDelivery, 'content'>, [string]>(
      'SELECT id,job,platform,recipient,archive,state,error,attempts FROM deliveries WHERE job=? ORDER BY created DESC,id DESC LIMIT 228'
    ).all(job))
  }
  /** Explicit operator reconciliation. Active send claims are never reset. */
  reconcile(job: string, id: string, attempts: number, decision: 'sent' | 'retry'): void {
    if (!Number.isSafeInteger(attempts) || attempts < 1 || !['sent', 'retry'].includes(decision)) throw new Error('Invalid delivery reconciliation')
    this.withDb(db => db.transaction(() => {
      const result = db.query("UPDATE deliveries SET state=?, error=NULL WHERE job=? AND id=? AND attempts=? AND state='uncertain'")
        .run(decision === 'sent' ? 'sent' : 'pending', job, id, attempts)
      if (!result.changes) throw new Error('Delivery changed or is not awaiting reconciliation')
      db.exec("DELETE FROM deliveries WHERE state='sent' AND id NOT IN (SELECT id FROM deliveries WHERE state='sent' ORDER BY created DESC,id DESC LIMIT 100)")
    }).immediate())
  }
  async send(job: string, id: string, sender: DeliverySender): Promise<void> {
    const claim = crypto.randomUUID()
    const delivery = this.withDb(db => db.transaction(() => {
      const row = db.query<OutboxDelivery, [string, string]>('SELECT * FROM deliveries WHERE job=? AND id=?').get(job, id)
      if (!row) throw new Error('Unknown delivery')
      if (row.state === 'sent') return null
      if (row.state !== 'pending') throw new Error('Delivery may already have been sent; reconcile it before retrying')
      db.query("UPDATE deliveries SET state='sending', attempts=attempts+1, executor_pid=?, claim_id=? WHERE id=?").run(this.pid, claim, id)
      return row
    }).immediate())
    if (!delivery) return
    try {
      await sender(delivery.platform, delivery.recipient, delivery.content)
    } catch (error) {
      this.withDb(db => db.query("UPDATE deliveries SET state='uncertain', error=? WHERE id=? AND state='sending' AND claim_id=?").run(String(error).slice(0, 4000), id, claim))
      throw error
    }
    this.withDb(db => db.transaction(() => {
      const settled = db.query("UPDATE deliveries SET state='sent', error=NULL WHERE id=? AND state='sending' AND claim_id=?").run(id, claim)
      if (!settled.changes) throw new Error('Delivery claim changed before completion; outcome requires reconciliation')
      // Keep bounded delivery receipts. Pending/uncertain payloads are never pruned.
      db.exec("DELETE FROM deliveries WHERE state='sent' AND id NOT IN (SELECT id FROM deliveries WHERE state='sent' ORDER BY created DESC,id DESC LIMIT 100)")
    }).immediate())
  }
  private withDb<T>(operation: (db: Database) => T): T {
    const db = new Database(this.path, { create: true, strict: true })
    try { db.exec('PRAGMA busy_timeout=0'); return operation(db) } finally { db.close() }
  }
}
