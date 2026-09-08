// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { useEffect, useState } from 'react'
import { useOptionalGateway } from '../app/gatewayContext.js'

export interface ActivityRow {
  id: string; kind: 'shell' | 'watcher' | 'schedule'; title: string; detail: string; state: string
  startedAt: number | null; endedAt: number | null; action: 'stop' | 'pause' | 'cancel' | null
  scope: 'session' | 'workspace'; revision?: string; nextRunAt?: string | null
  lastState?: string | null; lastEndedAt?: number | null; exitCode?: number | null
}
export interface ActivitySnapshot { rows: ActivityRow[]; omitted: number }
export function parseActivity(value: unknown): ActivitySnapshot {
  if (!value || typeof value !== 'object') throw new Error('Invalid activity response')
  const response = value as Record<string, unknown>
  if (response.ok !== true || !Array.isArray(response.rows)) throw new Error(typeof response.error === 'string' ? response.error : 'Activity unavailable')
  const rows = response.rows.map((value: unknown): ActivityRow => {
    if (!value || typeof value !== 'object') throw new Error('Invalid activity row')
    const r = value as Record<string, unknown>
    if (typeof r.id !== 'string' || !r.id || !['shell','watcher','schedule'].includes(String(r.kind)) || typeof r.title !== 'string' || typeof r.state !== 'string' || typeof r.detail !== 'string' || !['session','workspace'].includes(String(r.scope)) || ![null,'stop','pause','cancel'].includes(r.action as null | string)) throw new Error('Invalid activity row')
    for (const key of ['startedAt','endedAt']) if (r[key] !== null && (typeof r[key] !== 'number' || !Number.isFinite(r[key]) || Number(r[key]) < 0)) throw new Error('Invalid activity timestamp')
    return r as unknown as ActivityRow
  })
  return { rows, omitted: typeof response.omitted === 'number' && response.omitted >= 0 ? response.omitted : 0 }
}
export const ACTIVITY_NOTICE_MS = 30_000
export const isActive = (row: ActivityRow) => ['running','watching','cancelling'].includes(row.state)
export function recentOutcomes(rows: ActivityRow[], now: number) {
  return rows.filter(row => {
    const ended = row.kind === 'schedule' ? row.lastEndedAt : row.endedAt
    return typeof ended === 'number' && ended <= now && now - ended < ACTIVITY_NOTICE_MS
  })
}

/** Subscribe before reading; coalesce invalidations without losing one during a request. */
export function useActivity(sessionId: string | null) {
  const gateway = useOptionalGateway()
  const [snapshot, setSnapshot] = useState<{ sessionId: string; value: ActivitySnapshot | null; error: string } | null>(null)
  const [revision, setRevision] = useState(0)
  const [now, setNow] = useState(Date.now)
  useEffect(() => { const timer = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(timer) }, [])
  useEffect(() => {
    if (!gateway?.gw || !sessionId) return
    let active = true, pending = false, dirty = false
    let generation = 0
    const load = async () => {
      if (pending) { dirty = true; return }
      pending = true
      const attempt = generation
      try {
        const value = parseActivity(await gateway.gw.request('background.activity', { session_id: sessionId }))
        if (active && attempt === generation) setSnapshot({ sessionId, value, error: '' })
      } catch (error) {
        if (active && attempt === generation) setSnapshot({ sessionId, value: null, error: error instanceof Error ? error.message : String(error) })
      } finally {
        pending = false
        if (active && dirty) { dirty = false; void load() }
      }
    }
    const refresh = () => { void load() }
    const disconnected = () => { generation++; if (active) setSnapshot({ sessionId, value: null, error: 'Daemon disconnected' }) }
    gateway.gw.on('background_changed', refresh)
    gateway.gw.on('session.info', refresh)
    gateway.gw.on('close', disconnected)
    gateway.gw.on('exit', disconnected)
    void load()
    return () => {
      active = false
      gateway.gw.off('background_changed', refresh)
      gateway.gw.off('session.info', refresh)
      gateway.gw.off('close', disconnected)
      gateway.gw.off('exit', disconnected)
    }
  }, [gateway, sessionId, revision])
  const current = snapshot?.sessionId === sessionId ? snapshot : null
  return { rows: current?.value?.rows ?? [], omitted: current?.value?.omitted ?? 0, error: current?.error ?? '', loading: !current, now, refresh: () => setRevision(n => n + 1), gateway }
}
