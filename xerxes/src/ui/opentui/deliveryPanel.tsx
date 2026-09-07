// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useKeyboard } from '@opentui/react'
import type { ScrollBoxRenderable } from '@opentui/core'
import { useEffect, useRef, useState } from 'react'
import { useOptionalGateway } from '../app/gatewayContext.js'
import type { Theme } from '../theme.js'
import { Box, Text } from './primitives.js'
import { DialogHeader, DialogFooter } from './dialogChrome.js'
type Delivery = { id: string; platform: string; recipient: string; state: string; attempts: number; content?: string; error?: string | null }
function parse(value: unknown): Delivery {
  if (!value || typeof value !== 'object') throw new Error('Invalid delivery response')
  const row = value as Record<string, unknown>
  if (typeof row.id !== 'string' || typeof row.platform !== 'string' || typeof row.recipient !== 'string' || typeof row.attempts !== 'number' || !Number.isSafeInteger(row.attempts) || typeof row.state !== 'string' || !['pending', 'sending', 'sent', 'uncertain'].includes(row.state)) throw new Error('Invalid delivery record')
  return { id: row.id, platform: row.platform, recipient: row.recipient, state: row.state, attempts: row.attempts, content: typeof row.content === 'string' ? row.content : '', error: typeof row.error === 'string' ? row.error : null }
}
export function DeliveryPanel({ t, scheduleId, onClose }: { t: Theme; scheduleId: string; onClose: () => void }) {
  const gateway = useOptionalGateway()
  const [rows, setRows] = useState<Delivery[]>([])
  const [selected, setSelected] = useState('')
  const [detail, setDetail] = useState<Delivery | null>(null)
  const [refresh, setRefresh] = useState(0)
  const [error, setError] = useState('')
  const [confirm, setConfirm] = useState<'sent' | 'retry' | null>(null)
  const [busy, setBusy] = useState(false)
  const pending = useRef(false)
  const alive = useRef(true)
  const scroll = useRef<ScrollBoxRenderable | null>(null)
  useEffect(() => { alive.current = true; return () => { alive.current = false } }, [])
  useEffect(() => {
    let current = true
    if (gateway) void gateway.rpc<{ ok: boolean; deliveries?: unknown; error?: string }>('schedule.deliveries', { schedule_id: scheduleId }).then(result => {
      if (!result?.ok || !Array.isArray(result.deliveries)) throw new Error(result?.error || 'Deliveries unavailable')
      const values = result.deliveries.map(parse)
      if (current) { setRows(values); setSelected(id => values.some(row => row.id === id) ? id : values[0]?.id ?? '') }
    }).catch(failure => { if (current) setError(String(failure)) })
    return () => { current = false }
  }, [gateway, scheduleId, refresh])
  useEffect(() => {
    let current = true
    setDetail(null); setConfirm(null); scroll.current?.scrollTo(0)
    if (gateway && selected) void gateway.rpc<{ ok: boolean; delivery?: unknown; error?: string }>('schedule.delivery.inspect', { schedule_id: scheduleId, delivery_id: selected }).then(result => {
      if (!result?.ok) throw new Error(result?.error || 'Delivery unavailable')
      const value = parse(result.delivery)
      if (value.id !== selected) throw new Error('Delivery identity changed')
      if (current) setDetail(value)
    }).catch(failure => { if (current) setError(String(failure)) })
    return () => { current = false }
  }, [gateway, scheduleId, selected, refresh])
  const submit = (decision?: 'sent' | 'retry') => {
    if (!gateway || !detail || pending.current) return
    pending.current = true; setBusy(true); setConfirm(null)
    void gateway.rpc<{ ok: boolean; error?: string }>(decision ? 'schedule.delivery.resolve' : 'schedule.delivery.send', { schedule_id: scheduleId, delivery_id: detail.id, ...(decision ? { decision, attempts: detail.attempts } : {}) }).then(result => {
      if (!result?.ok) throw new Error(result?.error || 'Delivery action failed')
    }).catch(failure => { if (alive.current) setError(String(failure)) })
      .finally(() => { pending.current = false; if (alive.current) { setBusy(false); setRefresh(value => value + 1) } })
  }
  useKeyboard(key => {
    if (key.eventType === 'release' || !['escape', 'return', 'up', 'down', 's', 'a', 't', 'r', 'pageup', 'pagedown'].includes(key.name)) return
    key.preventDefault(); key.stopPropagation()
    if (key.name === 'escape') { if (confirm) setConfirm(null); else onClose(); return }
    if (busy) return
    if (confirm) { if (key.name === 'return') submit(confirm); return }
    if (key.name === 's' && detail?.state === 'pending') submit()
    else if (key.name === 'a' && detail?.state === 'uncertain') setConfirm('sent')
    else if (key.name === 't' && detail?.state === 'uncertain') setConfirm('retry')
    else if (key.name === 'r') { setError(''); setRefresh(value => value + 1) }
    else if (key.name === 'up' || key.name === 'down') {
      const index = rows.findIndex(row => row.id === selected)
      setSelected(rows[Math.max(0, Math.min(rows.length - 1, index + (key.name === 'up' ? -1 : 1)))]?.id ?? '')
    } else if (key.name === 'pageup' || key.name === 'pagedown') scroll.current?.scrollBy(key.name === 'pageup' ? -10 : 10)
  })
  return <Box flexDirection="column" flexGrow={1} minHeight={0}>
    <DialogHeader t={t} title={<> Deliveries · {Math.max(0, rows.findIndex(row => row.id === selected) + 1)}/{rows.length}</>} />
    {error ? <Text color={t.color.warn} wrap="wrap">{error}</Text> : null}
    <scrollbox ref={scroll} style={{ flexGrow: 1, minHeight: 0 }} contentOptions={{ flexDirection: 'column' }}>
      {detail ? <>
        <Text wrap="wrap">{detail.platform} → {detail.recipient || '(default destination)'}</Text>
        <Text wrap="wrap">{detail.state} · {detail.attempts} attempts</Text>
        {detail.error ? <Text color={t.color.warn} wrap="wrap">{detail.error}</Text> : null}
        <Text wrap="wrap">{detail.content}</Text>
      </> : <Text>No delivery selected.</Text>}
    </scrollbox>
    {confirm ? <Text color={t.color.warn} wrap="wrap">{confirm === 'sent' ? 'Confirm you verified this reached the destination.' : 'Allow retry after checking the destination. A duplicate message is possible.'} Enter confirms · Esc cancels</Text> : <>
      <Text color={t.ds.secondary}>{busy ? 'Working…' : detail?.state === 'pending' ? 'S send saved output' : detail?.state === 'uncertain' ? 'A mark sent · T allow retry' : 'No delivery action available'}</Text>
      <DialogFooter t={t}><Text color={t.ds.secondary}>↑↓ select · R refresh · Esc back</Text></DialogFooter>
    </>}
  </Box>
}
