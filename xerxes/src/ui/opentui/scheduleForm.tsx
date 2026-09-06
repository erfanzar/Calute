// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import type { TextareaRenderable } from '@opentui/core'
import { useEffect, useRef, useState } from 'react'
import { useOptionalGateway } from '../app/gatewayContext.js'
import type { Theme } from '../theme.js'
import { Box, Text } from './primitives.js'
export interface ScheduleDraft { max_total_tokens?: number | null; stop_condition?: string | null; target_session_id?: string | null; expires_at?: string | null; max_runs?: number | null; runs_started?: number; max_model_calls?: number | null; deliver?: string; recipient?: string; missed_run_policy?: "coalesce" | "skip"; misfire_grace_seconds?: number; timezone?: string; id: string; revision: string; prompt: string; schedule: string; paused: boolean; next_run_at?: string; timeout_seconds?: number | null; max_retries?: number | null; interval_seconds?: number | null }
export function ScheduleForm({ t, initial, onClose, onSaved, followup = false }: { t: Theme; followup?: boolean; initial?: ScheduleDraft; onClose: () => void; onSaved: (id: string) => void }) {
  const gateway = useOptionalGateway()
  const compact = useTerminalDimensions().height < 26
  const [field, setField] = useState(0)
  const [totalTokens, setTotalTokens] = useState(String(initial?.max_total_tokens ?? ''))
  const [stopCondition, setStopCondition] = useState(initial?.stop_condition ?? "")
  const [target, setTarget] = useState(initial?.target_session_id || (!initial && followup) ? 'session' : 'independent')
  const [expiresAt, setExpiresAt] = useState(initial?.expires_at ?? (!initial && followup ? new Date(Date.now() + 86400000).toISOString() : ''))
  const [maxRuns, setMaxRuns] = useState(initial?.max_runs == null ? (!initial && followup ? '10' : '') : String(initial.max_runs))
  const [prompt, setPrompt] = useState(initial?.prompt ?? '')
  const [timing, setTiming] = useState((initial?.interval_seconds != null || (!initial && followup)) ? 2 : initial && !initial.schedule ? 1 : 0)
  const once = timing === 1
  const intervalMode = timing === 2
  const [interval, setIntervalValue] = useState(String(initial?.interval_seconds ?? (followup ? 600 : 3600)))
  const [cron, setCron] = useState(initial?.schedule || '0 9 * * *')
  const [timezone, setTimezone] = useState(initial?.timezone ?? 'UTC')
  const [at, setAt] = useState(initial?.next_run_at ?? '')
  const [paused, setPaused] = useState(initial?.paused ?? true)
  const [timeout, setTimeoutValue] = useState(String(initial?.timeout_seconds ?? 300))
  const [retries, setRetries] = useState(String(initial?.max_retries ?? 3))
  const [missedPolicy, setMissedPolicy] = useState(initial?.missed_run_policy ?? "coalesce")
  const [grace, setGrace] = useState(String(initial?.misfire_grace_seconds ?? 300))
  const [deliver, setDeliver] = useState(initial?.deliver ?? "none")
  const [recipient, setRecipient] = useState(initial?.recipient ?? "")
  const [destinations, setDestinations] = useState<{ name: string; enabled: boolean }[]>([])
  const [destinationError, setDestinationError] = useState("")
  const [modelCalls, setModelCalls] = useState(String(initial?.max_model_calls ?? ""))
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [preview, setPreview] = useState('Checking next run…')
  const pending = useRef(false)
  const alive = useRef(true)
  const input = useRef<TextareaRenderable | null>(null)
  const editable = [0, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16].includes(field)
  useEffect(() => { alive.current = true; return () => { alive.current = false } }, [])
  useEffect(() => {
    let cancelled = false
    if (field !== 9 || !gateway) return
    setDestinationError('Loading configured channels…')
    void gateway.rpc<{ ok: boolean; destinations?: { name: string; enabled: boolean }[]; error?: string }>('schedule.options', {}).then(result => {
      if (cancelled) return
      if (!result?.ok || !Array.isArray(result.destinations)) throw new Error(result?.error || 'Could not load delivery channels')
      setDestinations(result.destinations)
      setDestinationError('')
    }).catch(error => { if (!cancelled) setDestinationError(String(error)) })
    return () => { cancelled = true }
  }, [field, gateway])
  useEffect(() => {
    let cancelled = false
    setPreview('Checking next run…')
    const timer = setTimeout(() => {
      if (!gateway) { setPreview('Next run unavailable: disconnected'); return }
      void gateway.rpc<{ ok: boolean; next_run_at?: string; error?: string }>('schedule.preview', {
        timezone, ...(intervalMode ? { interval_seconds: Number(interval) } : once ? { at } : { schedule: cron }),
      }).then(result => {
        if (cancelled) return
        if (!result?.ok || !result.next_run_at) throw new Error(result?.error || 'Preview unavailable')
        setPreview(`Next eligible run (UTC): ${result.next_run_at}`)
      }).catch(failure => { if (!cancelled) setPreview(`Next run: ${String(failure)}`) })
    }, 250)
    return () => { cancelled = true; clearTimeout(timer) }
  }, [gateway, timezone, intervalMode, once, interval, at, cron])
  useEffect(() => { if (editable) input.current?.setText(field === 16 ? totalTokens : field === 15 ? stopCondition : field === 13 ? expiresAt : field === 12 ? maxRuns : field === 11 ? modelCalls : field === 10 ? recipient : field === 9 ? deliver : field === 8 ? grace : field === 6 ? timezone : field === 0 ? prompt : field === 4 ? timeout : field === 5 ? retries : intervalMode ? interval : once ? at : cron) }, [field, editable, once, intervalMode])
  const save = () => {
    if (!gateway || pending.current) return
    if (!prompt.trim()) { setError('Enter a prompt.'); return }
    if (totalTokens.trim() && (!Number.isSafeInteger(Number(totalTokens)) || Number(totalTokens) < 1)) { setError('Lifetime token threshold must be a positive whole number.'); return }
    if (!Number.isSafeInteger(Number(timeout)) || Number(timeout) < 1 || Number(timeout) > 3600 || !Number.isSafeInteger(Number(retries)) || Number(retries) < 0 || Number(retries) > 10) { setError('Timeout: 1–3600 seconds; retries: 0–10.'); return }
    pending.current = true; setBusy(true); setError('')
    void gateway.rpc<{ ok: boolean; error?: string; job?: { id?: string } }>(initial ? 'schedule.update' : 'schedule.create', {
      ...(initial ? { schedule_id: initial.id, revision: initial.revision } : {}),
      ...(totalTokens.trim() || initial?.max_total_tokens != null ? { max_total_tokens: totalTokens.trim() ? Number(totalTokens) : null } : {}),
      prompt, paused, timezone, deliver, recipient, target, ...(stopCondition.trim() || initial?.stop_condition ? { stop_condition: stopCondition.trim() || null } : {}), expires_at: expiresAt.trim() || null, max_runs: maxRuns.trim() ? Number(maxRuns) : null, max_model_calls: modelCalls.trim() ? Number(modelCalls) : null, missed_run_policy: missedPolicy, misfire_grace_seconds: Number(grace), timeout_seconds: Number(timeout), max_retries: Number(retries), ...(intervalMode ? { interval_seconds: Number(interval) } : once ? { at } : { schedule: cron })
    }).then(result => {
      if (!result?.ok || typeof result.job?.id !== 'string') throw new Error(result?.error || 'Schedule could not be saved')
      if (alive.current) onSaved(result.job.id)
    }).catch(failure => { if (alive.current) setError(String(failure)) })
      .finally(() => { pending.current = false; if (alive.current) setBusy(false) })
  }
  useKeyboard(key => {
    if (key.eventType === 'release') return
    if (field === 9 && ['up', 'down'].includes(key.name)) {
      key.preventDefault(); key.stopPropagation()
      if (busy || !destinations.length) return
      const index = destinations.findIndex(value => value.name === deliver)
      const next = destinations[(index + (key.name === 'up' ? destinations.length - 1 : 1)) % destinations.length]!
      setDeliver(next.name)
      input.current?.setText(next.name)
      if (next.name === 'none') setRecipient('')
      return
    }
    if (!['escape', 'tab', 'return'].includes(key.name) && (editable || !['left', 'right', 'space'].includes(key.name))) return
    key.preventDefault(); key.stopPropagation()
    if (busy) return
    if (key.name === 'escape') onClose()
    else if (key.name === 'tab') setField(value => (value + (key.shift ? 16 : 1)) % 17)
    else if (key.name === 'return') save()
    else if (field === 14) setTarget(value => value === 'session' ? 'independent' : 'session')
    else if (field === 1) setTiming(value => (value + (key.name === 'left' ? 2 : 1)) % 3)
    else if (field === 7) setMissedPolicy(value => value === "skip" ? "coalesce" : "skip")
    else if (field === 3) setPaused(value => !value)
  })
  const labels = ['Prompt', 'Timing', intervalMode ? 'Interval seconds' : once ? 'ISO time with timezone' : 'Cron', 'State', 'Timeout seconds', 'One-shot retries', 'Recurring timezone', 'Missed runs', 'Lateness allowance seconds', 'Delivery channel', 'Recipient or room ID', 'Model calls per run (blank: unlimited)', 'Lifetime attempts (blank: unlimited)', 'Expiry ISO time (blank: none)', 'Run in', 'Stop condition (blank: none)', 'Lifetime token threshold (blank: unlimited)']
  const values = [prompt || '(required)', intervalMode ? 'Interval' : once ? 'One-shot' : 'Recurring', intervalMode ? interval : once ? at || '(required)' : cron, paused ? 'Paused' : 'Enabled', timeout, retries, timezone, missedPolicy === "skip" ? "Skip overdue occurrences" : "Run once after returning", grace, deliver, recipient || "(none)", modelCalls || "unlimited", maxRuns || "unlimited", expiresAt || "none", target === "session" ? "This conversation (attempt limit and expiry required)" : "Independent scheduled session", stopCondition || "none", totalTokens || "unlimited"]
  return <Box flexDirection="column" flexGrow={1} minHeight={0}>
    <Text color={t.ds.secondary}>Attempts used: {initial?.runs_started ?? 0} · retries and manual runs count</Text>
    <Text bold>{initial ? 'Edit schedule' : followup ? 'New conversation follow-up' : 'New schedule'}</Text>
    <scrollbox style={{ flexGrow: 1, minHeight: 0 }} contentOptions={{ flexDirection: 'column' }}>
      {labels.map((label, index) => compact && Math.abs(index - field) > 1 ? null : <Text key={label} color={field === index ? t.color.accent : t.ds.secondary} wrap="wrap">{field === index ? '› ' : '  '}{label}: {values[index]}</Text>)}
      {editable ? <textarea key={field} ref={input} focused={!busy} minHeight={1} maxHeight={3} focusedBackgroundColor={t.color.statusBg} focusedTextColor={t.color.text} onContentChange={() => {
        const value = input.current?.plainText ?? ''
        if (field === 16) setTotalTokens(value)
        else if (field === 15) setStopCondition(value)
        else if (field === 0) setPrompt(value)
        else if (field === 4) setTimeoutValue(value)
        else if (field === 5) setRetries(value)
        else if (field === 13) setExpiresAt(value)
        else if (field === 12) setMaxRuns(value)
        else if (field === 11) setModelCalls(value)
        else if (field === 9) setDeliver(value)
        else if (field === 10) setRecipient(value)
        else if (field === 8) setGrace(value)
        else if (field === 6) setTimezone(value)
        else if (intervalMode) setIntervalValue(value)
        else if (once) setAt(value)
        else setCron(value)
      }} /> : null}
      {field === 9 ? <Text color={t.ds.secondary} wrap="wrap">↑/↓ choose: {destinations.map(value => `${value.name}${value.enabled ? "" : " (disabled)"}`).join(" · ") || "none"}. {destinationError}</Text> : null}
      {field === 16 ? <Text color={t.ds.secondary} wrap="wrap">Blocks new calls at measured usage across all attempts. In-flight calls may overshoot. Unknown historical usage blocks execution.</Text> : null}
      {field === 15 ? <Text color={t.ds.secondary} wrap="wrap">Session follow-ups only. The model checks this condition and records its evidence before stopping future wakes.</Text> : null}
      {error ? <Text color={t.color.warn} wrap="wrap">{error}</Text> : null}
      <Text color={t.ds.secondary} wrap="wrap">{preview}{paused ? ' · Paused: enable to run.' : ''}</Text>
      <Text color={t.ds.secondary} wrap="wrap">Overlap: forbidden. Skip applies after the lateness allowance; missed one-shots pause for review.</Text>
      <Text color={t.ds.secondary} wrap="wrap">New jobs start paused. Enable when ready. The owning daemon must be online.</Text>
    </scrollbox>
    <Text color={t.ds.secondary}>{busy ? 'Saving…' : 'Tab field · ←→ choose · Enter save'}</Text>
    <Text color={t.ds.secondary}>Esc back</Text>
  </Box>
}
