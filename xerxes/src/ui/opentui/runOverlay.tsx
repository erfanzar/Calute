// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import type { ScrollBoxRenderable } from '@opentui/core'
import { useEffect, useRef, useState } from 'react'
import { useOptionalGateway } from '../app/gatewayContext.js'
import { patchOverlayState } from '../app/overlayStore.js'
import { cancelRun, acknowledgeRun, inspectRun, listRunPage, type RunDetail, type RunSummary, type UpcomingRun, type RunAttention } from '../lib/runs.js'
import type { Theme } from '../theme.js'
import { overlayPanelSize } from './overlayLayout.js'
import { Box, Text } from './primitives.js'

export function RunOverlay({ t, scheduleId, onClose }: { t: Theme; scheduleId?: string; onClose?: () => void }) {
  const gateway = useOptionalGateway()
  const terminal = useTerminalDimensions()
  const size = overlayPanelSize(terminal, { maxWidth: 180, minWidth: 32 })
  const kinds = ['', 'agent', 'terminal', 'schedule', 'monitor']
  const states = ['', 'running', 'failed', 'interrupted', 'cancelled', 'succeeded']
  const [kindIndex, setKindIndex] = useState(0)
  const [stateIndex, setStateIndex] = useState(0)
  const kind = kinds[kindIndex]!
  const state = states[stateIndex]!
  const [hasMore, setHasMore] = useState(false)
  const [pages, setPages] = useState<{ startedAt: number; id: string }[]>([])
  const before = pages.at(-1)
  const [attention, setAttention] = useState<RunAttention[]>([])
  const [attentionTotal, setAttentionTotal] = useState(0)
  const [upcoming, setUpcoming] = useState<UpcomingRun[]>([])
  const [upcomingTotal, setUpcomingTotal] = useState(0)
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [selected, setSelected] = useState('')
  const [detail, setDetail] = useState<RunDetail | null>(null)
  const [unread, setUnread] = useState(false)
  const [workspaceScope, setWorkspaceScope] = useState(true)
  const scope = workspaceScope ? 'workspace' : 'session'
  const [listError, setListError] = useState('')
  const [detailError, setDetailError] = useState('')
  const [actionError, setActionError] = useState('')
  const error = [listError, detailError, actionError].filter(Boolean).join('\n')
  const [loading, setLoading] = useState(true)
  const [refresh, setRefresh] = useState(0)
  const cancelling = useRef(false)
  const [cancelPending, setCancelPending] = useState(false)
  const [acknowledging, setAcknowledging] = useState(false)
  const alive = useRef(true)
  const scroll = useRef<ScrollBoxRenderable | null>(null)
  useEffect(() => { alive.current = true; return () => { alive.current = false } }, [])
  useEffect(() => {
    let current = true
    let timer: ReturnType<typeof setTimeout> | undefined
    if (!gateway) { setLoading(false); setListError('Run history requires a connected daemon.'); return }
    const load = async () => {
      try {
        const page = await listRunPage(gateway.rpc, unread, scope, scheduleId, before, { ...(kind ? { kind } : {}), ...(state ? { state } : {}) })
        if (!current) return
        const rows = page.runs
        setAttention(page.attention)
        setAttentionTotal(page.attentionTotal)
        setUpcoming(page.upcoming)
        setUpcomingTotal(page.upcomingTotal)
        setHasMore(page.hasMore)
        setRuns(rows)
        setSelected(previous => rows.some(run => run.id === previous) ? previous : rows[0]?.id ?? '')
        setListError('')
      } catch (failure) { if (current) setListError(String(failure)) }
      finally {
        if (current) { setLoading(false); timer = setTimeout(() => { void load() }, 2000) }
      }
    }
    setLoading(true)
    void load()
    return () => { current = false; if (timer) clearTimeout(timer) }
  }, [gateway, unread, refresh, scope, scheduleId, before, kind, state])
  const selectedRun = runs.find(run => run.id === selected)
  useEffect(() => {
    let current = true
    let timer: ReturnType<typeof setTimeout> | undefined
    setDetail(null)
    setDetailError('')
    scroll.current?.scrollTo(0)
    const load = async () => {
      if (!gateway || !selected) return
      try {
        const value = await inspectRun(gateway.rpc, selected, scope)
        if (current) {
          setDetail(value)
          setDetailError('')
          if (value.state === 'running' || (value.reactionHealth && ['waiting', 'queued', 'running', 'cancelling', 'awaiting-cleanup'].includes(value.reactionHealth.state))) timer = setTimeout(() => { void load() }, 2000)
        }
      } catch (failure) { if (current) setDetailError(String(failure)) }
    }
    void load()
    return () => { current = false; if (timer) clearTimeout(timer) }
  }, [gateway, selected, selectedRun?.revision, refresh, scope])
  const acknowledge = () => {
    if (!gateway || !detail || !detail.unread || acknowledging) return
    setActionError('')
    setAcknowledging(true)
    void acknowledgeRun(gateway.rpc, detail, scope).then(() => {
      if (alive.current) setRefresh(value => value + 1)
    }).catch(failure => { if (alive.current) setActionError(String(failure)) })
      .finally(() => { if (alive.current) setAcknowledging(false) })
  }
  const cancelSelected = () => {
    if (!gateway || !detail?.cancelLabel || cancelling.current) return
    setActionError('')
    cancelling.current = true
    setCancelPending(true)
    void cancelRun(gateway.rpc, detail, scope).then(() => { if (alive.current) setRefresh(value => value + 1) })
      .catch(failure => { if (alive.current) setActionError(String(failure)) })
      .finally(() => { cancelling.current = false; if (alive.current) setCancelPending(false) })
  }
  useKeyboard(key => {
    if (key.eventType === 'release') return
    if (key.name === 'escape') { key.preventDefault(); key.stopPropagation(); onClose ? onClose() : patchOverlayState({ runs: false }); return }
    if (!['up', 'down', 'pageup', 'pagedown', 'home', 'end', 'a', 'u', 'r', 'w', 'n', 'p', 'k', 's', 'x', 't', 'e'].includes(key.name)) return
    key.preventDefault(); key.stopPropagation()
    const index = runs.findIndex(run => run.id === selected)
    if (key.name === 'up' || key.name === 'down') setSelected(runs[Math.max(0, Math.min(runs.length - 1, index + (key.name === 'up' ? -1 : 1)))]?.id ?? '')
    else if (key.name === 'n') { if (hasMore && !loading && runs.length) { const last = runs.at(-1)!; setPages(value => [...value, { startedAt: last.startedAt, id: last.id }]) } }
    else if (key.name === 'k') { if (!scheduleId) { setPages([]); setKindIndex(value => (value + 1) % kinds.length) } }
    else if (key.name === 's') { setPages([]); setStateIndex(value => (value + 1) % states.length) }
    else if (key.name === 'p') setPages(value => value.slice(0, -1))
    else if (key.name === 'u') { setPages([]); setUnread(value => !value) }
    else if (key.name === 'w') { setPages([]); setWorkspaceScope(value => !value) }
    else if (key.name === 'r') setRefresh(value => value + 1)
    else if (key.name === 't' && !scheduleId) patchOverlayState({ runs: false, schedules: true })
    else if (key.name === 'e' && attentionTotal > 0) { onClose ? onClose() : patchOverlayState({ runs: false }) }
    else if (key.name === 'a') acknowledge()
    else if (key.name === 'x') cancelSelected()
    else if (key.name === 'home') scroll.current?.scrollTo(0)
    else if (key.name === 'end') scroll.current?.scrollTo(Number.MAX_SAFE_INTEGER)
    else scroll.current?.scrollBy((key.name === 'pageup' ? -1 : 1) * Math.max(1, size.height - 10))
  })
  const wide = size.width >= 100
  const nextRows = workspaceScope && !scheduleId ? upcoming.slice(0, wide ? 3 : 1) : []
  const attentionRows = attention.slice(0, wide ? 3 : 1)
  const availableHeight = size.height - 7 - (attentionRows.length ? attentionRows.length + 1 : 0) - (nextRows.length ? nextRows.length * 2 + 1 : 0)
  const listHeight = wide ? availableHeight : Math.max(3, Math.floor(availableHeight / 3))
  const visibleCount = Math.max(1, Math.floor(listHeight / 2))
  const selectedIndex = runs.findIndex(run => run.id === selected)
  const start = Math.max(0, selectedIndex - visibleCount + 1)
  return <box position="absolute" left={0} top={0} width="100%" height="100%" zIndex={150} backgroundColor="#000000cc" alignItems="center" justifyContent="center">
    <Box width={size.width} height={size.height} flexDirection="column" paddingX={1} borderStyle="round" borderColor={t.color.border} backgroundColor={t.color.statusBg}>
      <Text bold color={t.color.text}>{scheduleId ? 'Schedule history' : 'Runs'} · {workspaceScope ? 'Workspace' : 'Session'}{unread ? ' · Unread' : ''} · {runs.length} · Page {pages.length + 1}</Text>
      <Text color={t.ds.secondary}>K {scheduleId ? 'schedule' : kind || 'all kinds'} · S {state || 'all states'}</Text>
      {attentionRows.length ? <Box flexDirection="column" flexShrink={0} onMouseDown={() => { onClose ? onClose() : patchOverlayState({ runs: false }) }}>
        <Text color={t.color.warn} wrap="truncate-end">Session attention · {attentionTotal} · E chat</Text>
        {attentionRows.map(item => <Text key={item.id} color={t.color.text} wrap="truncate-end">{item.kind}: {item.title}</Text>)}
      </Box> : null}
      {nextRows.length ? <Box flexDirection="column" flexShrink={0} onMouseDown={() => patchOverlayState({ runs: false, schedules: true })}>
        <Text color={t.color.accent} wrap="truncate-end">Scheduled next · {upcomingTotal} · T manage</Text>
        {nextRows.map(job => <Box key={job.id} flexDirection="column" flexShrink={0}><Text color={t.color.text} wrap="truncate-end">{job.title}</Text><Text color={t.ds.secondary} wrap="truncate-end">{job.nextRunAt}</Text></Box>)}
      </Box> : null}
      {error ? <Text color={t.color.warn} wrap="wrap">{error}</Text> : null}
      <Box flexDirection={wide ? 'row' : 'column'} flexGrow={1} minHeight={0}>
        <Box width={wide ? '38%' : '100%'} height={wide ? '100%' : listHeight} flexDirection="column" paddingRight={1}>
          {runs.length ? runs.slice(start, start + visibleCount).map(run => <Box key={run.id} flexDirection="column" height={2} backgroundColor={run.id === selected ? t.ds.selected : undefined} onMouseDown={() => setSelected(run.id)}>
            <Text color={run.unread ? t.color.accent : t.color.text} wrap="truncate-end">{run.unread ? '●' : '·'} {run.title}</Text>
            <Text color={t.ds.secondary} wrap="truncate-end">{run.kind} · {run.state}{workspaceScope ? ` · ${run.ownerSessionId.slice(0, 8)}` : ''}</Text>
          </Box>) : <Text color={t.ds.secondary} wrap="wrap">{loading ? 'Loading runs…' : unread ? 'No unread results.' : 'No recorded runs in this session yet.'}</Text>}
        </Box>
        <scrollbox ref={scroll} style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }} contentOptions={{ flexDirection: 'column' }}>
          {wide || !detail ? <Text bold color={t.color.text} wrap="wrap">{detail?.title ?? (selected ? 'Loading result…' : 'Select a run to inspect its result.')}</Text> : null}
          {detail ? <Box flexDirection="column" flexShrink={0}>
            <Text color={t.ds.secondary} wrap="wrap">{detail.kind} · {detail.state} · {detail.workspace}</Text>
            {detail.tokenUsage ? <Text color={t.ds.secondary} wrap="wrap">Tokens: {detail.tokenUsage.inputTokens} input · {detail.tokenUsage.outputTokens} output{detail.tokenUsage.complete ? '' : ' · partial; some usage unavailable'}</Text> : detail.kind === 'schedule' ? <Text color={t.ds.secondary}>Token usage unavailable.</Text> : null}
            {detail.cancelLabel ? <Box onMouseDown={cancelSelected}><Text color={t.color.warn}>{cancelPending ? 'Requesting cancellation…' : `X · ${detail.cancelLabel}`}</Text></Box> : null}
            {wide ? <Text color={t.ds.secondary} wrap="wrap">{detail.id}</Text> : null}
            {detail.reactionHealth ? <Box flexDirection="column">
              <Text color={t.color.accent} wrap="wrap">Reactions: {detail.reactionHealth.state} · {detail.reactionHealth.attempts}/{detail.reactionHealth.maxReactions} attempts · {detail.reactionHealth.pendingEvents} queued events</Text>
              {detail.reactionHealth.usage ? <Text color={t.ds.secondary} wrap="wrap">Measured reaction tokens: {detail.reactionHealth.usage.inputTokens} input · {detail.reactionHealth.usage.outputTokens} output{detail.reactionHealth.usage.complete ? "" : " · incomplete usage"}</Text> : null}
              {detail.reactionHealth.lastError ? <Text color={t.color.warn} wrap="wrap">{detail.reactionHealth.lastError}</Text> : null}
            </Box> : null}
            {detail.error ? <Text color={t.color.warn} wrap="wrap">{detail.error}</Text> : null}
            {detail.outputTruncated ? <Text color={t.ds.secondary}>Earlier output was omitted.</Text> : null}
            <Text color={t.color.text} wrap="wrap">{detail.output || 'No output recorded.'}</Text>
          </Box> : null}
        </scrollbox>
      </Box>
      <Text color={t.ds.secondary}>{acknowledging ? 'Acknowledging…' : size.width < 70 ? 'N/P pages · ↑↓ · K kind · S state' : 'K kind · S state · N/P pages · ↑↓ select · U unread · A acknowledge · W scope · R refresh'}</Text>
      <Text color={t.ds.secondary}>{size.width < 70 ? 'A ack · U unread · W scope · Esc close' : 'PgUp/PgDn scroll result · Esc close'}</Text>
    </Box>
  </box>
}
