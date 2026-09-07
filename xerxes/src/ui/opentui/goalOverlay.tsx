// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */

import type { KeyBinding, ScrollBoxRenderable, TextareaRenderable } from '@opentui/core'
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { patchOverlayState } from '../app/overlayStore.js'
import { useOptionalGateway } from '../app/gatewayContext.js'
import { useTurnSelector } from '../app/turnStore.js'
import { $uiState } from '../app/uiStore.js'
import type { Theme } from '../theme.js'
import { overlayPanelSize } from './overlayLayout.js'
import { Box, Text } from './primitives.js'
import { SessionFollowups, openSessionFollowups } from './sessionFollowups.js'

export interface GoalOverlayProps { t: Theme }

interface GoalEvidence {
  kind: 'tool' | 'user-decision'
  toolCallId?: string
  decisionId?: string
  summary: string
  recordedAt: number
}

interface GoalCriterion {
  id: string
  description: string
  evidence?: GoalEvidence
}

interface InspectedGoal {
  id: string
  revision: number
  objective: string
  currentMilestone?: string
  phase: string
  roundsStarted: number
  maxGoalRounds: number
  maxDurationMs?: number
  maxTotalTokens?: number
  createdAt?: number
  blockedReason?: { code: string; message: string }
  criteria: GoalCriterion[]
  activation?: 'armed' | 'disarmed'
}

interface GoalContinuation {
  version: 1
  id: string
  sessionId: string
  goalId: string
  revision: number
  state: 'queued' | 'running' | 'settled' | 'interrupted' | 'cancelled'
  queuedAt: number
  startedAt?: number
  settledAt?: number
  ownerId?: string
  round?: number
  reason?: string
}

interface GoalTokenUsage {
  inputTokens: number
  outputTokens: number
  measuredCalls: number
  settledCalls: number
  pendingCalls: number
  complete: boolean
}

interface GoalInspection {
  sessionId: string
  goal: InspectedGoal | null
  tokenUsage: GoalTokenUsage | null
  continuation: GoalContinuation | null
}

const DECISION_KEY_BINDINGS: KeyBinding[] = [
  { name: 'return', action: 'submit' },
  { name: 'enter', action: 'submit' },
  { name: 'kpenter', action: 'submit' },
  { name: 'linefeed', action: 'submit' },
]

function parseGoalInspection(value: unknown, expectedSessionId: string): GoalInspection {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid goal inspection response')
  const row = value as Record<string, unknown>
  if (row.ok !== true || row.session_id !== expectedSessionId) throw new Error(typeof row.error === 'string' ? row.error : 'Invalid goal inspection response')
  const rawGoal = row.goal
  const rawTokenUsage = row.token_usage
  let tokenUsage: GoalTokenUsage | null = null
  if (rawTokenUsage !== undefined && rawTokenUsage !== null) {
    if (!rawTokenUsage || typeof rawTokenUsage !== 'object' || Array.isArray(rawTokenUsage)) {
      throw new Error('Invalid goal inspection token usage')
    }
    const usage = rawTokenUsage as Record<string, unknown>
    const integerFields = ['inputTokens', 'outputTokens', 'measuredCalls', 'settledCalls', 'pendingCalls'] as const
    if (integerFields.some(field => typeof usage[field] !== 'number'
      || !Number.isSafeInteger(usage[field]) || (usage[field] as number) < 0)
      || typeof usage.complete !== 'boolean') {
      throw new Error('Invalid goal inspection token usage')
    }
    const inputTokens = usage.inputTokens as number
    const outputTokens = usage.outputTokens as number
    if (inputTokens > Number.MAX_SAFE_INTEGER - outputTokens) throw new Error('Invalid goal inspection token usage')
    tokenUsage = { inputTokens, outputTokens, measuredCalls: usage.measuredCalls as number,
      settledCalls: usage.settledCalls as number, pendingCalls: usage.pendingCalls as number,
      complete: usage.complete as boolean }
  }
  const rawContinuation = row.continuation
  let continuation: GoalContinuation | null = null
  if (rawContinuation !== undefined && rawContinuation !== null) {
    if (!rawContinuation || typeof rawContinuation !== 'object' || Array.isArray(rawContinuation)) throw new Error('Invalid goal continuation')
    const value = rawContinuation as Record<string, unknown>
    const states = ['queued', 'running', 'settled', 'interrupted', 'cancelled'] as const
    if (value.version !== 1 || typeof value.id !== 'string' || !value.id.trim() || value.sessionId !== expectedSessionId
      || typeof value.goalId !== 'string' || !value.goalId.trim() || typeof value.revision !== 'number' || !Number.isSafeInteger(value.revision) || value.revision < 1
      || !states.includes(value.state as typeof states[number]) || typeof value.queuedAt !== 'number' || !Number.isSafeInteger(value.queuedAt) || value.queuedAt < 0
      || (value.startedAt !== undefined && (typeof value.startedAt !== 'number' || !Number.isSafeInteger(value.startedAt) || value.startedAt < 0))
      || (value.settledAt !== undefined && (typeof value.settledAt !== 'number' || !Number.isSafeInteger(value.settledAt) || value.settledAt < 0))
      || (value.ownerId !== undefined && (typeof value.ownerId !== 'string' || !value.ownerId.trim()))
      || (value.round !== undefined && (typeof value.round !== 'number' || !Number.isSafeInteger(value.round) || value.round < 1))
      || (value.reason !== undefined && (typeof value.reason !== 'string' || !value.reason.trim()))) throw new Error('Invalid goal continuation')
    continuation = { version: 1, id: value.id, sessionId: expectedSessionId, goalId: value.goalId as string, revision: value.revision as number,
      state: value.state as GoalContinuation['state'], queuedAt: value.queuedAt as number,
      ...(value.startedAt === undefined ? {} : { startedAt: value.startedAt as number }), ...(value.settledAt === undefined ? {} : { settledAt: value.settledAt as number }),
      ...(value.ownerId === undefined ? {} : { ownerId: value.ownerId as string }), ...(value.round === undefined ? {} : { round: value.round as number }), ...(value.reason === undefined ? {} : { reason: value.reason as string }) }
  }
  if (rawGoal === null) return { sessionId: expectedSessionId, goal: null, tokenUsage, continuation: null }
  if (!rawGoal || typeof rawGoal !== 'object' || Array.isArray(rawGoal)) throw new Error('Invalid goal inspection goal')
  const goal = rawGoal as Record<string, unknown>
  if (typeof goal.id !== 'string' || !goal.id.trim() || typeof goal.revision !== 'number' || !Number.isSafeInteger(goal.revision) || goal.revision < 1
    || typeof goal.objective !== 'string' || typeof goal.phase !== 'string' || typeof goal.roundsStarted !== 'number' || !Number.isSafeInteger(goal.roundsStarted) || goal.roundsStarted < 0
    || typeof goal.maxGoalRounds !== 'number' || !Number.isSafeInteger(goal.maxGoalRounds) || goal.maxGoalRounds < 1) throw new Error('Invalid goal inspection goal')
  const hasDuration = goal.maxDurationMs !== undefined
  if (hasDuration && (typeof goal.maxDurationMs !== 'number' || !Number.isSafeInteger(goal.maxDurationMs) || goal.maxDurationMs < 1
    || typeof goal.createdAt !== 'number' || !Number.isSafeInteger(goal.createdAt) || goal.createdAt < 0
    || goal.createdAt > Number.MAX_SAFE_INTEGER - goal.maxDurationMs)) throw new Error('Invalid goal inspection time limit')
  if (goal.maxTotalTokens !== undefined
    && (typeof goal.maxTotalTokens !== 'number' || !Number.isSafeInteger(goal.maxTotalTokens) || goal.maxTotalTokens < 1)) {
    throw new Error('Invalid goal inspection token limit')
  }
  if (goal.activation !== undefined && goal.activation !== 'armed' && goal.activation !== 'disarmed') throw new Error('Invalid goal inspection activation')
  if (goal.currentMilestone !== undefined && (typeof goal.currentMilestone !== 'string' || !goal.currentMilestone.trim() || goal.currentMilestone.length > 1_000)) throw new Error('Invalid goal inspection milestone')
  const criteria: GoalCriterion[] = []
  if (goal.criteria !== undefined) {
    if (!Array.isArray(goal.criteria) || goal.criteria.length > 128) throw new Error('Invalid goal inspection criteria')
    for (const item of goal.criteria) {
      if (!item || typeof item !== 'object' || Array.isArray(item)) throw new Error('Invalid goal inspection criterion')
      const criterion = item as Record<string, unknown>
      if (typeof criterion.id !== 'string' || !criterion.id.trim() || typeof criterion.description !== 'string' || !criterion.description.trim()) {
        throw new Error('Invalid goal inspection criterion')
      }
      let evidence: GoalEvidence | undefined
      if (criterion.evidence !== undefined) {
        if (!criterion.evidence || typeof criterion.evidence !== 'object' || Array.isArray(criterion.evidence)) throw new Error('Invalid goal inspection evidence')
        const rawEvidence = criterion.evidence as Record<string, unknown>
        const userDecision = rawEvidence.kind === 'user-decision'
        if (rawEvidence.kind !== undefined && rawEvidence.kind !== 'user-decision' && rawEvidence.kind !== 'tool-result') throw new Error('Invalid goal inspection evidence')
        if (userDecision ? 'toolCallId' in rawEvidence : 'decisionId' in rawEvidence) throw new Error('Invalid goal inspection evidence')
        if (typeof rawEvidence.summary !== 'string' || !rawEvidence.summary.trim()
          || typeof rawEvidence.recordedAt !== 'number' || !Number.isFinite(rawEvidence.recordedAt) || rawEvidence.recordedAt < 0
          || (userDecision ? typeof rawEvidence.decisionId !== 'string' || !rawEvidence.decisionId.trim() : typeof rawEvidence.toolCallId !== 'string' || !rawEvidence.toolCallId.trim())) throw new Error('Invalid goal inspection evidence')
        evidence = userDecision
          ? { kind: 'user-decision', decisionId: rawEvidence.decisionId as string, summary: rawEvidence.summary, recordedAt: rawEvidence.recordedAt }
          : { kind: 'tool', toolCallId: rawEvidence.toolCallId as string, summary: rawEvidence.summary, recordedAt: rawEvidence.recordedAt }
      }
      criteria.push({ id: criterion.id, description: criterion.description, ...(evidence ? { evidence } : {}) })
    }
  }
  let blockedReason: InspectedGoal['blockedReason']
  if (goal.blockedReason !== undefined) {
    if (!goal.blockedReason || typeof goal.blockedReason !== 'object' || Array.isArray(goal.blockedReason)) throw new Error('Invalid goal inspection blocker')
    const reason = goal.blockedReason as Record<string, unknown>
    if (typeof reason.code !== 'string' || !reason.code.trim() || typeof reason.message !== 'string' || !reason.message.trim()) throw new Error('Invalid goal inspection blocker')
    blockedReason = { code: reason.code, message: reason.message }
  }
  return {
    sessionId: expectedSessionId,
    goal: {
      id: goal.id,
      revision: goal.revision,
      objective: goal.objective,
      ...(goal.currentMilestone === undefined ? {} : { currentMilestone: goal.currentMilestone }),
      phase: goal.phase,
      roundsStarted: goal.roundsStarted,
      maxGoalRounds: goal.maxGoalRounds,
      ...(hasDuration ? { maxDurationMs: goal.maxDurationMs as number, createdAt: goal.createdAt as number } : {}),
      ...(goal.maxTotalTokens === undefined ? {} : { maxTotalTokens: goal.maxTotalTokens }),
      ...(blockedReason ? { blockedReason } : {}),
      criteria,
      ...(goal.activation === undefined ? {} : { activation: goal.activation as 'armed' | 'disarmed' }),
    },
    tokenUsage,
    continuation: continuation && continuation.goalId === goal.id ? continuation : null,
  }
}

function formatDuration(durationMs: number): string {
  const totalSeconds = Math.max(0, Math.floor(durationMs / 1_000))
  const days = Math.floor(totalSeconds / 86_400)
  const hours = Math.floor((totalSeconds % 86_400) / 3_600)
  const minutes = Math.floor((totalSeconds % 3_600) / 60)
  const seconds = totalSeconds % 60
  if (days) return `${days}d ${hours}h`
  if (hours) return `${hours}h ${minutes}m`
  if (minutes) return `${minutes}m ${seconds}s`
  return `${seconds}s`
}

function continuationDisplay(continuation: GoalContinuation | null, compact: boolean): string | null {
  if (!continuation) return null
  const round = continuation.round === undefined ? '' : ` round ${continuation.round}`
  if (compact) {
    if (continuation.state === 'queued') return `CONTINUATION · queued${continuation.reason ? ` · ${continuation.reason}` : ''}`
    if (continuation.state === 'running') return `CONTINUATION ·${round || ' running'}`
    if (continuation.state === 'interrupted') return 'CONTINUATION · interrupted · resume required'
    return `CONTINUATION · ${continuation.state}`
  }
  if (continuation.state === 'queued') return `CONTINUATION · queued · waiting${continuation.reason ? ` · ${continuation.reason}` : ''}`
  if (continuation.state === 'running') return `CONTINUATION · running${round}`
  if (continuation.state === 'interrupted') return `CONTINUATION · interrupted · last round not rerun automatically${continuation.reason ? ` · ${continuation.reason}` : ''}`
  return `CONTINUATION · ${continuation.state}${continuation.reason ? ` · ${continuation.reason}` : ''}`
}

function tokenUsageDisplay(
  goal: InspectedGoal | null | undefined,
  usage: GoalTokenUsage | null,
  compact: boolean,
): { readonly text: string; readonly warning: boolean } {
  if (usage === null) {
    return {
      text: compact
        ? goal?.maxTotalTokens === undefined ? 'No usage recorded' : `No usage recorded · cap ${goal.maxTotalTokens}`
        : goal?.maxTotalTokens === undefined ? 'TOKEN USAGE · No usage recorded' : `TOKEN BUDGET · No usage recorded · cap ${goal.maxTotalTokens}`,
      warning: true,
    }
  }
  const total = usage.inputTokens + usage.outputTokens
  const complete = usage.complete && usage.pendingCalls === 0
  const status = complete ? 'usage complete' : 'usage unknown'
  return {
    text: compact
      ? goal?.maxTotalTokens === undefined
        ? `${total} tokens · ${usage.pendingCalls} pending · ${complete ? 'complete' : 'unknown usage'}`
        : `${total}/${goal.maxTotalTokens} · ${usage.pendingCalls} pending · ${complete ? 'complete' : 'unknown'}`
      : goal?.maxTotalTokens === undefined
        ? `TOKEN USAGE · ${total} tokens · ${usage.pendingCalls} pending calls · ${status}`
        : `TOKEN BUDGET · ${total} / ${goal.maxTotalTokens} tokens · ${usage.pendingCalls} pending calls · ${status}`,
    warning: !complete,
  }
}

/** A bounded goal inspector; long checklists scroll without hiding close controls. */
export function GoalOverlay({ t }: GoalOverlayProps) {
  const { info, sid } = useStore($uiState)
  const gateway = useOptionalGateway()
  const todos = useTurnSelector(state => state.todos)
  const terminal = useTerminalDimensions()
  const scroll = useRef<ScrollBoxRenderable | null>(null)
  const [inspection, setInspection] = useState<GoalInspection | null>(null)
  const [inspectionError, setInspectionError] = useState('')
  const [inspectionLoading, setInspectionLoading] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const [selectedCriterion, setSelectedCriterion] = useState(0)
  const [decisionCriterion, setDecisionCriterion] = useState<number | null>(null)
  const [decisionDraft, setDecisionDraft] = useState('')
  const [decisionMessage, setDecisionMessage] = useState('')
  const [decisionBusy, setDecisionBusy] = useState(false)
  const submittingDecision = useRef(false)
  const inspectionEpoch = useRef(0)
  const decisionInput = useRef<TextareaRenderable | null>(null)
  const decisionContext = useRef<{ sessionId: string; goalId: string; revision: number; criterionId: string; description: string; generation: number } | null>(null)
  const liveSession = useRef<string | null>(sid)
  const sessionGeneration = useRef(0)
  const preserveDraft = useRef(false)
  const inspectedSession = useRef<string | null>(null)
  liveSession.current = sid
  const size = overlayPanelSize({ ...terminal, height: Math.min(terminal.height, 48) }, { maxWidth: 104, minWidth: 36 })

  useEffect(() => () => { sessionGeneration.current += 1 }, [])

  useEffect(() => {
    let current = true
    let requestInFlight = false
    let timer: ReturnType<typeof setInterval> | undefined
    if (!gateway || !sid) {
      inspectedSession.current = null
      setInspection(null)
      setInspectionError('')
      setInspectionLoading(false)
      setDecisionCriterion(null)
      setSelectedCriterion(0)
      decisionContext.current = null
      submittingDecision.current = false
      setDecisionBusy(false)
      return () => { current = false }
    }
    if (inspectedSession.current !== sid) {
      inspectedSession.current = sid
      sessionGeneration.current += 1
      setInspection(null)
      setDecisionCriterion(null)
      setSelectedCriterion(0)
      decisionContext.current = null
      setDecisionDraft('')
      setDecisionMessage('')
      submittingDecision.current = false
      setDecisionBusy(false)
      preserveDraft.current = false
    }
    const inspect = (): void => {
      if (!current || requestInFlight || submittingDecision.current) return
      const epoch = inspectionEpoch.current
      requestInFlight = true
      setInspectionLoading(true)
      void gateway.rpc<Record<string, unknown>>('goal.inspect', {})
        .then(value => {
          if (!current || epoch !== inspectionEpoch.current) return
          const next = parseGoalInspection(value, sid)
          setInspection(next)
          setInspectionError('')
        })
        .catch(error => {
          if (current && epoch === inspectionEpoch.current) setInspectionError(error instanceof Error ? error.message : String(error))
        })
        .finally(() => {
          requestInFlight = false
          if (current) setInspectionLoading(false)
        })
    }
    inspect()
    timer = setInterval(inspect, 3_000)
    return () => {
      current = false
      if (timer !== undefined) clearInterval(timer)
    }
  }, [gateway, sid, refresh])

  const cancelDecision = () => {
    if (decisionBusy) return
    setDecisionCriterion(null)
    decisionContext.current = null
    setDecisionDraft('')
    setDecisionMessage('')
    preserveDraft.current = false
  }

  const beginDecision = (index: number) => {
    const criterion = inspection?.goal?.criteria[index]
    const goal = inspection?.goal
    if (!criterion || !goal || !sid || decisionBusy) return
    decisionContext.current = { sessionId: sid, goalId: goal.id, revision: goal.revision, criterionId: criterion.id, description: criterion.description, generation: sessionGeneration.current }
    setDecisionCriterion(index)
    if (!preserveDraft.current) setDecisionDraft('')
    preserveDraft.current = false
    setDecisionMessage('')
  }

  useEffect(() => {
    if (decisionCriterion !== null && decisionInput.current && decisionDraft) decisionInput.current.setText(decisionDraft)
  }, [decisionCriterion])

  const submitDecision = async () => {
    if (submittingDecision.current) return
    const context = decisionContext.current
    const summary = decisionInput.current?.plainText.trim() ?? decisionDraft.trim()
    if (!gateway || !context || !summary) {
      setDecisionMessage('Enter a note before accepting the criterion.')
      return
    }
    setDecisionDraft(decisionInput.current?.plainText ?? decisionDraft)
    submittingDecision.current = true
    inspectionEpoch.current += 1
    setDecisionBusy(true)
    setDecisionMessage('Submitting human decision…')
    try {
      const result = await gateway.rpc<Record<string, unknown>>('goal.decision', {
        session_id: context.sessionId, goal_id: context.goalId, revision: context.revision,
        criterion_id: context.criterionId, summary,
      })
      const next = parseGoalInspection(result, context.sessionId)
      if (liveSession.current !== context.sessionId || sessionGeneration.current !== context.generation) return
      setInspection(next)
      setDecisionCriterion(null)
      decisionContext.current = null
      setDecisionDraft('')
      setDecisionMessage('Criterion accepted by you.')
      setRefresh(value => value + 1)
    } catch (error) {
      if (liveSession.current !== context.sessionId || sessionGeneration.current !== context.generation) return
      setDecisionMessage(`${error instanceof Error ? error.message : String(error)} · Draft kept.`)
    } finally {
      if (liveSession.current === context.sessionId && sessionGeneration.current === context.generation) {
        submittingDecision.current = false
        setDecisionBusy(false)
      }
    }
  }

  useKeyboard(key => {
    if (key.eventType === 'release') return
    if (decisionCriterion !== null) {
      if (key.name === 'escape' || key.name === 'esc') { key.preventDefault(); key.stopPropagation(); cancelDecision() }
      else if (key.name === 'r' && key.ctrl && !decisionBusy) {
        key.preventDefault(); key.stopPropagation(); preserveDraft.current = true
        setDecisionCriterion(null); decisionContext.current = null; setDecisionMessage('Draft kept. Refreshing goal details…'); setRefresh(value => value + 1)
      }
      else if ((key.name === 'return' || key.name === 'enter') && !key.shift) { key.preventDefault(); key.stopPropagation(); void submitDecision() }
      return
    }
    if (key.name === 'l') {
      key.preventDefault(); key.stopPropagation(); openSessionFollowups(); return
    }
    if (key.name === 'escape') {
      key.preventDefault()
      key.stopPropagation()
      patchOverlayState({ goal: false })
      return
    }
    if (key.name === 'r') {
      key.preventDefault(); key.stopPropagation()
      if (!inspectionLoading) setRefresh(value => value + 1)
      return
    }
    if (key.name === 'tab' && criteria.length) {
      key.preventDefault(); key.stopPropagation()
      setSelectedCriterion(value => (value + (key.shift ? criteria.length - 1 : 1)) % criteria.length)
      return
    }
    if ((key.name === 'a' || key.name === 'return') && criteria.length && inspected?.phase !== 'complete') {
      key.preventDefault(); key.stopPropagation(); beginDecision(selectedCriterion); return
    }
    const rows = Math.max(1, size.height - 6)
    const delta = key.name === 'down' ? 1 : key.name === 'up' ? -1
      : key.name === 'pagedown' ? rows : key.name === 'pageup' ? -rows : 0
    if (delta || key.name === 'home' || key.name === 'end') {
      key.preventDefault()
      key.stopPropagation()
      if (key.name === 'home') scroll.current?.scrollTo(0)
      else if (key.name === 'end') scroll.current?.scrollTo(Number.MAX_SAFE_INTEGER)
      else scroll.current?.scrollBy(delta)
    }
  })

  const done = todos.filter(todo => todo.status === 'completed').length
  const active = todos.filter(todo => todo.status === 'in_progress').length
  const inspected = inspection?.goal
  const goalObjective = inspection ? inspected?.objective : info?.goal
  const goalPhase = inspection ? inspected?.phase : info?.goal_phase
  const criteria = inspected?.criteria ?? []
  useEffect(() => {
    setSelectedCriterion(value => criteria.length ? Math.min(value, criteria.length - 1) : 0)
  }, [criteria.length])
  const now = Date.now()
  const timeLimit = inspected?.maxDurationMs !== undefined && inspected.createdAt !== undefined
    ? {
        elapsedMs: Math.min(inspected.maxDurationMs, Math.max(0, now - inspected.createdAt)),
        remainingMs: Math.max(0, inspected.createdAt + inspected.maxDurationMs - now),
        expired: now >= inspected.createdAt + inspected.maxDurationMs,
      }
    : null
  const tokenUsage = inspection?.tokenUsage ?? null
  const tokenUsageLine = inspection ? tokenUsageDisplay(inspected, tokenUsage, terminal.width < 60) : null
  const continuationLine = continuationDisplay(inspection?.continuation ?? null, terminal.width < 60)
  return (
    <box position="absolute" left={0} top={0} width="100%" height="100%" zIndex={150}
      backgroundColor="#000000cc" alignItems="center" justifyContent="center">
      <Box backgroundColor={t.color.statusBg} borderColor={t.color.border} borderStyle="round"
        flexDirection="column" width={size.width} height={size.height} paddingX={2} paddingY={terminal.height >= 24 ? 1 : 0}>
        <Text bold color={t.color.text}>Goal & Todos</Text>
        <Text color={t.ds.secondary}>{goalPhase ?? 'No active goal'}{inspected ? ` · rounds ${inspected.roundsStarted}/${inspected.maxGoalRounds === Number.MAX_SAFE_INTEGER ? 'unlimited' : inspected.maxGoalRounds}` : ''} · {done}/{todos.length} done · {active} active</Text>
        {timeLimit ? <Text color={timeLimit.expired ? t.color.warn : t.ds.secondary}>TIME LIMIT · elapsed {formatDuration(timeLimit.elapsedMs)} · remaining {formatDuration(timeLimit.remainingMs)}{timeLimit.expired ? ' · EXPIRED' : ''}</Text> : null}
        {tokenUsageLine ? <Text color={tokenUsageLine.warning ? t.color.warn : t.ds.secondary}>{tokenUsageLine.text}</Text> : null}
        {continuationLine ? <Text color={inspection?.continuation?.state === 'interrupted' ? t.color.warn : t.ds.secondary} wrap="wrap">{continuationLine}{inspection?.continuation?.state === 'queued' && inspected?.activation === 'disarmed' ? ' · /goal resume required' : ''}</Text> : null}
        {inspectionLoading ? <Text color={t.ds.secondary}>Refreshing goal details…</Text> : null}
        {inspectionError && terminal.width >= 60 ? <Text color={t.color.warn} wrap="wrap">Goal details unavailable: {inspectionError}</Text> : null}
        <scrollbox ref={scroll} style={{flexGrow: 1, flexShrink: 1, minHeight: 0}}
          contentOptions={{flexDirection: 'column', paddingRight: 1}}>
          <Box flexDirection="column" flexShrink={0}>
            <Box flexDirection="column" flexShrink={0} paddingY={terminal.height < 24 ? 0 : 1} borderSides={['bottom']} borderColor={t.color.border}>
              <Text bold color={t.color.accent}>OBJECTIVE</Text>
              <Text color={t.color.text} wrap="wrap">{goalObjective || 'No goal set. Use /goal to create one.'}</Text>
            </Box>
            {inspected?.currentMilestone ? <Box flexDirection="column" flexShrink={0} paddingY={terminal.height < 24 ? 0 : 1}>
              <Text bold color={t.ds.secondary}>{inspected.phase === 'complete' ? 'LAST MILESTONE' : 'CURRENT MILESTONE'}</Text>
              <Text color={t.color.text} wrap="wrap">{inspected.currentMilestone}</Text>
            </Box> : null}
            {inspected?.blockedReason ? <Box flexDirection="column" paddingY={terminal.height < 24 ? 0 : 1}>
              <Text bold color={t.color.warn}>BLOCKER · {inspected.blockedReason.code}</Text>
              <Text color={t.color.warn} wrap="wrap">{inspected.blockedReason.message}</Text>
            </Box> : null}
            {inspection ? <Box flexDirection="column" flexShrink={0} paddingY={terminal.height < 24 ? 0 : 1}>
              <Text bold color={t.ds.secondary}>ACCEPTANCE CRITERIA</Text>
              {criteria.length ? criteria.map(criterion => (
                <Box key={criterion.id} flexDirection="column" flexShrink={0} marginTop={1}>
                  <Text color={criterion.evidence ? t.color.ok : selectedCriterion === criteria.indexOf(criterion) ? t.color.accent : t.color.text} wrap="wrap">{selectedCriterion === criteria.indexOf(criterion) ? '›' : criterion.evidence ? '✓' : '◇'} {criterion.description}</Text>
                  {criterion.evidence ? <Text color={criterion.evidence.kind === 'user-decision' ? t.color.ok : t.ds.secondary} wrap="wrap">{criterion.evidence.kind === 'user-decision' ? `Human decision (user confirmed): ${criterion.evidence.summary} · decision ${criterion.evidence.decisionId}` : `Evidence (model-assessed relevance; tool relevance, not human certification): ${criterion.evidence.summary} · tool ${criterion.evidence.toolCallId}`}</Text> : <Text color={t.ds.secondary}>Pending evidence</Text>}
                </Box>
              )) : <Text color={t.ds.secondary}>No declared criteria.</Text>}

            </Box> : null}
            {gateway && sid ? <Box flexDirection="column" flexShrink={0} paddingY={terminal.height < 24 ? 0 : 1}>
              <Text bold color={t.ds.secondary}>FOLLOW-UPS · L manage</Text>
              <SessionFollowups t={t} sessionId={sid} expanded />
            </Box> : null}
            <Box paddingTop={terminal.height < 24 ? 0 : 1}><Text bold color={t.ds.secondary}>TASK PLAN</Text></Box>
            {todos.length ? todos.map(todo => (
              <Box key={todo.id} flexDirection="row" flexShrink={0} marginTop={1} gap={1} paddingX={1} paddingY={terminal.height < 24 ? 0 : 1} backgroundColor={todo.status === 'in_progress' ? t.ds.selected : undefined}>
                <Text color={todo.status === 'completed' ? t.color.ok : todo.status === 'in_progress' ? t.color.accent : t.color.muted}>
                  {todo.status === 'completed' ? '✓' : todo.status === 'in_progress' ? '◌' : '◇'}
                </Text>
                <Box flexDirection="column" flexGrow={1} minWidth={0}>
                  <Text color={todo.status === 'completed' ? t.ds.secondary : t.color.text} wrap="wrap">{todo.content}</Text>
                  <Text color={t.ds.secondary}>{todo.status === 'in_progress' ? 'In progress' : todo.status === 'completed' ? 'Completed' : todo.status === 'cancelled' ? 'Cancelled' : 'Planned'}</Text>
                </Box>
              </Box>
            )) : <Text color={t.ds.secondary}>No todos yet.</Text>}
          </Box>
        </scrollbox>
        {decisionCriterion !== null && decisionContext.current ? <Box flexDirection="column" flexShrink={0} marginTop={1} paddingX={1} paddingY={terminal.height < 24 ? 0 : 1} borderStyle="round" borderColor={t.color.accent}>
          <Text bold color={t.color.accent}>ACCEPT CRITERION · HUMAN DECISION</Text>
          <Text wrap="wrap">{decisionContext.current.description}</Text>
          <textarea key={`${decisionContext.current.sessionId}:${decisionContext.current.goalId}:${decisionContext.current.revision}:${decisionContext.current.criterionId}`} ref={decisionInput} focused={!decisionBusy} keyBindings={DECISION_KEY_BINDINGS} minHeight={2} maxHeight={3} placeholder="Explain why you accept this criterion…" placeholderColor={t.color.muted} onContentChange={() => setDecisionDraft(decisionInput.current?.plainText ?? '')} onSubmit={() => { void submitDecision() }} />
          <Text color={decisionMessage.includes('kept') || decisionMessage.startsWith('Enter') ? t.color.warn : t.ds.secondary} wrap="wrap">{decisionMessage || 'Enter accept · Ctrl+R review latest · Esc cancel'}</Text>
        </Box> : null}
        <Text color={t.ds.secondary}>{decisionCriterion !== null ? 'Enter accept · Ctrl+R review · Esc cancel' : terminal.width < 60 ? criteria.length && inspected?.phase !== 'complete' ? 'Tab/A accept · Esc close' : '↑↓ scroll · Esc close' : 'Tab select · A/Enter accept · ↑↓ scroll · R refresh · PgUp/PgDn page · L follow-ups · Esc close'}</Text>
      </Box>
    </box>
  )
}
