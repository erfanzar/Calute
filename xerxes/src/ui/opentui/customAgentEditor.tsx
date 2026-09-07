// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import type { TextareaRenderable } from '@opentui/core'
import { useEffect, useRef, useState } from 'react'
import { useOptionalGateway } from '../app/gatewayContext.js'
import type { Theme } from '../theme.js'
import { Box, Text } from './primitives.js'
import { DialogHeader, DialogFooter, DialogEmpty } from './dialogChrome.js'
import { overlayPanelSize } from './overlayLayout.js'

type Row = { id: string; description: string; error?: string }
type Draft = { id: string; content: string; revision: string | null }
export function CustomAgentEditor({ t, onClose }: { t: Theme; onClose: () => void }) {
  const gateway = useOptionalGateway()
  const terminal = useTerminalDimensions()
  const size = overlayPanelSize(terminal, { maxWidth: 104, minWidth: 32, maxHeight: 34 })
  const [rows, setRows] = useState<Row[]>([])
  const [selected, setSelected] = useState(0)
  const [draft, setDraft] = useState<Draft | null>(null)
  const [refresh, setRefresh] = useState(0)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const input = useRef<TextareaRenderable | null>(null)
  const alive = useRef(true)
  useEffect(() => { alive.current = true; return () => { alive.current = false } }, [])
  useEffect(() => {
    let current = true
    if (!gateway) { setError('Connect to a daemon to edit agents.'); return }
    void gateway.rpc('agentPreset.projectList', {}).then(result => {
      if (!current) return
      if (!result?.ok || !Array.isArray(result.agents)) throw new Error(String(result?.error || 'Could not load agents'))
      const agents = result.agents.map((value: unknown): Row => {
        if (!value || typeof value !== 'object' || !('id' in value) || typeof value.id !== 'string' || !('description' in value) || typeof value.description !== 'string') throw new Error('Invalid custom agent record')
        return { id: value.id, description: value.description, ...('error' in value && typeof value.error === 'string' ? { error: value.error } : {}) }
      })
      setRows(agents)
    }).catch(failure => { if (current) setError(String(failure)) })
    return () => { current = false }
  }, [gateway, refresh])
  useEffect(() => { if (draft) input.current?.setText(draft.content) }, [draft?.id])
  const edit = async () => {
    if (!gateway || !rows[selected]) return
    setBusy(true); setError('')
    try {
      const result = await gateway.rpc('agentPreset.projectRead', { id: rows[selected]!.id })
      if (!result?.ok || typeof result.content !== 'string' || typeof result.revision !== 'string') throw new Error(String(result?.error || 'Could not read agent'))
      if (alive.current) setDraft({ id: rows[selected]!.id, content: result.content, revision: result.revision })
    } catch (failure) { if (alive.current) setError(String(failure)) }
    finally { if (alive.current) setBusy(false) }
  }
  const save = async () => {
    if (!gateway || !draft) return
    setBusy(true); setError('')
    const content = input.current?.plainText ?? draft.content
    const id = draft.revision === null ? '' : draft.id
    try {
      const result = await gateway.rpc('agentPreset.projectWrite', { id, content, revision: draft.revision })
      if (!result?.ok) throw new Error(String(result?.error || 'Could not save agent'))
      if (alive.current) { setDraft(null); setRefresh(value => value + 1) }
    } catch (failure) { if (alive.current) setError(String(failure)) }
    finally { if (alive.current) setBusy(false) }
  }
  const beginNew = () => { if (busy) return;  setError(''); setDraft({ id: 'new-agent', revision: null, content: '---\nname: new-agent\ndescription: Describe when to delegate to this specialist.\n---\nYou are a specialist. Describe your instructions here.\n' })  }
  useKeyboard(key => {
    if (key.eventType === 'release') return
    if (key.name === 'escape') { key.preventDefault(); key.stopPropagation(); if (!busy) { setError(''); if (draft) setDraft(null); else onClose() }; return }
    if (draft) {
      if (key.ctrl && key.name === 's') { key.preventDefault(); key.stopPropagation(); if (!busy) void save() }
      return
    }
    if (!['n', 'return', 'up', 'down'].includes(key.name)) return
    key.preventDefault(); key.stopPropagation()
    if (busy) return
    if (key.name === 'up') setSelected(value => Math.max(0, value - 1))
    else if (key.name === 'down') setSelected(value => Math.min(Math.max(0, rows.length - 1), value + 1))
    else if (key.name === 'return') void edit()
    else beginNew()
  })
  const count = Math.max(1, size.height - 16)
  const start = Math.max(0, selected - count + 1)
  return <box position="absolute" left={0} top={0} width="100%" height="100%" zIndex={150} backgroundColor="#000000cc" alignItems="center" justifyContent="center">
    <Box width={!draft && !rows.length ? Math.min(88, size.width) : size.width} height={!draft && !rows.length ? Math.min(24, size.height) : size.height} flexDirection="column" paddingX={1} borderStyle="round" borderColor={t.color.border} backgroundColor={t.color.statusBg}>
      <DialogHeader t={t} title={draft ? `Edit agent · ${draft.id}` : 'Custom agents · this project'} subtitle="Specialists with a purpose. Build a team for this repository." />
      <Text color={t.ds.secondary} wrap="wrap">{draft ? 'Edit Markdown and YAML frontmatter. New agents use the name field as their filename.' : '.xerxes/agents · specialists available for delegation'}</Text>
      {error ? <Text color={t.color.warn} wrap="wrap">{error}</Text> : null}
      {draft ? <textarea key={draft.id} ref={input} flexGrow={1} minHeight={1} focused={!busy} initialValue={draft.content} focusedBackgroundColor={t.color.statusBg} focusedTextColor={t.color.text} /> : <Box flexGrow={1} minHeight={0} flexDirection="column">
        {rows.length ? rows.slice(start, start + count).map((row, index) => <Text key={row.id} color={start + index === selected ? t.color.accent : t.color.text} wrap="truncate-end">{start + index === selected ? '› ' : '  '}{row.id} · {row.error ? 'Needs repair' : row.description}</Text>) : <DialogEmpty t={t} title="No custom agents." description="Give a specialist a role, instructions and tools." action="Press N to create one." onAction={beginNew} symbol="✦" />}
      </Box>}
      <DialogFooter t={t}><Text color={t.ds.secondary} wrap="wrap">{busy ? 'Saving / loading…' : !draft && !rows.length ? 'N new · Esc close' : draft ? 'Ctrl+S save · Esc discard draft' : '↑↓ select · Enter edit · N new · Esc close'}</Text></DialogFooter>
    </Box>
  </box>
}
