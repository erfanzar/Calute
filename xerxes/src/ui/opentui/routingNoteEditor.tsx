// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useKeyboard } from '@opentui/react'
import type { TextareaRenderable } from '@opentui/core'
import { useEffect, useRef, useState } from 'react'
import { useOptionalGateway } from '../app/gatewayContext.js'
import type { Theme } from '../theme.js'
import { Box, Text } from './primitives.js'
interface Note { note: string; revision: number }
interface Response { ok: boolean; routing_note?: Note; error?: string }
export function RoutingNoteEditor({ t, profile, model, onClose }: { t: Theme; profile: string; model: string; onClose: () => void }) {
  const gateway = useOptionalGateway()
  const [notes, setNotes] = useState<Note[]>([])
  const [field, setField] = useState(0)
  const [reload, setReload] = useState(0)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const input = useRef<TextareaRenderable | null>(null)
  const pending = useRef(false), alive = useRef(true)
  useEffect(() => {
    let cancelled = false
    alive.current = true
    setError('')
    if (!gateway) { setError('Gateway unavailable'); return }
    void Promise.all((model ? ['', model] : ['']).map(value => gateway.rpc<Response>('model.routing_note.get', { provider_profile: profile, model: value }))).then(results => {
      if (cancelled) return
      setNotes(results.map(result => { if (!result?.ok || !result.routing_note) throw new Error(result?.error || 'Could not load routing notes'); return result.routing_note }))
    }).catch(failure => { if (!cancelled) setError(String(failure)) })
    return () => { cancelled = true; alive.current = false }
  }, [gateway, profile, model, reload])
  useEffect(() => { input.current?.setText(notes[field]?.note ?? '') }, [field, notes.length])
  const save = () => {
    const current = notes[field]
    if (!gateway || !current || pending.current) return
    if (current.note.length > 2000) { setError('Notes must be at most 2000 characters.'); return }
    pending.current = true; setBusy(true); setError('')
    void gateway.rpc<Response>('model.routing_note.save', { provider_profile: profile, model: field ? model : '', ...current }).then(result => {
      if (!result?.ok || !result.routing_note) throw new Error(result?.error || 'Could not save routing note')
      const saved = { note: result.routing_note.note, revision: result.routing_note.revision }
      if (alive.current) { setNotes(previous => previous.map((value, index) => index === field ? saved : value)); input.current?.setText(saved.note); setError('Saved. Agents can read this preference through model discovery.') }
    }).catch(failure => { if (alive.current) setError(String(failure)) }).finally(() => { pending.current = false; if (alive.current) setBusy(false) })
  }
  useKeyboard(key => {
    if (key.eventType === 'release') return
    if (!['escape', 'tab', 'f2', 'f5'].includes(key.name)) return
    key.preventDefault(); key.stopPropagation()
    if (busy) return
    if (key.name === 'escape') onClose()
    else if (key.name === 'tab' && notes.length) setField(value => (value + 1) % notes.length)
    else if (key.name === 'f2') save()
    else if (key.name === 'f5' && !notes.length) setReload(value => value + 1)
  })
  return <Box flexDirection="column" flexGrow={1} minHeight={0}>
    <Text bold>Routing preferences</Text>
    <Text wrap="wrap">{profile} · {field ? model : 'All models in this provider profile'}</Text>
    <Text wrap="wrap">User guidance for agent selection. These notes do not enforce limits. Blank saves remove guidance.</Text>
    <scrollbox style={{ flexGrow: 1, minHeight: 0 }} contentOptions={{ flexDirection: 'column' }}>
      {notes.length ? <textarea key={field} ref={input} focused={!busy} minHeight={4} maxHeight={10} onContentChange={() => { const note = input.current?.plainText ?? ''; setNotes(previous => previous.map((value, index) => index === field ? { ...value, note } : value)) }} /> : <Text>{error ? 'Notes unavailable · F5 retry' : 'Loading notes…'}</Text>}
      {error ? <Text wrap="wrap" color={t.color.warn}>{error}</Text> : null}
    </scrollbox>
    <Text wrap="wrap">{busy ? 'Saving…' : 'Tab provider/model · F2 save selected note · Esc back'}</Text>
  </Box>
}
