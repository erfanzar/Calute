// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import { useEffect, useRef, useState } from 'react'
import type { TextareaRenderable } from '@opentui/core'

import { useGateway } from '../app/gatewayContext.js'
import { connectRemoteMachine, parseRemoteMachine, type RemoteMachine } from '../lib/machineHandoff.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'
import { windowItems } from './overlayLayout.js'
import { InfoRow, ModalShell } from './pickerChrome.js'
import { MachineBrowser } from './machineBrowser.js'

export interface MachinePickerProps {
  t: Theme
  onCancel: () => void
  connect?: typeof connectRemoteMachine
}

export function MachinePicker({ t, onCancel, connect = connectRemoteMachine }: MachinePickerProps) {
  const gateway = useGateway()
  const { height, width } = useTerminalDimensions()
  const [machines, setMachines] = useState<RemoteMachine[]>([])
  const [index, setIndex] = useState(0)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [editing, setEditing] = useState(false)
  const [field, setField] = useState(0)
  const [draft, setDraft] = useState(['', '', ''])
  const [browser, setBrowser] = useState<'hosts' | 'folders' | null>(null)
  const input = useRef<TextareaRenderable | null>(null)
  const active = useRef(true)
  const controller = useRef<AbortController | null>(null)

  useEffect(() => {
    active.current = true
    void gateway.rpc('slash.exec', { command: 'machine list' }).then(raw => {
      if (!active.current) return
      const result = asRpcResult(raw)
      if (!result?.ok || !Array.isArray(result.machines)) throw new Error(result?.error ?? 'Invalid machine list response')
      setMachines(result.machines.map(parseRemoteMachine))
    }).catch(cause => { if (active.current) setError(rpcErrorMessage(cause)) })
      .finally(() => { if (active.current) setLoading(false) })
    return () => { active.current = false; controller.current?.abort() }
  }, [gateway])

  const open = async () => {
    const selected = machines[index]
    if (!selected || busy || loading) return
    setBusy(true)
    setError('')
    setNotice('')
    const cancellation = new AbortController()
    controller.current = cancellation
    try {
      const raw = await gateway.rpc('slash.exec', { command: `machine connect ${selected.alias}` })
      if (!active.current || cancellation.signal.aborted) return
      const result = asRpcResult(raw)
      if (!result?.ok) throw new Error(result?.error ?? 'Could not connect to machine')
      await connect(parseRemoteMachine(result.machine), { signal: cancellation.signal })
      if (active.current) setNotice('Remote session closed. You are back in your local workspace.')
    } catch (cause) {
      if (active.current) setError(rpcErrorMessage(cause))
    } finally {
      controller.current = null
      if (active.current) setBusy(false)
    }
  }

  const add = () => { setDraft(['', '', '']); setField(0); setError(''); setNotice(''); setEditing(true) }
  const save = async () => {
    if (busy) return
    const values = [...draft]
    values[field] = input.current?.plainText ?? values[field]!
    setDraft(values)
    // The daemon's quoted-argument grammar has no escapes. Choose the other
    // quote for paths containing quotes; refuse ambiguous input explicitly.
    const quote = (value: string) => {
      if (value.includes('"') && value.includes("'")) throw new Error('Use a path without mixed quote characters')
      const delimiter = value.includes('"') ? "'" : '"'
      return delimiter + value + delimiter
    }
    setBusy(true); setError('')
    try {
      const result = asRpcResult(await gateway.rpc('slash.exec', { command: `machine add ${values.map(quote).join(' ')}` }))
      if (!active.current) return
      if (!result?.ok || !Array.isArray(result.machines)) throw new Error(result?.error ?? 'Could not save workspace')
      const saved = result.machines.map(parseRemoteMachine)
      setMachines(saved); setIndex(Math.max(0, saved.findIndex(machine => machine.alias === values[0])))
      setEditing(false); setNotice('Workspace saved. Press Enter to connect.')
    } catch (cause) { if (active.current) setError(rpcErrorMessage(cause)) }
    finally { if (active.current) setBusy(false) }
  }

  useKeyboard(key => {
    if (browser) return
    if (['escape', 'up', 'down', 'return', 'enter', 'tab'].includes(key.name) || (!editing && key.name === 'n')) {
      key.preventDefault(); key.stopPropagation()
    }
    if (key.name === 'escape') {
      if (editing) { if (!busy) { setEditing(false); setError('') }; return }
      controller.current?.abort(); onCancel(); return
    }
    if (busy || loading) return
    if (editing) {
      if (key.name === 'f2' && field > 0) {
        key.preventDefault(); key.stopPropagation()
        const values = [...draft]; values[field] = input.current?.plainText ?? values[field]!
        setDraft(values)
        if (field === 2 && !values[1]?.trim()) { setError('Choose an SSH host first'); return }
        setError(''); setBrowser(field === 1 ? 'hosts' : 'folders'); return
      }
      if (key.name === 'tab' || key.name === 'up' || key.name === 'down') {
        const text = input.current?.plainText
        setDraft(old => old.map((value, i) => i === field ? text ?? value : value))
        setField(old => (old + (key.shift || key.name === 'up' ? 2 : 1)) % 3)
      }
      if (key.name === 'return' || key.name === 'enter') void save()
      return
    }
    if (key.name === 'n') { add(); return }
    if (key.name === 'up') setIndex(old => Math.max(0, old - 1))
    if (key.name === 'down') setIndex(old => Math.min(Math.max(0, machines.length - 1), old + 1))
    if (key.name === 'return' || key.name === 'enter') { if (!machines.length) add(); else void open() }
  })

  const compact = height < 25 || width < 65
  const visible = Math.max(1, Math.min(5, Math.floor((height - 16) / 3)))
  const window = windowItems(machines, index, visible)
  const labels = ['Workspace name', 'SSH host', 'Project folder']
  const examples = ['e.g. training-server', 'e.g. me@gpu-host or an SSH alias', 'e.g. /home/me/projects/my-app']
  if (browser) return <MachineBrowser t={t} target={browser === 'folders' ? draft[1] : undefined} initialPath={browser === 'folders' ? draft[2] : ''} onCancel={() => setBrowser(null)} onSelect={value => { setDraft(old => old.map((item, i) => i === (browser === 'hosts' ? 1 : 2) ? value : item)); setBrowser(null) }} />
  return (
    <ModalShell height={height} width={width} panelHeight={Math.min(height, editing ? 26 : machines.length ? 16 + Math.min(visible, machines.length) * 3 : compact ? 18 : 26)} panelWidth={Math.min(88, Math.max(1, width - 4))} t={t} title={editing ? 'Add workspace' : 'Remote workspaces'} headerRight={<text flexShrink={0} fg={t.color.muted}>{editing ? 'SSH' : `${machines.length} saved`}</text>}>
      <box paddingLeft={2} paddingRight={2} flexDirection="column" flexGrow={1} minHeight={0}>
      <box border={['bottom']} borderColor={t.color.border} paddingBottom={compact ? 0 : 1} marginBottom={compact ? 0 : 1}><text flexShrink={0} fg={t.color.muted} truncate wrapMode="none">Your projects, on any machine.</text></box>
      {loading ? <InfoRow color={t.color.muted}>Loading saved machines…</InfoRow> : null}
      {editing ? <box flexDirection="column" flexGrow={1} minHeight={0}>
        {labels.map((label, i) => <box key={label} flexDirection="column" flexShrink={0} marginBottom={compact ? 0 : 1} paddingLeft={1} paddingRight={1} backgroundColor={field === i ? t.color.completionCurrentBg : undefined}>
          <text flexShrink={0} fg={field === i ? t.color.accent : t.color.muted}>{`${i + 1}  ${label}${i > 0 ? ' · F2 browse' : ''}`}</text>
          {field === i ? <textarea key={field} ref={input} initialValue={draft[i]} focused={!busy} height={1} minHeight={1} maxHeight={1} placeholder={examples[i]} focusedBackgroundColor={t.color.completionCurrentBg} focusedTextColor={t.color.text} /> : <text flexShrink={0} fg={draft[i] ? t.color.text : t.color.muted} truncate wrapMode="none">{draft[i] || examples[i]}</text>}
        </box>)}
      </box> : !loading && !machines.length ? <box flexDirection="column" flexGrow={1} justifyContent="center" alignItems="center" gap={compact ? 0 : 1}>
        <text flexShrink={0} fg={t.color.accent}>{'┌─────┐      ┌─────┐\n│  >_ │ ──── │  >_ │\n└─────┘      └─────┘'}</text>
        <text flexShrink={0} fg={t.color.text}><b>No remote workspaces yet</b></text>
        {!compact ? <text flexShrink={0} fg={t.color.muted}>Bring a remote project into your terminal.</text> : null}
        <box backgroundColor={t.color.completionCurrentBg} paddingLeft={2} paddingRight={2} onMouseDown={add}><text flexShrink={0} fg={t.color.accent}><b>+ Add workspace   ↵</b></text></box>
      </box> : <box flexDirection="column" flexGrow={1} minHeight={0}>{window.items.map((machine, position) => <box key={machine.alias} flexDirection="column" flexShrink={0} paddingLeft={1} paddingRight={1} paddingBottom={1} backgroundColor={index === window.offset + position ? t.color.completionCurrentBg : undefined} onMouseDown={() => setIndex(window.offset + position)}>
        <text flexShrink={0} fg={t.color.accent} truncate wrapMode="none">{`${index === window.offset + position ? '›' : ' '} ${machine.alias} · ${machine.target}`}</text>
        <text flexShrink={0} fg={t.color.muted} truncate wrapMode="none">{`  ${machine.workspacePath}`}</text>
      </box>)}</box>}
      {error ? <text flexShrink={0} fg={t.color.error} wrapMode="word">{error}</text> : null}
      {notice ? <text flexShrink={0} fg={t.color.text} wrapMode="word">{notice}</text> : null}
      {!compact && !editing ? <box marginTop={1}><text flexShrink={0} fg={t.color.muted}>Local rendering · remote execution over SSH. Exit to return here.</text></box> : null}
      <box border={['top']} borderColor={t.color.border} paddingTop={compact ? 0 : 1} marginTop={1} flexShrink={0}><text flexShrink={0} fg={t.color.muted} truncate wrapMode="none">{busy ? editing ? 'Saving…' : 'Connecting…' : editing ? 'Tab next · Enter save · Esc back' : machines.length ? '↑↓ select · Enter connect · N add · Esc close' : 'Enter add workspace · Esc close'}</text></box>
      </box>
    </ModalShell>
  )
}
