// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import { useEffect, useRef, useState } from 'react'

import { useGateway } from '../app/gatewayContext.js'
import { connectRemoteMachine, parseRemoteMachine, type RemoteMachine } from '../lib/machineHandoff.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'
import { windowItems } from './overlayLayout.js'
import { InfoRow, ModalShell } from './pickerChrome.js'

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

  useKeyboard(key => {
    if (['escape', 'up', 'down', 'return', 'enter'].includes(key.name)) {
      key.preventDefault(); key.stopPropagation()
    }
    if (key.name === 'escape') { controller.current?.abort(); onCancel(); return }
    if (busy || loading) return
    if (key.name === 'up') setIndex(old => Math.max(0, old - 1))
    if (key.name === 'down') setIndex(old => Math.min(Math.max(0, machines.length - 1), old + 1))
    if (key.name === 'return' || key.name === 'enter') void open()
  })

  const visible = Math.max(1, Math.min(6, Math.floor((height - 13) / 2)))
  const window = windowItems(machines, index, visible)
  return (
    <ModalShell height={height} width={width} panelHeight={Math.min(height, 13 + Math.min(visible, machines.length) * 2)} panelWidth={Math.min(104, Math.max(1, width - 4))} t={t} title="Remote workspaces">
      <InfoRow color={t.color.muted}>Connect over SSH · exit the remote session to return here</InfoRow>
      {loading ? <InfoRow color={t.color.muted}>Loading saved machines…</InfoRow> : null}
      {!loading && machines.length === 0 ? <InfoRow color={t.color.text}>No remote workspaces saved yet.</InfoRow> : null}
      {window.items.map((machine, position) => <box key={machine.alias} flexShrink={0} paddingLeft={2} paddingRight={2} backgroundColor={index === window.offset + position ? t.color.completionCurrentBg : undefined} onMouseDown={() => setIndex(window.offset + position)}>
        <text fg={t.color.accent} truncate wrapMode="none">{`${index === window.offset + position ? '›' : ' '} ${machine.alias} · ${machine.target}`}</text>
        <text fg={t.color.muted} truncate wrapMode="none">{`  ${machine.workspacePath}`}</text>
      </box>)}
      <InfoRow color={t.color.muted}>/machine add &lt;name&gt; &lt;ssh-target&gt; &lt;absolute-path&gt;</InfoRow>
      <InfoRow color={t.color.muted}>/machine remove &lt;name&gt;</InfoRow>
      {error ? <InfoRow color={t.color.error}>{error}</InfoRow> : null}
      {notice ? <InfoRow color={t.color.text}>{notice}</InfoRow> : null}
      <InfoRow color={t.color.muted}>{busy ? 'Connecting…' : '↑/↓ select · Enter connect · Esc close'}</InfoRow>
    </ModalShell>
  )
}
