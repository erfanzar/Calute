// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useEffect, useRef, useState } from 'react'
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import { useGateway } from '../app/gatewayContext.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'
import { ModalShell } from './pickerChrome.js'
import { windowItems } from './overlayLayout.js'

export function MachineBrowser({ t, target, initialPath = '', onSelect, onCancel }: {
  t: Theme; target?: string; initialPath?: string; onSelect: (value: string) => void; onCancel: () => void
}) {
  const gateway = useGateway()
  const { width, height } = useTerminalDimensions()
  const [path, setPath] = useState(initialPath)
  const [items, setItems] = useState<string[]>([])
  const [index, setIndex] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [truncated, setTruncated] = useState(false)
  const generation = useRef(0)
  const load = async (folder: string) => {
    const request = ++generation.current
    setLoading(true); setError('')
    try {
      const command = target ? `machine browse ${target}${folder ? ` ${Buffer.from(folder).toString('base64url')}` : ''}` : 'machine hosts'
      const result = asRpcResult(await gateway.rpc('slash.exec', { command }))
      if (request !== generation.current) return
      if (!result?.ok) throw new Error(result?.error ?? 'Could not load choices')
      const entries = target ? result.directories : result.hosts
      if (!Array.isArray(entries) || !entries.every(entry => typeof entry === 'string')) throw new Error('Invalid machine browser response')
      if (target && (typeof result.path !== 'string' || !result.path.startsWith('/'))) throw new Error('Invalid remote folder')
      if (target) setPath(result.path as string)
      setItems(entries as string[]); setIndex(0); setTruncated(result.truncated === true)
    } catch (cause) { if (request === generation.current) setError(rpcErrorMessage(cause)) }
    finally { if (request === generation.current) setLoading(false) }
  }
  useEffect(() => { void load(initialPath); return () => { generation.current++ } }, [gateway, target])
  const parent = () => path.replace(/\/+$/, '').replace(/\/[^/]*$/, '') || '/'
  const open = () => {
    if (!items[index]) return
    if (target) void load(`${path === '/' ? '' : path}/${items[index]}`)
    else onSelect(items[index]!)
  }
  useKeyboard(key => {
    key.preventDefault(); key.stopPropagation()
    if (key.name === 'escape') { onCancel(); return }
    if (loading) return
    if (key.name === 'r') { void load(path); return }
    if (key.name === 'up') setIndex(old => Math.max(0, old - 1))
    if (key.name === 'down') setIndex(old => Math.min(items.length - 1, old + 1))
    if (key.name === 'home') setIndex(0)
    if (key.name === 'end') setIndex(Math.max(0, items.length - 1))
    if (key.name === 'return' || key.name === 'enter' || key.name === 'right') { if (!error) open() }
    if (target && (key.name === 'backspace' || key.name === 'left')) void load(parent())
    if (target && key.name === 'space' && !error) onSelect(path)
  })
  const visible = Math.max(1, Math.min(16, height - 12))
  const window = windowItems(items, index, visible)
  return <ModalShell height={height} width={width} panelHeight={Math.min(height, 30)} panelWidth={Math.min(88, Math.max(1, width - 4))} t={t} title={target ? 'Choose project folder' : 'Choose SSH host'}>
    <box flexDirection="column" flexGrow={1} minHeight={0} paddingLeft={1} paddingRight={1}>
      <text fg={t.color.accent} truncate wrapMode="none">{target ? `${target} · ${path || 'Home directory'}` : '~/.ssh/config · included files'}</text>
      <box flexDirection="column" flexGrow={1} minHeight={0} marginTop={1}>
        {loading ? <text fg={t.color.muted}>Loading…</text> : error ? <text fg={t.color.error}>{error}</text> : !items.length ? <text fg={t.color.muted}>{target ? 'No subfolders. Space selects this folder.' : 'No SSH aliases found. Esc returns to manual entry.'}</text> : window.items.map((item, i) => <box key={item} height={1} backgroundColor={index === window.offset + i ? t.color.completionCurrentBg : undefined} onMouseDown={() => setIndex(window.offset + i)}><text fg={index === window.offset + i ? t.color.accent : t.color.text} truncate wrapMode="none">{`${index === window.offset + i ? '›' : ' '} ${target ? '▸ ' : ''}${item}`}</text></box>)}
      </box>
      {truncated ? <text fg={t.color.muted}>First 1,000 folders shown; use a more specific path.</text> : null}
      <text fg={t.color.muted}>{target ? '↵ open · Space choose · ← parent' : '↑↓ select · Enter choose'}</text>
      <text fg={t.color.muted}>R reload · Esc back</text>
    </box>
  </ModalShell>
}
