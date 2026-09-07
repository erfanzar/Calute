// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useTerminalDimensions } from '@opentui/react'
import type { ScrollBoxRenderable } from '@opentui/core'
import { useEffect, useRef, type ReactNode } from 'react'
import type { Theme } from '../theme.js'
import { Box, Text } from './primitives.js'

export interface SettingsFormField { id: number; label: string; value: string; group: string }

/** Common presentation for operational forms; values and actions stay with their owners. */
export function SettingsFormLayout({ t, title, subtitle, fields, selected, onSelect, editor, help, compactHelp, summary, error, busy }: {
  t: Theme; title: string; subtitle: string; fields: SettingsFormField[]; selected: number;
  onSelect: (id: number) => void; editor?: ReactNode; help: ReactNode; compactHelp?: ReactNode; summary?: ReactNode; error: string; busy: string;
}) {
  const terminal = useTerminalDimensions()
  const compact = terminal.height < 26
  const wide = terminal.width >= 100 && !compact
  const scroll = useRef<ScrollBoxRenderable | null>(null)
  const index = fields.findIndex(row => row.id === selected)
  const active = fields[index]
  useEffect(() => {
    const timer = setTimeout(() => scroll.current?.scrollChildIntoView(`setting-field-${selected}`), 0)
    return () => clearTimeout(timer)
  }, [selected])
  const visible = compact ? fields.slice(Math.max(0, index - 1), index + 2) : fields
  return <Box flexDirection="column" flexGrow={1} minHeight={0} paddingX={terminal.width >= 70 ? 1 : 0}>
    <Box borderStyle="single" borderSides={['bottom']} borderColor={t.color.border} paddingBottom={compact ? 0 : 1} flexShrink={0} flexDirection="column">
      <Text bold color={t.color.text}><span fg={t.color.accent}>✦ </span>{title}</Text>
      {!compact ? <Text color={t.ds.meta} wrap="wrap">{subtitle}</Text> : null}
    </Box>
    <Box flexDirection={wide ? 'row' : 'column'} flexGrow={1} minHeight={0} paddingTop={compact ? 0 : 1} gap={wide ? 2 : 0}>
      <scrollbox ref={scroll} style={{ flexGrow: 1, flexBasis: 0, minHeight: compact ? 3 : 4, minWidth: 0 }} contentOptions={{ flexDirection: 'column' }}>
        {visible.map((row, position) => <Box key={row.id} id={`setting-field-${row.id}`} flexDirection="column" flexShrink={0}>
          {position === 0 || visible[position - 1]?.group !== row.group ? <Box marginTop={position && !compact ? 1 : 0} paddingBottom={compact ? 0 : 1}><Text bold color={t.color.accent}>{row.group}</Text></Box> : null}
          <Box paddingX={1} backgroundColor={selected === row.id ? t.color.selectionBg : undefined} onClick={() => { if (!busy) onSelect(row.id) }}>
            <Text color={selected === row.id ? t.color.text : t.ds.secondary} wrap="wrap">{selected === row.id ? '› ' : '  '}{row.label}: {row.value}</Text>
          </Box>
        </Box>)}
      </scrollbox>
      <Box width={wide ? '38%' : '100%'} flexShrink={0} flexDirection="column" minHeight={0}>
        <Box borderStyle="round" borderColor={t.color.accent} backgroundColor={t.color.completionMetaBg} paddingX={1} paddingY={compact ? 0 : 1} flexDirection="column" flexShrink={0}>
          <Text color={t.color.accent}>{`EDIT · ${index + 1}/${fields.length}`}</Text>
          <Text bold color={t.color.text} wrap="wrap">{active?.label}</Text>
          {editor || <Text color={t.color.text} wrap="wrap">{active?.value}</Text>}
          {!compact ? <Text color={t.ds.meta} wrap="wrap">{editor ? 'Type to edit · Tab to continue' : '← → or Space to choose'}</Text> : null}
        </Box>
        {!compact ? <Box paddingTop={1} flexDirection="column" flexShrink={0}>{help}</Box> : null}
        {compact && compactHelp ? <Box flexDirection="column" flexShrink={0}>{compactHelp}</Box> : null}
        {!compact && summary ? <Box marginTop={1} borderStyle="single" borderSides={['top']} borderColor={t.color.border} paddingTop={1} flexDirection="column"><Text color={t.color.accent}>RUN PREVIEW</Text>{summary}</Box> : null}
      </Box>
    </Box>
    {error ? <Text color={t.color.warn} wrap="wrap">{error}</Text> : null}
    <Box borderStyle="single" borderSides={['top']} borderColor={t.color.border} marginTop={compact ? 0 : 1} paddingTop={compact ? 0 : 1} flexShrink={0} flexDirection="column">
      <Text color={t.color.text}>{busy || 'Tab / Shift+Tab field · ←→ choose · Enter save'}</Text>
      <Text color={t.ds.meta}>Esc back</Text>
    </Box>
  </Box>
}
