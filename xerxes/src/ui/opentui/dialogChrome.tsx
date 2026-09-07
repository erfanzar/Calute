// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useTerminalDimensions } from '@opentui/react'
import type { ReactNode } from 'react'
import type { Theme } from '../theme.js'
import { Box, Text } from './primitives.js'

/** One visual hierarchy for operational dialogs; small terminals spend rows on content. */
export function DialogHeader({ t, title, subtitle }: { t: Theme; title: ReactNode; subtitle?: ReactNode }) {
  const { height } = useTerminalDimensions()
  if (height < 26) return <Text bold color={t.color.text} wrap="truncate-end">{title}</Text>
  return <Box flexDirection="column" flexShrink={0} borderStyle="single" borderSides={['bottom']} borderColor={t.color.border} paddingTop={height >= 26 ? 1 : 0} paddingBottom={height >= 26 ? 1 : 0} marginBottom={height >= 26 ? 1 : 0}>
    <Text bold color={t.color.text} wrap="truncate-end"><span fg={t.color.accent}>✦ </span>{title}</Text>
    {subtitle && height >= 26 ? <Text color={t.ds.secondary} wrap="truncate-end">{subtitle}</Text> : null}
  </Box>
}

export function DialogFooter({ t, children }: { t: Theme; children: ReactNode }) {
  const { height } = useTerminalDimensions()
  if (height < 26) return <Box flexDirection="column" flexShrink={0}>{children}</Box>
  return <Box flexDirection="column" flexShrink={0} borderStyle="single" borderSides={['top']} borderColor={t.color.border} paddingTop={height >= 26 ? 1 : 0} marginTop={height >= 26 ? 1 : 0}>{children}</Box>
}

export function DialogEmpty({ t, title, description, action, onAction, symbol = '◇' }: { t: Theme; title: string; description: string; action?: string; onAction?: () => void; symbol?: string }) {
  const { height, width } = useTerminalDimensions()
  const roomy = height >= 26 && width >= 70
  return <Box flexDirection="column" flexGrow={1} minHeight={0} justifyContent="center" alignItems="center" paddingX={roomy ? 2 : 0} gap={roomy ? 1 : 0} overflow="hidden">
    {roomy ? <Text color={t.color.accent}>{symbol}</Text> : null}
    <Text bold color={t.color.text} wrap="wrap">{title}</Text>
    {roomy || !action ? <Text color={t.ds.secondary} wrap={roomy ? 'wrap' : 'truncate-end'}>{description}</Text> : null}
    {action ? <Box paddingX={1} backgroundColor={t.color.completionCurrentBg} onMouseDown={onAction}><Text color={t.color.accent} wrap="wrap">{action}</Text></Box> : null}
  </Box>
}

export function DialogSection({ t, children }: { t: Theme; children: ReactNode }) {
  const { height } = useTerminalDimensions()
  return <Box flexShrink={0} marginTop={height >= 26 ? 1 : 0}><Text bold color={t.color.accent}>{children}</Text></Box>
}

export function SettingRow({ t, label, children, selected = false }: { t: Theme; label: string; children: ReactNode; selected?: boolean }) {
  const { width, height } = useTerminalDimensions()
  if (height < 26) return <Box flexShrink={0} backgroundColor={selected ? t.color.completionCurrentBg : undefined}><Text color={selected ? t.color.accent : t.color.text} wrap="wrap">{selected ? '› ' : '  '}{label}: {children}</Text></Box>
  const wide = width >= 85
  return <Box flexDirection={wide ? 'row' : 'column'} flexShrink={0} paddingX={1} marginBottom={height >= 26 ? 1 : 0} backgroundColor={selected ? t.color.completionCurrentBg : undefined}>
    <Box width={wide ? 27 : '100%'} flexShrink={0}><Text color={selected ? t.color.accent : t.ds.secondary} wrap="wrap">{selected ? '› ' : '  '}{label}</Text></Box>
    <Box flexGrow={1} minWidth={0}><Text color={t.color.text} wrap="wrap">{children}</Text></Box>
  </Box>
}
