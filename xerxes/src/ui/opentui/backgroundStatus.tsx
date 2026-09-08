// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { patchOverlayState } from '../app/overlayStore.js'
import type { Theme } from '../theme.js'
import { Box, Text } from './primitives.js'
import { recentOutcomes, useActivity } from './useActivity.js'

export function BackgroundStatus({ sessionId, t }: { sessionId: string | null; t: Theme }) {
  const { rows, error, now } = useActivity(sessionId)
  const shells = rows.filter(row => row.kind === 'shell' && row.state === 'running').length
  const watchers = rows.filter(row => row.kind === 'watcher' && row.state === 'watching').length
  const running = rows.filter(row => row.kind === 'schedule' && ['running','cancelling'].includes(row.state)).length
  const scheduled = rows.filter(row => row.kind === 'schedule' && row.state === 'scheduled').length
  const recent = recentOutcomes(rows, now)
  const failed = recent.filter(row => ['failed','interrupted'].includes(row.kind === 'schedule' ? row.lastState ?? '' : row.state)).length
  const finished = recent.length - failed
  const labels = [shells ? `${shells} ${shells === 1 ? 'shell' : 'shells'} running` : '', watchers ? `${watchers} ${watchers === 1 ? 'watcher' : 'watchers'} active` : '', running ? `${running} scheduled ${running === 1 ? 'run' : 'runs'} active` : '', scheduled ? `${scheduled} scheduled` : ''].filter(Boolean)
  return <Box flexDirection="row" flexWrap="wrap" flexShrink={0}>
    {error ? <Box onClick={() => patchOverlayState({ activity: true })}><Text color={t.color.muted}> · background status unavailable</Text></Box> : null}
    {labels.map(label => <Box key={label} onClick={() => patchOverlayState({ activity: true })}><Text color={t.color.brandGold}>{' · ' + label}</Text></Box>)}
    {finished > 0 ? <Box onClick={() => patchOverlayState({ activity: true })}><Text color={t.color.ok}>{` · ✓ ${finished} finished`}</Text></Box> : null}
    {failed > 0 ? <Box onClick={() => patchOverlayState({ activity: true })}><Text color={t.color.warn}>{` · ! ${failed} failed`}</Text></Box> : null}
  </Box>
}
