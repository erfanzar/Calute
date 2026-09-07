// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useEffect, useState } from 'react'
import { useOptionalGateway } from '../app/gatewayContext.js'
import { patchOverlayState } from '../app/overlayStore.js'
import type { Theme } from '../theme.js'
import { Box, Text } from './primitives.js'

export const BACKGROUND_STATUS_POLL_MS = 2000
interface Counts { shells: number; watchers: number }
export function parseBackgroundStatus(value: unknown): Counts {
  if (!value || typeof value !== 'object') throw new Error('Invalid background status')
  const row = value as Record<string, unknown>
  if (row.ok !== true || !Number.isSafeInteger(row.shells) || Number(row.shells) < 0 || !Number.isSafeInteger(row.watchers) || Number(row.watchers) < 0) throw new Error('Background status unavailable')
  return { shells: Number(row.shells), watchers: Number(row.watchers) }
}

/** Poll while idle too; never infer background work from model prose. */
export function BackgroundStatus({ sessionId, t }: { sessionId: string | null; t: Theme }) {
  const gateway = useOptionalGateway()
  const [snapshot, setSnapshot] = useState<{ sessionId: string; counts: Counts | null } | null>(null)
  useEffect(() => {
    if (!gateway || !sessionId) return
    let active = true
    let timer: ReturnType<typeof setTimeout> | undefined
    const refresh = async () => {
      try {
        const counts = parseBackgroundStatus(await gateway.gw.request('background.status', { session_id: sessionId }))
        if (active) setSnapshot({ sessionId, counts })
      } catch {
        // Do not leave a stale "running" badge after losing the daemon.
        if (active) setSnapshot({ sessionId, counts: null })
      } finally {
        if (active) timer = setTimeout(() => { void refresh() }, BACKGROUND_STATUS_POLL_MS)
      }
    }
    void refresh()
    return () => { active = false; if (timer) clearTimeout(timer) }
  }, [gateway, sessionId])
  if (!gateway || !snapshot || snapshot.sessionId !== sessionId) return null
  if (!snapshot.counts) return <Text color={t.color.muted}> · background status unavailable</Text>
  const { shells, watchers } = snapshot.counts
  return <Box flexDirection="row" flexWrap="wrap" flexShrink={0}>
    {shells > 0 ? <Box onClick={() => patchOverlayState({ terminals: true })}><Text color={t.color.brandGold}>{` · ${shells} ${shells === 1 ? 'shell' : 'shells'} running`}</Text></Box> : null}
    {watchers > 0 ? <Box onClick={() => patchOverlayState({ monitors: true })}><Text color={t.color.brandGold}>{` · ${watchers} ${watchers === 1 ? 'watcher' : 'watchers'} active`}</Text></Box> : null}
  </Box>
}
