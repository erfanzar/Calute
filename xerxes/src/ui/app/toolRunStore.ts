// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Expansion state for collapsed tool runs. Kept outside the message
// components for the same reason thinking blocks are: transcript rows
// unmount and remount as they leave the virtualized window, and a toggle
// that lived in the row would silently reset when you scrolled past it.
//
// Deliberately separate from the thinking store rather than sharing its
// `allExpanded` flag — Ctrl+T means "show me the reasoning", and it should
// not also unfold every tool run in the session.

import { atom } from 'nanostores'

export interface ToolRunVisibility {
  allExpanded?: boolean
  overrides: Record<string, boolean>
}

const buildState = (): ToolRunVisibility => ({ overrides: {} })

export const $toolRunVisibility = atom<ToolRunVisibility>(buildState())

export const getToolRunVisibility = () => $toolRunVisibility.get()

/** Runs collapse by default; a click or F9 opens them. */
export const toolRunExpanded = (visibility: ToolRunVisibility, runId: string): boolean =>
  visibility.overrides[runId] ?? visibility.allExpanded ?? false

export const toggleToolRun = (runId: string) => {
  const current = $toolRunVisibility.get()

  $toolRunVisibility.set({
    ...current,
    overrides: { ...current.overrides, [runId]: !toolRunExpanded(current, runId) }
  })
}

/** Keyboard access to every folded run, including rows outside the viewport. */
export const toggleAllToolRuns = () =>
  $toolRunVisibility.set({
    allExpanded: !$toolRunVisibility.get().allExpanded,
    overrides: {}
  })

export const resetToolRunVisibility = () => $toolRunVisibility.set(buildState())
