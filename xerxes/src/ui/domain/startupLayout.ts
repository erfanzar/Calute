// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export interface StartupLayoutState {
  busy: boolean
  hasLiveTurn: boolean
  pendingInteraction: boolean
  transcriptEmpty: boolean
}

/**
 * The welcome art is only an idle-session surface. A daemon-owned turn (for
 * example a skill shorthand) can become busy or request approval before it
 * contributes a transcript row, so those signals must switch to the session
 * layout independently of transcript hydration.
 */
export const shouldShowStartupWelcome = ({
  busy,
  hasLiveTurn,
  pendingInteraction,
  transcriptEmpty
}: StartupLayoutState): boolean => transcriptEmpty && !busy && !hasLiveTurn && !pendingInteraction

/** Conversation fills the available pane, excluding its side gutters. */
export const contentColumnWidth = (columns: number): number => Math.max(1, Math.floor(columns) - 4)

/** The idle welcome remains a compact centered surface. */
export const welcomeColumnWidth = (columns: number): number => Math.min(120, contentColumnWidth(columns))
