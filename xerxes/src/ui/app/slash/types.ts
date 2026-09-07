// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { MutableRefObject } from 'react'

import type { SlashHandlerContext, UiState } from '../interfaces.js'

export interface SlashRunCtx extends SlashHandlerContext {
  flight: number
  guarded: <T>(fn: (r: T) => void) => (r: null | T) => void
  guardedErr: (e: unknown) => void
  sid: null | string
  slashFlightRef: MutableRefObject<number>
  stale: () => boolean
  ui: UiState
}

export interface SlashCommand {
  /** Local panel commands remain discoverable without a daemon catalog. */
  group?: string
  aliases?: string[]
  help?: string
  name: string
  run: (arg: string, ctx: SlashRunCtx, cmd: string) => void
  usage?: string
}
