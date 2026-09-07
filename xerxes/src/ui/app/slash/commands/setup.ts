// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { launchXerxesCommand } from '../../../lib/externalCli.js'
import { withTerminalSuspended } from '../../../lib/terminalRuntime.opentui.js'
import { runExternalSetup } from '../../setupHandoff.js'
import { runNativeSlash } from '../nativeSlash.js'
import type { SlashCommand } from '../types.js'

export const setupCommands: SlashCommand[] = [
  {
    help: 'inspect the repo and create project instructions, specialists, skills and commands',
    name: 'init',
    group: 'tools',
    run: (arg, ctx) => runNativeSlash(ctx, `init ${arg}`.trim(), 'Repository setup')
  },
  {
    help: 'run full setup wizard (launches `xerxes setup`)',
    name: 'setup',
    run: (arg, ctx) =>
      void runExternalSetup({
        args: ['setup', ...arg.split(/\s+/).filter(Boolean)],
        ctx,
        done: 'setup complete — starting session…',
        launcher: launchXerxesCommand,
        suspend: withTerminalSuspended
      })
  }
]
