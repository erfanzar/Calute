// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { runMachineCommand } from '../src/daemon/machineCommand.js'

test('machine commands persist validated workspaces and resolve them without claiming SSH connectivity', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-machines-'))
  const file = join(root, 'machines.json')
  try {
    expect(await runMachineCommand(file, '')).toMatchObject({ ok: true, machines: [] })
    expect(await runMachineCommand(file, 'add compute me@host "/srv/my project"')).toMatchObject({ ok: true })
    expect(await runMachineCommand(file, 'connect compute')).toEqual({ ok: true, machine: { alias: 'compute', target: 'me@host', workspacePath: '/srv/my project' } })
    expect(await runMachineCommand(file, 'add compute other /tmp')).toMatchObject({ ok: false })
    expect(await runMachineCommand(file, 'add bad -oProxyCommand=evil /tmp')).toMatchObject({ ok: false })
    expect(await runMachineCommand(file, 'add bad host ../tmp')).toMatchObject({ ok: false })
    expect(await runMachineCommand(file, 'add bad host "/unclosed')).toMatchObject({ ok: false })
    expect(await runMachineCommand(file, 'remove compute')).toMatchObject({ ok: true, machines: [] })
    expect(await runMachineCommand(file, 'connect compute')).toMatchObject({ ok: false })
    await Bun.write(file, '{broken')
    expect(await runMachineCommand(file, 'add no host /tmp')).toMatchObject({ ok: false })
    expect(await Bun.file(file).text()).toBe('{broken')
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('concurrent machine mutations preserve both entries', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-machine-lock-'))
  const file = join(root, 'machines.json')
  try {
    const results = await Promise.all([runMachineCommand(file, 'add a host /a'), runMachineCommand(file, 'add b host /b')])
    expect(results.every(result => result.ok)).toBe(true)
    expect((await runMachineCommand(file, 'list')).machines).toHaveLength(2)
  } finally { await rm(root, { recursive: true, force: true }) }
})
