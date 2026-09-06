// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), '..')

function spawnCli(args: string[]): { process: ReturnType<typeof Bun.spawn>; stdout: Promise<string>; stderr: Promise<string> } {
  const child = Bun.spawn({
    cmd: [process.execPath, 'src/cli.ts', ...args],
    cwd: resolve(rootDir),
    stdout: 'pipe',
    stderr: 'pipe',
  })
  return {
    process: child,
    stdout: new Response(child.stdout).text(),
    stderr: new Response(child.stderr).text(),
  }
}

test("schedule CLI rejects unsupported sources and legacy storage before creating inert triggers", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-schedule-cli-"))
  try {
    const unsupported = spawnCli(["schedule", "create", "--schedule", "webhook:/hooks/build", "--objective", "run build", "--socket", join(directory, "absent.sock")])
    expect(await unsupported.process.exited).not.toBe(0)
    expect(await unsupported.stderr).toMatch(/webhook|unsupported/)
    const legacy = spawnCli(["schedule", "create", "--id", "old-trigger", "--owner", "user", "--schedule", "interval:60", "--objective", "work", "--directory", directory])
    expect(await legacy.process.exited).not.toBe(0)
    expect(await legacy.stderr).toMatch(/legacy|migrat/)
    expect(await Bun.file(join(directory, "scheduler.jsonl")).exists()).toBe(false)
    const missing = spawnCli(["schedule", "list", "--project-dir", directory, "--socket", join(directory, "absent.sock")])
    expect(await missing.process.exited).not.toBe(0)
    expect(await missing.stderr).toMatch(/daemon|socket/)
  } finally { await rm(directory, { recursive: true, force: true }) }
})
