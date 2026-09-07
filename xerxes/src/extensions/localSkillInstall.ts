// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { lstat, mkdir, readdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { basename, dirname, join, resolve } from 'node:path'
import { parseSkillMarkdown } from './skills.js'
import { withFileLock } from '../session/daemonTranscript.js'
import { scanSkill } from './skillsGuard.js'

/** Install an explicitly selected local skill with its assets; never replace an existing skill. */
export async function installLocalSkill(source: string, root: string, reserved: readonly string[]): Promise<string> {
  await mkdir(root, { recursive: true })
  if (!(await lstat(root)).isDirectory()) throw new Error('Skill install root must be a real directory')
  return withFileLock(join(root, '.install.lock'), () => installLocalSkillUnlocked(source, root, reserved), { waitMs: 30_000, staleMs: 60_000, label: 'skill installation' })
}

async function installLocalSkillUnlocked(source: string, root: string, reserved: readonly string[]): Promise<string> {
  const selected = resolve(source)
  const sourceRoot = basename(selected) === 'SKILL.md' ? dirname(selected) : selected
  if (!(await lstat(sourceRoot)).isDirectory()) throw new Error('Skill source must be a directory or SKILL.md file; symlinks are not accepted')
  const manifest = join(sourceRoot, 'SKILL.md')
  if (!(await lstat(manifest)).isFile() || (await lstat(manifest)).size > 1_048_576) throw new Error('SKILL.md must be a regular file smaller than 1 MiB')
  const skill = parseSkillMarkdown(await readFile(manifest, 'utf8'), manifest)
  const name = skill.metadata.name
  if (!/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}$/.test(name)) throw new Error('Invalid skill name')
  if (reserved.includes(name)) throw new Error(`Skill '${name}' is already discovered; refusing to shadow or replace it`)
  await mkdir(root, { recursive: true })
  if (!(await lstat(root)).isDirectory()) throw new Error('Skill install root must be a real directory')
  const destination = join(root, name)
  try { await lstat(destination); throw new Error(`Skill '${name}' is already installed`) }
  catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error }
  const staging = join(root, `.install-${crypto.randomUUID()}`)
  let files = 0
  let bytes = 0
  const copy = async (from: string, to: string): Promise<void> => {
    if (++files > 512) throw new Error('Skill exceeds 512 entries')
    const info = await lstat(from)
    if (info.isDirectory()) {
      await mkdir(to)
      for (const child of await readdir(from)) await copy(join(from, child), join(to, child))
    } else if (info.isFile()) {
      bytes += info.size
      if (bytes > 16 * 1_048_576) throw new Error('Skill exceeds 16 MiB')
      await writeFile(to, await readFile(from), { mode: info.mode & 0o755 })
    } else throw new Error(`Skill contains a symlink or unsupported file: ${from}`)
  }
  try {
    await copy(sourceRoot, staging)
    const scan = await scanSkill(staging)
    if (!scan.isSafe) throw new Error(`Skill failed security scan: ${scan.summary}`)
    await rename(staging, destination)
    return destination
  } finally { await rm(staging, { recursive: true, force: true }) }
}
