// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { createHash, randomUUID } from 'node:crypto'
import { existsSync, lstatSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { withFileLock } from '../session/daemonTranscript.js'
import { parseAgentMarkdown } from './definitions.js'

function directory(cwd: string): string {
  const base = join(cwd, '.xerxes')
  const agents = join(base, 'agents')
  for (const path of [base, agents]) if (existsSync(path) && lstatSync(path).isSymbolicLink()) throw new Error('Agent editing does not follow symbolic links.')
  return agents
}
function target(cwd: string, id: string): string {
  if (!/^[a-z][a-z0-9-]{0,63}$/.test(id)) throw new Error('Agent ID must be a lowercase slug, up to 64 characters.')
  const path = join(directory(cwd), `${id}.md`)
  if (existsSync(path) && !lstatSync(path).isFile()) throw new Error('Agent file must be a regular file.')
  return path
}
function revision(content: string): string { return createHash('sha256').update(content).digest('hex') }
export function listProjectAgents(cwd: string) {
  const dir = directory(cwd)
  if (!existsSync(dir)) return []
  return readdirSync(dir, { withFileTypes: true }).filter(entry => entry.isFile() && /^[a-z][a-z0-9-]{0,63}\.md$/.test(entry.name)).map(entry => {
    const id = entry.name.slice(0, -3)
    try { return { id, description: parseAgentMarkdown(join(dir, entry.name), 'project').description } }
    catch (error) { return { id, description: '', error: String(error) } }
  }).sort((a, b) => a.id.localeCompare(b.id))
}
export function readProjectAgent(cwd: string, id: string) {
  const content = readFileSync(target(cwd, id), 'utf8')
  return { id, content, revision: revision(content) }
}
export async function writeProjectAgent(cwd: string, id: string, content: string, expected: string | null) {
  const dir = directory(cwd)
  if (content.length > 256_000 || !content.trim()) throw new Error('Agent definition must contain 1–256000 characters.')
  mkdirSync(dir, { recursive: true })
  return withFileLock(join(dir, '.editor.lock'), async () => {
    directory(cwd)
    const temporary = join(dir, `.draft-${randomUUID()}.md`)
    try {
      writeFileSync(temporary, content, { encoding: 'utf8', mode: 0o600, flag: 'wx' })
      const definition = parseAgentMarkdown(temporary, 'project')
      const resolvedId = id || definition.name
      const path = target(cwd, resolvedId)
      if (definition.name !== resolvedId) throw new Error(`Frontmatter name must be ${resolvedId}.`)
      if (existsSync(path) ? expected !== revision(readFileSync(path, 'utf8')) : expected !== null) throw new Error('Agent changed on disk. Reopen it before saving; your draft has been preserved.')
      renameSync(temporary, path)
      return readProjectAgent(cwd, resolvedId)
    } finally { rmSync(temporary, { force: true }) }
  }, { waitMs: 5000, staleMs: 30000, label: 'project agents' })
}
