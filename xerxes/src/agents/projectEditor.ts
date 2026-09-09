// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { createHash, randomUUID } from 'node:crypto'
import { existsSync, lstatSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { withFileLock } from '../session/daemonTranscript.js'
import { parseAgentMarkdown, parseAgentMarkdownContent } from './definitions.js'

function directory(cwd: string): string {
  const base = join(cwd, '.xerxes')
  const agents = join(base, 'agents')
  for (const path of [base, agents]) if (existsSync(path) && lstatSync(path).isSymbolicLink()) throw new Error('Agent editing does not follow symbolic links.')
  return agents
}
function target(cwd: string, id: string): string {
  const parts = id.split('/')
  if (parts.length > 17 || parts.some(part => !part || part.startsWith('.') || /[\\\0]/.test(part)) || !/^[a-z][a-z0-9-]{0,63}$/.test(parts.at(-1)!)) throw new Error('Agent ID must be a relative path ending in a lowercase slug, up to 64 characters.')
  let parent = directory(cwd)
  for (const part of parts.slice(0, -1)) {
    parent = join(parent, part)
    if (!lstatSync(parent).isDirectory()) throw new Error('Agent editing does not follow symbolic links or non-directories.')
  }
  const path = join(parent, `${parts.at(-1)}.md`)
  if (existsSync(path) && !lstatSync(path).isFile()) throw new Error('Agent file must be a regular file.')
  return path
}
function revision(content: string): string { return createHash('sha256').update(content).digest('hex') }
export function listProjectAgents(cwd: string) {
  const dir = directory(cwd)
  if (!existsSync(dir)) return []
  function files(relative = '', depth = 0): string[] {
    if (depth > 16) return []
    return readdirSync(join(dir, relative), { withFileTypes: true }).flatMap(entry => {
      const name = relative ? `${relative}/${entry.name}` : entry.name
      if (entry.isDirectory() && !entry.name.startsWith('.') && !entry.name.includes('\\')) return files(name, depth + 1)
      return entry.isFile() && /^[a-z][a-z0-9-]{0,63}\.md$/.test(entry.name) ? [name.slice(0, -3)] : []
    })
  }
  return files().map(id => {
    try { return { id, description: parseAgentMarkdown(target(cwd, id), 'project').description } }
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
      // Filename-based names must use the actual destination, not .draft-UUID.
      const definition = parseAgentMarkdownContent(content, id ? target(cwd, id) : temporary, 'project')
      if (!definition.description.trim()) throw new Error('Agent frontmatter.description is required for runtime discovery.')
      const resolvedId = id || definition.name
      const path = target(cwd, resolvedId)
      const filenameName = resolvedId.split('/').at(-1)!
      if (definition.name !== filenameName && (!existsSync(path) || parseAgentMarkdown(path, 'project').name !== definition.name)) throw new Error(`Frontmatter name must be ${filenameName} or the existing declared name.`)
      if (existsSync(path) ? expected !== revision(readFileSync(path, 'utf8')) : expected !== null) throw new Error('Agent changed on disk. Reopen it before saving; your draft has been preserved.')
      renameSync(temporary, path)
      return readProjectAgent(cwd, resolvedId)
    } finally { rmSync(temporary, { force: true }) }
  }, { waitMs: 5000, staleMs: 30000, label: 'project agents' })
}
