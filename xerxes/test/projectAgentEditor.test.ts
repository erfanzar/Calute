// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, expect, test } from 'bun:test'
import { mkdtempSync, rmSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { listProjectAgents, readProjectAgent, writeProjectAgent } from '../src/agents/projectEditor.js'
import { loadAgentDefinitions } from '../src/agents/definitions.js'
const roots: string[] = []
afterEach(() => { for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true }) })
function root() { const dir = mkdtempSync(join(tmpdir(), 'agent-editor-')); roots.push(dir); return dir }
const content = '---\nname: reviewer\ndescription: Review code.\n---\nFind defects.\n'
test('creates a delegatable specialist and updates only a matching revision', async () => {
  const cwd = root()
  expect(listProjectAgents(cwd)).toEqual([])
  const first = await writeProjectAgent(cwd, 'reviewer', content, null)
  expect(loadAgentDefinitions({ cwd }).get('reviewer')?.description).toBe('Review code.')
  expect(listProjectAgents(cwd)[0]?.id).toBe('reviewer')
  const next = await writeProjectAgent(cwd, 'reviewer', content.replace('Find defects.', 'Review carefully.'), first.revision)
  expect(next.revision).not.toBe(first.revision)
  await expect(writeProjectAgent(cwd, 'reviewer', content, first.revision)).rejects.toThrow('changed on disk')
  expect(readProjectAgent(cwd, 'reviewer').content).toContain('Review carefully.')
})
test('rejects invalid definitions and traversal without overwriting an existing agent', async () => {
  const cwd = root()
  const first = await writeProjectAgent(cwd, 'reviewer', content, null)
  await expect(writeProjectAgent(cwd, 'reviewer', content.replace('name: reviewer', 'name: other'), first.revision)).rejects.toThrow('name must be reviewer')
  await expect(writeProjectAgent(cwd, '../outside', content, null)).rejects.toThrow('slug')
  expect(readFileSync(join(cwd, '.xerxes/agents/reviewer.md'), 'utf8')).toBe(content)
})
test('serializes concurrent writers and uses the parsed quoted YAML name for creation', async () => {
  const cwd = root()
  const first = await writeProjectAgent(cwd, '', content.replace('name: reviewer', 'name: "reviewer"'), null)
  const results = await Promise.allSettled([
    writeProjectAgent(cwd, 'reviewer', content + 'One', first.revision),
    writeProjectAgent(cwd, 'reviewer', content + 'Two', first.revision),
  ])
  expect(results.filter(result => result.status === 'fulfilled')).toHaveLength(1)
  expect(results.filter(result => result.status === 'rejected')).toHaveLength(1)
})
