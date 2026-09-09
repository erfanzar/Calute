// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, mkdir, rm, symlink, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { listProjectAgents, readProjectAgent, writeProjectAgent } from '../src/agents/projectEditor.js'
import { loadAgentDefinitions } from '../src/agents/definitions.js'

test('edits filename-named agents authored by project setup without requiring name frontmatter', async () => {
  const root = await mkdtemp(join(tmpdir(), 'agent-editor-'))
  try {
    await mkdir(join(root, '.xerxes/agents'), { recursive: true })
    const content = '---\ntools: [ReadFile]\ndescription: Review code\n---\nFind defects.\n'
    await writeFile(join(root, '.xerxes/agents/reviewer.md'), content)
    const original = readProjectAgent(root, 'reviewer')
    const saved = await writeProjectAgent(root, 'reviewer', content + 'Check error paths.\n', original.revision)
    expect(saved.content).toContain('Check error paths.')
    await expect(writeProjectAgent(root, 'reviewer', content, original.revision)).rejects.toThrow('changed on disk')
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('lists and edits nested specialists discovered by the runtime', async () => {
  const root = await mkdtemp(join(tmpdir(), 'agent-editor-'))
  try {
    await mkdir(join(root, '.xerxes/agents/research'), { recursive: true })
    const content = '---\nname: investigator\ndescription: Investigate failures\n---\nFind defects.\n'
    await writeFile(join(root, '.xerxes/agents/research/debugger.md'), content)
    expect(loadAgentDefinitions({ cwd: root, userDirectory: join(root, 'absent') }).get('investigator')?.description).toBe('Investigate failures')
    expect(listProjectAgents(root)).toEqual([{ id: 'research/debugger', description: 'Investigate failures' }])
    const original = readProjectAgent(root, 'research/debugger')
    const saved = await writeProjectAgent(root, original.id, content + 'Check logs.\n', original.revision)
    expect(saved.content).toContain('Check logs.')
    expect(() => readProjectAgent(root, '../outside')).toThrow()
    await symlink(join(root, '.xerxes/agents/research'), join(root, '.xerxes/agents/linked'))
    expect(listProjectAgents(root)).toHaveLength(1)
    expect(() => readProjectAgent(root, 'linked/debugger')).toThrow('symbolic links')
    await expect(writeProjectAgent(root, 'linked/debugger', content, original.revision)).rejects.toThrow('symbolic links')
    await expect(writeProjectAgent(root, saved.id, content, original.revision)).rejects.toThrow('changed on disk')
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('edits a valid specialist whose declared name differs from its filename', async () => {
  const root = await mkdtemp(join(tmpdir(), 'agent-editor-'))
  try {
    await mkdir(join(root, '.xerxes/agents'), { recursive: true })
    const content = '---\nname: investigator\ndescription: Investigate failures\n---\nFind defects.\n'
    await writeFile(join(root, '.xerxes/agents/debugger.md'), content)
    const original = readProjectAgent(root, 'debugger')
    expect((await writeProjectAgent(root, original.id, content + 'Check logs.\n', original.revision)).content).toContain('Check logs.')
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('rejects agent drafts that runtime discovery would skip', async () => {
  const root = await mkdtemp(join(tmpdir(), 'agent-editor-'))
  try {
    for (const content of ['---\nname: reviewer\n---\nReview code.', '---\nname: reviewer\ndescription: " "\n---\nReview code.']) {
      await expect(writeProjectAgent(root, '', content, null)).rejects.toThrow('description')
    }
  } finally { await rm(root, { recursive: true, force: true }) }
})
