// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, readFile, rm, symlink, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { loadAgentDefinitions, subagentCatalogForAgent } from '../src/agents/definitions.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { defaultSkillDiscoveryRoots, SkillRegistry, trustedHashWorkspaceSkills } from '../src/extensions/skills.js'
import { expandSkillInstructions } from '../src/extensions/skillInjection.js'
import { bootstrapSubagentsForAgent } from '../src/runtime/bootstrap.js'
import { registerFileTools } from '../src/tools/fileTools.js'
import { WorkspacePathResolver } from '../src/tools/pathSafety.js'
import { registerProjectSetupTool } from '../src/tools/projectSetup.js'
import type { JsonObject } from '../src/types/toolCalls.js'

test('setup creates discoverable specialists and trusted workflows, preserves existing files and detects edits', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-project-setup-'))
  const skillsDirectory = join(root, 'trusted-home')
  const trustPaths = { skillsDirectory }
  const registry = new ToolRegistry()
  const paths = new WorkspacePathResolver(root)
  registerFileTools(registry, paths)
  registerProjectSetupTool(registry, paths, trustPaths)
  const execute = (artifacts: JsonObject[]) => registry.execute({ id: 'setup', type: 'function', function: { name: 'create_project_setup', arguments: { artifacts } } }, { metadata: {} })
  const artifacts = [
    { kind: 'agent', name: 'kernel-expert', description: 'Use for GPU kernel layout and correctness.', instructions: 'Inspect src/kernels before reviewing changes.' },
    { kind: 'skill', name: 'kernel-audit', description: 'Review kernel correctness.', instructions: 'Read src/kernels and tests/kernels; report uncovered cases.' },
    { kind: 'command', name: 'repo-test', description: 'Run the repository test workflow.', instructions: 'Run bun test for $ARGUMENTS and report failures.' },
  ]
  try {
    expect(JSON.parse(await execute(artifacts))).toMatchObject({ created: [
      '.xerxes/agents/kernel-expert.md', '.xerxes/skills/kernel-audit/SKILL.md', '.xerxes/commands/repo-test.md',
    ], skipped: [] })
    const definitions = loadAgentDefinitions({ cwd: root, userDirectory: join(root, 'empty') })
    expect(bootstrapSubagentsForAgent(definitions)).toContainEqual({ name: 'kernel-expert', description: 'Use for GPU kernel layout and correctness.' })
    expect(subagentCatalogForAgent(definitions, 'objective')['kernel-expert']?.resolvedProfile).toBe('kernel-expert')
    expect(subagentCatalogForAgent(definitions, 'researcher')['kernel-expert']).toBeUndefined()
    expect(subagentCatalogForAgent(definitions, 'kernel-expert').reviewer).toBeDefined()
    expect(definitions.get('kernel-expert')?.tools).toEqual(['ReadFile', 'ListDir', 'GlobTool', 'GrepTool'])
    const skills = () => new SkillRegistry({ workspaceTrust: trustedHashWorkspaceSkills(trustPaths) })
    const roots = defaultSkillDiscoveryRoots({ cwd: root }).filter(entry => entry.workspace)
    const initial = skills()
    expect((await initial.refresh(...roots)).sort()).toEqual(['kernel-audit', 'repo-test'])
    expect(initial.get('repo-test')?.instructions).toContain('$ARGUMENTS')
    expect(initial.get('repo-test')?.allowCommandExecution).toBe(true)
    const commandPath = join(root, '.xerxes/commands/repo-test.md')
    const original = await readFile(commandPath, 'utf8')
    expect(JSON.parse(await execute(artifacts))).toMatchObject({ created: [], skipped: expect.arrayContaining(['.xerxes/commands/repo-test.md']) })
    expect(await readFile(commandPath, 'utf8')).toBe(original)
    await writeFile(commandPath, original + '\nA local edit.\n')
    const edited = skills()
    expect((await edited.refresh(...roots)).sort()).toEqual(['kernel-audit', 'repo-test'])
    expect(edited.get('repo-test')?.instructions).toContain('A local edit.')
    expect(edited.get('repo-test')?.allowCommandExecution).toBe(false)
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('existing project workflows load without creating files or trusting shell preprocessing', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-autoload-'))
  try {
    const files = {
      '.xerxes/commands/inspect.md': '---\ndescription: Inspect the project\n---\nInspect $ARGUMENTS.',
      '.xerxes/skills/audit/SKILL.md': '---\nname: audit\ndescription: Audit the project\n---\nRead the tests.',
      '.agents/skills/legacy/SKILL.md': '---\nname: legacy\n---\nRead the legacy tests.',
    }
    for (const [path, content] of Object.entries(files)) {
      const absolute = join(root, path)
      await mkdir(absolute.slice(0, absolute.lastIndexOf('/')), { recursive: true })
      await writeFile(absolute, content)
    }
    const registry = new SkillRegistry({ workspaceTrust: trustedHashWorkspaceSkills({ skillsDirectory: join(root, 'trust') }) })
    const roots = defaultSkillDiscoveryRoots({ cwd: root }).filter(entry => entry.workspace)
    expect((await registry.refresh(...roots)).sort()).toEqual(['audit', 'inspect'])
    expect(registry.get('legacy')).toBeUndefined()
    const command = registry.get('inspect')!
    expect(command.allowCommandExecution).toBe(false)
    expect(await expandSkillInstructions(command.instructions, { cwd: root, args: 'src', allowCommandExecution: command.allowCommandExecution !== false })).toBe('Inspect src.')
    expect(await Bun.file(join(root, 'trust/.hub/trusted_hashes.json')).exists()).toBe(false)
    for (const [path, content] of Object.entries(files)) expect(await readFile(join(root, path), 'utf8')).toBe(content)
    await rm(join(root, '.xerxes/commands/inspect.md'))
    expect(await registry.refresh(...roots)).toEqual(['audit'])
    expect(registry.get('inspect')).toBeUndefined()
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('setup validates a complete batch, refuses symlink escapes and supports cancellation before mutation', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-setup-boundary-'))
  const outside = await mkdtemp(join(tmpdir(), 'xerxes-setup-outside-'))
  const registry = new ToolRegistry()
  const paths = new WorkspacePathResolver(root)
  registerFileTools(registry, paths)
  registerProjectSetupTool(registry, paths, { skillsDirectory: join(root, 'trust') })
  const artifact = { kind: 'agent', name: 'reader', description: 'Read files.', instructions: 'Inspect the repository.' }
  const call = (artifacts: JsonObject[]) => ({ id: 'setup', type: 'function' as const, function: { name: 'create_project_setup', arguments: { artifacts } } })
  try {
    await expect(registry.execute(call([artifact, { ...artifact, name: '../escape' }]), { metadata: {} })).rejects.toThrow()
    expect(await Bun.file(join(root, '.xerxes/agents/reader.md')).exists()).toBe(false)
    await expect(registry.execute(call([{ ...artifact, tools: ['ImaginaryTool'] }]), { metadata: {} })).rejects.toThrow('registered tool names')
    await expect(registry.execute(call([{ ...artifact, kind: 'command', name: 'init' }]), { metadata: {} })).rejects.toThrow('collides')
    await expect(registry.execute(call([artifact]), { metadata: {} }, AbortSignal.abort(new Error('cancel setup')))).rejects.toThrow('cancelled before execution')
    expect(await Bun.file(join(root, '.xerxes/agents/reader.md')).exists()).toBe(false)
    await mkdir(join(root, '.xerxes'))
    await symlink(outside, join(root, '.xerxes/agents'))
    await expect(registry.execute(call([artifact]), { metadata: {} })).rejects.toThrow('outside workspace')
    expect(await Bun.file(join(outside, 'reader.md')).exists()).toBe(false)
  } finally { await rm(root, { recursive: true, force: true }); await rm(outside, { recursive: true, force: true }) }
})

test('automatic specialist discovery does not widen user presets or expose nested-only aliases', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-setup-catalog-'))
  try {
    await mkdir(join(root, '.xerxes/agents'), { recursive: true })
    await writeFile(join(root, '.xerxes/agents/specialist.md'), '---\ndescription: Project specialist\n---\nReview carefully.')
    await writeFile(join(root, '.xerxes/agents/reviewer.md'), '---\ndescription: Project-specific reviewer\n---\nUse this repository review checklist.')
    const definitions = loadAgentDefinitions({ cwd: root, userDirectory: join(root, 'empty') })
    const main = definitions.get('default')!
    const specialist = definitions.get('specialist')!
    definitions.set('@catalog:hidden:/child.yaml', { ...specialist, name: 'hidden' })
    expect(subagentCatalogForAgent(definitions, 'default').hidden).toBeUndefined()
    expect(subagentCatalogForAgent(definitions, 'default').reviewer).toEqual({ description: 'Project-specific reviewer', resolvedProfile: 'reviewer' })
    expect(subagentCatalogForAgent(definitions, 'objective').reviewer?.resolvedProfile).toBe('reviewer')
    definitions.set('default', { ...main, source: 'user' })
    expect(subagentCatalogForAgent(definitions, 'default').specialist).toBeUndefined()
    definitions.set('default', { ...main, source: 'user', subagents: { specialist: { path: '/pinned.yaml', description: 'Pinned', resolvedProfile: '@catalog:hidden:/child.yaml' } } })
    expect(subagentCatalogForAgent(definitions, 'default').specialist?.resolvedProfile).toBe('@catalog:hidden:/child.yaml')
  } finally { await rm(root, { recursive: true, force: true }) }
})
