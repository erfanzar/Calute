// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'

import { resolveCommand } from '../bridge/commands.js'
import { BUILTIN_AGENTS } from '../agents/definitions.js'
import { ToolRegistry } from '../executors/toolRegistry.js'
import { parseSkillMarkdown, skillInstructionsAreSafe } from '../extensions/skills.js'
import { recordTrustedSkillContent, type SkillGuardPaths } from '../extensions/skillsGuard.js'
import type { JsonObject } from '../types/toolCalls.js'
import { WorkspacePathResolver } from './pathSafety.js'

/** Create validated repo-specific definitions without overwriting existing setup. */
export function registerProjectSetupTool(
  registry: ToolRegistry,
  paths: WorkspacePathResolver,
  trustPaths: SkillGuardPaths = {},
): void {
  registry.register({ type: 'function', function: {
    name: 'create_project_setup',
    description: 'After inspecting the repository, create relevant project agents, reusable skills and slash commands under .xerxes. Use only when the user requests repository setup. Existing files are preserved. Newly created skills/commands are trusted by their exact content hash; editing them requires renewed trust. Run /reload after creation, or use /init which refreshes automatically. Does not execute the instructions.',
    parameters: { type: 'object', additionalProperties: false, required: ['artifacts'], properties: {
      artifacts: { type: 'array', minItems: 1, maxItems: 24, items: {
        type: 'object', additionalProperties: false, required: ['kind', 'name', 'description', 'instructions'], properties: {
          kind: { type: 'string', enum: ['agent', 'skill', 'command'] },
          name: { type: 'string', pattern: '^[a-z][a-z0-9-]{0,63}$' },
          description: { type: 'string', minLength: 1, maxLength: 1000, description: 'When to choose this specialist or workflow.' },
          instructions: { type: 'string', minLength: 1, maxLength: 16000, description: 'Repository-specific Markdown instructions grounded in inspected paths and commands.' },
          tools: { type: 'array', minItems: 1, maxItems: 32, items: { type: 'string' }, description: 'Agent-only explicit tool allowlist. Omit for read-only file exploration tools. Use exact registered names.' },
        },
      } },
    } },
  } }, async (input, _context, signal) => {
    signal?.throwIfAborted()
    const artifacts = prepareArtifacts(input, registry)
    // Resolve every target before writing any file; reject path escapes early.
    const targets = await Promise.all(artifacts.map(async artifact => ({ ...artifact, path: await paths.resolve(artifact.relativePath) })))
    const created: string[] = []
    const skipped: string[] = []
    for (const target of targets) {
      signal?.throwIfAborted()
      await mkdir(dirname(await paths.recheck(target.path)), { recursive: true })
      const path = await paths.recheck(target.path)
      try {
        await writeFile(path, target.content, { encoding: 'utf8', flag: 'wx' })
      } catch (error) {
        if (error && typeof error === 'object' && 'code' in error && error.code === 'EEXIST') {
          skipped.push(target.relativePath)
          continue
        }
        throw error
      }
      created.push(target.relativePath)
      if (target.kind !== 'agent') {
        // Hash the bytes we authored, never a reread that could trust another writer's edit.
        await recordTrustedSkillContent(path, target.content, trustPaths)
      }
    }
    return { created, skipped, next: 'Reload to discover the new agents, skills and commands. /init does this after its setup turn.' }
  }, 'default', {
    concurrencySafe: false, defer: false, destructive: false, interruptBehavior: 'block',
    maxResultBytes: 16000, openWorld: false, readOnly: false,
  })
}

function prepareArtifacts(input: JsonObject, registry: ToolRegistry) {
  if (!Array.isArray(input.artifacts) || input.artifacts.length < 1 || input.artifacts.length > 24) throw new Error('Provide 1 to 24 setup artifacts')
  const seen = new Set<string>()
  const workflowNames = new Set<string>()
  return input.artifacts.map(value => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Each artifact must be an object')
    const { kind, name, description, instructions, tools } = value
    if (kind !== 'agent' && kind !== 'skill' && kind !== 'command') throw new Error('Unknown artifact kind')
    if (typeof name !== 'string' || !/^[a-z][a-z0-9-]{0,63}$/.test(name)) throw new Error('Artifact name must be a lowercase slug')
    if (kind === 'agent' && BUILTIN_AGENTS.has(name)) throw new Error(`Agent '${name}' is a shipped profile; choose a project-specific name such as repo-${name}`)
    if (typeof description !== 'string' || !description.trim() || description.length > 1000) throw new Error('Provide a description up to 1000 characters')
    if (typeof instructions !== 'string' || !instructions.trim() || instructions.length > 16000) throw new Error('Provide instructions up to 16000 characters')
    if (kind !== 'agent' && (resolveCommand(name) || workflowNames.has(name))) throw new Error(`Workflow name '${name}' collides with another command or skill`)
    if (kind !== 'agent') workflowNames.add(name)
    const relativePath = kind === 'skill' ? `.xerxes/skills/${name}/SKILL.md` : `.xerxes/${kind === 'agent' ? 'agents' : 'commands'}/${name}.md`
    if (seen.has(relativePath)) throw new Error(`Duplicate artifact ${relativePath}`)
    seen.add(relativePath)
    const selectedTools = tools ?? ['ReadFile', 'ListDir', 'GlobTool', 'GrepTool']
    if (kind === 'agent' && (!Array.isArray(selectedTools) || selectedTools.length < 1 || selectedTools.length > 32 || selectedTools.some(tool => typeof tool !== 'string' || !registry.definitions().some(entry => entry.function.name === tool)))) throw new Error(`Agent '${name}' requires registered tool names`)
    const content = ['---', ...(kind === 'agent' ? [`tools: ${JSON.stringify(selectedTools)}`] : [`name: ${JSON.stringify(name)}`]), `description: ${JSON.stringify(description)}`, '---', instructions.trim(), ''].join('\n')
    if (!skillInstructionsAreSafe(parseSkillMarkdown(content, relativePath))) throw new Error(`Unsafe instructions in ${relativePath}`)
    return { kind, relativePath, content }
  })
}
