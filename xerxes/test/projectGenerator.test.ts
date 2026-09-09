// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { generateProjectAgent } from '../src/agents/projectGenerator.js'

const markdown = '---\nname: jax-reviewer\ndescription: "Review JAX shapes and sharding."\n---\nVerify shapes and provide regression tests.'
test('generates a validated unsaved draft from a description', async () => {
  let prompt = ''
  const draft = await generateProjectAgent('Review JAX code', async value => { prompt = value; return '```markdown\n' + markdown + '\n```' })
  expect(prompt).toContain('Review JAX code')
  expect(draft).toEqual({ id: 'jax-reviewer', content: markdown + '\n', revision: null })
})
test('rejects missing descriptions before calling a model', async () => {
  let called = false
  await expect(generateProjectAgent(' ', async () => { called = true; return markdown })).rejects.toThrow('Describe the agent')
  expect(called).toBe(false)
})
test('rejects malformed output and propagates provider failures', async () => {
  for (const output of ['hello', '---\nname: ../escape\n---\nDo things', '---\nname: reviewer\n---\n']) {
    await expect(generateProjectAgent('review', async () => output)).rejects.toThrow()
  }
  await expect(generateProjectAgent('review', async () => { throw new Error('Provider unavailable') })).rejects.toThrow('Provider unavailable')
})
