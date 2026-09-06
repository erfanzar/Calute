// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { parseAgentIntelligenceConfig, resolveAgentIntelligenceModel } from '../src/agents/intelligence.js'

test('unconfigured delegation inherits and explicit models override the user default', () => {
  expect(resolveAgentIntelligenceModel(parseAgentIntelligenceConfig(undefined))).toBeUndefined()
  const config = parseAgentIntelligenceConfig({ default: 'smart', smart: ' custom-model ' })
  expect(resolveAgentIntelligenceModel(config)).toBe('custom-model')
  expect(resolveAgentIntelligenceModel(config, undefined, ' exact ')).toBe('exact')
  expect(Object.isFrozen(config)).toBe(true)
})

test('invalid tier configuration is actionable instead of silently falling back', () => {
  for (const invalid of [null, [], 'smart', { default: 'smart' }, { default: 'bogus' }, { light: '' }, { smart: 3 }, { smrat: 'model' }]) {
    expect(() => parseAgentIntelligenceConfig(invalid)).toThrow()
  }
  expect(() => resolveAgentIntelligenceModel({}, 'smart')).toThrow('runtime.agent_intelligence.smart')
})

test('structured tiers retain profile and reasoning while legacy strings remain supported', async () => {
  const { resolveAgentIntelligenceSettings } = await import('../src/agents/intelligence.js')
  const config = parseAgentIntelligenceConfig({ smart: { model: 'deep', provider_profile: 'work', reasoning_effort: 'high' }, light: 'fast' })
  expect(resolveAgentIntelligenceSettings(config, 'smart')).toEqual({ model: 'deep', provider_profile: 'work', reasoning_effort: 'high' })
  expect(resolveAgentIntelligenceSettings(config, 'light')).toEqual({ model: 'fast' })
  expect(() => parseAgentIntelligenceConfig({ smart: { model: 'deep', api_key: 'must-not-store' } })).toThrow()
})
