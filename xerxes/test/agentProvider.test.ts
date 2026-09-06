// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { agentProviderResolver, agentProviderRouteResolver, providerRouteIdentity } from '../src/daemon/agentProvider.js'
import type { ProviderProfile } from '../src/bridge/profiles.js'

test('child provider resolver isolates credentials and limits for each selected profile', () => {
  const profile: ProviderProfile = { name: 'child', provider: 'openai', api_key: 'fixture-child-key', base_url: 'https://child.invalid/v1', model: 'fixture', sampling: {}, model_overrides: { fixture: { context_limit: 8192, max_output_tokens: 1024 } } }
  const calls: unknown[] = []
  const resolver = agentProviderResolver({ get: name => name === 'child' ? profile : undefined }, (model, overrides) => {
    calls.push({ model, overrides })
    return { async *stream() { yield { content: 'ready' } } }
  })
  const selected = resolver('child', 'fixture')
  expect(calls).toEqual([{ model: 'fixture', overrides: { provider: 'openai', api_key: 'fixture-child-key', base_url: 'https://child.invalid/v1' } }])
  expect(selected.contextLimit?.('fixture')).toBe(8192)
  expect(selected.maxOutputTokens?.('fixture')).toBe(1024)
  expect(() => resolver('missing', 'fixture')).toThrow('unavailable')
  expect(calls).toHaveLength(1)
})

test('saved routes permit credential rotation but reject endpoint changes before client creation', () => {
  let profile: ProviderProfile = { name: 'child', provider: 'openai', api_key: 'old-key', base_url: 'https://one.invalid/v1', model: 'fixture', sampling: {} }
  const store = { get: () => profile }
  const route = agentProviderRouteResolver(store)('child', 'fixture')
  const credentials: unknown[] = []
  const resolver = agentProviderResolver(store, (_model, overrides) => {
    credentials.push(overrides?.api_key)
    return { async *stream() { yield { content: 'ready' } } }
  })
  profile = { ...profile, api_key: 'rotated-key' }
  resolver('child', 'fixture', route)
  expect(credentials).toEqual(['rotated-key'])
  expect(route).toMatch(/^[a-f0-9]{64}$/)
  profile = { ...profile, base_url: 'https://two.invalid/v1' }
  expect(() => resolver('child', 'fixture', route)).toThrow('route changed')
  expect(credentials).toHaveLength(1)
})

test('routing identity distinguishes providers and transport while honoring inline connections', () => {
  const original = providerRouteIdentity('fixture', { provider: 'openai', baseUrl: 'https://inline.invalid/v1' })
  expect(providerRouteIdentity('fixture', { provider: 'openai', baseUrl: 'https://inline.invalid/v1' })).toBe(original)
  expect(providerRouteIdentity('fixture', { provider: 'openai', baseUrl: 'https://profile.invalid/v1' })).not.toBe(original)
  expect(providerRouteIdentity('fixture', { provider: 'openai-codex', baseUrl: 'https://inline.invalid/v1' })).not.toBe(original)
  expect(providerRouteIdentity('fixture', { provider: 'openai', baseUrl: 'https://inline.invalid/v1', responsesApi: true })).not.toBe(original)
})
