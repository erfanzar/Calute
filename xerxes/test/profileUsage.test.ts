// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { profileQuota } from '../src/auth/profileUsage.js'
const kimi = { name: 'mine', provider: 'kimi-code', api_key: 'fixture-key', base_url: 'https://api.kimi.com/coding/v1', model: 'fixture' }
test('profile usage uses only the selected key and preserves percentages below one percent', async () => {
  const result = await profileQuota(kimi, { environment: { XERXES_KIMI_USAGE_URL: 'https://wrong.invalid' }, fetchImplementation: async (url, init) => {
    expect(url).toBe('https://api.kimi.com/coding/v1/usages')
    expect(new Headers(init?.headers).get('Authorization')).toBe('Bearer fixture-key')
    return Response.json({ usages: [{ scope: 'LIMIT_5H', used_percent: 0.5 }] })
  } })
  expect(result).toMatchObject({ status: 'available', remaining_tokens: null, provider_profile: 'mine', scope: 'profile_credentials', windows: [{ used_percent: 0.5 }] })
  expect(JSON.stringify(result)).not.toContain('fixture-key')
})
test('unbound endpoints and missing keys never fall back to a shared account', async () => {
  let calls = 0
  for (const profile of [{ ...kimi, api_key: '' }, { ...kimi, base_url: 'https://custom.invalid' }, { ...kimi, provider: 'openai-codex' }]) {
    expect((await profileQuota(profile, { fetchImplementation: async () => { calls++; throw new Error('unexpected') } })).status).toBe('unknown')
  }
  expect(calls).toBe(0)
})
test('usage failures are sanitized, cancellation propagates, and Z.ai preserves reset timestamps', async () => {
  const failed = await profileQuota(kimi, { fetchImplementation: async () => new Response('fixture-key secret response', { status: 401 }) })
  expect(failed.status).toBe('unknown')
  expect(JSON.stringify(failed)).not.toContain('fixture-key')
  await expect(profileQuota(kimi, { signal: AbortSignal.abort(new Error('cancelled')) })).rejects.toThrow('cancelled')
  const controller = new AbortController()
  await expect(profileQuota(kimi, { signal: controller.signal, fetchImplementation: async () => { controller.abort(new Error('during fetch')); throw new Error('network') } })).rejects.toThrow('during fetch')
  const result = await profileQuota({ ...kimi, provider: 'zhipu', base_url: 'https://api.z.ai/api/coding/paas/v4' }, { fetchImplementation: async () => Response.json({ data: { limits: [{ type: 'TIME_LIMIT', percentage: 12, nextResetTime: 1788480000000 }] } }) })
  expect(result).toMatchObject({ status: 'available', windows: [{ used_percent: 12, resets_at: new Date(1788480000000).toISOString() }] })
})

test('model-facing quota rejects ambiguous or out-of-range values instead of guessing or clamping', async () => {
  for (const row of [{ percentage: 0.5 }, { used_percent: -1 }, { used_percent: 101 }, { used_percent: '50' }]) {
    const result = await profileQuota(kimi, { fetchImplementation: async () => Response.json({ usages: [{ scope: 'LIMIT_5H', ...row }] }) })
    expect(result.status).toBe('unknown')
  }
  const zai = { ...kimi, provider: 'zhipu', base_url: 'https://api.z.ai/api/coding/paas/v4' }
  expect((await profileQuota(zai, { fetchImplementation: async () => Response.json({ data: { limits: [{ percentage: 120 }] } }) })).status).toBe('unknown')
  const valid = await profileQuota(zai, { fetchImplementation: async () => Response.json({ data: { limits: [{ type: 'TOKENS_LIMIT', unit: 99, percentage: 10, remaining: 5000 }] } }) })
  expect(valid).toMatchObject({ status: 'available', remaining_tokens: null, windows: [{ label: 'tokens_limit', detail: 'tokens_limit', used_percent: 10 }] })
  expect(JSON.stringify(valid)).not.toContain('weekly')
  expect(JSON.stringify(valid)).not.toContain('5000')
})

test('builtin Codex usage binds to its stored-login route and labels shared account scope', async () => {
  const codex = { name: 'codex', provider: 'openai-codex', base_url: 'https://chatgpt.com/backend-api/codex', api_key: '', model: 'fixture' }
  let credentials = 0
  const options = {
    codexCredential: async () => { credentials++; return { accessToken: 'oauth-secret', accountId: 'workspace-secret', planType: 'fixture' } },
    fetchImplementation: async (url: string, init?: RequestInit) => {
      expect(url).toBe('https://chatgpt.com/backend-api/wham/usage')
      expect(new Headers(init?.headers).get('chatgpt-account-id')).toBe('workspace-secret')
      return Response.json({ rate_limit: { primary_window: { used_percent: 15, limit_window_seconds: 18000, reset_after_seconds: 600 } } })
    },
  }
  const result = await profileQuota(codex, options)
  expect(result).toMatchObject({ status: 'available', scope: 'shared_codex_login', remaining_tokens: null, windows: [{ label: '5-hour', used_percent: 15, reset_after_seconds: 600 }] })
  expect(JSON.stringify(result)).not.toContain('secret')
  for (const profile of [{ ...codex, name: 'custom' }, { ...codex, api_key: 'different-key' }, { ...codex, base_url: 'https://custom.invalid' }]) expect((await profileQuota(profile, options)).status).toBe('unknown')
  expect(credentials).toBe(1)
  expect((await profileQuota(codex, { ...options, fetchImplementation: async () => Response.json({ rate_limit: { primary_window: { used_percent: -5 } } }) })).status).toBe('unknown')
})

test('Codex quota cancellation stops waiting for shared credential refresh without making a usage call', async () => {
  const controller = new AbortController()
  let entered!: () => void, release!: () => void, fetches = 0
  const ready = new Promise<void>(resolve => { entered = resolve })
  const held = new Promise<void>(resolve => { release = resolve })
  const result = profileQuota({ name: 'codex', provider: 'openai-codex', base_url: 'https://chatgpt.com/backend-api/codex', api_key: '', model: 'fixture' }, {
    signal: controller.signal,
    codexCredential: async () => { entered(); await held; return { accessToken: 'fixture', accountId: undefined, planType: undefined } },
    fetchImplementation: async () => { fetches++; throw new Error('unexpected') },
  })
  await ready
  controller.abort(new Error('stop quota lookup'))
  try { await expect(result).rejects.toThrow('stop quota lookup'); expect(fetches).toBe(0) }
  finally { release() }
})
