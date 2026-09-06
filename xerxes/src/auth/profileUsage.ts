// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { CodexSession, codexAuthHeaders, type CodexCredential } from './codexAuth.js'
import { fetchCodexUsage, fetchKimiUsage, fetchZaiUsage, type UsageProfile, type UsageRequestOptions } from './usage.js'
export type ProfileQuota = {
  status: 'unknown'; remaining_tokens: null; reason: string
} | {
  status: 'available'; remaining_tokens: null; source: 'provider_usage_endpoint';
  scope: 'profile_credentials' | 'shared_codex_login'; provider_profile: string; observed_at: string;
  windows: { label: string; used_percent: number; resets_at: string | null; reset_after_seconds: number | null; detail: string | null }[]
}
export const unknownProfileQuota = (reason: string): ProfileQuota => ({ status: 'unknown', remaining_tokens: null, reason })
/** Stop waiting without cancelling a shared OAuth refresh used by other requests. */
async function usageCredential(resolve: (signal?: AbortSignal) => Promise<CodexCredential>, signal: AbortSignal): Promise<CodexCredential> {
  signal.throwIfAborted()
  const pending = resolve(signal)
  return new Promise((accept, reject) => {
    const abort = () => { signal.removeEventListener('abort', abort); reject(signal.reason) }
    signal.addEventListener('abort', abort, { once: true })
    pending.then(value => { signal.removeEventListener('abort', abort); accept(value) }, error => { signal.removeEventListener('abort', abort); reject(error) })
    if (signal.aborted) abort()
  })
}
/** Only the built-in Codex route binds to the shared login used for inference. */
export async function profileQuota(profile: UsageProfile, options: UsageRequestOptions & { readonly codexCredential?: (signal?: AbortSignal) => Promise<CodexCredential> } = {}): Promise<ProfileQuota> {
  options.signal?.throwIfAborted()
  let host: string
  try { const url = new URL(profile.base_url); host = url.protocol === 'https:' && !url.port ? url.hostname : '' } catch { host = '' }
  const codex = profile.name === 'codex' && profile.provider === 'openai-codex' && profile.base_url === 'https://chatgpt.com/backend-api/codex' && !profile.api_key.trim()
  const kimi = ['kimi', 'kimi-code'].includes(profile.provider) && ['api.kimi.com', 'api.moonshot.cn', 'api.moonshot.ai'].includes(host)
  const zai = ['zhipu', 'zai', 'zai-coding', 'zai-coding-cn'].includes(profile.provider) && ['api.z.ai', 'open.bigmodel.cn'].includes(host)
  if (!codex && !kimi && !zai) return unknownProfileQuota('No profile-bound subscription usage adapter for this provider endpoint.')
  if (!codex && !profile.api_key.trim()) return unknownProfileQuota('This profile has no dedicated API key; shared login usage is not attributed to it.')
  try {
    // Fixed provider endpoints: user environment overrides cannot redirect profile credentials.
    const request = { ...options, strictUnits: true, environment: {}, signal: options.signal ? AbortSignal.any([options.signal, AbortSignal.timeout(10000)]) : AbortSignal.timeout(10000) }
    const credential = codex ? await usageCredential(options.codexCredential ?? (signal => new CodexSession().credential(signal)), request.signal) : undefined
    request.signal.throwIfAborted()
    const report = credential ? await fetchCodexUsage(codexAuthHeaders(credential), request) : kimi ? await fetchKimiUsage(profile.api_key, request) : await fetchZaiUsage(profile.api_key, { ...request, host: host === 'open.bigmodel.cn' ? 'cn' : 'global' })
    options.signal?.throwIfAborted()
    if (!Number.isSafeInteger(report.fetchedAt) || report.fetchedAt < 0 || report.windows.length < 1 || report.windows.length > 32) throw new Error('Invalid usage report')
    const windows = report.windows.map(window => {
      if (!Number.isFinite(window.usedPercent) || window.usedPercent < 0 || window.usedPercent > 100
        || window.label.length > 256 || (window.detail?.length ?? 0) > 512
        || (window.resetsAt !== undefined && (!Number.isSafeInteger(window.resetsAt) || window.resetsAt < 0))
        || (window.resetAfterSeconds !== undefined && (!Number.isFinite(window.resetAfterSeconds) || window.resetAfterSeconds < 0))) throw new Error('Invalid usage window')
      return { label: window.label, used_percent: window.usedPercent, resets_at: window.resetsAt === undefined ? null : new Date(window.resetsAt).toISOString(), reset_after_seconds: window.resetAfterSeconds ?? null, detail: window.detail ?? null }
    })
    return { status: 'available', remaining_tokens: null, source: 'provider_usage_endpoint', scope: codex ? 'shared_codex_login' : 'profile_credentials', provider_profile: profile.name, observed_at: new Date(report.fetchedAt).toISOString(), windows }
  } catch {
    options.signal?.throwIfAborted()
    return unknownProfileQuota('Provider usage is unavailable or its response is unsupported; no remaining-token estimate was made.')
  }
}
