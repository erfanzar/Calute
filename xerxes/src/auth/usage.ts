// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Subscription usage windows for quota-based providers (Claude Pro/Max,
 * ChatGPT Codex plans, Kimi Code, Z.ai coding plan). Each fetcher maps the
 * provider's quota endpoint onto one normalized report: a plan label plus
 * rolling windows (5-hour, weekly) with used percent and reset times.
 *
 * Endpoints are constants with environment overrides; when a provider
 * changes shape or a subscription does not expose a window, the fetcher
 * returns an actionable error rather than fabricating numbers.
 */

import { ConfigurationError } from '../core/errors.js'
import { AnthropicOAuthSession } from './anthropicOAuth.js'
import { codexAuthHeaders, CodexSession } from './codexAuth.js'
import { KimiCodingOAuthSession } from './kimiCodingOAuth.js'

export const CLAUDE_USAGE_URL = 'https://api.anthropic.com/api/oauth/usage'
export const CODEX_USAGE_URL = 'https://chatgpt.com/backend-api/wham/usage'
export const KIMI_USAGE_URL = 'https://api.kimi.com/coding/v1/usages'
export const ZAI_USAGE_URL = 'https://api.z.ai/api/monitor/usage/quota/limit'

export type UsageFetch = (url: string, init?: RequestInit) => Promise<Response>

export interface UsageWindow {
  /** Human label, e.g. "5-hour", "weekly". */
  readonly label: string
  /** 0–100 percentage of the window's allowance already consumed. */
  readonly usedPercent: number
  /** Epoch milliseconds when the window resets, when the provider reports it. */
  readonly resetsAt?: number
  /** Seconds until reset, when the provider reports only a duration. */
  readonly resetAfterSeconds?: number
  /** Optional provider detail, e.g. "opus", "sonnet", "tokens". */
  readonly detail?: string
}

export interface ProviderUsageReport {
  readonly provider: string
  readonly planType?: string
  readonly windows: readonly UsageWindow[]
  readonly fetchedAt: number
}

export interface UsageRequestOptions {
  /** Model-facing reports reject ambiguous units and malformed percentages. */
  readonly strictUnits?: boolean
  readonly fetchImplementation?: UsageFetch
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly signal?: AbortSignal
}

const recordOf = (value: unknown): Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}

const finite = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) ? value : undefined

const stringOf = (value: unknown): string | undefined =>
  typeof value === 'string' && value.trim() ? value.trim() : undefined

/** "2026-09-03T12:00:00Z" or epoch seconds/ms → epoch ms, when parseable. */
function epochMs(value: unknown): number | undefined {
  if (typeof value === 'string') {
    const parsed = Date.parse(value)
    return Number.isNaN(parsed) ? undefined : parsed
  }
  const number = finite(value)
  if (number === undefined) return undefined
  return number < 10_000_000_000 ? number * 1_000 : number
}

async function fetchJson(
  provider: string,
  url: string,
  headers: Record<string, string>,
  options: UsageRequestOptions,
): Promise<Record<string, unknown>> {
  options.signal?.throwIfAborted()
  const deadline = new AbortController()
  const timer = setTimeout(() => deadline.abort(new Error('Usage request timed out')), 10000)
  const signal = options.signal ? AbortSignal.any([options.signal, deadline.signal]) : deadline.signal
  try {
    const fetcher = options.fetchImplementation ?? (globalThis.fetch as UsageFetch)
    const response = await fetcher(url, { headers, method: 'GET', redirect: 'error', signal })
    signal.throwIfAborted()
    if (!response.ok) {
      await response.body?.cancel()
      throw new ConfigurationError(provider, provider + ' usage request failed (' + response.status + ')')
    }
    const text = await usageResponseText(response, signal)
    signal.throwIfAborted()
    return recordOf(JSON.parse(text))
  } finally { clearTimeout(timer) }
}

export const MAX_USAGE_RESPONSE_BYTES = 262144
async function usageResponseText(response: Response, signal: AbortSignal): Promise<string> {
  if (Number(response.headers.get('content-length')) > MAX_USAGE_RESPONSE_BYTES) {
    await response.body?.cancel()
    throw new Error('Usage response exceeds size limit')
  }
  if (!response.body) throw new Error('Usage response has no body')
  const reader = response.body.getReader(), decoder = new TextDecoder('utf-8', { fatal: true })
  let bytes = 0, text = ''
  const abort = () => { void reader.cancel(signal.reason).catch(() => {}) }
  signal.addEventListener('abort', abort, { once: true })
  try {
    signal.throwIfAborted()
    for (;;) {
      const chunk = await reader.read()
      signal.throwIfAborted()
      if (chunk.done) break
      bytes += chunk.value.byteLength
      if (bytes > MAX_USAGE_RESPONSE_BYTES) throw new Error('Usage response exceeds size limit')
      text += decoder.decode(chunk.value, { stream: true })
    }
    return text + decoder.decode()
  } catch (error) {
    try { await reader.cancel(error) } catch { /* Preserve the original transport/parse failure. */ }
    throw error
  } finally { signal.removeEventListener('abort', abort); reader.releaseLock() }
}

const percentFromFraction = (value: number): number =>
  value <= 1 ? value * 100 : value

function claudeWindow(key: string, body: Record<string, unknown>, label: string, detail?: string): UsageWindow | undefined {
  const window = recordOf(body[key])
  const utilization = finite(window.utilization)
  if (utilization === undefined) return undefined
  const resetsAt = epochMs(window.resets_at)
  return {
    label,
    usedPercent: Math.max(0, Math.min(100, utilization)),
    ...(resetsAt !== undefined ? { resetsAt } : {}),
    ...(detail ? { detail } : {}),
  }
}

/** Claude Pro/Max: GET api.anthropic.com/api/oauth/usage (OAuth bearer + oauth beta flag). */
export async function fetchClaudeUsage(
  accessToken: string,
  options: UsageRequestOptions = {},
): Promise<ProviderUsageReport> {
  const url = options.environment?.XERXES_CLAUDE_USAGE_URL?.trim() || CLAUDE_USAGE_URL
  const body = await fetchJson('claude', url, {
    Authorization: `Bearer ${accessToken}`,
    'anthropic-beta': 'oauth-2025-04-20',
  }, options)
  const windows = [
    claudeWindow('five_hour', body, '5-hour'),
    claudeWindow('seven_day', body, 'weekly'),
    claudeWindow('seven_day_opus', body, 'weekly', 'opus'),
    claudeWindow('seven_day_sonnet', body, 'weekly', 'sonnet'),
  ].filter((window): window is UsageWindow => window !== undefined)
  if (!windows.length) {
    throw new ConfigurationError(
      'claude',
      'Claude usage response carried no recognized windows. If the endpoint moved, set XERXES_CLAUDE_USAGE_URL.',
    )
  }
  return { provider: 'claude', windows, fetchedAt: Date.now() }
}

function codexWindow(window: Record<string, unknown>, fallbackLabel: string, detail?: string, strict = false): UsageWindow | undefined {
  const usedPercent = finite(window.used_percent)
  if (strict && Object.keys(window).length && (usedPercent === undefined || usedPercent < 0 || usedPercent > 100)) throw new ConfigurationError('codex', 'Invalid explicit usage percentage')
  if (usedPercent === undefined) return undefined
  const seconds = finite(window.limit_window_seconds)
  const label = seconds === 18_000 ? '5-hour'
    : seconds === 604_800 ? 'weekly'
    : fallbackLabel
  const resetAfterSeconds = finite(window.reset_after_seconds)
  const resetsAt = epochMs(window.reset_at)
  return {
    label,
    usedPercent: Math.max(0, Math.min(100, usedPercent)),
    ...(resetsAt !== undefined ? { resetsAt } : {}),
    ...(resetAfterSeconds !== undefined ? { resetAfterSeconds } : {}),
    ...(detail ? { detail } : {}),
  }
}

/** ChatGPT Codex plans: GET chatgpt.com/backend-api/wham/usage (bearer + account header). */
export async function fetchCodexUsage(
  headers: Record<string, string>,
  options: UsageRequestOptions = {},
): Promise<ProviderUsageReport> {
  const url = options.environment?.XERXES_CODEX_USAGE_URL?.trim() || CODEX_USAGE_URL
  const body = await fetchJson('codex', url, headers, options)
  const rateLimit = recordOf(body.rate_limit)
  const windows = [
    codexWindow(recordOf(rateLimit.primary_window), options.strictUnits ? 'primary' : '5-hour', undefined, options.strictUnits),
    codexWindow(recordOf(rateLimit.secondary_window), options.strictUnits ? 'secondary' : 'weekly', undefined, options.strictUnits),
  ].filter((window): window is UsageWindow => window !== undefined)
  // Model-specific limits (e.g. Codex Spark) ride a side list with their own
  // 5-hour/weekly pair; tag them so they do not read as the plan's totals.
  const additional = Array.isArray(body.additional_rate_limits) ? body.additional_rate_limits : []
  for (const entry of additional) {
    const record = recordOf(entry)
    const name = stringOf(record.limit_name)
    const nested = recordOf(record.rate_limit)
    for (const window of [
      codexWindow(recordOf(nested.primary_window), options.strictUnits ? 'primary' : '5-hour', name, options.strictUnits),
      codexWindow(recordOf(nested.secondary_window), options.strictUnits ? 'secondary' : 'weekly', name, options.strictUnits),
    ]) {
      if (window) windows.push(window)
    }
  }
  if (!windows.length) {
    throw new ConfigurationError(
      'codex',
      'Codex usage response carried no rate-limit windows. If the endpoint moved, set XERXES_CODEX_USAGE_URL.',
    )
  }
  const planType = stringOf(body.plan_type)
  return {
    provider: 'codex',
    ...(planType ? { planType } : {}),
    windows,
    fetchedAt: Date.now(),
  }
}

/** Kimi Code subscription: GET api.kimi.com/coding/v1/usages (device-flow bearer). */
export async function fetchKimiUsage(
  accessToken: string,
  options: UsageRequestOptions = {},
): Promise<ProviderUsageReport> {
  const url = options.environment?.XERXES_KIMI_USAGE_URL?.trim() || KIMI_USAGE_URL
  const body = await fetchJson('kimi', url, { Authorization: `Bearer ${accessToken}` }, options)
  const windows = kimiWindows(body, options.strictUnits)
  if (!windows.length) {
    throw new ConfigurationError(
      'kimi',
      'Kimi usage response carried no recognized windows. If the endpoint moved, set XERXES_KIMI_USAGE_URL.',
    )
  }
  const planType = stringOf(body.plan) ?? stringOf(body.plan_type) ?? stringOf(body.membership)
  return {
    provider: 'kimi',
    ...(planType ? { planType } : {}),
    windows,
    fetchedAt: Date.now(),
  }
}

function kimiWindows(body: Record<string, unknown>, strict = false): UsageWindow[] {
  const windows: UsageWindow[] = []
  // Tolerant across the shapes the coding subscription has returned: a
  // top-level list of scope-tagged rows, or keyed five-hour/weekly records.
  const rows = Array.isArray(body.usages) ? body.usages
    : Array.isArray(body.data) ? body.data
    : []
  for (const row of rows) {
    const record = recordOf(row)
    if (strict && (finite(record.used_percent) === undefined || (record.used_percent as number) < 0 || (record.used_percent as number) > 100)) throw new ConfigurationError('kimi', 'Usage row has no valid explicit percentage')
    const used = finite(record.used_percent) ?? finite(record.percentage)
    const scope = (stringOf(record.scope) ?? stringOf(record.type) ?? '').toLowerCase()
    if (used === undefined || !scope) continue
    const resetsAt = epochMs(record.reset_at ?? record.resets_at)
    windows.push({
      label: strict ? scope : scope.includes('week') ? 'weekly' : scope.includes('5') ? '5-hour' : scope,
      usedPercent: Math.max(0, Math.min(100, finite(record.used_percent) !== undefined ? used : percentFromFraction(used))),
      ...(resetsAt !== undefined ? { resetsAt } : {}),
    })
  }
  for (const [key, label] of [['five_hour', '5-hour'], ['weekly', 'weekly'], ['seven_day', 'weekly']] as const) {
    if (strict && body[key] !== undefined) {
      const percent = finite(recordOf(body[key]).utilization)
      if (percent === undefined || percent < 0 || percent > 100) throw new ConfigurationError('kimi', 'Usage window has no valid percentage')
    }
    const window = claudeWindow(key, body, label)
    if (window) windows.push(window)
  }
  return windows
}

/**
 * Z.ai coding plan: GET api.z.ai/api/monitor/usage/quota/limit (API-key
 * bearer). The CN host answers the same path under open.bigmodel.cn.
 */
export async function fetchZaiUsage(
  apiKey: string,
  options: UsageRequestOptions & { readonly host?: 'cn' | 'global' } = {},
): Promise<ProviderUsageReport> {
  const fallback = options.host === 'cn'
    ? ZAI_USAGE_URL.replace('api.z.ai', 'open.bigmodel.cn')
    : ZAI_USAGE_URL
  const url = options.environment?.XERXES_ZAI_USAGE_URL?.trim() || fallback
  const body = await fetchJson('zai', url, { Authorization: `Bearer ${apiKey}` }, options)
  const data = recordOf(body.data)
  const limits = Array.isArray(data.limits) ? data.limits : []
  const windows = limits.flatMap(limit => {
    const record = recordOf(limit)
    const used = finite(record.percentage)
    if (options.strictUnits && (used === undefined || used < 0 || used > 100)) throw new ConfigurationError('zai', 'Usage window has no valid percentage')
    if (used === undefined) return []
    const type = stringOf(record.type) ?? 'quota'
    const unit = finite(record.unit)
    const label = options.strictUnits ? type.toLowerCase() : type === 'TOKENS_LIMIT' ? (unit === 5 ? '5-hour' : 'weekly')
      : type === 'TIME_LIMIT' ? '5-hour'
      : type.toLowerCase()
    const nextReset = epochMs(record.nextResetTime)
    const remaining = finite(record.remaining)
    return [{
      label,
      usedPercent: Math.max(0, Math.min(100, used)),
      ...(nextReset !== undefined ? { resetsAt: nextReset } : {}),
      detail: options.strictUnits ? type.toLowerCase() : remaining !== undefined ? `${remaining} remaining` : type.toLowerCase(),
    }]
  })
  if (!windows.length) {
    throw new ConfigurationError(
      'zai',
      'Z.ai usage response carried no quota limits. If the endpoint moved, set XERXES_ZAI_USAGE_URL.',
    )
  }
  return { provider: 'zai', windows, fetchedAt: Date.now() }
}

export const SUBSCRIPTION_USAGE_PROVIDERS = ['claude', 'codex', 'kimi', 'zai'] as const
export type SubscriptionUsageProvider = (typeof SUBSCRIPTION_USAGE_PROVIDERS)[number]

const PROVIDER_ALIASES: Readonly<Record<string, SubscriptionUsageProvider>> = {
  anthropic: 'claude',
  claude: 'claude',
  chatgpt: 'codex',
  codex: 'codex',
  'openai-codex': 'codex',
  kimi: 'kimi',
  'kimi-code': 'kimi',
  bigmodel: 'zai',
  zai: 'zai',
  'zai-coding': 'zai',
  'zai-coding-cn': 'zai',
  zhipu: 'zai',
}

export interface SubscriptionUsageError {
  readonly provider: SubscriptionUsageProvider
  readonly message: string
}

export interface SubscriptionUsageCollection {
  readonly errors: readonly SubscriptionUsageError[]
  readonly reports: readonly ProviderUsageReport[]
}

/** Minimal profile shape the collector needs from the daemon's profile store. */
export interface UsageProfile {
  readonly name: string
  readonly provider: string
  readonly api_key: string
  readonly base_url: string
  readonly model: string
}

/**
 * Resolve subscription quota for every configured profile (and OAuth
 * session), not just the active one. A profile whose provider has no quota
 * endpoint is skipped; a profile whose fetch fails lands in `errors`.
 */
export async function collectSubscriptionUsage(
  provider?: string,
  options: UsageRequestOptions & { readonly profiles?: readonly UsageProfile[] } = {},
): Promise<SubscriptionUsageCollection> {
  const normalized = provider?.trim().toLowerCase()
  const targets = normalized
    ? [PROVIDER_ALIASES[normalized] ?? (() => {
        throw new ConfigurationError(
          'usage',
          `unknown usage provider '${provider}'; expected one of ${SUBSCRIPTION_USAGE_PROVIDERS.join(', ')}`,
        )
      })()]
    : [...SUBSCRIPTION_USAGE_PROVIDERS]
  const reports: ProviderUsageReport[] = []
  const errors: SubscriptionUsageError[] = []
  await Promise.all(targets.map(async target => {
    try {
      reports.push(await fetchOne(target, options))
    } catch (error) {
      errors.push({ provider: target, message: error instanceof Error ? error.message : String(error) })
    }
  }))
  const order = new Map(SUBSCRIPTION_USAGE_PROVIDERS.map((name, index) => [name, index]))
  reports.sort((a, b) => (order.get(a.provider as SubscriptionUsageProvider) ?? 0) - (order.get(b.provider as SubscriptionUsageProvider) ?? 0))
  errors.sort((a, b) => (order.get(a.provider) ?? 0) - (order.get(b.provider) ?? 0))
  return { errors, reports }
}

async function fetchOne(
  provider: SubscriptionUsageProvider,
  options: UsageRequestOptions & { readonly profiles?: readonly UsageProfile[] },
): Promise<ProviderUsageReport> {
  const environment = options.environment ?? process.env
  const profiles = options.profiles ?? []
  const profile = profiles.find(p => PROVIDER_ALIASES[p.provider] === provider)

  if (provider === 'claude') {
    const credential = await new AnthropicOAuthSession({ environment }).credential(options.signal)
    return fetchClaudeUsage(credential.access, { ...options, environment })
  }
  if (provider === 'codex') {
    const credential = await new CodexSession({ environment }).credential(options.signal)
    return fetchCodexUsage(codexAuthHeaders(credential), { ...options, environment })
  }
  if (provider === 'kimi') {
    // Profile API key first (Kimi Code profiles store the coding key), then
    // the device-flow OAuth session.
    const apiKey = profile?.api_key?.trim() || environment.KIMI_CODE_API_KEY?.trim()
    if (apiKey) return fetchKimiUsage(apiKey, { ...options, environment })
    const credential = await new KimiCodingOAuthSession({ environment }).credential(options.signal)
    return fetchKimiUsage(credential.access, { ...options, environment })
  }
  // zai
  const apiKey = profile?.api_key?.trim()
    || environment.ZHIPU_API_KEY?.trim()
    || environment.ZAI_API_KEY?.trim()
  if (!apiKey) {
    throw new ConfigurationError(
      'zai',
      'No Z.ai API key found. Add a zai-coding profile or set ZHIPU_API_KEY.',
    )
  }
  const isCn = profile?.base_url.includes('bigmodel.cn')
    || environment.XERXES_ZAI_USAGE_URL?.includes('bigmodel.cn')
  return fetchZaiUsage(apiKey, {
    ...options,
    environment,
    host: isCn ? 'cn' : 'global',
  })
}

/** Compact human rendering: `Claude (max) — 5-hour 42% (resets 15:00), weekly 12%`. */
export function formatUsageReport(report: ProviderUsageReport, now = Date.now()): string {
  const name = report.provider[0]!.toUpperCase() + report.provider.slice(1)
  const plan = report.planType ? ` (${report.planType})` : ''
  const windows = report.windows.map(window => {
    const reset = window.resetsAt !== undefined
      ? `resets ${new Date(window.resetsAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
      : window.resetAfterSeconds !== undefined
        ? `resets in ${Math.max(1, Math.round((window.resetAfterSeconds - Math.max(0, now - report.fetchedAt) / 1_000) / 60))}m`
        : undefined
    const detail = window.detail ? ` ${window.detail}` : ''
    return `${window.label}${detail} ${Math.round(window.usedPercent)}%${reset ? ` (${reset})` : ''}`
  })
  return `${name}${plan} — ${windows.join(', ')}`
}
