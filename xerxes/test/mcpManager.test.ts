// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  MCPCapabilityNotFoundError,
  MCPManager,
  type MCPClientPort,
} from '../src/mcp/manager.js'
import {
  MCPReconnectError,
  ReconnectPolicy,
  reconnectWithBackoff,
  scrubCredentials,
} from '../src/mcp/reconnect.js'
import type {
  MCPPrompt,
  MCPPromptResult,
  MCPResource,
  MCPResourceContentsResult,
  MCPServerConfig,
  MCPTool,
  MCPToolCallResult,
} from '../src/mcp/types.js'
import type { JsonObject } from '../src/types/toolCalls.js'

interface ClientFixture {
  readonly prompts?: readonly MCPPrompt[]
  readonly resources?: readonly MCPResource[]
  readonly tools?: readonly MCPTool[]
  readonly onCallTool?: (signal?: AbortSignal) => void | Promise<void>
  readonly onConnect?: () => void | Promise<void>
  readonly onDisconnect?: () => void | Promise<void>
}

class FakeMCPClient implements MCPClientPort {
  readonly calls: Array<{ readonly arguments_: JsonObject; readonly name: string }> = []
  readonly config: MCPServerConfig
  readonly prompts: readonly MCPPrompt[]
  readonly resources: readonly MCPResource[]
  readonly tools: readonly MCPTool[]
  connected = false
  disconnects = 0

  constructor(config: MCPServerConfig, private readonly fixture: ClientFixture = {}) {
    this.config = config
    this.prompts = fixture.prompts ?? []
    this.resources = fixture.resources ?? []
    this.tools = fixture.tools ?? []
  }

  async connect(): Promise<void> {
    await this.fixture.onConnect?.()
    this.connected = true
  }

  async disconnect(): Promise<void> {
    this.disconnects += 1
    await this.fixture.onDisconnect?.()
    this.connected = false
  }

  async callTool(
    name: string,
    arguments_: JsonObject = {},
    options: { readonly signal?: AbortSignal } = {},
  ): Promise<MCPToolCallResult> {
    this.calls.push({ name, arguments_ })
    await this.fixture.onCallTool?.(options.signal)
    return { content: [{ type: 'text', text: this.config.name + ':' + name }] }
  }

  async readResource(uri: string): Promise<MCPResourceContentsResult> {
    return { contents: [{ uri, text: this.config.name + ':resource' }] }
  }

  async getPrompt(name: string, arguments_: JsonObject = {}): Promise<MCPPromptResult> {
    return {
      messages: [{
        role: 'user',
        content: { type: 'text', text: this.config.name + ':' + name + ':' + String(arguments_.name) },
      }],
    }
  }
}

const ALPHA_TOOLS: readonly MCPTool[] = [
  { name: 'echo', description: 'First echo', inputSchema: { type: 'object' } },
  { name: 'alpha_only', inputSchema: { type: 'object' } },
]
const BETA_TOOLS: readonly MCPTool[] = [
  { name: 'echo', description: 'Second echo', inputSchema: { type: 'object' } },
  { name: 'beta_only', inputSchema: { type: 'object' } },
]

test('MCPManager owns multiple live clients, deduplicates tools, and routes capabilities', async () => {
  const clients: FakeMCPClient[] = []
  const manager = new MCPManager({
    clientFactory: config => {
      const client = new FakeMCPClient(config, config.name === 'alpha'
        ? {
            tools: ALPHA_TOOLS,
            resources: [{ uri: 'memo://alpha', name: 'Alpha' }],
            prompts: [{ name: 'brief' }],
          }
        : {
            tools: BETA_TOOLS,
            resources: [{ uri: 'memo://beta', name: 'Beta' }],
            prompts: [{ name: 'brief' }, { name: 'beta_prompt' }],
          })
      clients.push(client)
      return client
    },
  })

  expect(await manager.addServer({ name: 'alpha' })).toBeTrue()
  expect(await manager.start({ name: 'beta' })).toBeTrue()
  expect(await manager.addServer({ name: 'alpha' })).toBeFalse()
  expect(await manager.addServer({ name: 'off', enabled: false })).toBeFalse()
  expect(clients).toHaveLength(2)
  expect(manager.listServers()).toEqual(['alpha', 'beta'])
  expect(manager.getAllTools()).toEqual([
    { name: 'echo', description: 'First echo', inputSchema: { type: 'object' }, serverName: 'alpha' },
    { name: 'alpha_only', inputSchema: { type: 'object' }, serverName: 'alpha' },
    { name: 'beta_only', inputSchema: { type: 'object' }, serverName: 'beta' },
  ])
  expect(manager.getAllResources()).toEqual([
    { uri: 'memo://alpha', name: 'Alpha', serverName: 'alpha' },
    { uri: 'memo://beta', name: 'Beta', serverName: 'beta' },
  ])
  expect(manager.getAllPrompts()).toEqual([
    { name: 'brief', serverName: 'alpha' },
    { name: 'brief', serverName: 'beta' },
    { name: 'beta_prompt', serverName: 'beta' },
  ])
  expect(manager.getCapabilitiesSummary()).toEqual({
    alpha: { tools: 2, resources: 1, prompts: 1 },
    beta: { tools: 2, resources: 1, prompts: 2 },
  })
  expect(manager.status('beta')).toEqual({
    name: 'beta',
    connected: true,
    tools: 2,
    resources: 1,
    prompts: 2,
  })

  await expect(manager.callTool('echo', { message: 'hello' })).resolves.toEqual({
    content: [{ type: 'text', text: 'alpha:echo' }],
  })
  await expect(manager.readResource('memo://beta')).resolves.toEqual({
    contents: [{ uri: 'memo://beta', text: 'beta:resource' }],
  })
  await expect(manager.getPrompt('beta_prompt', { name: 'Ada' })).resolves.toEqual({
    messages: [{ role: 'user', content: { type: 'text', text: 'beta:beta_prompt:Ada' } }],
  })
  expect(clients[0]?.calls).toEqual([{ name: 'echo', arguments_: { message: 'hello' } }])
  await expect(manager.callTool('missing')).rejects.toBeInstanceOf(MCPCapabilityNotFoundError)

  expect(await manager.stop('alpha')).toBeTrue()
  expect(manager.getServer('alpha')).toBeUndefined()
  expect(clients[0]?.disconnects).toBe(1)
  await manager.stopAll()
  expect(manager.listServers()).toEqual([])
  expect(clients[1]?.disconnects).toBe(1)
})

test('MCPManager propagates AbortSignal and does not serialize calls to independent servers', async () => {
  let alphaStarted!: () => void
  let releaseAlpha!: () => void
  const alphaStart = new Promise<void>(resolve => { alphaStarted = resolve })
  const alphaGate = new Promise<void>(resolve => { releaseAlpha = resolve })
  let receivedSignal: AbortSignal | undefined
  const manager = new MCPManager({
    clientFactory: config => new FakeMCPClient(config, config.name === 'alpha'
      ? {
          tools: [{ name: 'alpha_tool', inputSchema: { type: 'object' } }],
          onCallTool: signal => {
            receivedSignal = signal
            alphaStarted()
            return alphaGate
          },
        }
      : { tools: [{ name: 'beta_tool', inputSchema: { type: 'object' } }] }),
  })
  await Promise.all([manager.addServer({ name: 'alpha' }), manager.addServer({ name: 'beta' })])

  const controller = new AbortController()
  const alphaCall = manager.callTool('alpha_tool', {}, { signal: controller.signal })
  await alphaStart
  await expect(Promise.race([
    manager.callTool('beta_tool'),
    new Promise<'timed out'>(resolve => setTimeout(() => resolve('timed out'), 100)),
  ])).resolves.toEqual({ content: [{ type: 'text', text: 'beta:beta_tool' }] })
  expect(receivedSignal).toBe(controller.signal)

  releaseAlpha()
  await alphaCall
  await manager.disconnectAll()
})

test('MCPManager serializes lifecycle changes with in-flight calls on the same server', async () => {
  let callStarted!: () => void
  let releaseCall!: () => void
  const started = new Promise<void>(resolve => { callStarted = resolve })
  const gate = new Promise<void>(resolve => { releaseCall = resolve })
  const manager = new MCPManager({
    clientFactory: config => new FakeMCPClient(config, {
      tools: [{ name: 'hold', inputSchema: { type: 'object' } }],
      onCallTool: () => {
        callStarted()
        return gate
      },
    }),
  })
  await manager.addServer({ name: 'alpha' })

  const call = manager.callTool('hold')
  await started
  const removal = manager.removeServer('alpha')
  await new Promise(resolve => setTimeout(resolve, 10))
  expect(manager.getServer('alpha')).toBeDefined()

  releaseCall()
  await call
  await expect(removal).resolves.toBeTrue()
})

test('MCPManager bounds pending operations independently for each server', async () => {
  let releaseAlpha!: () => void
  const alphaGate = new Promise<void>(resolve => { releaseAlpha = resolve })
  const manager = new MCPManager({
    maxPendingOperationsPerServer: 2,
    clientFactory: config => new FakeMCPClient(config, config.name === 'alpha'
      ? { tools: [{ name: 'alpha_tool', inputSchema: { type: 'object' } }], onCallTool: () => alphaGate }
      : { tools: [{ name: 'beta_tool', inputSchema: { type: 'object' } }] }),
  })
  await Promise.all([manager.addServer({ name: 'alpha' }), manager.addServer({ name: 'beta' })])

  const active = manager.callTool('alpha_tool')
  const queued = manager.callTool('alpha_tool')
  await expect(manager.callTool('alpha_tool')).rejects.toMatchObject({
    name: 'MCPServerQueueFullError',
    serverName: 'alpha',
    limit: 2,
  })
  await expect(manager.callTool('beta_tool')).resolves.toEqual({
    content: [{ type: 'text', text: 'beta:beta_tool' }],
  })

  releaseAlpha()
  await Promise.all([active, queued])
  await manager.disconnectAll()
})

test('MCPManager reconnects through fresh factory clients with injected backoff', async () => {
  const clients: FakeMCPClient[] = []
  const sleeps: number[] = []
  let connection = 0
  const manager = new MCPManager({
    clientFactory: config => {
      connection += 1
      const attempt = connection
      const client = new FakeMCPClient(config, {
        tools: [{ name: 'echo', inputSchema: { type: 'object' } }],
        onConnect: () => {
          if (attempt === 2 || attempt === 3) {
            throw new Error('authorization: bearer secret-token-value-' + attempt)
          }
        },
      })
      clients.push(client)
      return client
    },
    reconnect: {
      policy: new ReconnectPolicy({ maxAttempts: 4, baseSeconds: 1, factor: 2, maxSeconds: 8 }),
      sleep: seconds => {
        sleeps.push(seconds)
      },
    },
  })

  expect(await manager.addServer({ name: 'alpha' })).toBeTrue()
  expect(await manager.reconnect('alpha')).toBeTrue()
  expect(sleeps).toEqual([1, 2])
  expect(clients).toHaveLength(4)
  expect(clients[0]?.disconnects).toBe(1)
  expect(manager.lastFailure('alpha')).toBeUndefined()
  expect(manager.getServer('alpha')).toBe(clients[3])
})

test('MCPManager removes an exhausted reconnect candidate and retains only a redacted failure', async () => {
  const sleeps: number[] = []
  let factoryCalls = 0
  const manager = new MCPManager({
    clientFactory: config => {
      factoryCalls += 1
      return new FakeMCPClient(config, {
        onConnect: () => {
          if (factoryCalls > 1) {
            throw new Error('api_key=super-secret-key-12345678')
          }
        },
      })
    },
    reconnect: {
      policy: { maxAttempts: 3, baseSeconds: 0.5, factor: 2, maxSeconds: 5 },
      sleep: seconds => {
        sleeps.push(seconds)
      },
    },
  })

  expect(await manager.addServer({ name: 'alpha' })).toBeTrue()
  expect(await manager.reconnect('alpha')).toBeFalse()
  expect(manager.getServer('alpha')).toBeUndefined()
  expect(manager.listServers()).toEqual([])
  expect(sleeps).toEqual([0.5, 1])
  expect(manager.lastFailure('alpha')).toEqual({
    name: 'alpha',
    operation: 'reconnect',
    attempt: 3,
    error: 'api_key=[redacted]',
  })
})

test('reconnectWithBackoff uses seconds, validates policy values, and scrubs terminal errors', async () => {
  const delays: number[] = []
  let attempts = 0
  await expect(reconnectWithBackoff(
    () => {
      attempts += 1
      if (attempts < 3) {
        throw new Error('token=very-secret-token-value-123')
      }
      return 'connected'
    },
    {
      policy: { maxAttempts: 4, baseSeconds: 2, factor: 3, maxSeconds: 5 },
      sleep: seconds => {
        delays.push(seconds)
      },
    },
  )).resolves.toBe('connected')
  expect(delays).toEqual([2, 5])
  expect(scrubCredentials('password hunter2 sk-abcdefghijklmnop')).toBe('password=[redacted] [redacted]')
  expect(() => new ReconnectPolicy({ maxAttempts: 0 })).toThrow(RangeError)

  await expect(reconnectWithBackoff(
    () => {
      throw new Error('authorization: bearer secret-token-value-123')
    },
    { policy: { maxAttempts: 1 } },
  )).rejects.toMatchObject({
    name: 'MCPReconnectError',
    attempts: 1,
    message: 'authorization: bearer=[redacted]',
  } satisfies Partial<MCPReconnectError>)
})

test('failed and disabled MCP configurations stay discoverable and failed initial connections can reconnect', async () => {
  let fail = true
  const manager = new MCPManager({
    clientFactory: config => new FakeMCPClient(config, { tools: ALPHA_TOOLS, onConnect: () => { if (fail) throw new Error('authentication failed: private-health-token') } }),
    reconnect: { policy: { maxAttempts: 1 } },
  })
  expect(await manager.addServer({ name: 'broken', env: { API_KEY: 'private-health-token' } })).toBe(false)
  expect(await manager.addServer({ name: 'off', enabled: false })).toBe(false)
  expect(manager.listServers()).toEqual([])
  expect(manager.listConfiguredServers()).toEqual(['broken', 'off'])
  expect(manager.status('broken')).toMatchObject({ connected: false, state: 'failed', tools: 0 })
  expect(JSON.stringify(manager.listStatus())).not.toContain('private-health-token')
  expect(manager.status('off')?.state).toBe('disabled')
  expect(await manager.reconnect('off')).toBe(false)
  fail = false
  expect(await manager.reconnect('broken')).toBe(true)
  expect(manager.status('broken')).toMatchObject({ connected: true, tools: 2 })
  expect(manager.status('broken')?.lastError).toBeUndefined()
  await manager.disconnectAll()
  expect(manager.listConfiguredServers()).toEqual([])
})

test('removing a configuration during reconnect prevents late candidate resurrection', async () => {
  const started = Promise.withResolvers<void>()
  const finish = Promise.withResolvers<void>()
  const clients: FakeMCPClient[] = []
  const manager = new MCPManager({ clientFactory: config => {
    const retry = clients.length > 0
    const client = new FakeMCPClient(config, { onConnect: async () => { if (retry) { started.resolve(); await finish.promise } } })
    clients.push(client)
    return client
  } })
  await manager.addServer({ name: 'alpha' })
  const reconnecting = manager.reconnect('alpha')
  await started.promise
  expect(await manager.removeServer('alpha')).toBe(true)
  finish.resolve()
  expect(await reconnecting).toBe(false)
  expect(manager.status('alpha')).toBeUndefined()
  expect(manager.getAllTools()).toEqual([])
  expect(clients[1]?.disconnects).toBe(1)
})

test('reusing a configuration object cannot let an older reconnect replace its new registration', async () => {
  const started = Promise.withResolvers<void>()
  const finish = Promise.withResolvers<void>()
  const clients: FakeMCPClient[] = []
  const manager = new MCPManager({
    clientFactory: config => {
      const candidate = new FakeMCPClient(config, clients.length === 1 ? {
        onConnect: async () => { started.resolve(); await finish.promise },
      } : {})
      clients.push(candidate)
      return candidate
    },
    reconnect: { policy: { maxAttempts: 1 } },
  })
  const config = { name: 'alpha' }
  await manager.addServer(config)
  const reconnecting = manager.reconnect('alpha')
  await started.promise
  await manager.removeServer('alpha')
  await manager.addServer(config)
  finish.resolve()
  expect(await reconnecting).toBe(false)
  expect(manager.getServer('alpha')).toBe(clients[2])
  expect(clients[1]?.disconnects).toBe(1)
  await manager.disconnectAll()
})

test('removing a server interrupts its retry sleep without waiting for the sleeper', async () => {
  const sleeping = Promise.withResolvers<void>()
  let calls = 0
  const manager = new MCPManager({
    clientFactory: config => new FakeMCPClient(config, { onConnect: () => {
      calls += 1
      if (calls > 1) throw new Error('offline')
    } }),
    reconnect: { sleep: () => { sleeping.resolve(); return new Promise<void>(() => {}) } },
  })
  await manager.addServer({ name: 'alpha' })
  const reconnecting = manager.reconnect('alpha')
  await sleeping.promise
  await manager.removeServer('alpha')
  expect(await reconnecting).toBe(false)
  expect(calls).toBe(2)
  expect(manager.listStatus()).toEqual([])
}, 1000)

test('reconnect cancellation skips future attempts and redacts arbitrary abort reasons', async () => {
  const controller = new AbortController()
  let attempts = 0
  const retry = reconnectWithBackoff(() => {
    attempts += 1
    throw new Error('offline')
  }, {
    signal: controller.signal,
    onError: () => controller.abort('private-cancel-secret'),
  })
  await expect(retry).rejects.toMatchObject({ name: 'AbortError', message: 'MCP reconnect cancelled' })
  expect(attempts).toBe(1)
  await expect(reconnectWithBackoff(() => { throw new Error('must not run') }, { signal: controller.signal }))
    .rejects.toMatchObject({ name: 'AbortError' })
})

test('disconnectAll interrupts native retry timers and retains no registration', async () => {
  const failed = Promise.withResolvers<void>()
  let attempts = 0
  const manager = new MCPManager({
    clientFactory: config => new FakeMCPClient(config, { onConnect: () => {
      attempts += 1
      if (attempts > 1) throw new Error('offline')
    } }),
    reconnect: { policy: { baseSeconds: 60 }, onError: () => { failed.resolve() } },
  })
  await manager.addServer({ name: 'alpha' })
  const reconnecting = manager.reconnect('alpha')
  await failed.promise
  // Let the rejection hook finish and the native retry timer start.
  await Bun.sleep(0)
  await manager.disconnectAll()
  expect(await reconnecting).toBe(false)
  expect(manager.listConfiguredServers()).toEqual([])
  expect(attempts).toBe(2)
}, 1000)

test.each([false, true])('queued MCP cancellation settles before preceding work and never dispatches (namespaced=%s)', async namespaced => {
  const gate = Promise.withResolvers<void>()
  const entered = Promise.withResolvers<void>()
  let client!: FakeMCPClient
  const manager = new MCPManager({
    maxPendingOperationsPerServer: 2,
    clientFactory: config => client = new FakeMCPClient(config, {
      tools: ALPHA_TOOLS,
      onCallTool: async () => { entered.resolve(); await gate.promise },
    }),
  })
  await manager.addServer({ name: 'alpha' })
  const first = manager.callTool('echo')
  await entered.promise
  const controller = new AbortController()
  const queued = namespaced
    ? manager.callServerTool('alpha', client, 'echo', {}, { signal: controller.signal })
    : manager.callTool('echo', {}, { signal: controller.signal })
  controller.abort('private cancellation reason')
  try {
    await expect(queued).rejects.toMatchObject({ name: 'AbortError', message: 'MCP operation cancelled before execution' })
    expect(client.calls).toHaveLength(1)
    // Tombstones remain bounded until preceding work drains.
    await expect(manager.callTool('echo')).rejects.toMatchObject({ name: 'MCPServerQueueFullError' })
  } finally { gate.resolve(); await first }
  await Bun.sleep(0)
  await manager.callTool('echo')
  expect(client.calls).toHaveLength(2)
  await manager.disconnectAll()
}, 1000)

test('replacement preserves working MCP calls on validation and connection failure', async () => {
  const clients: FakeMCPClient[] = []
  const manager = new MCPManager({ clientFactory: config => {
    const client = new FakeMCPClient(config, { tools: ALPHA_TOOLS,
      onConnect: () => { if (config.command === 'bad') throw new Error('candidate-private-secret') },
    })
    clients.push(client); return client
  } })
  try {
    await manager.addServer({ name: 'alpha', command: 'good' })
    const previous = manager.getServer('alpha')!
    await expect(manager.replaceServer({ name: 'alpha', command: 'bad', enabled: 'false' })).rejects.toThrow('enabled must be a boolean')
    expect(clients).toHaveLength(1)
    await expect(manager.replaceServer({ name: 'alpha', command: 'bad', env: { TOKEN: 'candidate-private-secret' } })).rejects.not.toThrow('candidate-private-secret')
    expect(manager.getServer('alpha')).toBe(previous)
    expect(manager.status('alpha')?.connected).toBe(true)
    expect(manager.lastFailure('alpha')?.operation).toBe('replace')
    expect(clients[1]?.disconnects).toBe(1)
    await manager.callServerTool('alpha', previous, 'echo')
    expect(clients[0]?.calls).toHaveLength(1)
    expect(await manager.replaceServer({ name: 'alpha', command: 'new' })).toBe(true)
    expect(clients[0]?.disconnects).toBe(1)
    expect(manager.getServer('alpha')).toBe(clients[2])
    await expect(manager.callServerTool('alpha', previous, 'echo')).rejects.toThrow('changed or disconnected')
    expect(await manager.replaceServer({ name: 'alpha', command: 'new', enabled: false })).toBe(true)
    expect(manager.status('alpha')?.state).toBe('disabled')
    expect(clients).toHaveLength(3)
    expect(clients[2]?.disconnects).toBe(1)
  } finally { await manager.disconnectAll() }
})

test.each(['cancel', 'remove', 'replace'] as const)('staged MCP replacement handles %s without losing or resurrecting registrations', async action => {
  const entered = Promise.withResolvers<void>(), release = Promise.withResolvers<void>()
  const clients: FakeMCPClient[] = []
  const manager = new MCPManager({ clientFactory: config => {
    const client = new FakeMCPClient(config, { tools: ALPHA_TOOLS,
      onConnect: async () => { if (config.command === 'candidate') { entered.resolve(); await release.promise } },
    })
    clients.push(client); return client
  } })
  const controller = new AbortController()
  try {
    await manager.addServer({ name: 'alpha', command: 'old' })
    const previous = manager.getServer('alpha')!
    const pending = manager.replaceServer({ name: 'alpha', command: 'candidate' }, controller.signal)
    const result = pending.then(() => 'success', error => error as Error)
    await entered.promise
    await manager.callServerTool('alpha', previous, 'echo')
    expect(clients[0]?.calls).toHaveLength(1)
    await expect(manager.replaceServer({ name: 'alpha', command: 'other' })).rejects.toThrow('already in progress')
    if (action === 'cancel') controller.abort('private-abort-secret')
    else {
      await manager.removeServer('alpha')
      if (action === 'replace') await manager.addServer({ name: 'alpha', command: 'newer' })
    }
    release.resolve()
    const failure = await result
    expect(failure).toBeInstanceOf(Error)
    expect(String(failure)).not.toContain('private-abort-secret')
    if (action === 'cancel') expect((failure as Error).name).toBe('AbortError')
    expect(clients[1]?.disconnects).toBe(1)
    expect(manager.getServer('alpha')?.config.command).toBe(action === 'cancel' ? 'old' : action === 'replace' ? 'newer' : undefined)
    if (action !== 'remove') expect(await manager.replaceServer({ name: 'alpha', command: 'final' })).toBe(true)
  } finally { release.resolve(); await manager.disconnectAll() }
})

test('queued legacy capability calls do not dispatch to a retired client', async () => {
  const entered = Promise.withResolvers<void>(), release = Promise.withResolvers<void>()
  const client = new FakeMCPClient({ name: 'alpha', command: 'bun' }, {
    tools: ALPHA_TOOLS, resources: [{ uri: 'memo://alpha', name: 'memo' }], prompts: [{ name: 'brief' }],
    onCallTool: async () => { entered.resolve(); await release.promise },
  })
  const manager = new MCPManager({ clientFactory: () => client })
  try {
    await manager.addServer(client.config)
    const first = manager.callTool('echo')
    await entered.promise
    const removing = manager.removeServer('alpha')
    const queued = [manager.callTool('echo'), manager.readResource('memo://alpha'), manager.getPrompt('brief')]
    const results = Promise.allSettled(queued)
    release.resolve()
    await first; await removing
    for (const result of await results) {
      expect(result.status).toBe('rejected')
      if (result.status === 'rejected') expect(String(result.reason)).toContain('changed or disconnected')
    }
    expect(client.calls).toHaveLength(1)
  } finally { release.resolve(); await manager.disconnectAll() }
})

test('replacement reports old-client teardown failure without discarding the installed candidate', async () => {
  const clients: FakeMCPClient[] = []
  const manager = new MCPManager({ clientFactory: config => {
    const client = new FakeMCPClient(config, { onDisconnect: () => {
      if (config.command === 'old') throw new Error('old-private-secret')
    } })
    clients.push(client); return client
  } })
  try {
    await manager.addServer({ name: 'alpha', command: 'old', env: { TOKEN: 'old-private-secret' } })
    const controller = new AbortController(); controller.abort('private-abort-secret')
    await expect(manager.replaceServer({ name: 'alpha', command: 'new' }, controller.signal)).rejects.toMatchObject({ name: 'AbortError' })
    expect(clients).toHaveLength(1)
    expect(await manager.replaceServer({ name: 'alpha', command: 'new' })).toBe(true)
    expect(manager.getServer('alpha')).toBe(clients[1])
    expect(manager.status('alpha')?.connected).toBe(true)
    expect(manager.lastFailure('alpha')?.operation).toBe('disconnect')
    expect(manager.lastFailure('alpha')?.error).not.toContain('old-private-secret')
  } finally { await manager.disconnectAll() }
})
