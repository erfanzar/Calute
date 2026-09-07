// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Structural port keeps this example portable outside a Xerxes checkout. */
interface Registry {
  registerTool(name: string, callback: (...args: unknown[]) => unknown,
    meta: { name: string; description: string; version: string }): void
}

export function register(registry: Registry): void {
  registry.registerTool('text_stats', (...args: unknown[]) => {
    if (args.length !== 1 || typeof args[0] !== 'string') {
      throw new Error('text_stats expects exactly one string in args')
    }
    const text = args[0]
    return {
      characters: Array.from(text).length,
      words: text.trim() ? text.trim().split(/\s+/u).length : 0,
      lines: text === '' ? 0 : text.split(/\r\n|\r|\n/u).length,
    }
  }, { name: 'text-statistics', description: 'Count Unicode characters, whitespace-separated words and lines', version: '1.0.0' })
}
