// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** Lifecycle notifications only: output bytes never trigger a UI refresh. */
export class ActivityChanges {
  private readonly listeners = new Set<() => void>()
  subscribe(listener: () => void): () => void {
    this.listeners.add(listener)
    return () => { this.listeners.delete(listener) }
  }
  notify(): void {
    for (const listener of this.listeners) {
      try { listener() } catch (error) { console.error('Activity observer failed:', error) }
    }
  }
}
