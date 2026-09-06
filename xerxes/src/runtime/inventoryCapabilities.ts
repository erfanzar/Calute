// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { profileContextLimit, profileMaxOutputTokens, resolvedProfileModelCapabilities, type ProviderProfile } from '../bridge/profiles.js'
import type { InventoryModel } from './modelInventory.js'

/** Resolve live metadata using the same override precedence as agent execution. */
export function inventoryCapabilities(profile: ProviderProfile, model: InventoryModel): InventoryModel {
  const context = model.context_limit ?? profileContextLimit(profile, model.id)
  const output = model.max_output_tokens ?? profileMaxOutputTokens(profile, model.id)
  const resolved = resolvedProfileModelCapabilities({
    ...profile,
    model_capabilities: {
      ...profile.model_capabilities,
      [model.id]: {
        ...(context === undefined ? {} : { context_limit: context }),
        ...(output === undefined ? {} : { max_output_tokens: output }),
      },
    },
  }, model.id)
  return {
    id: model.id,
    ...(resolved.contextLimit === undefined ? {} : { context_limit: resolved.contextLimit }),
    ...(resolved.maxOutputTokens === undefined ? {} : { max_output_tokens: resolved.maxOutputTokens }),
    context_source: resolved.contextSource,
    output_source: resolved.outputSource,
  }
}
