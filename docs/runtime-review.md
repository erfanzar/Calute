# Delegation runtime review

This is the historical review from the initial intelligence-tier implementation.
Its limitations and test totals describe that earlier snapshot, not the current
checkpoint. Provider/profile routing, reasoning controls and durable recovery
have since changed; see [implementation status](feature-gap-implementation-status.md)
and the [configuration guide](configuration-guide.md) for current behavior.

This review covers model selection and the surrounding delegation path: Claude-compatible
tools → native subagent host → subagent manager → child turn runner, including batch
registration, cancellation, persisted model selection, and retry. It is not an exhaustive
review of every provider, channel, or external integration.

## Changes made

- Added user-owned intelligence-to-model mappings without changing the session wire format.
  Model selection happens before registration, so existing snapshots and retries retain the
  resolved model. Defaults preserve inheritance unless configured otherwise.
- Preflight every swarm's tier selection before starting children. A typo or unmapped tier
  in a later entry cannot start earlier children and then force a rollback.
- Reject already-cancelled single/background delegation, matching swarm behavior. Previously
  `AgentTool` could start detached work after the caller had cancelled.
- Corrected the system prompt's unlimited-batch claim: the runtime caps batches at 32 and
  uses a registration concurrency pool. The prompt now describes the real boundary.

## Design observations and next improvements

1. **Registration concurrency and execution concurrency are distinct.** `agentOps.ts` limits
   registration concurrency; `subagentManager.ts` also enforces live-agent and depth limits.
   Make both visible in fleet telemetry so a queued child can report whether it is waiting
   for registration, a runtime slot, or provider capacity.
2. **Tiers are preferences, not measured efficiency.** Record per-task latency, token use,
   retries, and completion outcome before proposing automatic tier escalation. A fast cheap
   child that needs multiple retries may be less efficient than one stronger child.
3. **Provider routing stays explicit.** The native host uses the current provider client.
   Cross-provider tiers would need a per-child provider/profile resolver with capability,
   credential, and context-limit checks; accepting arbitrary profile names here would be misleading.
4. **Reasoning effort should be capability-aware.** A future tier can bundle model plus
   reasoning effort, but only after routing validates supported effort values per model.
   The present feature deliberately selects the model only.
5. **Durable task telemetry is best-effort in parts of the manager.** Some attempt-record
   failures are intentionally swallowed. Surface degraded persistence as an event or metric
   while keeping child execution independent from the telemetry store; do not claim durable
   recovery is healthy when its backing store is failing.

Validation uses deterministic local provider/manager fixtures. No external model calls are
needed to verify routing, tier validation, cancellation, or persistence contracts.

## Verification in this worktree

- Focused delegation and native-host tests: 93 passed, including provider-request model
  selection, persisted snapshots, retries, invalid tiers, and cancelled spawning.
- Prompt/bootstrap checks: 15 passed; the default prompt remains within its existing size budget.
- Full runtime suite: 2,941 passed, 2 skipped, 1 failed. The existing bundled-skill discovery
  test still expects 119 entries and discovers 117; this change does not alter its scanner or assets.
- UI suite: 1,033 passed. Root type checks, runtime/UI build, gateway smoke checks, docs build,
  and `git diff --check` passed.
