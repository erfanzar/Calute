# Xerxes feature-gap roadmap

Planning audit refreshed: September 5, 2026. Current status includes uncommitted implementation foundations verified by source inspection. This refresh changes the plan only; it does not certify runtime behavior or production readiness.

Xerxes has more infrastructure than its interface reveals. The priority is to make background work observable, controllable, and reliable, then connect monitoring and scheduling to it. Building another scheduler or another agent manager would deepen the fragmentation.

This audit compares the current working tree with official Claude Code and OpenAI documentation. “Not found” means no complete default integration was identified in the inspected code; it does not prove that no extension can provide it. Existing dirty changes are included in the inventory. This was a source audit, not a live-provider or exhaustive end-to-end test.

## What the comparison actually means

Claude Code supports background Bash tasks and a task-management flow. Its scheduled-task documentation distinguishes session loops from more durable scheduling, and describes using event monitoring when available. These are separate lifecycles, not interchangeable forms of cron. [Interactive mode](https://code.claude.com/docs/en/interactive-mode), [scheduled tasks](https://code.claude.com/docs/en/scheduled-tasks).

Claude's Monitor tool can turn background process output into events that the agent handles while other work continues. Availability depends on the provider and configuration. Its LSP capability also requires an installed/configured language server. [Tools reference](https://code.claude.com/docs/en/tools-reference).

OpenAI's app scheduling supports recurring work, run review, and local project execution. Local-file tasks require the computer to be on and the app running. Its managed worktrees let independent tasks work in separate checkouts and move work back for review. These are app/environment capabilities; this plan does not attribute them all to the bare Codex CLI. [Automations](https://learn.chatgpt.com/docs/automations?surface=app), [worktrees](https://learn.chatgpt.com/docs/environments/git-worktrees).

Claude also offers prompt-linked rewind and lifecycle hooks; its agent teams are explicitly experimental. Xerxes already has corresponding lower-level building blocks, so those comparisons call for integration rather than wholesale replacement. [Checkpointing](https://code.claude.com/docs/en/checkpointing), [hooks](https://code.claude.com/docs/en/hooks), [agent teams](https://code.claude.com/docs/en/agent-teams).

## Inventory and recommended priority

P0 means reliability prerequisite; P1 means the first usable release; P2 means the next capability release; P3 means later expansion. S/M/L are relative implementation sizes, not calendar commitments.

| Capability | Xerxes today | Main gap | Priority / size |
| --- | --- | --- | --- |
| Background shells | Implemented, tools and F8 terminal panel | Completion delivery, lifetime clarity, log recovery | P1 / M |
| Unified runs and attention inbox | Durable run history and workspace/session Runs overlay | Kind-specific controls, navigation and queued model delivery | P1 / M |
| Durable scheduled jobs | Daemon cron plus separate durable trigger scheduler | Reconcile execution paths, cancellation and management | P0–P1 / L |
| Event-driven monitoring | Terminal watches, durable events, bounded reactions and creation/inspection UI | Total budgets, more sources and restart reconciliation | P2 / L |
| Session loops and follow-ups | Recurring cron available | Same-session idle wakeups with expiry and budgets | P2 / M |
| PR/CI monitoring workflows | Headless tools and integration building blocks | Connected watch → diagnose → verify workflow | P2 / M |
| Isolated agent workspaces | Worktree primitives; agent isolation rejected | Managed per-agent worktree lifecycle | P2 / L |
| Swarm coordination | Agents, messages, task dependencies and leases | Clear task ownership, recovery and results presentation | P2 / M |
| Intelligence and efficiency controls | Tier mapping in current checkout | Discoverable controls and actual outcome telemetry | P1–P2 / M |
| Checkpoint/rewind workflow | Snapshot, rollback, diff primitives | Previewable timeline and clear restoration scope | P2 / M |
| Goals and completion evidence | Goal tools, driver and F10 view | Observable criteria, budgets and verified completion | P1–P2 / M |
| Context and session controls | Resume, branching and compaction | Context provenance and selective control | P2 / M |
| Hook management | Hook runners and configuration | Inspector, testing and failure visibility | P2 / S–M |
| Semantic code navigation | LSP host adapter seam; edit diagnostics | Default language-server integration | P3 / L |
| Integration health/discovery | MCP, skills, plugins and host ports | Explain what is available and why something is not | P1 / M |
| Remote continuation | SSH, daemon and channel primitives | Authenticated continuation of the same session | P3 / L |

## 1. Background shells that feel dependable

**Existing:** [backgroundCommands.ts](/Users/erfan/Documents/Projects/Xerxes-Agents/xerxes/src/tools/backgroundCommands.ts), [processTools.ts](/Users/erfan/Documents/Projects/Xerxes-Agents/xerxes/src/tools/processTools.ts), and [terminalRegistry.ts](/Users/erfan/Documents/Projects/Xerxes-Agents/xerxes/src/runtime/terminalRegistry.ts). Commands can background explicitly or after foreground patience expires; check/kill tools and an F8 panel exist. Buffers are bounded. Subprocess handles are process-local. The current checkout also persists bounded terminal output and outcomes in runHistory.ts and exposes archived output through F8. Remaining work is durable log cursors, retention, stronger process identity and model completion delivery.

**Desired workflow:** “Run the tests; keep working; tell me when they finish.” The agent receives one completion event and the user can open output or stop the process immediately.

**Implementation:**

1. Keep the existing command manager as process owner. Add durable run metadata and bounded log files with output cursors, timestamps, owner session and exit reason.
2. Publish completion/failure events into the owning session's mailbox. Reading output must not repeatedly reintroduce the entire log into model context.
3. Extend F8 with running/finished filters, command and cwd, elapsed time, output search, interrupt and kill. Offer attach/input only for a real compatible PTY; ordinary background pipes must not pretend to be interactive terminals.
4. Display lifetime explicitly: session-owned, daemon-owned, or interrupted after restart. Reconnecting a UI can rediscover a live daemon process; restarting the daemon cannot magically restore its pipes.

**Acceptance:** Test cancellation of child process trees, interleaved stdout/stderr, output overflow, client reconnect, daemon restart, and cross-session access denial. Completion must wake the owner once without an LLM polling loop. Depends on the shared run/event foundation.

## 2. A runs view and persistent attention inbox

**Existing:** runtime/runHistory.ts and ui/opentui/runOverlay.tsx now provide persisted terminal, schedule, agent and monitor results, workspace/session scope, unread filtering, inspection and revision acknowledgement. Live notification delivery exists. Kind-specific controls, approval aggregation, future scheduled work, pagination and durable model wakeups are still incomplete. `/tasks` already aliases the agent view; preserve that meaning.

**Desired workflow:** `/runs` shows “3 agents working, 1 test command finished, 1 approval needed, next scheduled run tomorrow.” Selecting an item opens its existing detailed view.

**Implementation:** Build a read model over existing owners, not a second execution engine. Give each item a kind, ID, parent, workspace, state, last meaningful activity, elapsed time and supported actions. Add running/waiting/needs-attention/finished filters and persistent unread acknowledgements. Deep-link to the session, terminal output, agent result, or schedule run. Notifications should default to failures, requested completions and required decisions; unchanged status stays quiet.

**Acceptance:** A reconnect preserves unread items; acknowledging a result does not stop the run; stale actions fail clearly; empty and populated states work at narrow and wide terminal sizes. Do not turn successful foreground tool calls into inbox spam. Depends on typed run identities and event sequencing.

## 3. One scheduling product, using the existing engines

**Existing:** [cron/scheduler.ts](/Users/erfan/Documents/Projects/Xerxes-Agents/xerxes/src/cron/scheduler.ts) executes persisted cron jobs through the daemon. `/cron` supports management and manual runs. [runtime/scheduler.ts](/Users/erfan/Documents/Projects/Xerxes-Agents/xerxes/src/runtime/scheduler.ts) and [scheduleCommand.ts](/Users/erfan/Documents/Projects/Xerxes-Agents/xerxes/src/runtime/scheduleCommand.ts) separately persist trigger/task events. A complete default automatic runner for that second store was not identified. The remote cron tool also has an injected host-port contract; its existence is not proof of default model exposure.

**Current reliability foundation:** The checkout now passes AbortSignal to the runner, retains active ownership until the actual promise settles, caps concurrency and shares admission with manual runs. Preserve these changes and verify them through the complete scheduling product. Remaining gaps include reconciling stores, delivery recovery, timezone/misfire policy and discoverable management. Do not reintroduce timeout-based lease release while work remains active.

**Implementation:**

1. Establish the cron runner as the initial execution backend and adapt durable triggers into it. Inventory both stores and define a reversible migration with stable IDs; never activate both copies of the same schedule.
2. Pass AbortSignal through runner, turn and tools. Keep the lease until work has actually stopped; cap global and per-workspace concurrency. A timeout must not authorize an overlapping retry.
3. Add explicit timezone, next-fire preview, overlap policy, missed-run policy, maximum duration, retry limit, model budget and notification destination. Handle clock changes and DST deliberately.
4. Add `/schedules` management with create/edit/pause/run-now/history. Retain existing CLI and `/cron` compatibility. Expose a structured model tool through the actual bootstrap path.
5. Show whether the daemon is online. Persisting a schedule does not mean it can execute while the machine is off. Host service installation should be a separate explicit setup action.

**Acceptance:** Fake-clock tests for DST, missed fires, restart, duplicate delivery, lease contention, cancellation and retry overlap; migration test across both stores; failed deliveries remain visible without rerunning already-completed work. This is the first reliability milestone.

## 4. Event-driven monitors

**Existing:** runtime/terminalMonitors.ts and tools/monitorTools.ts implement terminal watches with literal matching, duplicate suppression, expiry, event limits and owner-scoped notifications. The current checkout also contains durable event cursors, reactionMailbox.ts and reactionDispatcher.ts for serialized model reactions, attempt/time limits, cancellation and partial parent-turn usage accounting. Bare /monitors opens an inspector with a creation form backed by monitor.create. These are implementation foundations, not a completed reliability claim: whole-reaction token budgets, automatic pending-event reconciliation after restart, stronger executor identity, policy editing and file/WebSocket/webhook sources remain incomplete.

**Desired workflow:** “Watch the build logs and tell me if compilation breaks.” Watching should consume no model turns while nothing relevant changes.

**Implementation:** Finish recovery and budget admission around the existing durable mailbox rather than introducing another queue. Account for child-agent and auxiliary provider calls before advertising a total token cap; unknown usage must remain explicit. Reconcile persisted-but-undelivered events without retrying uncertain external effects. Provide bounded evidence retrieval when a burst exceeds the prompt excerpt. Preserve the existing start/list/stop and inspector, then add policy replacement/editing and source health. Add file changes next, followed by websocket and authenticated webhook adapters. Apply deterministic matching, debounce and deduplication before waking the model.

**Acceptance:** Thousands of irrelevant lines produce zero model invocations; duplicate events do not cause duplicate actions; reconnect resumes from a cursor or declares a gap; stopping cleans up the source; a busy conversation receives a queued event rather than a concurrent turn. Depends on runs, mailboxes and cancellation.

## 5. Session loops and bounded follow-ups

**Existing:** Cron can recur, but a first-class same-session idle follow-up flow was not identified.

**Desired workflow:** “Check back in ten minutes; stop once the deployment is healthy.” This differs from “start a new daily report task.”

**Implementation:** Add a session-targeted schedule type on the unified runner. Proposed `/loop` and follow-up tools specify interval/deadline, stop condition, expiry, maximum checks and total token budget. Wake only when the session can accept a turn; coalesce missed checks. Prefer a monitor where a real event source exists. Display next wake, checks used, last outcome and cancel control beside the session's goal.

**Acceptance:** No overlapping turns, no unlimited catch-up after sleep, visible expiration, cancellation while queued, restored schedules clearly distinguished from non-restorable live processes. Depends on items 3 and 4; fixed-interval support can ship before adaptive timing.

## 6. PR and CI watch-to-review workflows

**Existing:** Headless execution, tools and integration building blocks are available. A complete unified PR/check monitor to bounded repair-and-review flow was not found in this audit.

**Desired workflow:** “Watch this PR's checks. If they fail, diagnose the failure and prepare a fix.”

**Implementation:** Provide templates over monitors and schedules rather than a separate CI orchestrator. Track repository, PR, exact head SHA and check-run ID. Fetch failure evidence, run a bounded diagnosis in an isolated workspace, and attach test results and diff to a review item. Invalidate conclusions when the PR head changes. Keep read-only diagnosis and authorized editing as explicit modes; commit/push behavior follows the user's actual authorization.

**Acceptance:** Duplicate webhooks do not duplicate repair runs; an updated PR cancels or supersedes stale work; inaccessible logs yield a clear blocked state; flaky reruns have a cap. Offline tests use injected Git/CI ports. Depends on monitors, schedules and isolated workspaces.

## 7. Managed workspaces for parallel agents

**Existing:** Workspace/worktree primitives exist, but [agentOps.ts](/Users/erfan/Documents/Projects/Xerxes-Agents/xerxes/src/tools/claudeTools/agentOps.ts) currently rejects requested agent isolation. Manual worktree support is not equivalent to integrated per-agent isolation.

**Desired workflow:** Spawn two coding agents without both editing the same checkout, then inspect each result before combining changes.

**Implementation:** Wire the native worktree host into spawn. Resolve the requested starting state explicitly, allocate a checkout, run configured setup, and bind cwd and tool boundaries to that checkout. Record ownership on the agent and run. Present changed files and validation with an explicit apply/handoff action. Keep dirty worktrees until reviewed; cleanup must verify ownership and preserve unmerged work. Non-Git projects need an explicit supported fallback or an actionable rejection.

**Acceptance:** Conflicting edits remain isolated; cancellation during setup cleans only owned temporary resources; failed merge preserves both results; startup failures are surfaced; no agent silently falls back to the shared checkout. Depends on run identity, not the whole scheduling release.

## 8. Swarm task ownership and recovery

**Existing:** Subagents, messaging, waits and retry exist. [durableTaskRuntime.ts](/Users/erfan/Documents/Projects/Xerxes-Agents/xerxes/src/tasks/durableTaskRuntime.ts) already implements dependency checks and leases. Some pause/resume and saved replay paths are not exposed in the current UI.

**Desired workflow:** See each agent's assigned task, dependency, current blocker, result and reviewer—not repeated raw tool output.

**Implementation:** Connect existing task and lease records to agent cards. Show queued/assigned/running/blocked/review/completed states with a concrete reason. Surface lease expiry and recovery actions. Define pause carefully: defer new work, cancel current work, and suspend a process are different actions. Group an agent's evidence and artifacts into a result card; make dependencies and accepted results visible. Extend replay to durable saved trees where the persistence contract supports it.

**Acceptance:** Two agents cannot claim the same active lease; expired attempts cannot complete a newer attempt; dependency failure blocks dependents visibly; restart reconstructs the tree without inventing live agents. Use existing task semantics rather than copying experimental competitor team behavior.

## 9. Agent intelligence and measurable efficiency

**Existing:** [intelligence.ts](/Users/erfan/Documents/Projects/Xerxes-Agents/xerxes/src/agents/intelligence.ts) in the current checkout maps tiers to model behavior. Configuration is ahead of discoverable UI controls.

**Desired workflow:** Set swarm defaults to economical, balanced or maximum capability, allow per-agent overrides, and see what actually ran.

**Implementation:** Add a settings/picker surface exposing tier → actual provider/model/reasoning mapping. Validate against the available model catalog and show fallback decisions. Display inherited versus overridden settings on spawn and agent detail. Record tokens, elapsed time, retries, tool errors, tests and accepted outcome per task. Use these as transparent efficiency measures; do not invent a single “intelligence score.” Optional escalation from a cheaper model requires a bounded policy and visible reason.

**Acceptance:** Missing models produce an explained fallback or error; explicit overrides survive resume; budget limits apply across retries/escalation; absent pricing remains unknown rather than a fabricated cost. Compare policies using a fixed repository task suite before changing defaults.

## 10. Checkpoint and rewind as a usable workflow

**Existing:** [snapshots.ts](/Users/erfan/Documents/Projects/Xerxes-Agents/xerxes/src/session/snapshots.ts) and [snapshotDiff.ts](/Users/erfan/Documents/Projects/Xerxes-Agents/xerxes/src/session/snapshotDiff.ts) provide substantial infrastructure. `/snapshot`, `/snapshots` and rollback exist; automatic pre-turn capture is opt-in. A UI rollback-diff path explicitly reports that snapshot diffs are not exposed.

**Implementation:** Add a turn-linked timeline with changed files and diff preview. Separate restoring files, branching conversation and combined restoration so the user knows the scope. Offer selected-file restore, pre-restore backup and recovery from that backup. Detect concurrent edits since the checkpoint instead of silently replacing another agent's work. Show excluded files and capture failures.

**Acceptance:** Selected-file restore leaves unrelated edits intact; failure preserves the original work; conversation branching does not imply filesystem rollback; opted-out snapshots are never advertised as available. Depends on existing snapshot APIs and workspace ownership.

## 11. Goals with completion evidence

**Existing:** Goal tools, round driver, budgets and F10 inspector exist. This is an exposure and evidence problem, not a missing goal engine.

**Implementation:** Present objective, acceptance criteria, current milestone, blockers, remaining budget and next continuation in one view. Link each completed criterion to tests, files or an explicit user decision. Keep agent claims distinct from checked evidence. Allow editing criteria without losing history, with revision checks against stale updates. Align goal continuation with the same queued session-wakeup rules used by loops.

**Acceptance:** A tool failure cannot silently complete a criterion; stale revisions are rejected; exceeding budget stops further rounds with the incomplete state preserved. Design populated, blocked, completed and budget-exhausted states before implementation.

## 12. Context controls users can understand

**Existing:** Resume, branch/fork, compaction and retrieval are present.

**Implementation:** Add a context inspector showing instructions, conversation, retrieved memory and tool evidence with approximate token contribution and provenance. Let users pin useful facts, exclude stale retrieved material and branch from a selected turn. Show what compaction preserved and removed. Keep mandatory policy instructions outside optional exclusion controls. Make oversized tool output an artifact with targeted excerpts rather than a recurring context burden.

**Acceptance:** Branching preserves the source session; excluded retrieval does not silently return on the next turn; compaction retains pending tool/approval state; estimates are labelled and missing provider usage stays unknown. Depends on session persistence and context assembly, not new model features.

## 13. Hooks you can inspect and debug

**Existing:** [shellHooks.ts](/Users/erfan/Documents/Projects/Xerxes-Agents/xerxes/src/extensions/shellHooks.ts), hook runners and bootstrap loading already exist, including workspace trust handling.

**Implementation:** Add `/hooks` inventory showing lifecycle event, source configuration, enabled/trusted state and last result. Provide schema validation, a preview/dry-run mode where meaningful, bounded output, timeouts and a recent-failures view. Clearly distinguish observational hooks from hooks that can block an operation. Do not reorder security checks to make hook integration easier.

**Acceptance:** Malformed configuration points to the exact source; hook timeout cannot hang a turn; disabled hooks do not execute; untrusted workspace configuration cannot silently become trusted. Independent smaller release after capability inventory.

## 14. Default semantic navigation and diagnostics

**Existing:** LSP tools accept an injected host adapter; that is not a bundled working language-server connection. [editDiagnostics.ts](/Users/erfan/Documents/Projects/Xerxes-Agents/xerxes/src/runtime/editDiagnostics.ts) already provides checker-based diagnostics.

**Implementation:** Add a Bun-native JSON-RPC stdio language-server host with per-workspace lifecycle, configured server commands and capability discovery. Implement document versions, cancellation, definitions, references, hover and symbols before broader refactoring actions. Feed version-matched diagnostics into the existing edit-diagnostics presentation. Missing server binaries should yield installation/configuration guidance, not a fictional successful fallback.

**Acceptance:** Fake-server tests cover crashes, malformed responses, stale diagnostics, cancellation and root scoping. Real-server checks remain opt-in. Keep checker diagnostics usable when LSP is unavailable. Larger independent capability; avoid blocking automation work on it.

## 15. Capability discovery and integration health

**Existing:** MCP, skills, plugins and host-injected tools are present. A tool definition alone does not guarantee the current session can execute it.

**Implementation:** Generate a capability inventory from actual bootstrap registration and connected hosts. Show enabled, unavailable, denied and disconnected states with a reason and configuration source. Add health/reconnect controls and validate configuration before persisting it. Existing UI messages explicitly leave skill inspection/install/search, plugin enable/disable and tool configuration unavailable; expose these through supported native handlers with source provenance, transactional updates and rollback. Do not advertise installation just because listing works. Make help/completion reflect supported capabilities while preserving documented native-command fallback. Track model-catalog freshness separately from provider connectivity.

**Acceptance:** A host-dependent tool is never listed as ready with no host; reconnect does not duplicate registration; settings errors preserve the old working configuration; provider/catalog failure remains visible. This should ship early because it prevents misleading feature discovery across the whole product.

## 16. Continue the same session remotely

**Existing:** [remote/ssh.ts](/Users/erfan/Documents/Projects/Xerxes-Agents/xerxes/src/remote/ssh.ts), daemon interfaces and channels are useful building blocks. They do not by themselves establish a phone/browser continuation experience for the same live session.

**Implementation:** Define an authenticated attach protocol with session identity, event cursor, reconnect and a single control owner. Start with a remote terminal client over an explicit connection. Route approvals to an identified active controller and reconcile stale clients. Add a browser/mobile client only after attach semantics work. Public tunneling or cloud hosting is separate infrastructure with explicit setup, not an implied consequence of enabling remote access.

**Acceptance:** Reconnecting loses no acknowledged events; an old client cannot answer a new approval; revoked clients disconnect; remote access never changes workspace/tool policy. Depends on event cursors, ownership and compatibility negotiation. Defer behind reliable local background work.

## Shared implementation contract

Use small adapters around the current managers. A shared run record needs identity, kind, owner session/workspace, parent ID, attempt number, timestamps, terminal outcome, budget, evidence links and capability-specific actions. A durable mailbox needs ordered event IDs, acknowledgement cursors, deduplication and bounded retention. Keep execution ownership with the existing process, agent or scheduler component.

Do not promise exactly-once external side effects. Use at-least-once delivery with idempotency keys and explicit uncertainty after crashes. Record queued, running, waiting, cancelling, succeeded, failed and interrupted states distinctly; cancellation is complete only after the owner confirms it.

Preserve the v35 public contract and old persisted records. Add capabilities through an explicitly reviewed compatible extension/negotiation strategy; do not quietly change wire formats. New slash commands need daemon contracts, TUI completion/restoration and documentation together.

## Delivery sequence and proof required

1. **Foundation:** capability audit; preserve and harden cron cancellation/admission and existing run adapters; implement the durable event mailbox. Prove ownership, cancellation and restart behavior before new automation features.
2. **First product release:** background command completion, `/runs` and attention inbox, schedule management and migration, integration health. A user can start work, leave the chat, return and understand every outcome.
3. **Reactive work:** event monitors, session follow-ups and PR/CI templates. Prove unchanged sources cause no model work and noisy sources stay bounded.
4. **Parallel execution:** managed worktrees, swarm task presentation, intelligence controls and outcome telemetry. Prove isolation and safe result integration.
5. **Workflow depth:** checkpoint timeline, context/goal evidence, hook inspector; then LSP and remote continuation as independent larger projects.

For every release, test deterministic offline ports first, then daemon-to-TUI flows with populated fixtures. Cover keyboard navigation, scroll, cancellation, overlays restoring drafts/transcripts, and both narrow terminals and the user's wide display. Capture actual rendered states at realistic resolution; do not accept a small empty-screen smoke test as proof of usable agents, goals or runs. Live provider/CI calls remain explicit opt-in tests. Run the repository's Bun gates for cross-cutting implementation changes.

The recommended first slice is **background runs + completion inbox + reliable cron execution**. It makes existing features useful immediately and supplies the ownership/event machinery needed for monitoring and future autonomous work.

## 17. Model-visible provider inventory and routing guidance

User-requested extension: expose the configured model choices to the agent before
it delegates work. Reuse the provider profile store, existing model discovery and
reasoning-option validation rather than maintaining a second hard-coded catalog.

Implementation:

1. Add a read-only `list_available_models` native tool with provider/model filters
   and bounded pagination. Return configured provider profiles, stable spawn
   identifiers, discovery status and observation time. Distinguish configuration
   from verified access; a discovery failure must not imply an empty inventory.
2. For each model, include supported reasoning levels, context window and maximum
   output where known, plus capability/source provenance. Preserve unknown values.
   Validate the chosen provider/model/reasoning combination again at spawn time.
3. Let users edit provider/model routing notes through settings. Return those
   notes as preferences, alongside separately enforced restrictions and budgets.
   Without notes, the agent chooses according to task complexity and capabilities.
   Discovery never changes the main conversation model or writes settings.
4. Keep context capacity, user-controlled task budgets and provider subscription
   usage separate. Quota adapters should report only available authoritative
   measurements, with units, scope, observation time, reset time and availability
   reason. Never translate usage percentages, dollars or request limits into an
   invented remaining-token balance. Unknown quota is explicit; local measured
   consumption does not stand in for the account's remaining allowance.
5. Exercise discovery failures, unavailable quota, stale catalogs, invalid spawn
   selections, configured custom endpoints, user notes and bounded tool output
   with deterministic fixtures. Add opt-in live checks only for implemented host
   adapters. Render settings at wide and narrow terminal sizes and run the root
   gate before declaring this feature complete.

Discovery and routing-note settings are implemented and offline-verified; see the implementation status log for exact evidence. Authoritative quota adapters, stricter spawn selection validation, broader host wiring, and live acceptance remain incomplete.
