# Feature roadmap implementation evidence

Scope: all 16 areas in [the roadmap](/Users/erfan/Documents/Projects/Xerxes-Agents/docs/feature-gap-roadmap.md). This tracker is not a completion declaration.

## In progress: scheduling reliability foundation

- Scheduler runners now receive AbortSignal. Timeout aborts the runner and retains the active job identity until the underlying promise settles, preventing overlapping automatic retries of that identity.
- Stop cancels active runners even when invoked through manual tick rather than a polling interval.
- Scheduled daemon turns receive cancellation through admission and provider execution; goal continuation checks cancellation between rounds.
- Added focused tests for retained ownership after timeout, retry after actual settlement, and stop cancellation. These pass alongside existing cron hardening tests.
- Manual `/cron run` now shares scheduler ownership and admission with automatic runs. Both respect global and per-project concurrency limits (four each by default), retain slots during unresolved cancellation, and defer automatic jobs without consuming their fire time.
- Daemon shutdown retains the cross-process cron lease until actual scheduled work settles. A process that never settles keeps the lease until process death allows stale-lease recovery.
- Explicit host-cancellation tests cover provider cancellation, rejected-submit ownership, and pre-cancelled admission. Added tests cover limits, deferred jobs, manual lease enforcement and idle draining.
- Package checks (runtime, UI, desktop and generated catalogs) passed. The broader daemon/cron suite passed 151 tests, including two new daemon-level timeout and shutdown-lease integration regressions. Logs: `/tmp/xerxes-cron-wiring-tests.log` and `/tmp/xerxes-cron-wiring-check.log`.
- Integration tests exposed and fixed host cancellation bypassing child cancellation/goal disarming, and a startup-order bug that skipped the initial due-job tick. The cron command test now uses its own lease and archive rather than depending on the user's running daemon.

Still required in this foundation: mailbox adapters and durable delivery/retry semantics separate from execution retries. Current onComplete failures are still logged without durable redelivery.

## In progress: durable run history

- Added SQLite RunHistory with owner-scoped inspect/list, terminal outcomes, bounded output tails, persistent unread revisions and stale-ack rejection. Dead-process records recover as interrupted rather than fictional running processes.
- Production daemon bootstraps history; cron turns record success, failure and cancellation. Provider error notifications now prevent a scheduled turn from being recorded as successful.
- Added optional v35 `run.list`, `run.inspect`, `run.acknowledge` methods and native `/runs` list/unread/inspect/ack command with TUI registry/help wiring. These are the initial text surfaces, not the planned complete dashboard.
- Focused store tests cover reopen, owner scoping, acknowledgement and interrupted recovery. A daemon RPC/slash test covers metadata-only listings and scoped acknowledgements. Broader tests are running in `/tmp/xerxes-history-integration-tests.log`; checks in `/tmp/xerxes-history-package-check.log`.
- Terminal adapter now records background, foreground and PTY runs, batches live output checkpoints and persists final tails, original terminal kind and exit code. F8 lists/inspects archived rows without exposing live process controls. Successful foreground runs do not create unread items.
- Owner-scoped completion notifications are wired to local and WebSocket transports. Acknowledgement is explicit; notification delivery does not clear unread history. These events do not yet wake a model turn.
- Store/terminal/bundle tests passed (17 tests); broader daemon/store/terminal suite passed 112 tests before the final notification integration test, which also passed separately. Package checks passed. Full root gate is running as exec session 55793, output `/tmp/xerxes-roadmap-full-gate.log`.
- Remaining: agent adapter, mailbox delivery, dashboard, cross-session review, log cursors/retention, process-identity strengthening and actual rendered UI verification.

## Full gate finding

The earlier root gate passed checks then stopped in runtime tests: 2,950 passed, two skipped, one failed. UI tests/build were not reached. The two flagged inline credential examples in bundled skills were revised: Jina requires an explicitly configured authenticated integration, and Airtable's status-check instruction refers to its existing request pattern without repeating the flagged inline example. The scanner was unchanged. All bundled skill asset tests now pass; full gate rerun is pending.

The rerun completed successfully before dashboard work: checks, 2,957 runtime tests (two skipped), 1,034 UI tests, and build passed. Evidence: `/tmp/xerxes-roadmap-full-gate.log`, completed exec session 55793.

## In progress: runs dashboard

Bare `/runs` now opens a dedicated list/detail overlay. Selection survives list refresh, unread filtering and explicit revision acknowledgement are wired, live detail refresh does not reset scrolling, errors remain visible, and Escape closes the overlay. User-opened runs survive turn completion. Native text subcommands remain supported.

Six focused render/interaction tests pass at 220×65, 110×35, 60×24 and 40×18, covering populated output, acknowledgement, close, error state and persistence across turn completion. Narrow-render failures were fixed by compacting hints and duplicate metadata. These are real OpenTUI character-frame tests, not a desktop screenshot review. The full gate for this new dashboard is running; log `/tmp/xerxes-runs-dashboard-gate.log`.

Still outstanding: cross-session grouping, run-kind navigation/control, agent and monitor sources, model mailbox wakeups, stronger process identity/log retention, and the remaining roadmap areas. This is not a production-readiness declaration.

## Remaining roadmap requirements

1. Background shells: durable metadata/log cursors, completion events, F8 controls and lifecycle presentation.
2. Unified runs: read model, persistent attention inbox, acknowledgements and detail navigation.
3. Scheduling: reconcile stores, migration, management UI/tool wiring, timezone/misfire/overlap/budget policies.
4. Monitoring: sources, bounded deterministic filtering, event delivery and UI.
5. Session loops: idle wakeups, expiry, budgets, continuation and cancellation.
6. PR/CI: versioned monitoring, bounded diagnosis/repair and review evidence.
7. Workspaces: managed per-agent isolation, startup and result handoff/cleanup.
8. Swarms: task/lease integration, recovery, durable replay and result cards.
9. Intelligence: settings and model mapping UI, telemetry and bounded escalation.
10. Checkpoints: timeline, preview, scoped restoration and concurrent-edit handling.
11. Goals: criteria/evidence presentation, revision and continuation integration.
12. Context: provenance, selective controls, compaction transparency and branching.
13. Hooks: inventory, validation, diagnostics and bounded execution visibility.
14. LSP: real configured host, semantic operations, lifecycle and diagnostics integration.
15. Capabilities: actual bootstrap inventory, unavailable reasons and health controls.
16. Remote: authenticated same-session attach, cursors, control ownership and reconnection.

Final completion requires each roadmap acceptance criterion, real daemon/TUI integration verification, realistic populated terminal rendering and full Bun repository gates. No live-provider validation or production-readiness claim has been made.

## Agent adapter and workspace review

The production subagent host now records each real execution attempt, final
output, failure or cancellation into the parent's run history. Retrying creates
a new run ID and keeps prior results. `/runs` defaults to workspace scope with
W toggling session scope; matching RPC inspection/acknowledgement resolve the
workspace on the server rather than trusting a requested path.

Current evidence: 161 focused daemon/subagent/store tests passed, six dashboard
render/interaction tests passed, package checks and diff checks passed. Tests
cover separate retries, parent isolation, cancellation and workspace-bound
inspection/acknowledgement including rejection of a supplied unrelated path.
Logs: `/tmp/xerxes-agent-history-integration.log`,
`/tmp/xerxes-agent-runs-ui.log`, `/tmp/xerxes-agent-history-check.log`.

The dashboard full gate completed before these additions: 2,957 runtime tests,
1,040 UI tests, checks and build passed (`/tmp/xerxes-runs-dashboard-gate.log`,
exec session 6442). This is not evidence for a full gate after the agent adapter.

Next foundation work is event-driven monitors and bounded model mailbox
wakeups; also still needed are run-kind controls, durable log cursors/retention,
stronger process identity, schedule reconciliation and the remaining roadmap.

## Terminal monitoring foundation

Added a live terminal observer API and TerminalMonitors with literal matching,
line framing, duplicate suppression, TTL, per-watch event limits and per-session
and host admission limits. Watches retain bounded events in run history and
publish matching lines to attached owner sessions through the daemon. The
production tool host registers monitor_terminal/list_monitors/stop_monitor;
native `/monitors` lists/stops watches. Source exit, session eviction and daemon
shutdown detach observers without killing source commands.

Verification: 106 monitor/daemon tests passed, including tool ownership,
split-line matching, deduplication, expiry, source-exit flush and daemon
notification/stop integration. Package checks passed before the final readable
output adjustment and are being rerun. Logs: `/tmp/xerxes-monitor-daemon-tests.log`,
`/tmp/xerxes-monitor-final-unit.log`, `/tmp/xerxes-monitor-check.log`.

This does not yet implement idle-model wakeups, durable mailbox delivery,
file/WebSocket/webhook sources, or a dedicated monitor configuration view.
Further hardening is required for very long unterminated lines crossing buffer
limits, persistence failures, and detailed source-health/reconnect reporting.
The complete roadmap remains active.

## Monitor hardening follow-up

Terminal matching now scans bounded segments, retains the first matching
excerpt until the logical line ends, labels omitted context, and stops parsing
bursts at the configured event limit. Oversized lines no longer discard early
matches or matches split across writes. Storage failures detach the observer,
expose failed health and an error, attempt a durable failed outcome, and report
an unavailable store without blocking cleanup of other watches. The source
process remains alive. Completed failed watches share the retention bound.

Verification: 17 focused terminal/monitor tests passed; daemon contract suite
passed; runtime package check passed. Logs:
`/tmp/xerxes-monitor-hardening-check.log` and
`/tmp/xerxes-monitor-hardening-daemon.log`. These are scoped checks, not a new
full repository production-readiness gate. Durable mailbox/model wakeups and
the rest of the roadmap remain outstanding.

## Durable monitor evidence and immediate attention

Monitor matches now commit ordered per-run events and the display tail in one
SQLite transaction before notification delivery. Identical retries are
idempotent; conflicting sequence reuse and gaps fail. Running watches become
unread as matches arrive, and revision acknowledgement cannot hide a newer
match. `run.events` exposes bounded cursor pagination with the same authenticated
session/workspace scoping as inspection. This survives reconnect and storage
reopen, including evidence older than the short display tail.

Focused evidence/store/monitor tests passed (13 tests), runtime package check
passed. Daemon reconnect/cursor/ownership coverage and full gates are being
checked separately. Durable model wakeup claims, cancellation generations,
reaction budgets and outcomes remain unfinished; evidence persistence alone
must not be advertised as automatic model continuation.

## Durable reaction admission and executor contract

Added ReactionMailbox with immutable bounded policies, durable offered/consumed
cursors, burst coalescing, an atomic single-claim constraint per session, attempt
limits, deadlines and explicit outcomes. Cancellation revokes authority without
releasing an unresolved executor. Claims survive reopening; expired claims stay
fenced until the host has evidence that their executor stopped. Settlement is
transactional and rejects conflicting terminal outcomes.

Added an event-triggered ReactionDispatcher with an injected session-admission
port, cancellation before launch, deadline AbortSignal, and ownership retained
until the actual executor settles. Six tests cover durable/reopen/concurrent
claims, coalescing, bounded attempts, queued cancellation, an executor ignoring
abort until cleanup, and provider failure. These components are not yet connected
to production monitor tools or the daemon turn executor. Before enabling model
reactions, wire bounded user-configured authority, background-origin controls,
shared turn admission, evidence prompts, accounting and user-visible outcomes.
Do not expose these components as a working automatic reaction feature yet.

The preceding durable-event full root check/test/build gate completed with exit
0 (session 16265, `/tmp/xerxes-durable-events-full-gate.log`). It predates the
reaction components; their focused checks are tracked separately in
`/tmp/xerxes-reaction-check.log`. The full roadmap remains incomplete.

## Background turn authority

Added internal human/schedule/monitor origin controls through daemon runtime
submission and the real AgentTurnRunner. Background turns clear prior goal-round
identity and do not receive direct-human goal authority. The runtime rejects a
background origin combined with a goal-round claim. Scheduled execution now sets
schedule origin, including manual /cron run, and the server does not automatically
continue an unrelated session goal after a background turn.

Evidence: 40 real-loop AgentTurnRunner tests passed, runtime package checks
passed, the daemon/goal/composition integration suite passed, and a focused
socket test verified /cron run forwards schedule origin. Logs:
`/tmp/xerxes-origin-tests.log`, `/tmp/xerxes-origin-check.log`,
`/tmp/xerxes-origin-integration.log`, `/tmp/xerxes-cron-origin-test.log`.
Monitor reaction tools, daemon executor wiring and budget accounting still need
implementation before automatic reactions can be exposed to users.

## Daemon reaction executor adapter

The daemon now owns a ReactionDispatcher backed by the production mailbox file.
Its executor acquires the same session-operation queue before claiming work,
then uses the tracked turn path without reacquiring that lock. Reactions carry
monitor origin, bounded evidence marked as untrusted output, and readable
transcript labels. Reaction output and failure/cancellation are recorded as Runs.
User turn cancellation revokes session reaction authority; daemon shutdown aborts
and drains active reaction work using the tracked shutdown path. Events arriving
during a reaction drain through fresh admission after the current work settles.

110 daemon/mailbox/dispatcher tests and package checks passed. A socket test
configured a bounded policy, emitted real terminal evidence, observed a monitor
origin turn and its persisted answer, and verified the reaction limit. A separate
backlog test checks events offered during execution. Logs:
`/tmp/xerxes-reaction-wiring-integration.log`,
`/tmp/xerxes-reaction-wiring-check.log`, `/tmp/xerxes-reaction-backlog-test.log`.

Remaining: expose policy creation and per-watch cancellation via monitor tools
and UI, implement full token accounting and recovery of pending offers/claims,
and strengthen reaction health reporting. The existing monitor tools still
create notification-only watches; production policy creation is not yet exposed.
No full-goal completion or production-readiness claim is made.

## Reactive monitor tool controls and cancellation

Production monitor tools can now opt into bounded automatic reactions with
react=true, max_reactions (1–10) and reaction_timeout_seconds (1–120). Direct
user-turn authority is required; notification-only stays the default. Policy
configuration is persisted when creating the watch and is exposed in its summary.
The shared mailbox is passed through production construction to monitors and
the daemon dispatcher. Cancellation subscriptions revoke a matching active
reaction immediately; stopping another watch does not abort that executor.
stop_monitor and /monitors stop revoke the watched run's queued authority without
killing the source. Session eviction revokes all owner policies.

117 combined integration tests and package checks passed, plus a focused real
monitor/tool/dispatcher cancellation test. Logs:
`/tmp/xerxes-react-tool-integration.log`, `/tmp/xerxes-react-tool-check.log`,
`/tmp/xerxes-react-stop-test.log`. A full gate is being run separately.
Remaining: dedicated configuration UI, total-token accounting, policy/claim
health inspection, restart reconciliation and remaining roadmap features.
Reaction-count and time limits must not be advertised as a token/spending cap.

## Reaction health in tools, native commands and Runs

Added transactional policy/claim inspection distinguishing waiting, queued,
running, cancelling, awaiting-cleanup, cancelled, expired and exhausted. Counts,
pending evidence, active identity, last outcome/error and expiry are returned
without consuming or admitting work. Monitor summaries expose health to tools;
/monitors displays state and attempt limits. run.inspect exposes authorized
reaction_health, and Runs renders it with errors. Detail refresh remains active
for unresolved reactions even when the source watch has completed.

13 mailbox/monitor tests, seven UI render tests, a daemon socket health assertion,
and package checks passed. Logs: /tmp/xerxes-reaction-health-tests.log,
/tmp/xerxes-health-ui.log, /tmp/xerxes-health-daemon.log,
/tmp/xerxes-health-check.log. The prior reactive-controls full gate completed
successfully (session 82810, /tmp/xerxes-react-controls-full-gate.log); that gate
predates this health UI change. Token budgets, restart reconciliation, dedicated
configuration and remaining roadmap features are still incomplete.

## Recovery of exited reaction executors

New reaction claims record the executor PID with an additive SQLite migration.
On opening the production mailbox, claims whose recorded process is confirmed
gone become interrupted with an explicit uncertain-effects error. Every existing
reaction policy for that owner is revoked, including queued grants from other
watches; no attempts are refunded and no evidence is silently retried. This
recovers durable health, not the external operation or its child processes.

Live/reused PIDs, unreadable process state, invalid recorded PIDs and legacy
claims lacking executor identity remain fenced. Their deadline alone does not
prove cancellation. Stronger executor identity and deliberate recovery controls
are still needed for those cases. Pending unclaimed events are not yet restored
into a loaded session automatically.

Focused recovery/dispatcher tests and package checks passed; daemon integration
checks are logged at /tmp/xerxes-recovery-integration.log. Tests include dead
executor interruption and owner-wide revocation, live/reused PID fencing, stale
completion rejection, and old-schema claims with no identity. The complete
roadmap and production-readiness audit remain open.

## Reaction usage accounting groundwork

Reaction claims now store observed parent-turn input/output tokens and an
explicit completeness flag. Settlement charges the claim once; repeated
settlement cannot double-count usage. Historical and failed claims lacking
usage remain incomplete rather than appearing free. The daemon captures token
counter deltas under session admission and requires an explicit complete-usage
report from the runner. Tools/native health and Runs inspection expose totals;
Runs labels these parent-turn tokens and shows incomplete usage.

This is not yet total-budget enforcement. Child-agent and auxiliary-call usage,
provider-request admission and conservative handling of unknown totals remain
required before advertising a whole-reaction token/spending cap. Tests cover
idempotent accounting, rejected invalid counters, missing usage, daemon
propagation and rendered partial totals. Logs: /tmp/xerxes-usage-integration.log,
/tmp/xerxes-usage-daemon.log, /tmp/xerxes-usage-ui.log,
/tmp/xerxes-usage-check.log. The overall roadmap remains incomplete.

## Dedicated monitor inspector

Bare /monitors now opens a list/detail OpenTUI inspector with selected-watch
stop, refresh, evidence scrolling, expiry, reaction state and errors. It adapts
to a stacked layout on narrow terminals, preserves selection on refresh, keeps
action errors visible and survives turn completion. Explicit text subcommands
retain their original behavior. New monitor.list/inspect/stop RPCs use trusted
session ownership; list responses omit event bodies and detail returns the
latest 20 plus an omitted count. Source commands remain independent of stop.

Six real OpenTUI render/interaction tests passed at 220x65, 110x35, 60x24 and
40x18. The daemon contract test verifies list/inspect and cross-session stop
rejection. Package checks and diff checks passed. Logs:
/tmp/xerxes-monitor-view-tests.log, /tmp/xerxes-monitor-view-check.log,
/tmp/xerxes-monitor-rpc-test.log. Configuration creation/editing UI, total-token
budgets, recovery improvements and remaining roadmap features remain open.

## Monitor creation controls

The monitor inspector now opens a creation form with N. It lists live sources
owned by the session, accepts a literal match and expiry, and exposes an explicit
notification-only or automatic-reaction choice with attempt and timeout limits.
The additive monitor.create RPC validates settings and source ownership. Failed
submission preserves entered settings for correction and retry. Protocol and
configuration documentation describe the controls and clarify that limits are
not a total token or spending cap.

Ten OpenTUI monitor interaction tests passed, including creation at 150x40 and
40x18, no-source rejection, reaction selection, failed submission and retry.
Evidence: /tmp/xerxes-create-final-ui.log. The full root gate started in session
47841; its outcome is not yet certified here. Remaining roadmap work, including
schedule unification and total reaction budgets, is still open.

The monitor-creation full root gate completed successfully (session 47841):
check passed; runtime 2987 passed, 2 platform-specific skips, 0 failed; UI 1051
passed across 110 files; build exited successfully. Build output is recorded in
/tmp/xerxes-monitor-create-build.log. This validates the current worktree gate,
not live-provider behavior or completion of the remaining roadmap.

## Schedule-store integrity prerequisite

Cron no longer treats unreadable or malformed persisted jobs as an empty store.
Reads and mutations reject invalid JSON, invalid records and duplicate IDs,
leaving original bytes intact. Initialization only creates an absent file with
exclusive creation; other filesystem errors propagate. Updates validate the
result before writing, and an already-open store cannot silently recreate a
file removed underneath it. Valid legacy records remain readable.

Forty focused cron/store/scheduler tests passed in
/tmp/xerxes-cron-integrity.log; an additional missing-store boundary test passed
in /tmp/xerxes-cron-integrity-boundary.log. This removes a data-loss path before
schedule-management expansion. Cross-process writer serialization, migration of
the trigger store, timezone/misfire controls and the management UI remain open.

## Schedule management RPC foundation

Added workspace-scoped schedule.list/inspect/pause/resume/cancel/run over the
existing cron store and runner. Responses expose execution state and persisted
metadata; foreign-workspace jobs are rejected. Pause controls future admission,
whereas cancellation aborts only the active run and retains ownership until
actual cleanup settles. Manual execution retains existing lease, concurrency,
archive and delivery behavior. Fixed the no-session project fallback to honor
the configured daemon project before process cwd.

127 daemon/cron/integrity tests and runtime typecheck passed. Logs:
/tmp/xerxes-schedule-api-integration.log and /tmp/xerxes-schedule-api-check.log.
Protocol extension is documented. This is the API foundation; management UI,
creation/editing, trigger-store reconciliation and durable cross-process write
serialization remain pending. No claim of completed scheduling product yet.

## Schedule inspector wiring

Bare /schedules now opens the workspace schedule inspector with prompt,
execution state, UTC cadence, next/last run, and stored errors. P pauses/resumes,
G runs, X cancels, R refreshes and arrows select. Details scroll; narrow screens
stack the list. A pending manual run does not disable cancellation. Overlay
policy preserves chat/draft and keeps user-opened inspection through turn
completion. Native /schedules arguments alias existing cron management.

Five OpenTUI render/interaction tests passed at 220x65, 110x35, 60x24 and 40x18,
including cancel during a pending run and overlay restoration. Two daemon
contract tests passed for the slash alias and workspace API. Logs:
/tmp/xerxes-schedules-ui.log and /tmp/xerxes-schedules-slash.log. Root checks
passed; full test/build started as session 82800 with logs
/tmp/xerxes-schedules-test.log and /tmp/xerxes-schedules-build.log.
Creation/editing forms, schedule history navigation and reconciliation of the
separate durable trigger store remain incomplete.

## Schedule create/edit API

Added schedule.create and schedule.update with validated prompts, UTC cron or
explicit-timezone future one-shot times, paused state and workspace ownership.
Inspection returns an opaque persisted-record revision; stale edits fail and
active local executions cannot be edited. Updates preserve existing delivery
configuration and job identity. Tests cover successful create/edit, stale
updates, invalid cron, timezone-less dates and one-shot persistence. Runtime
check and daemon/store tests passed; logs are
/tmp/xerxes-schedule-edit-integration.log and
/tmp/xerxes-schedule-edit-boundaries.log. The UI creation/editing form remains
next, and cross-process revision/write coordination is still incomplete.

The preceding schedule-inspector full test/build gate (session 82800) completed
successfully, with 1056 UI tests across 111 files and verified runtime/UI builds.
Its logs are /tmp/xerxes-schedules-test.log and /tmp/xerxes-schedules-build.log;
it predates the create/edit API changes documented above.

## Schedule creation/editing form

Wired N create and E edit into /schedules. The form supports prompt, recurring
UTC cron versus one-shot timestamp, and paused/enabled state. New jobs default
to paused. Editing snapshots the revision; rejected saves keep the form and
cannot silently overwrite a concurrently changed record. Submission prevents
duplicate requests while pending. Documentation describes the workflow.

Eight form/panel tests passed, including creation at 150x40 and 40x18 and stale
edit rejection with retained input. UI typecheck passed. Logs:
/tmp/xerxes-schedule-form-tests.log and /tmp/xerxes-schedule-form-check.log.
Trigger migration, cross-process coordination, timezone scheduling beyond UTC,
policy controls and history navigation remain open; the roadmap is incomplete.
Full check/test/build for the schedule form is running as session 6819. Logs:
/tmp/xerxes-schedule-form-root-check.log,
/tmp/xerxes-schedule-form-root-test.log,
/tmp/xerxes-schedule-form-build.log. Completion has not yet been claimed.

## Cross-process cron writer coordination

All JobStore add/update/remove mutations now hold an immediate SQLite writer
transaction while reading and atomically replacing the JSON file. The sidecar
is a persistent coordination file, not a second job store. Contention fails
without waiting on the event loop; OS process exit releases locks without
PID-based stale-lock deletion. Expected revisions are rechecked under the lock.
Legacy executables bypassing this protocol must be stopped before concurrent
management. Schedule migration and execution-policy expansion remain open.

A real child-process test verifies lock contention, unchanged jobs after a
rejected write, and successful writes after SIGKILL of the lock holder. Another
test verifies stale-revision rejection across store instances. Runtime typecheck
passed. Combined daemon/cron verification is running in the current turn.
The prior schedule-form gate completed successfully (session 6819), including
1059 UI tests and verified builds; it predates this writer-lock change.
Combined verification completed: 144 daemon/cron/store tests passed, no failures,
in /tmp/xerxes-cron-writer-integration.log. Runtime typecheck passed in
/tmp/xerxes-cron-writer-check.log; diff whitespace checks are clean.

## Schedule run-history navigation

H in /schedules opens the Runs inspector filtered to the selected schedule.
Existing output/error inspection, acknowledgements and scrolling remain usable;
Escape returns to the schedule list. run.list now supports source/kind filters
inside the authorized scope, applied before the result limit. A test places 120
newer unrelated runs ahead of a schedule and confirms its history is retained.

Six history-store tests, 13 Runs/schedule UI tests, and the daemon filtering
contract test passed. Package checks passed. Logs:
/tmp/xerxes-schedule-history-test.log, /tmp/xerxes-schedule-history-ui.log,
/tmp/xerxes-schedule-history-rpc.log. Full root gate is running as session 58507
with /tmp/xerxes-history-full-{check,test,build}.log. Pagination beyond 100 runs,
trigger migration and remaining roadmap areas are not complete.

## Failed schedule-output delivery

Automatic schedule completion now records delivery failures separately from
prompt execution. A successful one-shot whose delivery fails remains paused
and inspectable rather than being removed; recurring jobs keep their cadence.
A typed delivery error carries the already-written output archive path. Missing
senders for a real channel fail explicitly. The schedule inspector shows stored
delivery failure and archive details. A later successful delivery clears the
current failure state. Automatic delivery retry/outbox retention and the crash
window between execution and persistence still require further work.

Focused tests cover one-shot non-reexecution, durable failure metadata, archived
output after sender failure, and missing-sender rejection. Logs:
/tmp/xerxes-delivery-failures.log and /tmp/xerxes-delivery-ui.log. The preceding
history full root gate completed successfully as session 58507; it predates
these delivery changes.

## Durable channel-delivery outbox foundation

Channel outputs now enter a SQLite outbox after archival and before sending.
Entries retain job identity, destination, payload, archive path, attempt count
and pending/sending/sent/uncertain state. Sending claims are transactional;
concurrent sends fail, sent receipts are idempotent, and sender errors remain
uncertain rather than retrying possibly completed external effects. Reopening
never resets an unresolved sending claim. The scheduler retains the delivery
ID with its failure metadata. Pending payloads survive archive pruning.

The outbox caps unresolved entries at 128, payloads at 1 MB, and sent receipts
at 100. It fails explicitly on capacity or lock contention. No live channel
messages were sent for validation. Thirty-four offline cron/outbox tests and
runtime typecheck passed. Logs: /tmp/xerxes-outbox-tests.log and
/tmp/xerxes-outbox-check.log. Delivery reconciliation/retry UI and source-run
crash recovery remain open, as do the other roadmap requirements.

## Delivery inspection and reconciliation API

Added workspace-scoped schedule delivery list/inspect/send/resolve RPCs. Lists
omit payloads, inspection returns retained output, and pending sends reuse the
native channel sender without invoking the model. Missing channel configuration
preserves pending state. Explicit uncertain-outcome reconciliation requires the
observed attempt count and cannot reset active sending claims. Retry decisions
only prepare pending state; sending remains a separate operation. Reconciled
sent receipts share bounded retention.

Offline tests cover payload scoping, stale reconciliation, concurrent-send
exclusion and pending preservation with no channel manager. Logs:
/tmp/xerxes-outbox-api-tests.log and /tmp/xerxes-outbox-api-contract.log.
UI controls, crashed sending-claim reconciliation and the remaining roadmap
still need implementation; no production-complete claim is made.

## Delivery panel

D in /schedules opens retained deliveries with destination, payload, attempt
count and state. S sends pending entries without a model turn. A/T reconcile
uncertain entries with an explicit confirmation; allowing retry does not send.
Active sending claims have no reset action. Errors remain visible after refresh
and Escape returns to the schedule inspector. Three new interaction tests at
150x40 and 40x18 cover state-aware controls, confirmation, no implicit send and
visible failure. Existing schedule tests also passed. Log:
/tmp/xerxes-delivery-panel-tests.log. Full-roadmap completion remains unproven.
The full root gate for delivery changes is running as session 49163, with logs
/tmp/xerxes-delivery-full-{check,test,build}.log. Its outcome is pending.

## Delivery executor recovery

Added an additive executor-PID/claim-ID migration to the outbox. Confirmed-dead
sending executors become uncertain on reopen, preserving their payload and
requiring explicit destination reconciliation. Live/reused PIDs and legacy
claims without identity remain fenced. Completion is conditional on the exact
claim ID, preventing an old sender from settling a newer retry. No automatic
resend or successful-delivery claim is inferred from process exit.

Six outbox/delivery tests and runtime typecheck passed. Tests cover dead-owner
recovery, live-owner fencing and late completion after retry admission. Logs:
/tmp/xerxes-outbox-recovery-tests.log and /tmp/xerxes-outbox-recovery-check.log.
The earlier full gate, session 49163, was confirmed still running this turn;
its completion is not yet certified. Stronger process identity, legacy/manual
recovery and remaining roadmap requirements remain open.

## Per-schedule execution limits

Added persisted per-job timeout and one-shot retry overrides, daemon RPC fields,
form controls and inspector labels. Old records inherit daemon defaults. Form
saves make displayed defaults explicit. Zero retries pauses a failed one-shot
after its first attempt; recurring failures retain cadence. Timeout still
holds admission until cleanup actually settles. This does not implement token
budgets or arbitrary retry policy.

Focused cron persistence/timeout tests, nine schedule UI tests, daemon schedule
contracts and package checks passed. Logs: /tmp/xerxes-schedule-limits-tests.log,
/tmp/xerxes-schedule-limits-ui.log, /tmp/xerxes-schedule-limits-daemon.log and
/tmp/xerxes-schedule-limits-check.log. The earlier delivery full gate completed
successfully (session 49163, 1063 UI tests); it predates these limit changes.

## Interval execution on the shared scheduler

Added first-class interval timing to CronJob, its runner, create/update/resume
RPCs, and schedule forms/inspection. Intervals persist with 1–86400 second
validation, cannot mix with cron or one-shot timing, and coalesce missed ticks
rather than creating catch-up bursts. Explicit mode conversion clears the old
interval; edits retain existing interval mode. Dispatch precision remains bound
by polling. This is an execution prerequisite for migration, not migration of
the legacy trigger store itself.

33 runtime tests, ten UI tests, daemon scheduling contracts and package checks
passed. Logs: /tmp/xerxes-interval-tests.log, /tmp/xerxes-interval-ui.log,
/tmp/xerxes-interval-daemon.log, /tmp/xerxes-interval-check.log. Remaining roadmap
work includes migration, scheduling timezones, goal/context/worktree features
and complete end-to-end verification.

## Legacy trigger-store mutation integrity

Legacy trigger create/enable/disable/remove now serialize with delivery and
mark-fired operations. All mutations hold a process-released SQLite writer
lock around the JSONL read/append path, with immediate contention errors rather
than event-loop blocking. This protects a future migration snapshot from
configuration writes bypassing delivery serialization. Explicit markFired
occurrence timestamps are preserved instead of being overwritten on append.

Tests cover concurrent configuration/delivery operations, supplied occurrence
times, and contention with a real child process followed by recovery after its
exit. Runtime typecheck passed. Evidence:
/tmp/xerxes-trigger-lock-tests.log and /tmp/xerxes-trigger-lock-check.log.
The preceding interval full root gate (session 33663) completed successfully.
Migration itself and the remaining roadmap are still incomplete.

## Time-trigger migration core

Added a migration handoff that durably records a destination and fences the
legacy source before creating its paused CronJob. The retained source cannot
be re-enabled, removed, or replaced through current APIs. Repeating migration
uses the same stable destination ID and does not overwrite an already imported
job's edits. Failed transfer remains fenced and can retry only the recorded
destination. Interval and compatible cron timing are converted; event/webhook
sources, dependency workflows and incompatible combined day constraints are
rejected rather than silently changing their meaning.

Eleven scheduler/migration tests and runtime typecheck passed. Logs:
/tmp/xerxes-migration-tests.log and /tmp/xerxes-migration-check.log. Operator
entry points, migration preview/rollback and unsupported cadence/source handling
remain unfinished. The complete roadmap is still active.

## Legacy migration operator entry points

`/schedules legacy` now previews convertible triggers, rejection reasons, and
recorded destinations. `/schedules migrate <trigger-id>` imports into the current
workspace as a paused job. `/cron` retains the same native aliases. Command help,
configuration documentation and protocol documentation describe the handoff.

Focused validation: 12 scheduler/migration tests passed, including unsupported
conversion without source mutation. The daemon schedule RPC test also passed
with preview/import assertions. The full check/test/build gate is being run in
this worktree; its result is recorded below when complete. Rollback, unsupported
source conversion, and remaining roadmap features are still unfinished.

Migration entry-point full gate completed successfully: root `bun run check`,
`bun run test`, `bun run build`, and `git diff --check`. Logs are
/tmp/xerxes-migration-full-check.log, /tmp/xerxes-migration-full-test.log, and
/tmp/xerxes-migration-full-build.log. This validates the current implementation;
it does not complete the remaining roadmap or substitute for live UI/provider
verification.

## Model-facing schedule management

The default daemon tool registry now exposes list_schedules and manage_schedule
through its normal tool discovery. Both use the same project schedule handler
as the UI/RPC, with trusted runtime session identity, optimistic edit revisions,
and typed timing limits. Model run-now uses the same scheduler admission and
isolated cron execution, result archive and delivery path. Abort requests cancel
without releasing active ownership early. Direct-user gating and the existing
persistent scheduling approval policy prevent background self-scheduling.

Focused tests cover identity, pre-dispatch cancellation, approval policy,
background activation rejection, workspace isolation, and stale edits. The full
root gate is being run; this addition does not complete the wider roadmap.

The root check/test/build gate completed successfully for the schedule tool
implementation, with 1064 UI tests passing. A subsequent focused daemon test
also verified model run-now isolation, overlapping-run rejection and abort
propagation with no successful archive. Evidence:
/tmp/xerxes-schedule-tools-full-check.log,
/tmp/xerxes-schedule-tools-full-test.log,
/tmp/xerxes-schedule-tools-full-build.log,
/tmp/xerxes-schedule-model-run-tests.log. `git diff --check` passed.
No live provider or native terminal visual verification was performed here.

## Scheduled workspace execution boundary

Found and repaired a routing bug: schedule turns previously opened sessions
without passing the job's project, so they could inherit the daemon's default
workspace. Execution now validates the stored directory, opens in that project,
and refuses to relocate an existing conversation. The runtime checks the
preserve-project condition inside its serialized session-open operation, closing
the concurrent opener race as well as the direct conflict path.

Focused tests exercise a daemon in project A running a job in project B,
conflicting session rejection without provider calls or conversation mutation,
and concurrent session opens. Full root validation is running for this change.

The follow-up tool audit found that shared file/process resolvers also retained
the daemon startup root. Core tools and the shared PTY manager now accept a
trusted active-workspace resolver; the production daemon binds it to the async
local turn session. Concurrent file reads and real Bun commands in two project
roots pass, including cross-project escape rejection. The full root gate was
restarted after these changes. Other specialized integration paths still need
the broader roadmap audit; this is not an all-tools isolation certification.

The final workspace-routing root gate passed: check, test, build, and
`git diff --check`. Evidence: /tmp/xerxes-workspace-routing-full-check.log,
/tmp/xerxes-workspace-routing-full-test.log,
/tmp/xerxes-workspace-routing-full-build.log. Focused evidence is in
/tmp/xerxes-schedule-project-tests.log and
/tmp/xerxes-workspace-routing-tests.log. No live-provider or native UI claim is
made; remaining roadmap work stays active.

## Workspace-specific shell hooks

Replaced the daemon's single startup-project shell hook runner with bounded
per-workspace runners. Agent turns, session start/end and overflow compaction
resolve the active session's configuration and command cwd. User configuration
remains shared; project configuration is still behind the explicit trust opt-in.
The opt-in parser now matches complete accepted words rather than substrings.
Static injected HookRunner support remains available for embedding hosts.

Eleven shell/workspace hook tests passed, exercising concurrent configurations,
actual shell cwd, disabled project hooks and lifecycle selection. Both static and
session-selected AgentTurnRunner hook tests passed. Full root validation is in
progress. This does not finish the planned hook inspector, testing UI or broader
integration audit.

Follow-up inspection found an unresolved hook timeout issue: the native executor
sends SIGTERM once and still awaits process exit plus both pipe readers. A hook
that ignores termination or leaves inherited pipes open can outlive its deadline.
Timeout escalation and bounded pipe shutdown remain required reliability work.

Workspace-hook routing root check/test/build and `git diff --check` passed.
Evidence: /tmp/xerxes-workspace-hooks-full-check.log,
/tmp/xerxes-workspace-hooks-full-test.log,
/tmp/xerxes-workspace-hooks-full-build.log. Focused logs:
/tmp/xerxes-workspace-hooks-tests.log and
/tmp/xerxes-workspace-hooks-turn-tests.log. Wider roadmap completion remains
unproven and the goal remains active.

## Bounded native hook deadlines

Repaired the previously noted timeout hang. The native hook executor now races
execution and output collection against a deadline, signals a dedicated POSIX
process group, escalates after 200 ms, and cancels pipe readers. Deadline expiry
is an error even if the shell reports success during cleanup; permission hooks
therefore fail closed. Windows bounds the caller and kills the direct process;
full descendant cleanup there is not claimed.

Thirteen shell/workspace hook tests passed, including an actual TERM-ignoring
shell (confirmed gone after escalation) and a zero-exit shell whose background
child retains stdout. Existing successful hooks and mutation hooks remain
covered. Runtime typecheck passed. Full root gate is running for the change.

Hook deadline root check/test/build and `git diff --check` passed. Evidence:
/tmp/xerxes-hook-timeout-full-check.log,
/tmp/xerxes-hook-timeout-full-test.log,
/tmp/xerxes-hook-timeout-full-build.log. Focused deadline evidence:
/tmp/xerxes-hook-timeout-tests.log. The broader roadmap remains incomplete.

## Monitor evidence reconciliation on session load

Added owner-scoped eligible-policy enumeration and durable event cursor lookup.
The dispatcher reconciles saved evidence before acquiring a normal bounded
reaction claim. Initialize/resume and session-open paths trigger reconciliation;
repeated attachment does not replay consumed events. Active uncertain claims,
revoked/expired/exhausted policies remain fenced. Execution separately verifies
that evidence belongs to the currently loaded workspace.

Twenty-one dispatcher/mailbox/history tests and a real daemon resume test passed.
The resume test reopens both SQLite stores after saving an event without offering
it, confirms one reaction, and resumes again without duplication. Full gate is
running. Automatic loading of all saved owners at daemon startup remains outside
this completed slice; broader monitor recovery and budget requirements remain.

Integration validation exposed two scope issues while adding recovery: monitor
sources record command subdirectories, so evidence scope now permits contained
subdirectories; and session.open omitted the configured daemon project fallback
when no active session existed. Both are corrected. Live monitor reaction and
restart recovery integration tests pass together. Earlier full gate attempts
failed on the live monitor test and are superseded only after a fresh gate.

Fresh final monitor-recovery root check/test/build passed after both scope fixes,
as did `git diff --check`. Authoritative logs:
/tmp/xerxes-monitor-recovery-final-check.log,
/tmp/xerxes-monitor-recovery-final-test.log,
/tmp/xerxes-monitor-recovery-final-build.log. Integration evidence:
/tmp/xerxes-monitor-recovery-integration-tests.log. Earlier failed gate logs do
not describe the final worktree. The wider roadmap remains active.

## Runs history pagination

Removed the UI's latest-100-only dead end with exclusive timestamp/id cursors
through RunHistory, run.list and the Runs/schedule-history overlay. Database
filtering precedes paging; equal timestamps are ordered by ID. The RPC reports
has_more and validates cursor pairs. N/P navigate older/previous pages; scope or
unread changes reset the page while polling retains its boundary.

Store tests cover 205 same-timestamp rows, a new arrival between pages and an
invalid cursor. Eight Runs UI tests pass, including older-page navigation and
scope reset. An initial full check caught an incorrect test-clock constructor;
the fixture was corrected and focused tests passed again. A fresh full gate is
running. This does not complete the remaining Runs controls or wider roadmap.

Runs pagination final root check/test/build and `git diff --check` passed.
Evidence: /tmp/xerxes-run-pages-full-check.log,
/tmp/xerxes-run-pages-full-test.log, /tmp/xerxes-run-pages-full-build.log.
Focused evidence: /tmp/xerxes-run-pages-tests.log and
/tmp/xerxes-run-pages-ui-tests.log. The broader goal remains active.

## Runs kind and status filters

Added combined store/RPC status filtering and kind/status controls in Runs.
Filtering precedes pagination, so old failures remain discoverable beneath more
than 100 newer results. Active filters are shown; K/S cycle kind/status and reset
page cursors. Schedule history preserves its schedule kind. Unknown RPC status
values fail explicitly.

Eight history tests, nine Runs UI tests and the scoped RPC integration test
passed. Coverage includes an old failed agent beneath 120 newer records, owner
isolation, and actual filter parameters sent from the panel. Full gate is
running; remaining Runs actions and other roadmap areas are incomplete.

Final Runs-filter root check/test/build and `git diff --check` passed. Evidence:
/tmp/xerxes-run-filters-full-check.log,
/tmp/xerxes-run-filters-full-test.log,
/tmp/xerxes-run-filters-full-build.log. Focused logs:
/tmp/xerxes-run-filters-tests.log, /tmp/xerxes-run-filters-ui-tests.log and
/tmp/xerxes-run-filters-rpc-tests.log. This does not complete the wider goal.

## Live Runs cancellation controls

Added daemon-advertised cancellation labels and revision-checked run.cancel.
The panel shows X/click controls only when available. Terminal and watch actions
use their live owner controls; monitor reaction cancellation revokes its watch
policy. Schedules use an active execution-ID map, so an older running-looking
history row cannot cancel a newer execution of the same job. A request remains
distinct from completed cleanup. Agent-kind controls are not yet exposed here.

Terminal scope/stale-request tests and an exact schedule execution cancellation
test passed; ten Runs UI tests passed including action/error presentation. Full
gate is running. Live iTerm inspection was attempted through Computer Use and
rejected because com.googlecode.iterm2 is not allowed by that tool's safety
restriction. No native UI verification is claimed and no bypass was attempted.

Runs actions final root check/test/build and `git diff --check` passed. Evidence:
/tmp/xerxes-run-actions-full-check.log,
/tmp/xerxes-run-actions-full-test.log,
/tmp/xerxes-run-actions-full-build.log. Focused logs:
/tmp/xerxes-run-actions-tests.log and /tmp/xerxes-run-actions-ui-tests.log.
The wider roadmap and live native verification remain incomplete.

## Strict one-shot timing validation

Added shared calendar-aware parsing for native slash and schedule RPC/model/form
inputs. JavaScript Date normalization could previously turn impossible dates
into a different execution day. Calendar days, leap years, clock fields and
offsets are now checked before parsing; slash entry also matches the RPC's
explicit-timezone and future-time requirements.

Thirteen parser cases and schedule RPC/slash integration tests passed, including
no store mutation for impossible or past inputs. The full root gate is running.
Timezone-aware recurring cron and the remaining roadmap are still incomplete.

One-shot validation final root check/test/build and `git diff --check` passed.
Evidence: /tmp/xerxes-schedule-time-full-check.log,
/tmp/xerxes-schedule-time-full-test.log,
/tmp/xerxes-schedule-time-full-build.log. Focused evidence:
/tmp/xerxes-schedule-time-tests.log and /tmp/xerxes-schedule-time-rpc-tests.log.
The overall goal remains active and incomplete.

## Recurring cron timezones

Added persisted IANA timezones with UTC defaults for legacy jobs. Native slash
creation, schedule RPC/model tools, editing, resume, and scheduler recurrence
use the same zone. The schedule form exposes the field and the inspector shows
it. Omitted update fields preserve the existing zone; invalid values fail
before persistence. Interval and one-shot timing keep their previous semantics.

The local-calendar calculation skips spring gaps and supports both occurrences
of fall repetitions, including half-hour DST shifts. Tests also cover fractional
UTC offsets, local date constraints, and a skipped date-line calendar day.
A persisted scheduler test executes both fall-back occurrences and verifies the
next day's instant. RPC tests cover preservation, invalid-zone rejection and
native slash creation. UI tests cover saved zones and keyboard field editing.

Focused tests passed; the final root gate is running. Timezone rules depend on
Bun's ICU data; no claim is made of exhaustive historical timezone validation.
The broader roadmap and native terminal verification remain incomplete.

Timezone final root `check`, `test`, `build`, and `git diff --check` passed.
Evidence: /tmp/xerxes-zone-full-check.log, /tmp/xerxes-zone-full-test.log,
/tmp/xerxes-zone-full-build.log. The UI suite passed 1068 tests across 113 files.
Focused logs: /tmp/xerxes-zone-focused.log, /tmp/xerxes-zone-ui.log,
/tmp/xerxes-zone-native.log. Overall goal remains active and incomplete.

## Scheduled execution receipts and conservative recovery

Scheduled occurrences now persist intent before invoking the model and record
completion before delivery. A recovered unfinished receipt pauses the schedule
instead of replaying it. Completion bookkeeping failures no longer enter model
retry handling; even if recording the recovery warning fails, the earlier
receipt fences a later tick/restart. Recurring advancement failures likewise
retain the receipt. Ordinary reported execution failures retain retry policy.

The Schedules panel explains recovery and possible duplicate effects. Explicit
resume acknowledges the receipt, retains it as previous execution metadata,
and clears the fence. Resume is rejected for a running/cancelling local job.
Recurring resume chooses a future occurrence; one-shot resume can repeat work.

Fault-injection tests cover failed completion receipt writes, failed one-shot
removal, failed recurring advancement, reopened stores/schedulers and normal
execution retry behavior. RPC tests cover receipt acknowledgment; UI tests cover
the warning and resume action. Focused tests passed; the full gate is running.
This does not claim exactly-once external effects, fsync durability, or fencing
of explicit manual execution. The broader roadmap remains incomplete.

Execution-receipt final root check/test/build and `git diff --check` passed.
Evidence: /tmp/xerxes-receipt-full-check.log,
/tmp/xerxes-receipt-full-test.log, /tmp/xerxes-receipt-full-build.log.
Focused fault/retry tests: /tmp/xerxes-receipt-focused.log; UI recovery action:
/tmp/xerxes-receipt-ui.log. The overall goal remains active and incomplete.

## Monitor usage across failures and children

Monitor reaction errors now carry measured usage through dispatcher settlement,
so failure/cancellation does not discard observed tokens. The daemon adds
cumulative counters for children started during the reaction, deduplicated by
agent/task, and excludes preexisting background children. The bounded collector
rejects invalid counters and keeps arithmetic within safe integer bounds.
Runs labels the result measured reaction tokens.

Children do not yet certify complete provider usage, so their totals remain
explicitly incomplete. This improves accounting but does not complete the
roadmap's whole-reaction hard budgets or auxiliary provider accounting.
Focused tests cover duplicate child events, invalid counters, preexisting
agents, failed/cancelled settlement, and the daemon's real monitor-to-Runs path
with and without children. Full gate is running.

Monitor-usage final root check/test/build and `git diff --check` passed.
Evidence: /tmp/xerxes-reaction-usage-full-check.log,
/tmp/xerxes-reaction-usage-full-test.log,
/tmp/xerxes-reaction-usage-full-build.log. Focused evidence:
/tmp/xerxes-reaction-usage-focused.log,
/tmp/xerxes-reaction-usage-final-focused.log and
/tmp/xerxes-reaction-usage-rpc.log. Goal remains active and incomplete.

## Reaction queue draining after per-watch interruption

Revalidated the dispatcher and found that a single controller covered the whole
owner queue. One reaction's timeout/cancellation aborted that controller and
stranded other offered evidence until another event arrived. Provider rejection
also exited before draining unrelated queued work.

Each claim now has its own cancellation controller combined with the owner's
stop signal. Per-watch revocation and deadlines cancel only that claim; session
cancellation/shutdown stop all admission. After actual executor cleanup and
persisted settlement, dispatch reacquires session admission and drains eligible
queued work. Failed claim evidence stays consumed, failure health is persisted,
and the dispatcher reports the first executor failure after draining. Settlement
storage failures still escape immediately, preserving uncertain ownership.

Focused tests pass for cancellation, deadlines, queued unrelated watches,
provider failure, usage preservation and recovery. Full root gate is running.
The broader plan remains incomplete.

Reaction-drain final root check/test/build and `git diff --check` passed.
Evidence: /tmp/xerxes-reaction-drain-full-check.log,
/tmp/xerxes-reaction-drain-full-test.log,
/tmp/xerxes-reaction-drain-full-build.log. Focused evidence:
/tmp/xerxes-reaction-drain-focused.log. Overall goal remains incomplete.

## Loaded hook inspection

Added `/hooks [list]` through the daemon and TUI command registry. It inspects
the active session's actual cached shell-hook runner, exposing loaded config
paths, trust state, load timestamp/errors and event/command/matcher/timeout/
blocking behavior. Unsupported runners report unavailable. Inspection does not
execute hooks, trust a workspace, or reload edited configuration.

Focused tests cover cached-versus-edited configuration, malformed workspace
config source paths, untrusted config exclusion, no command side effects and
native slash response. Full gate is running. Hook last-result history and a
controlled test mode remain outstanding; the broader goal remains incomplete.

Hook inspection root check/test/build and `git diff --check` passed.
Evidence: /tmp/xerxes-hooks-full-check.log, /tmp/xerxes-hooks-full-test.log,
/tmp/xerxes-hooks-full-build.log. Focused evidence: /tmp/xerxes-hooks-focused.log,
/tmp/xerxes-hooks-rpc.log, /tmp/xerxes-hooks-ui.log. Hook history/test mode and
remaining roadmap requirements are incomplete.

## TUI agent mode configuration

Added `/config` and `/config agents` for Light/Balanced/Smart provider profile,
model ID and reasoning effort. Settings persist with optimistic revisions in
an atomic SQLite store, take precedence over legacy daemon tier mappings, and
refresh default daemon tool registration on save. Profiles are listed without
credentials. Unconfigured tiers are removed from advertised tool schemas,
including nested swarm entries; server validation still rejects invalid calls.

Structured tier settings now travel into child descriptors, native run config,
provider transport, reasoning requests and retry snapshots. A selected provider
uses its own profile capability limits rather than the parent's. Existing model
string settings remain supported. The UI edits model IDs/profile names as text;
a catalog-backed picker and live-provider/native-terminal verification remain
outstanding. Default daemon integration is wired; standalone CLI/ACP parity for
the new settings store/provider selection still needs completion.

Focused configuration, persistence, native child-routing, RPC and UI tests pass.
The final full gate is running. The broader production-readiness goal remains
incomplete.

Agent configuration root check/test/build passed: 3069 runtime tests and 1072
UI tests in the full run. Final transport-limit adjustments passed root check
and the native transport test; the expanded settings UI test file passed all
three tests. Final source was rebuilt and `git diff --check` passed.
Evidence: /tmp/xerxes-agent-config-full-check.log,
/tmp/xerxes-agent-config-full-test.log, /tmp/xerxes-agent-config-full-build.log,
/tmp/xerxes-agent-config-final-check.log,
/tmp/xerxes-agent-provider-final.log,
/tmp/xerxes-agent-settings-ui-final.log,
/tmp/xerxes-agent-config-final-build.log.
Settings apply to subsequent turns; already-running turns retain their registry
and existing children keep captured configuration. Native graphical terminal
verification and the wider roadmap remain incomplete.

## Saved agent settings across CLI entry points

Revalidated all native CLI host constructors. One-shot and ACP previously
ignored the new settings store and lacked the provider resolver. Both now load
the same persisted settings as the daemon and use the shared agentProvider
resolver. Embedded hosts remain explicit injected-port consumers.

A real Bun one-shot subprocess test uses isolated local parent/child HTTP
endpoints. A model-issued smart-tier spawn reaches the selected child endpoint
with child credentials, configured model and high reasoning effort, then the
parent finishes. No external provider calls are made. Resolver tests also check
profile-specific context/output limits and missing-profile failures. Focused
tests passed. The full repository check/test/build gate and `git diff --check`
passed: 3071 runtime tests passed and 1073 UI tests passed. The ACP subprocess
test additionally verifies that its advertised intelligence choices load the
saved tier mapping. ACP child execution has not been independently exercised
by the new routing fixture; that fixture exercises the one-shot entry point.
Evidence: /tmp/xerxes-agent-parity-full-check.log,
/tmp/xerxes-agent-parity-full-test.log,
/tmp/xerxes-agent-parity-full-build.log. The final ACP assertion passed its
focused Bun test after the full gate.

## Agent configuration choices

The `/config` editor now supports Up/Down selection of saved provider profiles
and supported reasoning efforts, while retaining text entry. The additive
`agent.settings.options` RPC resolves the selected profile/model through the
same catalog and provider fallback used by save validation. It does not mutate
the parent profile or expose credentials. Stale choice responses are ignored
after changing the selected mode, provider, model, or closing the overlay.

An actual renderer keyboard test selects a provider and effort and checks the
saved payload. RPC tests cover available efforts, missing profiles and secret
exclusion. Root check/test/build and `git diff --check` passed; evidence is in
/tmp/xerxes-mode-choices-full-check.log,
/tmp/xerxes-mode-choices-full-test.log and
/tmp/xerxes-mode-choices-full-build.log. Model IDs still use text entry, and
native terminal/live-provider verification remains outstanding.

## Agent model discovery in configuration

The model field now uses the existing `fetch_models` daemon route for the
selected saved profile, or the active profile when inherited. Up/Down selects
discovered IDs; F5 refreshes. Only a bounded slice of the catalog is rendered.
Custom IDs are preserved, discovery does not select a model automatically, and
warnings/errors remain visible. Requests finishing after leaving the field or
changing profiles cannot replace its current choices.

Renderer tests exercise provider/model/effort selection through saving and
discovery failure followed by refresh without overwriting a custom ID. Root
check/test/build and `git diff --check` passed: 3071 runtime tests passed,
2 skipped, and 1075 UI tests passed. Evidence:
/tmp/xerxes-config-model-full-check.log,
/tmp/xerxes-config-model-full-test.log,
/tmp/xerxes-config-model-full-build.log. This verifies the offline integration;
native terminal and live-provider verification remain outstanding, as do the
other incomplete roadmap requirements.

## Provider-specific reasoning parity

Agent settings choices and save validation now share the main reasoning
picker's resolver, including authenticated Codex catalog discovery and its
offline fallback. The resolver accepts the selected child profile explicitly;
cached reasoning levels are keyed by profile name, endpoint and model instead
of model alone. Catalog discovery has an injected host port for deterministic
tests. Settings capability reads run outside the connection mutation queue.

An offline socket test gives two Codex profiles different ladders for the same
model, verifies choices and save/rejection behavior, and holds discovery pending
while proving a settings read can complete. Root check/test/build and
`git diff --check` passed: 3072 runtime tests passed, 2 skipped, 1075 UI tests
passed. Evidence: /tmp/xerxes-tier-reasoning-full-check.log,
/tmp/xerxes-tier-reasoning-full-test.log,
/tmp/xerxes-tier-reasoning-full-build.log. The expanded socket test also passed
separately in /tmp/xerxes-tier-reasoning-tests.log. Live catalog and native
terminal checks remain unverified; the full roadmap is not complete.

## Default agent mode and disabling tiers

The configuration editor now exposes the existing default policy: F2 cycles
inherit and configured modes. F4 disables the selected tier in the draft and
resets a matching default to inherit. Enter persists; Escape discards unsaved
changes. Clearing the default model manually produces an actionable validation
message. Reasoning discovery now runs only when the effort field is selected,
avoiding provider lookups for every character typed in other fields.

Renderer tests verify default selection, disabling the default, no premature
save, and no effort discovery while editing other fields. All seven focused
configuration tests passed. Root check/test/build and `git diff --check` passed;
evidence: /tmp/xerxes-mode-default-full-check.log,
/tmp/xerxes-mode-default-full-test.log,
/tmp/xerxes-mode-default-full-build.log. Native terminal verification and the
remaining roadmap requirements are still incomplete.

## Child configuration through recovery

The recovered reset/send path previously retained the model but dropped its
provider profile and reasoning effort. It now carries both into the replacement
child. Empty input is validated before consuming the restart permission; failed
spawn restores that permission. Successful spawn removes it and the old
tombstone. Retry wire responses now expose both optional configuration fields.

A focused native-host test serializes and restores a child snapshot, rejects an
empty continuation, then verifies that the next valid continuation uses the
saved child transport and effort without invoking the parent transport. It also
checks the retry response fields. Root check/test/build and `git diff --check`
passed; evidence: /tmp/xerxes-recovery-tier-check.log,
/tmp/xerxes-recovery-tier-test.log, /tmp/xerxes-recovery-tier-build.log,
/tmp/xerxes-recovery-tier-focused.log. This is deterministic recovery-path
coverage, not a live-provider or process-crash certification. The wider roadmap
remains incomplete.

## Assigned child settings in the inspector

Explicit provider profile and reasoning effort now flow from native manager
events through the daemon gateway adapter and UI progress state. Restored
snapshot rows retain the same fields, and partial live updates preserve them.
The agent list includes assigned configuration; the expanded inspector gives
profile and effort a wrapping line so compact-row truncation does not hide them.
Missing settings are omitted instead of inferred.

Native transport/event, gateway adapter, snapshot reconciliation and renderer
tests cover this path. All 68 focused UI tests passed. Root check/test/build and
`git diff --check` passed: 3073 runtime tests passed, 2 skipped, 1079 UI tests
passed. Evidence: /tmp/xerxes-agent-identity-full-check.log,
/tmp/xerxes-agent-identity-full-test.log,
/tmp/xerxes-agent-identity-full-build.log,
/tmp/xerxes-agent-identity-runtime.log and /tmp/xerxes-agent-identity-ui.log.
Native terminal verification and the broader roadmap remain incomplete.

## Incomplete child outcomes

Native child turns now treat explicit non-success stop reasons as failures,
including exhausted output/tool/objective guards and unconfigured-tool loops.
Previously only provider/context failures were handled, allowing partial text
from other stopped turns to mark an agent run successful. Failed run history now
retains partial output alongside the error for inspection.

Regression tests exercise repeated output truncation and unconfigured tool
requests through the native host, asserting failed handles, failed run history,
bounded provider calls and preserved partial output. The full suite passed:
3075 runtime tests, 2 skipped, and 1079 UI tests. Final root check/build and
`git diff --check` also passed after the last source adjustment. Evidence:
/tmp/xerxes-child-outcomes-focused.log,
/tmp/xerxes-child-outcomes-test.log,
/tmp/xerxes-child-outcomes-final-check.log,
/tmp/xerxes-child-outcomes-final-build.log. Remaining roadmap requirements are
still incomplete.

## Scheduled turn outcome propagation

The native turn adapter now includes `stop_reason` in its terminal status
update. Scheduled turns reject explicit non-success stop reasons before
success delivery, retaining partial output and the reason in failed run history.
Previously only error notifications were considered, so output/tool guard
exhaustion could be delivered as success. Legacy injected runners without a
stop reason retain their existing behavior.

Tests cover the real adapter's output-limit event and schedule execution with
output-limit, tool-budget and provider failures, asserting no success archive
or success timestamp. The first full run exposed an unrelated shutdown test
race reading a directory before creation; its bounded wait now treats ENOENT
as pending and still propagates other errors. Focused tests and the subsequent
full root check/test/build plus `git diff --check` passed. Evidence:
/tmp/xerxes-schedule-outcomes-focused.log,
/tmp/xerxes-schedule-outcomes-race.log,
/tmp/xerxes-schedule-outcomes-check.log,
/tmp/xerxes-schedule-outcomes-test.log,
/tmp/xerxes-schedule-outcomes-build.log. The wider roadmap remains incomplete.

## Monitor reaction stop reasons

Monitor reactions now reject explicit non-success terminal stop reasons instead
of treating an absence of error notifications as success. Failed reaction runs
retain partial output; mailbox settlement preserves measured parent/child usage
with completeness false and consumes the attempt under the existing limit.

Four integration scenarios cover success and incomplete outcomes with and
without child usage. They verify failed/completed mailbox outcomes, run history,
usage accumulation and no extra execution after the one-attempt limit. Root
check/test/build and `git diff --check` passed: 3081 runtime tests passed,
2 skipped, 1079 UI tests passed. Evidence:
/tmp/xerxes-monitor-outcomes-focused.log,
/tmp/xerxes-monitor-outcomes-check.log,
/tmp/xerxes-monitor-outcomes-test.log,
/tmp/xerxes-monitor-outcomes-build.log. Full reaction budgets, other incomplete
roadmap requirements, and native terminal verification remain open.

## Recent hook execution outcomes

`/hooks` now includes recent completed/denied/failed executions, timestamp,
duration and one-based hook index within each event. Each cached workspace
runner retains at most 100 immutable results; slash output shows the latest 20.
Matcher misses do not appear as successful executions. History contains no hook
input or output, remains memory-only, and resets on restart/cache eviction.
The wrapper preserves callback results and errors, including permission denials.

Focused Bun tests exercise real shell completion, denial and failure, matcher
exclusion, secret input/output exclusion, bounded retention and snapshot
stability. The daemon inspection contract includes empty recent history without
executing hooks. Root check/test/build and `git diff --check` passed. Evidence:
/tmp/xerxes-hook-results-focused.log, /tmp/xerxes-hook-results-check.log,
/tmp/xerxes-hook-results-test.log, /tmp/xerxes-hook-results-build.log.
Hook test mode and the other incomplete roadmap requirements remain open.

## Hook failure diagnostics

Shell hook failures now carry a typed timeout/exit category while preserving
their existing error messages and fail-closed behavior. Recent execution history
records only the category and optional exit code; other thrown failures are
classified as execution errors. `/hooks` renders these details without storing
stderr, stdout or input in its history.

All 15 focused hook tests passed, including real timeout cleanup, nonzero exit
classification and privacy assertions. Root check/test/build and
`git diff --check` passed: 3082 runtime tests passed, 2 skipped, and 1079 UI tests
passed. Evidence: /tmp/xerxes-hook-diagnostics-focused.log,
/tmp/xerxes-hook-diagnostics-check.log,
/tmp/xerxes-hook-diagnostics-test.log,
/tmp/xerxes-hook-diagnostics-build.log. Hook test mode and the broader roadmap
remain incomplete.

## Runs inspection error visibility

Runs now tracks list, inspection and action failures separately. A successful
background list poll no longer erases an inspection/cancellation failure.
Successful inspection retry clears its own error. Key-release events are
ignored to avoid duplicate keyboard actions. A renderer test waits through a
real list poll, confirms the inspection failure remains, then retries and
confirms recovery. The existing filter test now waits for its response to render.

Runtime tests passed (3082 passed, 2 skipped). Final UI tests passed (1080),
followed by root check/build and `git diff --check`. One full UI attempt exposed
a session-picker deferred-preview timing failure; that file passed in isolation
and the subsequent full UI run passed without changing it. This remains a
timing-test concern rather than a claimed production fix. Evidence:
/tmp/xerxes-runs-errors-focused.log, /tmp/xerxes-runs-errors-test.log,
/tmp/xerxes-runs-errors-session-check.log,
/tmp/xerxes-runs-errors-ui-final.log,
/tmp/xerxes-runs-errors-check-final.log,
/tmp/xerxes-runs-errors-build.log. The broader roadmap remains incomplete.

## Deterministic UI cancellation and pagination checks

The session-picker deferred-preview test now waits for the actual close callback
before delivering its stale response. Previously it raced standalone Escape
decoding and could resolve the preview before the close under parallel load.
The request-start assertion also proves there is a pending preview to invalidate.
The picker implementation already invalidates that request on close; no runtime
change was necessary. Runs pagination now waits for the requested row to render
instead of assuming two flush calls imply the asynchronous response has landed.

The picker tests passed in isolation and in full UI runs. The final full UI
suite passed all 1080 tests across 115 files, with UI type checking and
`git diff --check` passing. Evidence: /tmp/xerxes-picker-close-focused.log,
/tmp/xerxes-picker-close-ui.log, /tmp/xerxes-picker-close-check.log. Runtime tests
were not rerun for these test-only changes. The broader roadmap remains open.

## Schedule timing preview

The create/edit form now requests a debounced, read-only `schedule.preview` and
shows the next eligible UTC instant, validation errors, and an explicit paused
notice. The daemon shares its timing validator with create/update; preview does
not create, modify, or execute jobs. Late responses are discarded after timing
changes or unmount. Saving recalculates the time; this is a preview, not a reserved
execution slot.

Socket tests cover offset normalization, recurring timezone calculation, invalid
cron/timezone/interval/past timestamps, conflicting timing modes, and unchanged
persisted jobs. Rendered TUI tests cover visible success, errors and stale response
suppression. Full root check, test and build passed, including 1082 UI tests;
`git diff --check` passed. Evidence: /tmp/xerxes-preview-runtime.log,
/tmp/xerxes-preview-ui.log, /tmp/xerxes-preview-check.log,
/tmp/xerxes-preview-test.log, /tmp/xerxes-preview-build.log. Native terminal
verification and the remaining roadmap requirements are still open.

## Configurable missed-run policy

Schedule persistence, daemon create/update/list, the native model tool, and the
TUI form/inspector now expose `missed_run_policy` and `misfire_grace_seconds`.
Existing records retain coalesce behavior; omitted update fields preserve saved
values. Skip advances overdue recurring jobs and pauses missed one-shots for
review, recording the missed occurrence without claiming it ran. The default
allowance is 300 seconds, configurable from 1 to 86400. Admission still forbids
overlap, and unfinished execution receipts take precedence over skip handling.
Initialization and skipped occurrences no longer set a misleading last-run time.

Deterministic scheduler tests cover restart, clock rollback, exact grace boundary,
missed one-shots, coalescing, persisted validation and recovery receipts. Socket
and rendered TUI checks cover round-trip settings and keyboard changes. Full root
check/test/build passed: 3084 runtime tests, 2 skipped, 1083 UI tests.
`git diff --check` passed. Evidence: /tmp/xerxes-misfire-focused.log,
/tmp/xerxes-misfire-contract.log, /tmp/xerxes-misfire-ui.log,
/tmp/xerxes-misfire-check.log, /tmp/xerxes-misfire-test.log,
/tmp/xerxes-misfire-build.log. Model budgets, notification destination editing,
native terminal verification and the broader roadmap remain incomplete.

## Schedule delivery destination controls

The schedule form/inspector and structured model tool now expose delivery channel
and recipient. The read-only `schedule.options` RPC lists configured adapter names
and enabled state without credentials. New external destinations validate against
registered adapters and require a bounded, single-line recipient; omitted update
fields preserve the saved destination. Archive-only selection clears the recipient.
Saving never sends a test message. Runtime execution reuses archive/output routing
and the durable delivery outbox. Short terminals keep the current field and its
neighbors visible rather than pushing the editor below all eleven fields.

An in-memory channel adapter test verifies create, omitted-field preservation,
manual execution delivery, invalid-target rejection before persistence, and turning
forwarding off. Wide/narrow keyboard tests cover channel selection, recipient entry
and clearing. The first full run had loaded the pre-fix empty-recipient validator;
a fresh final-source gate passed check/test/build with 3085 runtime tests, 2 skipped,
and 1085 UI tests. `git diff --check` passed. Evidence:
/tmp/xerxes-destination-runtime.log, /tmp/xerxes-destination-ui-final.log,
/tmp/xerxes-destination-check-final.log, /tmp/xerxes-destination-test-final.log,
/tmp/xerxes-destination-build-final.log. No external messages were sent. Model
budgets, native terminal verification, and broader roadmap work remain open.

## Chat reattachment checklist and tool-row repair

A session snapshot now carries the latest todo list from its in-flight result or
persisted TodoWriteTool execution. Activation/resume restores it after clearing
the previous chat; a new turn keeps it until an explicit replacement/empty list.
Old transcripts can be read without migration. Live and replayed tool rows share
semantic formatting; the daemon preserves command/path context before truncation,
and the UI handles already-truncated legacy arguments without printing raw JSON.

Regression evidence covers switching original/other/original with 24 messages and
3/5 completed tasks, session isolation, live and completed snapshots, explicit
clearing, gateway forwarding, legacy JSON and bounded argument prefixes. Runtime
suite passed 3089 tests (2 skipped). Final UI suite passed all 1087 tests after
updating three assertions that specified the intentionally replaced behavior.
Type checks, build and git diff --check passed. Evidence:
/tmp/xerxes-restore-runtime.log, /tmp/xerxes-restore-ui-final.log,
/tmp/xerxes-restore-inflight.log, /tmp/xerxes-restore-check-final.log,
/tmp/xerxes-restore-test-verified.log, /tmp/xerxes-restore-ui-verified.log,
/tmp/xerxes-restore-build-final.log. The user's running session was not interrupted
or modified; native terminal visual verification remains open.

Before this reported regression took priority, a schedule logical model-call cap
was started: AsyncLocalStorage admission across runTurn and completeLlm, persisted
max_model_calls, and form/tool controls. Focused tests cover concurrent admission,
auxiliary/streaming calls, cancellation and scope closure. Complete native parent/
child scheduling integration and token-budget accounting still need verification;
this does not establish a production-ready total token cap. The roadmap remains
active.

## Native schedule model-call limit verification

The persisted max_model_calls setting is now exercised through schedule creation,
reload, the real AgentTurnRunner, and an actual native child host. Two admissions
allow parent delegation and child evidence but reject the parent's final model
request; three complete the work; four also admit optional title generation.
Required exhaustion produces a failed durable run and records admitted usage.
A discovered title-generation edge now skips optional work when no call remains,
instead of marking a successful three-call run exhausted after completion.
Fixture title clients keep these integration tests fully offline. TUI coverage
verifies setting and clearing the limit at a short terminal height.

The limit counts logical model requests through runTurn/completeLlm, including
native descendants and auxiliary completions. It is per execution attempt, not
per recurring schedule lifetime, and is not a token/billing/HTTP-retry cap. Total
token accounting, remaining roadmap features, and native terminal visual
verification remain open.

Full check/test/build passed: 3092 runtime tests, 2 skipped, 1088 UI tests.
`git diff --check` passed. Evidence: /tmp/xerxes-budget-native.log,
/tmp/xerxes-budget-ui.log, /tmp/xerxes-budget-check.log,
/tmp/xerxes-budget-test.log, /tmp/xerxes-budget-build.log.

## Scheduled attempt token accounting

Schedule execution now creates a shared call scope even when the call limit is unlimited. Streaming attempts and auxiliary completions record idempotent usage receipts; native descendants inherit the scope. The latest attempt persists observed input/output tokens, measured/settled/pending call counts, and completeness beside its admission count. The schedule inspector labels partial or unavailable usage explicitly.

Verification: the root check, test and build commands passed in this worktree (3,096 runtime tests passed, 2 skipped; 1,089 UI tests passed). Focused fixtures exercise real native child execution at limits 2/3/4 and unlimited, missing title usage, cancellation, idempotent receipts, and partial usage across a failed streaming attempt followed by retry. UI rendering verifies the partial-usage label. No live provider calls or native terminal visual checks were performed.

Remaining: this is a latest-attempt snapshot, not durable per-run historical accounting, a hard token cap, or billing reconciliation. Auxiliary/child calls pending at snapshot time keep the report partial; later completion does not rewrite that snapshot. Extending the ledger into run history and monitor reactions, adding budget admission policy for unknown usage, and the remaining roadmap requirements are still outstanding.

## Durable scheduled-attempt usage history

Each scheduled attempt now persists its token-usage snapshot atomically with its run output and terminal outcome. `run.inspect` and the Runs inspector expose the snapshot; later executions cannot replace it. The additive SQLite migration leaves legacy records unknown. Validation rejects negative/non-integer counters and inconsistent completeness before changing the run. Existing owner/workspace isolation and idempotent finalization remain intact.

A native-child integration test exposed finalization happening before optional title admission. Run finalization now occurs after the shared call scope closes, so run history and schedule metadata receive the same snapshot rather than a prematurely complete report.

Verification completed: root check, test, build, and diff whitespace check pass. Runtime: 3,098 passed, 2 skipped. UI: 1,091 passed. Focused tests cover failed/cancelled outcomes, restart persistence, separate attempts, legacy migration, invalid completeness, owner isolation, repeat finalization, native children, and rendered partial/complete history labels.

Remaining: live usage checkpoints for crash recovery, hard token budgets, shared accounting for monitor reactions, and the other unfinished roadmap items. Late completions after finalization do not rewrite historical snapshots; pending calls keep completeness false. No live-provider or native-terminal visual verification was performed in this step.

## Live scheduled-usage checkpoints and crash recovery

Scheduled attempts checkpoint shared model-call admission before provider execution and measured usage after settlement. The durable run read model accepts checkpoints only for the owning session's running record. Dead-owner recovery preserves them while marking the run interrupted. Late receipts cannot rewrite a finalized snapshot. Checkpoint failures poison further admission and surface as failures, including optional-call failures near finalization; they cannot be retried around. Cleanup still releases attempt signals and completion listeners when persistence throws.

Verification completed: root check, test, build and diff whitespace check pass (3,101 runtime tests passed, 2 skipped; 1,091 UI tests passed). A Bun subprocess test kills a real owner process after one measured and one pending call, reopens SQLite, and verifies interrupted state with preserved partial usage. Native schedule fixtures verify checkpoints are present before provider execution. Focused tests also cover failed persistence refusing all subsequent provider calls and late receipts leaving durable snapshots alone.

Remaining: monitor-reaction shared accounting and budget enforcement, hard token-budget policy, and the other unfinished roadmap requirements. Checkpoints retain settled observations; they cannot recover provider usage never reported before a crash. This is not billing reconciliation or a total token cap. The user's running daemon was not restarted or interrupted.

## Shared monitor-reaction call accounting

Monitor reactions now run inside the same shared provider-call scope used by schedules. Native child and auxiliary calls contribute once, with live run-history checkpoints and a terminal usage snapshot. The mailbox aggregate receives that snapshot's observed token totals. Failed runs remain failed and incomplete even when they emitted text. Missing provider usage and optional calls pending at finalization remain partial. Event-only injected runners retain observed parent/child counters as incomplete rather than falsely asserting full call coverage.

Verification: four deterministic native fixtures exercise monitor dispatch, actual parent and child turn runners, auxiliary completions, missing child usage, an explicitly pending title call, and a provider failure after reported usage. Full root check, test, build and whitespace checks pass: 3,105 runtime tests passed, 2 skipped; 1,091 UI tests passed. No external providers were called and the user's daemon was not restarted.

Remaining: mailbox crash recovery must reconcile attempt checkpoints into reaction aggregates, total token budgets still need admission/enforcement policy, and the broader roadmap remains incomplete. Historical snapshots do not grow after finalization merely because optional work later settles.

## Reaction aggregate checkpoint recovery

The reaction mailbox now checkpoints absolute attempt usage before the run-history projection. Live counters remain incomplete until final settlement; repeated checkpoints cannot double-charge, stale/foreign/decreasing checkpoints are rejected, and absent final counters cannot erase observed usage. Executor-death recovery preserves the aggregate, marks usage incomplete, and retains the existing no-replay cancellation fence.

Verification completed: root check, test, build and whitespace checks pass (3,107 runtime tests passed, 2 skipped; 1,091 UI tests passed). A real Bun subprocess is killed after persisting mailbox usage; repeated reopen retains the aggregate and rejects automatic reaction admission. Native monitor tests verify mailbox checkpoints already exist when the provider begins its next call. Unit checks cover exact claim ownership/ranges, monotonicity, idempotent snapshots, cancellation cleanup and settlement without final counters.

Remaining: the mailbox and run read model are separate transactions, so a crash between writes can leave the detailed run snapshot behind the aggregate. No historical backfill for pre-checkpoint interrupted claims has been performed. Total token-budget enforcement and the broader unfinished roadmap remain active. No live provider was used or user daemon interrupted.

## Hook selection preview

`/hooks preview <event> [tool-name]` now shows loaded hooks in execution order, matcher inclusion/skipping, timeout, command and permission-blocking capability. It shares hook configuration's event alias resolver and reports cached sources, configuration errors and workspace trust. The TUI command help exposes the syntax; errors use the existing gateway response path without erasing history. Preview is strictly selection-only: no shell command executes, no trust changes, and no predicted permission verdict is presented.

Verification: root check, test, build and whitespace checks passed (3,108 runtime tests passed, 2 skipped; 1,092 UI tests passed). Focused tests cover aliases, ordered matches, absent events, invalid events, untrusted workspace exclusion, no execution/side effects, the daemon slash contract and TUI forwarding/error preservation. Native terminal visual inspection and live-provider tests were not performed.

Remaining roadmap work includes hook failure filtering, automation token-budget enforcement, managed workspaces, follow-ups, and the other capabilities not yet proven complete. This preview does not execute hooks in a sandbox or establish their runtime correctness.

## Event-filtered hook failures

`/hooks failures [event]` now lists recent failures and permission denials newest first, with separate counts, optional native/alias event filtering, timeout/exit/execution classification and durations. It includes configuration errors and explicitly states that the 100-execution history is in memory and can age out or reset. Empty selections do not claim a lifetime clean history. The TUI help and daemon slash response expose it without changing trust or executing hooks.

Verification completed: root check, test, build and whitespace checks pass (3,108 runtime tests passed, 2 skipped; 1,093 UI tests passed). Existing real shell-hook tests now exercise failure filtering, aliases, denied-versus-failed counts, timeout results, privacy and aging. Daemon tests cover empty filtered results without command execution; TUI tests cover forwarding and transcript preservation.

This completes the currently identified hook inventory/selection-preview/recent-failure exposure work. It does not certify the broader roadmap, provider integrations or native visual acceptance. Automation budgets, managed agent workspaces, follow-ups and other unfinished requirements remain active.

## Managed agent worktree adapter foundation

Added a Bun-native Git adapter for the existing SubagentWorktreePort. It allocates unique branches/checkouts at committed HEAD, records task ownership and baseline in the repository's Git common directory, rejects non-Git workspaces and foreign/changed identities, and preserves dirty/untracked/ignored files and commits for review. Clean checkout removal retains the branch reference. Git commands disable checkout hooks, have a deadline and bounded output, and do not change process cwd. Failed setup retains its manifest and partial resources for recovery. Removed the manager's unconditional instruction to commit; commits now require user authorization.

Verification completed: root check, test, build and whitespace checks pass (3,111 runtime tests passed, 2 skipped; 1,093 UI tests passed). Real Git fixtures prove edit isolation, unchanged parent edits, committed-HEAD starting state, reload ownership checks, dirty/ignored/committed result preservation, foreign-path rejection, and non-Git failure. A real SubAgentManager cancellation test keeps its checkout until the runner settles before clean removal.

Incomplete integration: the native daemon host and public AgentTool isolation option are not enabled yet. Child tool/cwd/project boundaries, starting-state options, setup policy, result review/apply UI, and retained-workspace recovery still need wiring and verification. This adapter alone is not the managed-workspaces feature. The broader roadmap remains active.

## Native isolated spawn request integration

The native host now accepts an explicit immutable worktree adapter and binds each child execution to its allocated cwd through async-local session context. AgentTool and SpawnAgents entries accept `isolation: "worktree"`, validate unsupported values and forward the request to the native manager. Standalone runners reject isolation before registering or executing work. Native snapshots retain the isolation rule; recovered respawn reads that rule rather than silently selecting the profile default.

Real Git integration exercises two concurrent children through the public manager port with actual file tools, parent-path escape rejection, parent async context restoration, persisted child cwd and project ownership. A retry gets a new isolated checkout while preserving dirty prior results. Tool tests cover individual/batch forwarding, invalid inputs and unsupported runners without execution. Recovered-respawn routing is implemented but does not yet have a restart/worktree integration test.

Verification: root check, test, build and whitespace checks passed (3,113 runtime tests passed, 2 skipped; 1,094 UI tests passed). No live provider or native terminal visual acceptance was performed.

Still incomplete: default CLI wiring must route allocations through the correct host generation/project rather than binding the shared reconfigurable host to its first cwd. Starting-state selection, setup policy, review/apply/handoff and retained-checkout recovery remain required. This is progress on roadmap item 7, not completion of managed workspaces or the full plan.

## CLI workspace-aware isolation wiring

Daemon/TUI, one-shot and ACP hosts now supply the native worktree factory. One-shot and ACP file/process tool roots use the active async-local cwd, matching the daemon. Each worktree allocation captures its execution generation before asynchronous setup; cleanup dispatches to the adapter that allocated that checkout. Retained tasks pin their original execution generation for later resets/retries, instead of falling back to a newer project's provider/tools/cwd. Factories remain immutable across host reconfiguration; fixed adapters still reject ownership changes.

A real two-repository Git fixture changes the host project while allocation is waiting, proves the first child sees project A and the next sees B, then retries A after another reconfiguration and proves it still sees A. Existing real file-tool isolation, dirty-result preservation and cancellation tests remain passing. The older reset regression now requires the original execution configuration rather than a fallback to the newest provider.

Verification: root check, test, build and whitespace checks passed (3,114 runtime tests passed, 2 skipped; 1,094 UI tests passed). Live providers and native terminal visual behavior were not exercised. The CLI construction paths are wired and pass existing CLI suites; the new project-switch fixture exercises the native host directly.

Still required for roadmap item 7: starting-state selection, setup policy, result review/apply/handoff, retained-checkout recovery and restart integration coverage. Retained dirty allocations also need lifecycle bookkeeping integrated with that recovery UI. The broader plan remains active.

## Explicit isolated starting revision

AgentTool and each SpawnAgents entry now accept `worktree_ref` with `isolation: "worktree"`. It selects an existing Git branch, tag, commit or revision expression, defaults to HEAD and is resolved to a commit before allocation. The argument is validated before spawn; missing revisions fail without creating a checkout. The native manager retains the selection in execution config and snapshot rules, and recovered respawn restores the rule. Refs resolve per attempt; a full commit ID is required for a fixed baseline if a branch/tag may move. No parent uncommitted changes are copied by this option.

Verification completed: root check, test, build and whitespace checks passed (3,115 runtime tests passed, 2 skipped; 1,094 UI tests passed). Real Git fixtures cover tags/commit IDs/relative refs, invalid or missing revisions with no checkout allocation, and preservation of parent content. Native file-tool integration starts concurrent agents and their retry from an older tagged commit while the parent remains on newer content. Tool tests cover forwarding and rejecting invalid swarm configuration before registration. Restart recovery selection is wired but still lacks the full restart/worktree acceptance test.

Remaining: starting from parent uncommitted state, setup policy, workspace result review/apply/handoff and retained-workspace recovery. The broader roadmap is still incomplete; these tests do not establish native visual or live-provider acceptance.

## Parent working-file starting state

`worktree_source: "working-tree"` is wired through individual and swarm spawns, native execution config, snapshot rules and recovered-respawn selection. It requires worktree isolation and excludes worktree_ref. The native adapter captures tracked content/deletions and untracked non-ignored files via a unique temporary index, records the resulting Git tree without making a commit, and materializes it as unstaged/untracked child files. The parent index and files are not modified. Validation rejects observed HEAD/file changes and submodule/nested-repository snapshots; capture is not an atomic filesystem snapshot. Retries capture current parent files again. Inherited dirty content is retained for review.

Verification completed: root check, test, build and whitespace checks passed (3,116 runtime tests passed, 2 skipped; 1,094 UI tests passed). Real Git fixtures prove binary/deletion capture, ignored-file exclusion, byte-for-byte preservation of the parent index and its staged content, visibility through ordinary child git diff, and dirty cleanup refusal. Tool tests exercise forwarding and conflicting-option rejection; the native two-project fixture verifies dirty content selection through setup races and retry. Source mutation/submodule rejection and crash-during-capture behavior still need dedicated acceptance fixtures; no live provider or native visual acceptance was performed.

Managed workspaces still require setup policy, result review/apply/handoff and retained-workspace recovery. A further hardening review should cover inherited Git environment routing and temporary-index/ownership bookkeeping after interruption. The full roadmap remains active.

## Persisted workspace inspection and Git routing isolation

Added `/workspaces [list|after <cursor>|inspect <id>]` through the daemon slash handler, TUI registry and protocol/help documentation. Inventory reads persisted ownership records for the active session repository, pages at 100 records and keeps damaged/missing checkout errors visible. Inspection validates ownership and reports task/path/branch/base/HEAD/status plus the diff from the original commit or captured starting tree. A temporary intent-to-add index includes non-ignored untracked files without falsely reporting inherited untracked files as deletions. Neither the parent nor child index/files are changed. The commands perform no apply, merge or deletion and do not infer whether an agent is running.

Native Git operations now scrub inherited repository/index/object routing variables and disable optional index refresh writes. A subprocess fixture with Git routing pointed at a foreign repository still allocates in the selected repository and preserves the foreign index. Reload fixtures cover retained record discovery, original versus inherited baseline comparison, new untracked content, unchanged child index, invalid IDs and missing checkout errors. A daemon socket contract test verifies slash listing/inspection without applying results; the TUI test verifies forwarding and transcript preservation.

Verification: final root check, test, build and whitespace checks passed (3,119 runtime tests passed, 2 skipped; 1,095 UI tests passed). The first gate caught an optional-session type error; an explicit no-session failure fixed it, and the full suite was rerun successfully on final code. No native terminal visual acceptance or live-provider test was performed.

Remaining workspace requirements: setup policy, dedicated interactive review/apply/handoff, recovery and cleanup actions, broader source-capture failure fixtures, and interruption bookkeeping. Inventory currently refuses more than 4096 storage entries and diff output over 1 MiB; these errors are explicit, not truncated-success reports. The broader plan remains active.

## Interactive retained-workspace review

Bare `/workspaces` now opens a TUI panel with workspace selection, styled unified diffs using the existing diff rows, original baseline metadata, status, pagination and refresh. Wide terminals use a side list; narrow terminals stack it above the review. Keyboard selection, diff focus, vertical/horizontal scrolling and Escape are supported; mouse selection is available. User-opened review survives turn completion, blocks background modal hotkeys and preserves the chat. Explicit slash subcommands still provide textual inspection.

Added session-scoped `workspace.list` and `workspace.inspect` query RPCs so panel navigation does not append slash notifications. They reject absent sessions and use the selected session repository. The UI validates responses, discards stale selection results and exposes inspection/list errors rather than showing an old diff as current. Long lines scroll horizontally; parser caps show an explicit incomplete-display warning.

Verification completed: root check and full test passed (3,119 runtime tests passed, 2 skipped; 1,103 UI tests passed); final UI type check, rebuild and whitespace checks passed after layout refinements. Rendered fixtures exercise 220x65, 110x35, 60x24 and 40x18, stale replies, pagination, refresh recovery, Escape and horizontal access to a long-line tail. A scrollbar layout oscillation and incorrect test-harness arrow input were corrected before the final passing suite. Daemon socket tests cover query listing/inspection and rejection without a session. Native terminal visual acceptance remains unperformed.

This provides interactive inspection, not the complete result-integration workflow. Apply/handoff controls, setup policy, cleanup/recovery actions and remaining hardening requirements are still outstanding; the full roadmap remains active.

## Reviewed-patch integration check

Workspace reviews now carry a SHA-256 identity covering allocation ID, baseline/captured tree, source HEAD and the full binary-capable patch. Diff collection preserves patch whitespace exactly. `workspace.checkApply` requires that review identity, revalidates ownership/content, and runs Git apply-check against the active session repository. It reports destination/HEAD/check time and clean/conflict status without applying files or indexes. C / Check apply in the review panel exposes this check; switching selections or refreshing invalidates pending check results. Older daemons without a review identity remain inspection-only.

Verification completed: root check, full test, build and whitespace checks passed (3,120 runtime tests passed, 2 skipped; 1,104 UI tests passed). Real Git fixtures exercise text with trailing whitespace, binary patch checks, destination conflicts, unchanged parent/source files and stale binary-content rejection. Daemon socket tests validate the check RPC; UI tests verify exact review identity forwarding and discard results for a previous selection.

This is a non-mutating preflight, not apply authorization or a reservation on the destination. Actual apply still needs destination revalidation, rollback/partial-failure protection, recovery and user-facing confirmation. Setup policy, handoff/cleanup and broader roadmap requirements remain active. No live provider or native terminal visual acceptance was performed.

## Recoverable integration backend (not yet exposed as an apply action)

The native workspace service now binds apply to both the reviewed patch and a destination fingerprint. It prepares the result in a private Git staging directory, writes and syncs original file backups and expected states before touching the destination, preserves the parent index, and checks the resulting files and HEAD. Failure restores only files that still match the prepared result; differing content remains untouched and is recorded as needing recovery.

Explicit backend recovery validates record ownership, paths, bounded backup sizes and SHA-256 hashes before writing. It refuses a live or unverifiable integration-lock owner, can recover a dead local worker's prepared operation, and preserves concurrent changes. Real Git tests cover text, binary, rename/new files, parent index/source preservation, partial failure, stale destination rejection, damaged backups, live locks, repeat recovery, and a SIGKILL after the first destination write. The focused suite passed 9 tests; root type checks and build passed.

This is not yet a production-complete integration workflow. Apply/recovery still need daemon contracts, a review-bound confirmation UI and recovery inventory. Recovery interrupted while holding its own guard currently needs manual guard inspection; old hostless locks are conservatively unverifiable. Capture/write operations are not an atomic filesystem transaction against arbitrary external writers, and rollback can leave empty directories. Setup policy, handoff/cleanup, full restart acceptance and the broader roadmap remain unfinished. No user workspace result was applied and no native-terminal/live-provider acceptance was claimed.

Full repository verification for this backend completed: root check, test, build and `git diff --check` passed; UI suite passed 1,104 tests. The SIGKILL recovery fixture also passed in its dedicated nine-test suite.

## Checked apply through the daemon and TUI

Added `workspace.apply` for the active session repository. It requires the workspace ID, exact review ID, checked destination fingerprint and explicit confirmation. The workspace panel now offers A after a compatible successful check and shows the destination with Y/Esc confirmation. It blocks duplicate submissions and navigation during apply, invalidates checks after each attempt, and keeps confirmation/status outside the horizontally scrolling diff. Older check responses without a destination fingerprint do not enable apply. There is no automatic retry after an unknown RPC result.

Focused daemon socket validation passed for missing confirmation, stale destination rejection and actual apply while preserving the agent result. Eleven rendered UI tests passed, including narrow/wide confirmation, cancel, duplicate input protection, error display and check invalidation. Recovery RPC/UI and interrupted-recovery lock handling remain pending; the full roadmap is not complete. No native terminal visual acceptance was performed.

Full gate completed for checked apply: root check, test, build and whitespace checks passed. Runtime: 3,129 passed, 2 skipped. UI: 1,106 passed.

## Recovery controls and crash-released guards

Recovery now serializes callers with a SQLite immediate transaction held across the operation. The OS releases it when the recovery process dies; the database inode is retained. Live/unverifiable integration owners remain protected. Recovery re-reads the durable record after acquiring ownership, and a completed apply only releases a leftover lock without undoing its files. Legacy exclusive guard files are not automatically deleted.

`workspace.integrations` pages saved records for the active session repository, preserving unreadable/missing-artifact errors. `workspace.recover` requires explicit confirmation. I in the workspace panel opens recovery inventory; B requests original-file restoration for interrupted applies or lock release for completed ones, Y confirms, and Esc cancels/returns. Refresh after recovery updates status; duplicate submissions are blocked. Inventory metadata does not imply verified backup contents or a dead owner.

Focused validation passed: 12 backend tests (including SIGKILL of apply and recovery workers, concurrent recovery rejection and completed-result preservation) and 13 rendered workspace UI tests at narrow and wide sizes. The daemon contract exercises inventory, confirmation and restoration. Recovery before a prepared record exists, richer per-file recovery inspection, setup/handoff/cleanup and broader roadmap requirements remain unfinished. No native-terminal or live-provider acceptance was performed.

Final current-code gate passed: root check, full test, build and whitespace checks. Runtime: 3,132 passed, 2 skipped. UI: 1,108 passed. An earlier in-progress suite had loaded the prior recovery implementation before the completed-lock fix and failed the two updated cases; the fresh full suite above supersedes that mixed-version result.

## Interrupted preparation and terminal lock release

Apply now persists `preparing` before atomically publishing a fully written owner lock. No destination writes precede the durable `prepared` transition. A killed preparation can be listed and explicitly abandoned; recovery returns `abandoned` without restoring files. Apply re-reads preparation under ownership and refuses an operation already abandoned by recovery. Both apply and recovery publish new owner locks by exclusive hard-link creation, avoiding an empty JSON lock after a crash.

The recovery panel distinguishes abandoning preparation, restoring interrupted writes and releasing a terminal operation's leftover lock. Terminal `applied`, `rolled-back` and `abandoned` recovery preserves current files, including newer edits. Backend tests cover SIGKILL during preparation followed by abandonment and a successful fresh apply, and dead-lock release in all terminal states. Fifteen focused backend tests passed; type checks, build and whitespace checks passed.

An interruption before the first preparation record is published can leave an unavailable artifact directory, but no integration lock is published at that point. Damaged/legacy records remain visible and are not automatically deleted. Richer per-file recovery inspection, setup policy, handoff/cleanup and the wider roadmap remain unfinished; native terminal/live-provider acceptance is not claimed.

Full current-worktree gate passed for interrupted preparation: root check, full test, build and whitespace checks. Runtime: 3,135 passed, 2 skipped. UI: 1,110 passed, including confirmation/abandonment at narrow and wide sizes.

## Per-file recovery preview

Added `workspace.integration.inspect`, scoped to the active session repository. It verifies backup integrity and reports each affected path as restorable, already original, conflicting or preserved for a terminal operation. Changed HEAD and unreadable/different destination content remain conflicts. The recovery panel loads the preview on selection, rejects stale responses, pages file rows and allows horizontal access to long paths/reasons. Inspection never reserves or changes destination files; recovery rechecks them.

Focused native/socket tests pass for classification, unchanged files and corrupt-backup rejection. Sixteen workspace UI tests pass, including stale-inspection rejection and the existing apply/recovery confirmation paths. Type checks, build and whitespace checks pass. Workspace setup policy, handoff/cleanup, richer content diff presentation and broader roadmap requirements remain unfinished; no native terminal or live-provider acceptance is claimed.

Full current-worktree gate passed for recovery inspection: root check, full test, build and whitespace checks. Runtime: 3,136 passed, 2 skipped. UI: 1,111 passed.

## Allocation cancellation before configured setup

Tracing configured-setup integration exposed a retry cancellation bug: retry could resume/spawn after cancellation during allocation, and cancelled setup could accept another retry before settling. The manager now tracks a setup AbortController for initial allocation and retries, includes retry setup in wait/admission/shutdown bookkeeping, prevents overlapping retries while setup is pending, and preserves cancelled state instead of restoring the previous terminal status. Late successful allocations are cleaned only after allocation settles; cooperative allocation failures cannot start a runner.

The optional allocation signal passes through the native host to Git commands with the existing bounded timeout. Focused retry/worktree validation passed 18 tests, including cooperative and non-cooperative cancellation during retry and cleanup of a late clean checkout. Type checks, build and whitespace checks passed. This fixes a lifecycle prerequisite; configured setup commands, their policy/persistence/presentation, handoff/cleanup and the broader roadmap remain unfinished.

Full current-worktree gate passed for allocation cancellation: root check, full test, build and whitespace checks. Runtime: 3,138 passed, 2 skipped. UI: 1,111 passed.

## Configured workspace setup execution

Native isolated allocation now loads an opt-in per-repository argv command from the user-owned `workspace-setup.json` in Xerxes home. It runs after starting files are materialized and before agent startup, using the existing foreground executor with bounded output, a 1–120000 ms timeout and process-tree cancellation. Repository-local instructions do not automatically become executable setup policy. New allocations/retries reload the user policy.

Setup status and bounded output are atomically replaced in a private sidecar and exposed in workspace review. Failed setup stops startup, retaining checkout and evidence. Records lacking completion are displayed as incomplete rather than asserting a live process. Focused fixtures passed for checkout-scoped generated files, retained failure output, malformed policy rejection, timeout and cancellation; the final suite tests cancellation after a process actually starts.

Setup settings currently use the user configuration file; a dedicated settings editor, richer failed-setup retry controls, setup process restart reconciliation, handoff/cleanup and other roadmap requirements remain pending. No live dependency installation or native-terminal visual acceptance was performed.

Final full gate passed for configured setup: root check, full test, build and whitespace checks. Runtime: 3,142 passed, 2 skipped. UI: 1,111 passed. The first full run hit the existing 25-second slow-API-test deadline after an anomalous 351-second elapsed interval; that case then passed unchanged in isolation (11.5 seconds), and the fresh full suite passed without relaxing its timeout.

## Retry after workspace setup failure

The manager registers retryable runtime state before allocation/setup while leaving depth-limit rejection non-retryable. A typed setup failure carries the retained checkout identity into the task's displayed path. Retrying uses the same task identity with a new checkout and current setup configuration; failed checkout data stays available for review. A repeated setup failure updates the task's error/path instead of restoring a stale earlier failure. Cancellation checks now close a native handle only when one actually exists during pre-run setup.

The real Git/setup fixture proves no runner starts on initial or repeated setup failure, correction succeeds through retry in a different checkout, and previous failure markers remain intact. Focused setup/retry tests passed 15 cases; type checks, build and whitespace checks passed. Restart reconciliation, setup settings UI, handoff/cleanup and broader roadmap requirements remain open.

The setup-retry full gate also passed: 3,143 runtime tests passed, 2 skipped; 1,111 UI tests passed.

## Background command completion captures final output

Explicit background starts and foreground-to-background handoffs now wait for their stdout/stderr drains before closing the terminal mirror and publishing its durable result. The wait is bounded to one second after process exit; inherited pipes still open at that point are cancelled, and the saved output explicitly reports that capture stopped. Kill/release waits for this final capture when the process has exited. Handoff callbacks are only invoked when a terminal mirror actually exists.

A deterministic adopted-process fixture delays drain completion beyond process exit and verifies final stdout/stderr in the completion event and reopened history, one completion event across cleanup, and bounded cancellation of an indefinitely open pipe. Focused background/terminal tests passed 30 cases. This addresses completion ordering, not durable mailbox wakeups, log cursor recovery, process identity, retention or the rest of the background-shell roadmap.

Full gate passed for completion capture: root check, test, build and whitespace checks. Runtime: 3,145 passed, 2 skipped. UI: 1,111 passed. These are automated fixture results; native desktop and live-provider acceptance remain unverified.

## Command completion watches through the existing reaction mailbox

`monitor_terminal` and `monitor.create` now accept `trigger: completion` without match text. The monitor emits one bounded structured exit result, persists it as monitor evidence and uses the existing mailbox/dispatcher when an authorized direct-user turn requests `react: true`. A command that finishes before watch attachment is inspected and delivered immediately, avoiding the start/watch race. Output watches retain their prior default behavior. Completion evidence includes exit code, command, cwd, output tail and truncation metadata; process output remains untrusted in reaction prompts.

The TUI create form exposes a trigger selector and includes finished terminals. Existing monitor inspection, stopping and reaction controls apply. Focused tests cover before/after-exit attachment, exactly one evidence event and claim, cross-session denial, stopping before exit, native RPC creation/validation and the TUI completion selection. All five form tests passed; the combined focused daemon/completion filter passed nine cases.

This reuses durable evidence and reaction claim recovery; it does not recreate a watch or process pipes lost on daemon restart. Output log cursors, retention, process identity, automatic completion delivery without an explicit watch and broader roadmap acceptance remain open.

The final completion-watch gate passed: root check, build and whitespace checks; 3,149 runtime tests passed, 2 skipped; 1,112 UI tests passed. A further regression fixture verifies that JSON-escaped control-character output stays within the durable event limit, with explicit output/metadata truncation flags. No live-provider or native-desktop acceptance is claimed.

## Independent incremental terminal output cursors

Terminal mirrors now assign a per-run stream identity and expose independent UTF-16 offsets through native `terminal.output` and model `read_terminal_output`. Wrong-stream, negative, fractional and future offsets are rejected; IDs reused by a new process cannot silently replay the prior stream. Retention gaps report an exact dropped-character count. The existing consuming check-command buffer remains independent.

A private SQLite table atomically saves the bounded terminal tail and its total character count together with the run's displayed output. Checkpoints batch active output at 250 ms; final output is checkpointed before completion. Archive access uses `run:<run_id>` with the original stream identity, including after reopening the store. Older records without cursor metadata explicitly fall back to ordinary inspection. This is bounded-tail persistence, not an unlimited log archive or a zero-loss crash guarantee: uncheckpointed output can be lost, and a cursor beyond the recovered total is rejected.

Focused background/terminal/history tests passed 43 cases before the additional model-tool fixture; that fixture passed separately. The daemon archive/read/continue/invalid-cursor contract also passed. Full retention lifecycle, stronger process identity, restoring unfinished watches and remaining roadmap requirements are still pending.

Full cursor gate passed: root check, test, build and whitespace checks; 3,152 runtime tests passed, 2 skipped; 1,112 UI tests passed. Native desktop and live-provider acceptance were not performed.

## Confirmed background stop and late-fork cleanup

Background stop now preserves its process/terminal handles and reports an actionable error if exit cannot be confirmed, rather than hiding a still-running command. A successful signal to a live leader is followed by the existing bounded post-exit process-group sweep, covering helpers forked inside a TERM handler. Reaping a previously completed record does not sweep an old PID.

Real subprocess regressions cover a TERM handler that forks a signal-ignoring writer before exiting, and an injected signal-denial path that keeps the command visible and permits a later successful stop. The focused background suite passed 23 tests before the additional denial test; both new regression cases passed together. Windows process-tree containment, durable process identity and broader roadmap acceptance remain pending.

Full background-stop gate passed: root check, test, build and whitespace checks; 3,154 runtime tests passed, 2 skipped; 1,112 UI tests passed. No live-provider or native-desktop acceptance is claimed.

## F8 running/finished filters and retained-output search

The terminal list cycles All/Running/Finished with `f`, clearing the selected action when filters change. Detail `/` search filters retained output by case-insensitive literal text and displays original line numbers. Enter applies, Escape clears before leaving; pasted search text is bounded and does not route to terminal stdin. Empty filters and empty searches have explicit messages. The detail status now reflects the selected terminal rather than unrelated running terminals.

Keyboard/render fixtures at 220x65 and 60x24 cover filtering, opening a finished command while another runs, search, absence of process-control calls and Escape restoration. The complete UI suite passed 1,114 tests. This is renderer-based verification, not native desktop visual acceptance; search covers only retained output, not discarded history.

## Upcoming schedules in the workspace Runs overview

Workspace `run.list` now includes a bounded, ordered projection of unpaused schedules in the active workspace. It reports the total and first three next-due records without inventing execution-history entries. Session scope returns none. Existing run filters and acknowledgements remain about actual executions; upcoming schedules are separately labelled.

Runs displays up to three jobs on wide terminals and one on narrow terminals, with title and due time on separate lines. `T` or the section click opens existing schedule management. Older daemon responses remain history-only. The daemon contract verifies due ordering, paused exclusion and workspace isolation; 15 Runs UI cases pass, including upcoming rendering/navigation at 220x65 and 40x18. Approval aggregation, richer run navigation, scheduling service availability and other roadmap requirements remain open.

Full upcoming-Runs gate passed: root check, test, build and whitespace checks; 3,155 runtime tests passed, 2 skipped; 1,116 UI tests passed. Native desktop and live-provider acceptance remain unverified.

## Current-session attention in Runs

The interaction board now exposes a read-only summary of live approval and question waits, with bounded titles and no tool arguments. `run.list` includes up to three entries and their total for the current session, even when execution history uses workspace scope. Resolution and cancellation remove entries from the same underlying board; no duplicate approval authority or decision store was added.

Runs displays the summary and offers `E`/click to return to chat, leaving the existing approval/question controls responsible for decisions. The interaction-board fixture proves session isolation, argument omission and cancellation/response removal; the daemon contract proves current-session scoping in workspace Runs. Seventeen Runs UI tests pass, including wide/narrow attention rendering and navigation without an approval RPC. Broader cross-chat attention navigation and restart reconciliation remain open.

Full attention gate passed: 3,156 runtime tests passed, 2 skipped; 1,118 UI tests passed; build and whitespace checks passed. Final root type checks and the focused daemon contract passed after correcting test cleanup to use the public response method. Native desktop/live-provider acceptance was not performed.

## Commit observed output before returning incremental cursors

Durable terminal reads now validate the requested page and commit the current output tail/offset before returning a live cursor. Completed entries verify the acknowledged offset against persisted output. Persistence failures escape the read instead of returning a cursor that recovery cannot honor. Periodic checkpoints remain for output nobody has read; memory-only hosts retain their live-only behavior.

A disposable Bun subprocess emits a cursor and deliberately blocks its own timer; the parent SIGKILLs it and reopens history, proving that the cursor remains valid and the observed output survives with an interrupted outcome before any periodic checkpoint. A second fixture closes the store and verifies that a read fails. This does not guarantee survival of unobserved output, retention eviction, storage loss or power failure.

Full acknowledged-cursor gate passed: root check, test, build and whitespace checks; 3,158 runtime tests passed, 2 skipped; 1,118 UI tests passed. The crash fixture uses a real local Bun process; it is not a live-provider or native-desktop acceptance test.

## Run-owner command identity during recovery

Run history now additively stores the owning process command and compares it when a recorded PID remains alive during recovery. A different readable command marks the old run interrupted while preserving evidence. Matching identities, unreadable probes and legacy records without command metadata retain conservative PID-based behavior. Probes are cached per PID during each store initialization; the existing command probe now bounds execution to one second and output to 64000 bytes.

This strengthens detection of unrelated PID reuse but is not a process-birth identity: an identical restarted command cannot be distinguished, and a process that deliberately changes its displayed command invalidates the recorded identity. Other stores' ownership mechanisms are unchanged. Focused tests cover mismatch, match and unreadable identity. Stronger generation identity and the broader roadmap remain open.

Full run-identity gate passed: root check, test, build and whitespace checks; 3,161 runtime tests passed, 2 skipped; 1,118 UI tests passed. No native desktop or live-provider acceptance was performed.

## Snapshot rollback preserves unbacked ignored data

Review of rewind found a data-loss path: snapshot capture honors ignore rules, while full rollback previously ran workspace-wide `git clean -fdx`, removing ignored files absent from its own backup. Cleanup now derives literal removed-file paths from the pre-restore backup versus target snapshot and honors restored ignore rules. It refuses a backed-file path that has become a directory. Unbacked ignored files and unrelated ignored directory contents survive; backed source files absent from the target are removed and recoverable from the backup.

The real shadow-Git regression changes ignore rules, retains ignored data during rollback and undo, and restores bracket-bearing filenames literally. All 11 snapshot tests pass. This fixes cleanup scope; a timeline/diff UI, concurrent-edit validation and transactional failure recovery for the entire restore remain open. No claim is made that arbitrary external writers are excluded during restore.

The existing hardening test required ignored artifacts to be deleted both during rollback and during undo; both expectations were updated to preserve their contents, while the tracked-file restoration assertions remain. The final combined snapshot/hardening suite passes all 17 tests. Earlier full runs failed those obsolete deletion assertions; their failures are not reported as a passing gate.

Final snapshot-cleanup gate passed: root check, test, build and whitespace checks; 3,162 runtime tests passed, 2 skipped; 1,118 UI tests passed. No native desktop/live-provider acceptance was performed.

## Snapshot restore previews through the native slash path

`/rollback diff <id>` now previews current captured files to the target snapshot without restoring. The snapshot diff helper captures current new/deleted files into a temporary shadow index, preserving both the real workspace index and the persistent shadow index. It does not add a snapshot record. Ignored uncaptured files are excluded. The command catalog, TUI native forwarding/help, protocol and user guide now expose the preview; retired rollback RPC names remain unsupported.

Preview Git output is bounded to 2 MiB and fails explicitly above the limit; transcript output is capped at 100,000 characters with a truncation flag. Git collection also bounds ordinary stdout and stderr and cancels readers/kills the direct Git process on timeout or collection failure. Real-Git tests verify new/deleted files, unchanged workspace/index state, missing snapshot failure, oversized output failure and a successful subsequent preview. The daemon contract verifies restore direction and that source content stays unchanged. This remains advisory: concurrent external edits, revision-checked restore, a timeline UI and transactional full-restore recovery remain open. No native desktop or live-provider acceptance was performed.

Full preview gate passed: 3,164 runtime tests passed, 2 skipped; 1,119 UI tests passed; root type checks, build and whitespace checks passed. The updated command catalog also passed its focused three-test suite. An initial type-check failure on an explicit undefined optional property was corrected before the passing checks.

## Guard snapshot restore with the reviewed captured-tree revision

Snapshot previews now return a SHA-256 revision bound to the target commit and the current captured Git tree. The displayed `/rollback apply <id> <revision>` command checks that revision against the pre-restore backup tree before any workspace writes. Edits, additions, deletions or a different target reject with an actionable stale-preview error. Rejection can create a backup record; workspace files remain unchanged. A fresh preview permits restore. Legacy direct restore remains available without the guard for compatibility.

The native daemon path, TUI forwarding, completion catalog and docs expose the guarded command. Real-Git cases cover concurrent edit/create/delete, target mismatch, preservation on rejection and successful re-preview/restore. The daemon contract exercises rejection and acceptance, and TUI forwarding is covered. This does not lock arbitrary external writers during capture/restore, include ignored uncaptured data, or provide transactional recovery from a partially failed checkout; timeline UI and those recovery requirements remain open.

Validation passed: root type checks, test suite, build and whitespace checks. Runtime: 3,168 passed, 2 skipped. UI: 1,120 passed. No native desktop or live-provider acceptance was performed.

## Refuse known restore obstructions before writing workspace files

Full and selected-file restores now preflight destination paths before checkout. Directory/special-file leaf obstructions and non-directory or symlink ancestors are rejected before any workspace writes. Full restore also preflights its removed paths. This prevents a known partial-write sequence where Git restores an earlier file and then fails at a later blocked path. Obstructed file/directory transitions require explicit resolution; they are not automatically dismantled.

Real-Git regression cases place an edited earlier file ahead of a directory, file-parent or symlink-parent obstruction, and verify both full and selected-file refusal. Earlier edits, nested data and outside symlink-target data remain intact, with no new snapshot on the initial destination refusal. The focused snapshot/hardening suite passed 26 tests. This is preflight protection, not exclusion of racing external writers or transactional recovery from every mid-checkout I/O failure.

Root check, test, build and whitespace gates passed: 3,171 runtime tests passed, 2 skipped; 1,120 UI tests passed. Native desktop and live-provider acceptance remain unverified. Timeline UI and full restore failure recovery remain pending.

## Snapshot timeline with preview and guarded restore confirmation

`/snapshots` now opens a responsive timeline; `/snapshots list` retains text output. Entries expose capture time and optional session/turn coordinates. Selecting an entry requests a diff; Tab, arrows and page keys navigate retained preview output. A restore requires A then Y and submits the exact preview revision. Escape cancels confirmation first. Duplicate confirmation keys cannot dispatch a second in-flight restore. Errors preserve the panel and invalidate a failed restore preview until refresh. Turn completion preserves the user-opened timeline; chat content is not replaced.

Additive `snapshot.list` and `snapshot.preview` read RPCs resolve the current session workspace and emit no slash transcript notifications, avoiding repeated full diff dumps while browsing. The gateway maps the session ID to the native session key. Restore continues through the guarded slash command with observable completion/error notifications. The timeline explicitly separates filesystem restore from conversation state and reports ignored-file exclusions and truncated previews.

Renderer fixtures at 220x65, 100x30 and 40x18 cover content, confirmation cancellation and guarded restore. Additional cases cover stale asynchronous previews, failed restore invalidation, refresh after list failure and duplicate confirmation suppression. The daemon contract covers the read RPCs, input errors and matching preview revisions. An initial narrow-layout failure showed too little diff space; narrowing the list to one two-line entry corrected it. This is renderer-based acceptance, not native terminal visual verification. Selected-file controls, conversation/combined branching, full transactional recovery and remaining roadmap scope are not complete.

Final validation: root check, test, build and whitespace checks passed; 3,171 runtime tests passed, 2 skipped; 1,126 UI tests passed. UI type checks and build were rerun after final response validation changes.

## Selected-file preview and guarded restore/removal in the timeline

Snapshot preview RPCs now return literal changed paths and optionally scope diff/revision to one file. File revisions bind the target, literal path and captured entry including mode, allowing unrelated later edits to survive without rejecting the selected restore. Timeline F cycles files and all-files scope. The heading/confirmation explicitly distinguishes Restore from Remove. `snapshot.restoreFile` requires a file revision and saves a backup before restoring the selected leaf or removing a captured new file absent from the target. Legacy unguarded single-file restore still rejects absent target files. Directories, submodules, traversal and absent-from-both-tree requests are rejected. Literal whitespace and bracket filenames are preserved.

Real-Git cases verify file-scoped revision rejection, unrelated-edit preservation, literal filenames and recovering a guarded removal from its backup. Daemon contracts cover preview action/paths, guarded execution and missing revision rejection. Wide/narrow UI fixtures verify explicit removal confirmation and the file RPC instead of a full-workspace command. Adding the F shortcut initially reduced diff space at 40x18; the compact two-line footer fixed that renderer regression. Full I/O transaction recovery and arbitrary external-writer exclusion remain unfinished; no native terminal visual acceptance is claimed.

Final gate passed: 3,173 runtime tests passed, 2 skipped; 1,128 UI tests passed; root type checks, build and whitespace checks passed. UI type checks and build were rerun after the narrow footer adjustment.

## Durable restore-attempt records and backup review

Full and selected-file mutations now atomically write and fsync a private prepared restore record before workspace changes, then record completed or failed outcomes. Failures expose backup/attempt IDs and bounded error evidence; corrupt or oversized journals refuse mutation. A prepared record remains discoverable after reloading and means running-or-interrupted, not confirmed process liveness. Ordinary pruning pins unresolved targets and backups. The journal bounds retained completion history and stops new ordinary restores at 128 unresolved attempts while allowing an explicit recovery restore.

The timeline lists unfinished restore warnings and B selects the first backup for preview without starting recovery. Successful backup restoration covering the original full/file scope marks that attempt recovered. A fault-injected checkout fixture proves prepared evidence precedes writes, a partial failure remains discoverable, pruning preserves its backup, and explicit backup restoration repairs original content even at the unresolved-attempt limit. A corrupt-journal case verifies current files stay unchanged. This is durable recovery evidence and an explicit recovery path, not automatic transaction rollback, a real process-kill fixture, or exclusion of arbitrary external writers.

The final full gate passed after correcting recovery admission at the unresolved limit and temporary-journal cleanup: 3,175 runtime tests passed, 2 skipped; 1,129 UI tests passed; root type checks, build and whitespace checks passed. Native terminal and live-provider acceptance remain unverified. Full transaction guarantees and the remaining roadmap are still incomplete.

## Process-interruption recovery and bounded recovery retries

A real disposable Bun child is killed after an injected partial restore write and after prepared journal persistence. A fresh manager finds the prepared evidence, recovers the dead-owner repository lock and restores the original content from its backup. Dead owner PIDs no longer impose the full stale-lock grace delay; freshly incomplete lock files still retain it. This is process-kill validation at a controlled mutation point, not a claim about power loss or arbitrary Git/filesystem crash positions.

Repeated failed recovery retries previously accumulated unresolved records until the bounded journal could reject its own output. Retrying a recovery target/scope now retains the original recovery anchor plus the latest retry, and completed-history retention yields space to unresolved records. Writes enforce the same entry bound as reads. A regression retries failure four times without journal growth and then successfully recovers. Older retry backups become ordinary snapshots; the original anchor stays pinned.

Final root check, test, build and whitespace gates passed: 3,177 runtime tests passed, 2 skipped; 1,129 UI tests passed. An initial type-check failure in the new test fixture was corrected with a literal phase type. Automatic failed-restore reversal, broader transactional guarantees, native terminal acceptance and the remaining roadmap are still incomplete.

## Conditional automatic reversal of caught restore failures

Caught full/selected-file mutation failures now attempt bounded reversal using the target and pre-restore backup. A file already matching its backup is left alone; one still matching the intended restore target is restored to its backup state and checked afterward. Other contents and path obstructions are preserved as conflicts. Additions and deletions are handled as leaf operations. Recovery stops admitting further files after ten seconds, while in-flight Git retains its command timeout. Conflict details are bounded in the journal; fully verified reversal records reverted, and the original restore still reports failure with its recovery outcome.

Real-Git fixtures execute target checkout before injecting failure, covering overwritten files, restored target additions, removed prior files and a concurrent third version. Known writes reverse; the third version survives and keeps the backup pinned. Another fixture fails a selected-file deletion after unlink and proves the backup leaf is restored. The earlier repeated-recovery test now injects unknown partial contents so it continues to exercise unresolved retries rather than the new successful-reversal path.

This is conditional reversal for caught failures, not an exclusive filesystem transaction. Process death still requires explicit prepared-record recovery. Captured Git identity, ignored-data exclusions and arbitrary writers racing checks limit the guarantee. Full roadmap completion and native terminal/live-provider acceptance remain unproven.

Final root check, test, build and whitespace gate passed: 3,180 runtime tests passed, 2 skipped; 1,129 UI tests passed. The focused snapshot suite passed 29 tests.

## Runtime-backed tool discovery instead of count-only fallback

The default daemon had no injected toolCatalog despite constructing a native registry. `/tools` now falls back to the active runtime's tool inventory, with an explicit embedding catalog retaining precedence. Inventory distinguishes loaded schemas, deferred schemas, agent/mode-filtered tools and registered-but-unexposed schemas. It resolves transcript-revealed deferred tools and live registration changes without running a tool or calling a provider. Unknown enforcement profiles fail visibly rather than inventing a usable surface.

Additive `tool.inventory` exposes this read model without transcript notifications; the gateway maps session IDs to native keys. Results identify runtime-registry, host-catalog or unavailable source, and keep execution readiness not_checked/unknown. Text help/completion and `/tools` render the exposure labels while explicitly separating registration from authorization and connection readiness. Tool definitions alone still do not prove a connected host. Broader host health/provenance, reconnect controls and configuration management remain unfinished.

Focused runtime tests cover deferral revelation, live unregister, filtering, unavailable profiles and unexposed static schemas without provider calls. The daemon contract proves runtime fallback with no catalog injection and unavailable reporting without a session. Final root check, test, build and whitespace checks passed: 3,182 runtime tests passed, 2 skipped; 1,129 UI tests passed. The extra unexposed-schema assertion passed in the focused runtime test. Native desktop/live-provider acceptance remains unverified.

## Discoverable MCP failures and explicit reconnect

Failed initial connections and disabled configurations now remain in the manager's health inventory. The native configuration loader retains disabled entries so they reach status and reserve their user-configured names against project overrides; the manager does not launch disabled transports. Startup logs now report retained redacted connection failures rather than incorrectly labeling them disabled/duplicate.

Additive `mcp.status` and `/mcp [status|reconnect <name>]` expose health and explicit retry through daemon, command discovery and TUI slash handling. `/reload-mcp` retries all enabled configurations, including initial failures. These commands do not reread configuration files or activate disabled entries. Removing or replacing a registration prevents an older reconnect from restoring it or overwriting the replacement, including when an embedding host reuses the same configuration object. A superseded reconnect reports false and leaves the newer client intact. Existing backoff sleeps and already-started connection attempts are not immediately aborted by removal; late clients are disconnected before registration.

Focused configuration, lifecycle, command registry and slash-routing checks passed 63 tests. Final root check, test, build and whitespace gates passed: 3,186 runtime tests passed, 2 skipped; 1,131 UI tests passed. Initial test/API typo, command-count expectation, superseded-reconnect expectation and exact-optional fixture errors were corrected before the final gate. External MCP connectivity, native terminal acceptance, broader integration provenance/settings and the full roadmap remain unverified or incomplete.

## Cancel MCP retry waits on registration removal and shutdown

Registration lifetimes now own abort controllers. Removing/replacing a registration or disconnecting all servers cancels its pending reconnect backoff and prevents later attempts. The reconnect helper accepts an optional AbortSignal, checks it before attempts and after failure hooks, clears its native delay timer on abort, and observes custom sleeper rejections even if cancellation wins. Abort diagnostics use a fixed message instead of exposing arbitrary host abort reasons. Custom sleeper resources remain host-owned.

Already-started connects are intentionally awaited rather than abandoned: after they settle, the manager's registration identity check disconnects superseded clients. Immediate cancellation of active transport operations and bounded arbitrary embedding disconnect hooks remain outside this guarantee. Tests cover removal while an injected sleeper never settles, cancellation from an error hook/pre-cancelled calls, native 60-second backoff interrupted by disconnectAll, and existing late-candidate/replacement cases.

Focused MCP tests passed 25 tests. Final root test/build and whitespace gates passed: 3,189 runtime tests passed, 2 skipped; 1,131 UI tests passed. Root check passed again with the final test addition. Full roadmap completion and native/live integration acceptance remain outstanding.

## Native read-only skill inspection

Replaced the TUI's unavailable response for `/skills inspect <name>` with a native daemon handler. It refreshes normal discovery, exposes source path, frontmatter metadata, platform support and bounded literal instructions, and explicitly marks execution readiness not_checked. It never expands arguments/shell snippets or activates a model turn. Unknown names and malformed arguments fail visibly. Completion uses admitted registry names without invocation-only subcommand suffixes; help and protocol/configuration docs describe the supported surface. Existing workspace trust remains in force. Installation, remote search, configuration management and detailed rejected-candidate diagnostics remain unfinished.

A daemon regression inspects a skill containing an embedded touch command, verifies the marker never appears and the model transcript stays empty, checks source/literal content and name completion, and exercises an unknown skill. TUI forwarding has its own regression. Root check, test, build and whitespace gates passed: 3,190 runtime tests passed, 2 skipped; 1,132 UI tests passed. Native desktop acceptance and the full feature-gap roadmap remain outstanding.

## Current skill discovery diagnostics

`/skills diagnostics` now exposes registry discovery notes through the native handler and TUI. It reports kind, source path, optional name and bounded detail for rejected/shadowed/renamed candidates, with total/truncated fields. It refreshes discovery on every request so resolved conflicts do not linger. The projection is limited to 200 entries and 1,000 detail characters per entry. Missing skill inspection points to diagnostics. `/skills` argument completion now offers the supported list/inspect/diagnostics actions; invalid diagnostics arguments fail visibly. No trust changes, activation or dependency readiness are inferred.

A daemon fixture verifies two same-name sources identify the losing and winning paths, removing the loser clears the next report, completion exposes diagnostics, and extra arguments fail. TUI forwarding has a regression. Root check/test/build and whitespace gates passed: 3,191 runtime tests passed, 2 skipped; 1,133 UI tests passed. This exposes existing registry diagnostics; it does not implement skill installation, remote search, transactional settings or broader integration health. Native acceptance and the full roadmap remain incomplete.

## Preserve existing plugins during another module's registration

Plugin management inspection found that a module could call unregisterPlugin on a previously loaded plugin during register(), then fail; the rollback snapshot tracked names rather than removed values and could not restore the old capabilities. Registration now rejects removal of plugins outside that callback's newly registered names. Failed-module partial registrations still roll back. Own temporary registrations and ordinary host removal outside discovery remain supported. This protects public registry lifecycle calls, not arbitrary trusted JavaScript execution or direct mutation of returned objects.

Regression tests verify the existing tool and hook remain callable/retrievable, partial new tools disappear, the failure is observable, own cleanup works, and host removal remains available. Runtime/UI/desktop type checks and catalog checks passed; 19 focused extension/provider tests passed; build and whitespace checks passed. The full root test suite was not repeated for this isolated registry guard. Persisted plugin enable/disable, lifecycle-aware reload and dependency-safe management controls remain unfinished; no such controls are advertised as implemented.

## Plugin inventory provenance and honest host availability

Tracing the real bootstrap found that composeRuntimeFeatures is an available composition API but is not called by the normal CLI; the daemon creates an empty PluginRegistry unless an embedding host supplies one. `/plugins` now reports host-registry versus unconfigured explicitly, preserves existing name/slash-command fields and adds a capability inventory with execution_readiness not_checked. The registry captures the source path for successful module registrations and labels programmatic registrations as host-owned without inventing a file. `/plugins inspect <name>` and completion expose version, description, source, capability names and declared dependencies without calling a plugin. Unknown registrations fail visibly.

Native loading, module configuration persistence, lifecycle-aware reload and dependency-safe enable/disable remain unfinished. This inventory does not imply that registered tools, hooks, providers or channels are connected to the active turn loop. Module and host provenance regressions cover removal, and daemon tests cover injected/unconfigured hosts, completion and unchanged model transcripts. Root check/test/build and whitespace gates passed: 3,196 runtime tests passed, 2 skipped; 1,134 UI tests passed. Native desktop/live integrations and full roadmap completion remain unverified.

## Wire MCP tools into actual daemon turns

Tracing native plugin loading exposed a separate critical gap: the CLI constructed an MCP manager for status but never registered its tools with daemonRuntime. The CLI now passes the same manager into runner construction and publishes each connected server's schema through ToolRegistry. Stable provider-safe names combine readable server/tool segments with an exact-identity hash, so colliding sanitized names and same-named tools on different servers remain distinct. Ordinary schema validation, permissions and conservative tool capabilities apply; server safety hints do not confer trusted privileges. Calls require session context, route to their exact discovery client under the per-server queue, preserve AbortSignal, and surface isError results as failures. Superseded client handlers fail rather than silently dispatching against changed schemas.

Startup connection success and explicit reconnect rebuild the runner inventory. Shipped coding/creator profiles receive the discovered tool names; custom profiles and restricted mode ceilings remain unchanged. The CLI closes its manager on startup failure and normal shutdown and avoids rebuilding a runner once shutdown starts. Resource/prompt invocation, one-shot/ACP MCP loading, dynamic same-connection capability-change handling and native plugin loading remain unfinished.

A real spawned CLI daemon fixture uses a local Bun stdio MCP server and local fake provider. It proves schema publication, actual tool execution, successful reconnect without duplicate names, and execution from a fresh session after reconnect. Adapter tests cover exact server identity, invalid input, stale clients, cancellation, explicit MCP failure results and profile boundaries. The first CLI test exposed the static built-in profile filter; wiring that profile was required for actual model execution. A TypeScript fixture stream-type error was corrected before the final gate.

Final root check/test/build and whitespace checks passed: 3,200 runtime tests passed, 2 skipped; 1,134 UI tests passed. This is offline end-to-end CLI evidence, not a live external-provider or native desktop acceptance claim. The full roadmap remains incomplete.

## Prompt cancellation of queued MCP tool calls

Both the existing manager callTool path and namespaced runtime calls now pass AbortSignal into queue admission. Already-cancelled work never enters the queue. A waiting caller rejects promptly with a fixed cancellation error, and the queue checks cancellation again before dispatch. The bounded slot remains reserved until skipped, preventing unlimited cancelled nodes behind a stalled operation. Once execution begins, the waiting-only listener is removed and the transport owns cancellation/settlement. Namespaced lookup also consistently uses the normalized server key.

Regressions hold the preceding call unresolved, cancel the next call, verify prompt rejection and no dispatch, verify queue capacity remains enforced, then release the predecessor and execute new work. The first test attempted to construct an asynchronous Bun rejection assertion before aborting and stalled; its ordering was corrected. The combined MCP lifecycle/adapter/real CLI fixture passed 31 tests. Runtime/UI/desktop type and catalog checks, build and whitespace checks passed. The full root test suite was not repeated for this isolated queue change. Broader lifecycle, loading surfaces, native plugin work and roadmap completion remain outstanding.

## Shared MCP loading across CLI execution modes

Extracted the daemon's MCP user/project configuration policy into startConfiguredMcpServers. Daemon, one-shot, resumed one-shot and ACP now use the same loader, enabled/disabled handling, redacted failure reporting and user-over-project precedence. Project duplicates are resolved deterministically and names are compared after trimming. Workspace opt-in now matches only the complete values 1/true/yes/on, correcting the old partially anchored MCP-only expression. One-shot and ACP await discovery before building schemas and extend only shipped coding/creator profiles. Owned managers close on setup failure and shutdown, including resumed runtime failures.

The existing real CLI fixture now executes MCP calls in daemon, one-shot and ACP subprocesses against local fake provider/MCP servers. Its daemon branch also persists a session, stops the daemon, resumes with --resume in a new process and executes MCP again. Configuration tests prove workspace trust, disabled-user precedence, duplicate handling and redacted initial failures. An ACP fixture async-iterator typing issue was corrected using the explicit stream reader API.

Final root check/test/build and whitespace gates passed: 3,206 runtime tests passed, 2 skipped; 1,134 UI tests passed. Evidence is local offline integration, not live external-provider/native-desktop acceptance. Direct MCP resources/prompts, same-connection capability changes, native plugin lifecycle and the remaining roadmap are still incomplete.

## Validate MCP settings before transport construction

Replaced the MCP file loader's unchecked object cast with a reusable typed validator. It validates booleans, transport/endpoint compatibility, string arguments and environment/header records, native timeout bounds, client metadata and capabilities; unknown settings fail visibly. In particular, the string "false" cannot enable a server accidentally. URL transports must be explicit and URL credentials are rejected in favor of headers. Invalid entries produce field-specific warnings while valid siblings still load. JSON read/parse failures no longer interpolate exceptions that can quote credential-bearing source text.

Parser regressions cover malformed fields, invalid endpoints, secret-free errors and valid disabled siblings. Shared startup tests additionally prove malformed entries never connect in either workspace-trust mode while valid configured servers still connect. The full root check/test/build and whitespace gate passed: 3,222 runtime tests passed, 2 skipped; 1,134 UI tests passed. The focused configuration/startup/real CLI fixture passed 26 tests before the final startup assertions, which are covered by the full gate. Editable MCP settings, transactional configuration replacement/rollback, native desktop/live-provider acceptance and the remaining roadmap are still incomplete.

## Staged live MCP configuration replacement

Added MCPManager.replaceServer for the upcoming settings host. It validates unknown input, admits one pending replacement per server and connects the candidate outside the operation queue while the old client remains callable. Installation rechecks both registration and client identity under the queue, waits behind earlier calls, replaces configuration and capability routing together, and then disconnects the old client. Invalid settings, connection errors, cancellation and intervening removal/replacement preserve current state and clean up unused candidates. Disabled replacements retire the live client without constructing another transport. Teardown failures remain redacted and observable while the successfully installed candidate stays registered. Active connect/disconnect hooks must still settle; cancellation does not abandon host-owned resources.

The older tool/resource/prompt routing methods now recheck client identity at execution rather than dispatching queued requests to a retired client. Regressions cover working calls during candidate discovery, malformed settings with no factory call, failed connection cleanup, successful swaps, disable, pre-cancellation, cancellation during connection, removal/newer registration races, single-candidate admission, teardown diagnostics and queued stale calls. The focused lifecycle/adapter/real CLI subprocess suite passed 27 tests; root type/catalog checks, build and whitespace checks passed. The full root test suite was not repeated for this isolated manager change. This API only replaces live state: persisted settings, transaction coordination with disk, RPC/TUI settings controls, native acceptance and the broader roadmap remain unfinished.

## Revision-checked MCP settings persistence and live commit coordination

Added an existing-user-file McpSettingsStore and replaceMcpSettings host coordinator. Reads capture the exact bytes used for SHA-256 revision checks and validate the complete document; invalid or duplicate entries prevent an editor from dropping data silently. Saves validate the replacement, preserve sibling configurations, normalize supported file shapes to a server list, enforce a 1 MiB limit, reject symlink/multiple-link targets and write a private fsynced temporary file before rename. An exclusive sibling lock coordinates cooperating editors, and the source revision is checked again before rename. Interrupted locks are not silently stolen. Arbitrary external writers can still race the final check/rename; this is not an exclusive filesystem transaction.

The manager now accepts a synchronous host commit callback after candidate discovery and identity validation, immediately before its live swap. Connection failure, cancellation, a stale file or an occupied settings lock leaves the old client installed. A successful file rename precedes live installation; a process interrupted between them reads the new file at startup. Directory-sync and cleanup failures after rename are returned as warnings rather than falsely reporting a failed commit and retaining old live settings. Credential-bearing snapshots are host data and must not be emitted as transcript events.

Focused persistence/lifecycle/configuration/shared-startup/real CLI tests passed 53 tests. Root type/catalog checks, build and whitespace checks passed. Tests cover sibling preservation, file permissions, invalid/stale/symlink inputs, candidate connection failure, cancellation, external edits during discovery and another writer's lock. Full root tests were not repeated for this store-focused change. This remains a native host API: daemon injection, RPC authorization/redacted settings projections, TUI controls, lock recovery UX, native acceptance and the full roadmap are not yet complete.

## Native MCP settings RPC and CLI host wiring

The normal CLI daemon now injects its user-file settings store. Additive mcp.settings.get exposes source/revision, enabled/transport/timeout values, configured-field presence and connection health while keeping launch strings, URLs, environment and headers write-only. mcp.settings.save applies a patch to an existing user entry, preserves omitted fields, supports explicit null removal and rejects rename/stale revisions. Candidate connection and guarded persistence precede live replacement; successful saves refresh the runtime inventory. A refresh failure is returned as a warning on the committed settings rather than pretending the save failed. Settings operations do not emit transcript events. These privileged controls use the existing trusted/authenticated daemon transport, not a model tool.

A pending save owns an AbortController attached to its client. Disconnect aborts that candidate before commit, while active connection settlement and cleanup still belong to the transport. Socket regressions verify credential/launch omission, failed connections preserving disk and live state, omitted credential preservation on disable, stale revision rejection and client-disconnect cancellation. The latter waits for observed server-side disconnect before releasing its gated candidate. An unknown-typed revision assertion in the fixture was corrected before the final gate.

Full root check/test/build and whitespace checks passed: 3,236 runtime tests passed, 2 skipped; 1,134 UI tests passed. TUI settings controls, new-server/file creation, project settings editing, lock recovery UX, live/native acceptance and the full roadmap remain unfinished.

## MCP settings terminal editor

/config mcp now opens a protocol-backed settings overlay for existing user entries, while /config retains agent settings. Native slash fallback returns the masked settings read model; completion offers agents/mcp and help documents both. The overlay joins background-key blocking and overlay restoration state. Server and field navigation expose enabled, transport, command, argument array, URL, timeout, environment and headers. Saved launch/auth values remain hidden. Empty input preserves a saved field; Delete removes it; environment/header replacements are explicit whole records. Failed saves keep the draft. Dirty drafts prevent server switching; F5 explicitly discards/reloads. Loading/saving keeps the modal open until settlement. Successful saves clear private draft values and use the returned revision for subsequent changes.

Renderer tests cover toggling/saving rejected drafts and closing at 220x65 and 40x18, command entry and successful saves. Slash routing proves opening does not submit a model prompt; daemon tests verify native fallback remains masked and argument completion works. Initial Escape assertions ran before save settlement and were corrected to wait for the idle footer. An invalid internal RPC ID type was corrected before the full gate.

Full root check/test/build and whitespace checks passed: 3,236 runtime tests passed, 2 skipped; 1,138 UI tests passed. The focused overlay/slash suite passed 40 tests. This is renderer and offline integration evidence, not native iTerm visual acceptance. New-server/file creation, project editing, stale-lock recovery controls and the remaining roadmap are incomplete. The editor is in the newly built application; existing running daemons were not interrupted or restarted.

## Create user MCP servers from the terminal

/config mcp now opens empty when the user configuration file is missing. F2 enters a unique server name, then uses the existing fields and guarded save flow. The RPC accepts explicit create: true; settings and live-manager names must both be absent, so creating a user entry cannot silently overwrite a project registration. The manager stages new candidates without publishing them and invalidates pending creation on removal/shutdown. One candidate per name remains enforced. File creation uses a private fsynced temporary inode and a no-overwrite link operation, preserving a concurrently created destination; normal updates retain revision checks and rename. Candidate failure leaves both the missing file and absent live registration unchanged. Parent-directory setup remains host-owned.

Tests cover first-file success/permissions/single-link cleanup, failed connection, shutdown during discovery, existing registration collision and duplicate persisted names. The daemon contract creates a second disabled entry, preserving its sibling. A keyboard renderer test creates a server from the empty editor and verifies the create payload and successful row selection. Existing settings and UI regressions continue to pass.

Full root check/test/build and whitespace checks passed: 3,240 runtime tests passed, 2 skipped; 1,139 UI tests passed. This is offline/renderer verification; native terminal/live external-server acceptance, project editing, automatic stale-lock recovery and the broader roadmap remain incomplete. Active user daemons were not restarted.

## Preserve completed monitor state when notification delivery fails

Restart tracing confirmed that RunHistory recovers interrupted records while TerminalRegistry does not recover live process handles. TerminalMonitors currently keeps watch definitions/subscriptions only in memory. Automatic reattachment requires a recoverable source/output mechanism and remains incomplete; no synthetic successful recovery was added.

The trace exposed an independent completion bug: after appending the durable completion event and finishing the run, an onEvent exception entered fail(), changed the in-memory watch to failed and cancelled an already-accepted reaction. RunHistory correctly refused rewriting terminal success, producing an additional persistence error. Completion and output notification paths now contain observer failures uniformly, report bounded deliveryError separately and preserve the committed execution state and reaction. Later successful output delivery clears the health warning. The TUI inspector renders notification health alongside execution/reaction errors without replacing evidence.

Regressions cover attachment before and after source exit with a callback that queues a reaction and then throws, asserting one durable event, successful run state and exactly one retained claim. Another test verifies output delivery health clears on subsequent success. A 220x65 renderer fixture shows completed status, delivery warning and event evidence together. The focused runtime suite passed 16 tests and monitor UI suite passed 7 tests. Full root check/test/build and whitespace checks passed: 3,243 runtime tests passed, 2 skipped; 1,140 UI tests passed. Automatic delivery retry, durable delivery-health recovery, native terminal acceptance and the broader roadmap remain incomplete.

## Native LSP framing foundation

Tracing semantic navigation confirmed the CLI has no native language-server host; ClaudeSearchTools only forwards to an injected LspAdapter. Added a Bun-compatible incremental Content-Length codec using the official LSP 3.17 base protocol as the reference (https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#baseProtocol). It counts UTF-8 bytes, bounds headers to 8 KiB and message bodies to 4 MiB, handles split and consecutive frames, rejects duplicate/malformed length fields, non-ASCII/control headers, unsupported charset declarations, invalid UTF-8/JSON and incomplete EOF. Failure and EOF close the decoder instead of attempting stream resynchronization. Invalid body errors omit source contents.

Fourteen focused Bun tests passed, including every split boundary of a Unicode frame, one-byte streaming of multiple messages, framing/body limits, charset aliases, malformed inputs and post-EOF rejection. Root type/catalog checks, build and whitespace checks passed. Full root tests were not repeated for this isolated, not-yet-wired codec. Subprocess lifecycle, request cancellation, initialization/capabilities, document versioning, diagnostic freshness, workspace scoping and CLI integration remain required before native LSP can be advertised as available. No language server was installed or launched.

## Owned native LSP stdio connection

Added LspConnection around a configured Bun subprocess and the bounded LSP codec. Requests have correlated IDs, capped pending calls, timeouts and AbortSignal cancellation; queued cancelled requests are skipped before writing, cancellation notifications are sent best-effort and late replies are ignored. Writes are serialized and bounded by both bytes and message count. Malformed envelopes/frames, process exit and broken pipes reject pending requests. Stderr is drained without echoing potential credentials. Unsupported server-to-client requests receive MethodNotFound rather than fabricated support. Initialization/capability handling still belongs to the future language-server host.

Close attempts shutdown/exit within a bounded grace period, then terminates the owned process and cancels stream readers. New work is refused while closing. Tests exposed that signal termination can leave Bun exitCode null while signalCode identifies completion; the liveness check now recognizes both. This manages the configured subprocess, not arbitrary detached grandchildren.

Real local Bun fake-server tests cover concurrent Unicode replies, cancellation/late responses, timeout recovery, crashes, malformed frames/envelopes, missing executable guidance, queue saturation/recovery and a process ignoring shutdown/SIGTERM. The combined framing/connection suite passed 22 tests; root type/catalog checks, build and whitespace checks passed. Full root tests were not repeated for this isolated, not-yet-wired module. No external language server was installed. Workspace scoping, initialize/capabilities, document versions, diagnostics freshness, tool registration and CLI lifecycle wiring remain unfinished.

## Workspace-scoped LSP host and versioned documents

Added LspHost implementing the existing adapter boundary over LspConnection. Startup canonicalizes the workspace, performs initialize/initialized, requests UTF-16 positions and rejects incompatible position encoding or unsupported open/change synchronization. Definition/reference/hover/document-symbol calls check advertised capabilities. WorkspacePathResolver rejects traversal and static symlink escapes before reads; bounded regular-file UTF-8 reads cap source documents at 1 MiB. This scopes client reads, not the privileges of the configured server executable.

The host serializes document operations with a bounded queue and prompt queued cancellation, opens documents once, sends full replacement changes on content updates and uses monotonically increasing versions. It caps open documents at 32, closes evicted documents and does not reuse versions on reopening. Navigation rechecks file contents after the response and rejects externally changed source. Version-matched diagnostics are distinguished from unconfirmed results; absent/stale versions are ignored and malformed diagnostics cannot become fresh. Diagnostic results retain only bounded messages and validated range/severity fields. Startup/connection failure closes owned resources.

Tests cover initialization/capabilities, document updates, root scoping, stale/malformed diagnostics, queue cancellation, eviction versions, external edits and pre-cancelled startup. A real Bun stdio fake server proves initialization, navigation and versioned diagnostic publication through the native transport. Its first assertion expected the noncanonical macOS /var path and was corrected to the actual realpath workspace URI. The combined LSP suites passed 33 tests; root type/catalog checks, build and whitespace checks passed. Full root tests were not repeated for this isolated, not-yet-CLI-wired host. Configuration/discovery, CLI lifecycle and tool registration, edit-diagnostics integration and opt-in real language-server acceptance remain unfinished.

## LSP configuration validation and workspace routing

Added strict user-owned executable configuration parsing and LspManager. Configurations declare unique server names, language IDs, dot-prefixed suffixes, explicit commands/arguments/environment, enabled status and bounded timeouts. Unknown fields, invalid executable values and ambiguous enabled suffix ownership are rejected without echoing configuration secrets. Longest matching suffix selects the server. Parsing copies caller-owned arrays/environment to keep the validated configuration stable.

The manager lazily shares initialization per canonical workspace/server, isolates distinct workspaces, resolves source paths inside their workspace before launch, and caps live or initializing hosts at 32. Individual request cancellation settles independently of shared startup; runtime shutdown aborts startup, closes even a late-arriving host, rejects new work and reports close failures. Failed initialization removes its cache entry so a later call can retry.

The combined LSP suites passed 39 tests. An additional strengthened longest-suffix assertion passed with all 6 manager tests. Root check, build and git diff --check passed. The first manager cancellation test run hung because its Bun rejection matcher was evaluated before releasing the fake startup gate; replacing it with a nonblocking rejection capture fixed the test sequencing. That specific leftover test process was identified and terminated, then the suites completed normally. Full root tests were not repeated for these isolated, not-yet-CLI-wired modules. User-file loading, CLI lifecycle/tool wiring, settings UI, edit-diagnostics integration and opt-in real-server acceptance remain incomplete; these changes do not yet expose LSP to normal users.

## Native LSP configuration and CLI tool integration

User-home lsp.json now loads at runtime construction through a bounded regular-file reader (no implicit project executable configuration). Missing files disable LSP; invalid UTF-8/JSON, unsafe file types, oversized files and schema errors fail explicitly. Configured servers remain lazy. The native tool schema lists supported actions, requires a file path and explains UTF-16 offsets and unconfirmed diagnostics. Built-in default/creator tool lists gain LSPTool only with enabled configuration; custom/restricted definitions are not expanded automatically.

Daemon runner rebuilds share their runtime-owned LspManager. One-shot and ACP own equivalent managers; resumed sessions inherit daemon-runtime wiring. Calls choose the active session workspace. Shutdown and failure cleanup close language servers, and daemon audit closure remains in a finally path even when server cleanup fails. User configuration changes currently require restart; MCP reload does not reload LSP configuration.

Offline integration tests use the normal CLI subprocess, a local fake provider and a real Bun stdio LSP fixture. They prove provider schema exposure and actual hover execution for daemon, one-shot and ACP, plus reuse after daemon runner reload and execution after resuming a persisted session. Configuration tests cover missing files, no project auto-loading, malformed/oversized input, symlink rejection and lazy startup. The full root gate completed: 3,286 runtime tests passed, 2 skipped; 1,140 UI tests passed; root check, build and git diff --check passed. Evidence: /tmp/xerxes-lsp-runtime-{check,test,build}.log. No external provider or installed language server was exercised. Automatic edit-diagnostics integration, TUI LSP configuration/health and opt-in real-server acceptance remain unfinished.

## LSP diagnostic publication wait and source freshness

Before feeding native diagnostics into automatic edit reports, inspection found two correctness gaps: an immediate diagnostic request could precede the server publication, and its return path lacked the navigation path's concurrent-file-change check. LspHost now supports a bounded internal diagnostics wait (0–5000 ms); the configured native tool waits up to one second. Matching-version publication wakes the waiter, while stale/unversioned notifications do not. Timeout remains fresh:false; malformed publications remain unconfirmed. Cancellation and host shutdown release the wait; file contents and connection/cancellation state are rechecked before returning diagnostics. Source files open nonblocking before the regular-file check so named pipes cannot hang waiting for writers.

The six focused LSP/configuration/CLI suites passed 47 tests, followed by root check and build. A subsequently added real named-pipe regression passed together with all 16 host tests. git diff --check passed. Evidence: /tmp/xerxes-lsp-diagnostics-final-{tests,check,build}.log and /tmp/xerxes-lsp-fifo-tests.log. Full root tests were not repeated for this isolated host refinement; the prior full gate remains the last repository-wide test run. Automatic edit feedback and TUI LSP settings/health are still pending.

## Turn-owned semantic edit feedback

Inspection found that noteEditedPath/noteFileWillChange had no production callers and the daemon turn runner used workspace-global checker state. Added EditFeedback: each daemon turn owns its collector, wraps native file-writing tool execution, awaits the checker and bounded LSP baseline before mutation, and collects post-edit diagnostics only after successful execution. Fresh semantic findings are differenced against a fresh baseline; an unconfirmed baseline yields explicitly labelled current findings. Server errors and missing publications remain unconfirmed and do not suppress the independent checker result. Reports cap files/text and are appended through the existing persisted-conversation feedback path. Read-only turns no longer launch baseline checker work through this runner.

The first typecheck caught an incorrect assumption that internal tool arguments were JSON strings; the executor boundary already receives JsonObject. Corrected both implementation and fixture before running validation. Focused feedback/checker/turn-runner tests passed 55 tests. A separate real AgentTurnRunner test then proved baseline-before-write ordering and persisted semantic feedback (1 pass); it was added after the root test inventory had started and is not included in the full count below. Full root gate passed 3,296 runtime tests, 2 skips, and 1,140 UI tests; checks, build and git diff --check passed. Logs: /tmp/xerxes-edit-feedback-{tests-final,check-final,full-test,build}.log and /tmp/xerxes-edit-feedback-runner-test.log.

Automatic feedback is wired for daemon/resumed-session execution. One-shot/ACP retain explicit LSPTool but still need automatic feedback integration. Native terminal presentation/acceptance, LSP settings/health, and the rest of the broader roadmap remain incomplete. The collector does not claim it can attribute another concurrent external writer's changes to a particular turn or detect arbitrary shell mutations.

## Automatic edit feedback across ACP and one-shot entry points

ACP now creates a turn-owned EditFeedback collector, wraps its tool executor, persists the report and emits it through existing text_delta events before turn_end. The ACP tool loop now runs inside withActiveSession, so tools using the active-workspace resolver honor the ACP session cwd rather than the process startup workspace. One-shot execution wraps native edits and includes diagnostic feedback in text output, JSON response text, or stream-json text records before the terminal result. XERXES_EDIT_DIAGNOSTICS=0 disables automatic feedback in these paths; explicit LSPTool remains available.

The first ACP end-to-end feedback test exposed a macOS alias mismatch: the collector supplied absolute /var source paths to the canonical /private/var language-server root. The collector now supplies workspace-relative paths and retains host containment validation. Temporary local debugging output was removed before final verification. A local fake provider and real stdio fixture verify semantic findings after WriteFile, all three one-shot formats, and ACP event ordering. The ACP test opens a different workspace and verifies the write exists there and not at startup cwd.

Full root gate passed: 3,301 runtime tests, 2 skips; 1,140 UI tests; root check, build and git diff --check passed. Evidence: /tmp/xerxes-feedback-entry-{check,test,build}.log. These remain deterministic local-fixture tests, not installed-language-server or native-terminal acceptance. LSP settings/health UI, real-server opt-in acceptance, and the wider roadmap remain incomplete.

## Workspace-specific LSP health model

LspHost now exposes its live connected state. LspManager.health canonicalizes the requested workspace and reports disabled, idle, starting, ready, failed or closed per configured server without launching anything. Failed initialization remains visible in a bounded 32-entry history after its host entry is removed; retry clears that failure, and a disconnected initialized host reports failed. Responses contain server name, language ID, suffixes and fixed actionable failure text, never executable arguments/environment or raw server errors.

Regression coverage verifies lazy health reads, startup transition, failure visibility, retry, post-initialization disconnect, workspace isolation, shutdown and secret omission. The focused host/manager suites passed 23 tests; root check, build and git diff --check passed. Evidence: /tmp/xerxes-lsp-health-{tests,check,build}.log. Full root tests were not repeated for this isolated read-only health model. Daemon RPC and TUI wiring are still pending; this API is not yet a user-facing status screen. Disconnected hosts currently require runtime restart; no automatic replay/restart is claimed.

## User-facing LSP health inspection

Added lsp.status to the daemon RPC surface and /lsp [status] to the shared command catalog, daemon handler and TUI slash registry. The runtime resolves health through the selected session's cwd and its shared LspManager. Unavailable host integration and missing/unreadable workspace state produce explicit errors. Slash output lists server/language/suffix/state and fixed failure guidance using existing transcript output; it neither starts a server nor replaces the conversation. Protocol and configuration documentation describe the read-only contract.

CLI integration asserts idle status before the first tool call, ready status afterward, successful slash dispatch and rejection of unsupported arguments. A UI command test verifies active-session forwarding without transcript replacement. The first full gate found the expected command-count snapshot still at 76; updating it to 77 fixed the only runtime failure. The corrected complete gate passed 3,302 runtime tests, 2 skips; 1,141 UI tests; root check, build and git diff --check passed. Logs: /tmp/xerxes-lsp-status-verified-{check,test,build}.log. This is a health command, not the planned server configuration editor. Reconnect/settings controls and opt-in installed-server/native-terminal acceptance remain unfinished, as does the wider roadmap.

## Workspace/server release lifecycle

Added LspManager.release(root, name) as the lifecycle foundation for recovery controls. It cancels initialization, closes only the selected workspace/server, shares concurrent release calls, and leaves the entry present as stopping until cleanup completes. Requests cannot launch a replacement during cleanup or after a cleanup failure. A failed release retains failed health and permits a subsequent explicit cleanup retry; successful release returns to lazy idle state. Pending requests cannot execute against a late-arriving released host. Final manager shutdown waits for release already in progress and retries failed cleanup rather than racing an independent close.

The manager suite passed 10 tests, including cross-workspace isolation, blocking replacement during cleanup, cleanup failure/retry and release during initialization. Root check, build and git diff --check passed. Evidence: /tmp/xerxes-lsp-release-{tests,check,build}.log. Full root tests were not repeated for this manager-only lifecycle addition. The method is not yet exposed as a user-facing release/reconnect command; that wiring and server settings editing remain pending. The wire health state list now includes stopping.

## LSP release control exposed through daemon and TUI

Added /lsp release <name> and lsp.release RPC, wired through the runtime's selected-session workspace to LspManager.release. Unknown/empty names and unavailable hosts fail explicitly; cleanup errors return fixed guidance. The existing transcript output confirms successful release. The command does not change configuration, replay requests or eagerly launch another process. Concurrent requests sharing that workspace/server may be interrupted; other workspace hosts remain isolated.

The normal CLI fixture now verifies invalid release requests, slash release of a ready host, idle health after cleanup, idempotent RPC release and successful lazy reinitialization on a subsequent provider turn. UI forwarding preserves the selected session and does not replace transcript history. Full gate passed 3,305 runtime tests, 2 skips; 1,142 UI tests; root check, build and git diff --check passed. Logs: /tmp/xerxes-release-ui-{check,test,build}.log. LSP configuration editing and installed-server/native-terminal acceptance remain incomplete, along with the wider roadmap.

## LSP settings persistence foundation

Added host-only LspSettingsStore with validated complete-document saves, SHA-256 revisions, exclusive private edit locks, bounded regular-file reads, symlink/hardlink rejection, private temporary writes with fsync, initial-file no-overwrite linking and replacement rename. Stale revisions and invalid schemas cannot overwrite configuration. Post-commit directory synchronization or cleanup problems are warnings on the committed snapshot; interrupted locks are never stolen. This coordinates cooperating editors and detects external changes at precommit checks, but does not claim exclusion of arbitrary external writers between the final check and replacement.

Startup loadConfiguredLsp now uses the same reader, avoiding different validation between loading and editing. Snapshots contain host executable settings and must not be exposed raw to RPC/transcript clients. Focused settings, configuration and normal CLI suites passed 6 tests, root check/build and git diff --check passed. Evidence: /tmp/xerxes-lsp-settings-final-{tests,check,build}.log. Full root tests were not repeated for this persistence foundation. Save RPC, masked settings responses, live-manager replacement and TUI editing remain unfinished; this store alone does not expose editing to users.

## Masked LSP settings view and partial-edit validation

Added a client-safe settings view exposing revision, server identity/language/suffixes, enabled status, timeout and the presence of configured command/args/env fields. Saved executable commands, arguments and environment keys/values are omitted. Added pure preparation for create/update/remove requests: omitted fields preserve existing configuration, null clears optional args/env/timeout, names cannot be changed through updates, and the complete candidate document is validated for suffix ownership before persistence or host changes. Stale revisions and mismatched create/remove targets fail explicitly.

Focused settings view/store tests passed 5 tests; root check, build and git diff --check passed. A final strict string-action guard was checked with the 3 view tests again. Logs: /tmp/xerxes-lsp-settings-view-{tests,check,build}.log and /tmp/xerxes-lsp-settings-view-action-tests.log. Full root tests were not repeated for this isolated preparation layer. RPC save wiring, live host reconfiguration and the actual TUI editor remain incomplete; no raw snapshots are exposed to clients by this work.

## Live LSP reconfiguration and settings apply coordinator

LspManager.reconfigure validates a complete candidate, serializes settings changes, stops hosts whose configuration changed and retains unchanged hosts. New requests are rejected with retry guidance during the transition. A synchronous commit callback persists settings before publishing the new configuration; persistence failure keeps the previous configuration, with affected hosts available for lazy restart. Cancellation and manager shutdown prevent commit. Cleanup failure blocks save/replacement. Internal release uses stored canonical host keys so cleanup is not dependent on a workspace directory still existing.

saveLspSettings combines the validated partial-edit preparation, revisioned store save and live reconfiguration, returning only masked data. It does not eagerly start servers on save. Tests cover close-before-commit ordering, retention of unchanged hosts, persistence failure, cancellation, concurrent saves/requests, disk/live update and credential omission. The combined settings/manager suites passed 19 tests; root check, build and git diff --check passed. A subsequently added cleanup-failure regression passed with all 14 manager tests. Logs: /tmp/xerxes-lsp-apply-{tests,check,build}.log and /tmp/xerxes-lsp-reconfigure-failure-tests.log. Full root tests were not repeated for this backend-only coordinator. RPC/UI save and tool-inventory refresh wiring remain unfinished; these APIs do not yet constitute a user-facing editor.

## LSP settings RPC and live tool-inventory refresh

Wired lsp.settings.get/save through the daemon runtime to the shared LspSettingsStore/LspManager coordinator. Reads return masked settings. Save requests validate explicit name/revision/action/changes, cancel on connection closure before commit, persist/apply the candidate and refresh the native tool inventory. An inventory-refresh failure after commit becomes a warning on success rather than pretending disk/live settings were unchanged. RPC failures use fixed actionable text without raw configuration or server output. These operations require an active session and are not model-callable tools.

The normal daemon CLI fixture now begins with its only LSP server disabled and proves LSPTool is absent from inventory. A settings RPC enables it, refreshes inventory and makes it executable in the next provider turn. The fixture checks secret/argument omission and stale revision rejection; normal one-shot/ACP LSP tests remain passing. Full root gate passed 3,315 runtime tests, 2 skips; 1,142 UI tests; root check/build and git diff --check passed. Evidence: /tmp/xerxes-lsp-save-final-{check,test,build}.log. Actual /config LSP editor wiring and native-terminal/installed-server acceptance are still pending; backend RPC support is not the completed settings UI.

## Native /config lsp settings editor

Added LspSettingsOverlay and registered its overlay state, input blocking/restoration, rendering and /config lsp entry point. Shared command metadata, daemon completion and native config dispatch include lsp. The editor supports enabled/language/suffix/command/args/timeout/env fields, create, partial update, optional-field clearing and Y/N-confirmed removal. Successful saves use the server-returned rows/revision; failed saves retain the draft. Saved launch values remain hidden, and no model message is sent to open the editor.

Renderer tests exercise 220x65 and 40x18, failed-draft retention/close, create via keyboard, confirmed removal and revision reuse. The initial small-screen test exposed explanation text consuming the viewport; compact mode now shows the selected field directly. Daemon CLI tests cover /config lsp completion/dispatch alongside actual settings-save execution. Full gate passed 3,315 runtime tests, 2 skips; 1,147 UI tests; root check, build and git diff --check passed. Logs: /tmp/xerxes-lsp-editor-{check,test,build}.log. A final help-label wording correction was made after the gate. Native iTerm and installed-language-server acceptance remain unverified; the broader roadmap remains active.

## Installed clangd acceptance

Detected existing /usr/bin/clangd and exercised it without installing or downloading a server. The initial native-host probe returned hover, definition, references and symbols for C++ code, fresh clean diagnostics at version 1 and the expected undeclared-symbol error at version 2. Added lspRealServer.test.ts with explicit XERXES_TEST_CLANGD opt-in. It also verifies fresh clean version 3 after correction, automatic EditFeedback reporting a newly introduced semantic error, closed-host state and rejection after shutdown. Temporary workspaces and owned hosts are cleaned up.

Acceptance passed against Apple clangd 21.0.0 (clang-2100.1.1.101), arm64 macOS: 1 test, 12 assertions. The unset opt-in run skipped the test as intended. Root check and git diff --check passed; runtime implementation was unchanged, so the full gate/build were not repeated. Logs: /tmp/xerxes-clangd-acceptance-final.log, /tmp/xerxes-clangd-optout.log, /tmp/xerxes-clangd-version.log and /tmp/xerxes-clangd-check.log. This proves one installed server's host/diagnostic path, not universal language-server compatibility or native iTerm UI acceptance. The wider roadmap remains incomplete.

## Run-owner start-time recovery

RunHistory now migrates and records an optional owner_start value alongside the PID and command. Recovery marks unfinished records interrupted when the observed OS start identity differs, even when the command is identical; output and unread completion evidence survive. Matching or unreadable start identities preserve conservative recovery, and legacy rows retain command/liveness behavior. Probes validate positive integer PIDs, use bounded direct argv execution, normalize POSIX locale/timezone, and return unknown on inspection failure. This does not authorize signals against recovered PIDs or restore process-local handles. POSIX ps start times have second precision, so same-second PID reuse remains unresolved; Windows probe behavior has not been exercised on Windows.

Focused verification passed 19 tests / 77 assertions, including same-command replacement, unchanged/unknown identities, retained output and invalid PID inputs. The local macOS self-process probe returned a stable start identity. Full root check, test and build passed: 3,320 runtime tests, 3 skipped; 1,147 UI tests. git diff --check passed. Logs: /tmp/xerxes-run-start-{check,full-test,build,focused}.log. Runtime build a67d334d68c79faa. The full roadmap remains active.

## Background-command completion follow-up wiring

exec_command now accepts notify_on_completion in direct human turns. The daemon supplies a completion adapter through CoreToolsOptions and registerProcessTools; after explicit background launch or timeout adoption it creates one completion watch and one bounded reaction grant (24-hour expiry, one reaction, 60-second deadline). The existing event-driven dispatcher waits behind the current turn and delivers retained, explicitly untrusted evidence to the owning session. No polling or separate monitor tool call is needed. Foreground completions and default opted-out calls do not schedule reactions. Background-origin calls are denied before spawning, preserving the existing monitor authority boundary.

Unsupported hosts reject before starting a command. If watch admission fails after launch, the result preserves procId and reports completion_watch_error, avoiding a misleading command failure that could trigger duplicate work. The configuration guide documents cancellation, expiry and restart limits. This reuses process-local watches; restart reconciliation and token budgets remain incomplete.

Tests cover real command completion, timeout adoption, owner isolation, one durable claim, unsupported-host and non-human rejection, admission failure with a retained handle, foreground completion and opt-out. A normal CLI daemon with an offline local provider produced the command turn and a second completion turn containing the expected output, without a separate watch call. Full root check/test/build and git diff --check passed: 3,325 runtime tests, 3 skipped; 1,147 UI tests. Logs /tmp/xerxes-command-completion-{check,test,build}.log and /tmp/xerxes-cli-command-completion.log; build 88b01790209150d1. Full roadmap remains active.

## Reaction dispatcher completion handoff

Restart-path inspection exposed a live-delivery race in ReactionDispatcher: an event offered after an empty claim check could join the existing dispatch promise without causing another admission pass. A second window existed between the async drain returning and its separate promise.finally clearing the active map. Both could leave durable evidence pending until another external wake or reconciliation.

Repeated dispatch now records a pending wake and the drain rechecks after admission. Active ownership is released synchronously in the drain's own finally, closing the settlement microtask gap. This does not bypass mailbox claims, session admission, cancellation, expiry or uncertain-executor fencing. A regression test failed against the prior implementation (no reaction observed), then passed after the fix. A separate microtask-order test proves a fresh drain after settlement. Existing cancellation, coalescing, per-watch failure and normal CLI completion tests remain passing.

Full root check/test/build and git diff --check passed: 3,327 runtime tests, 3 skipped; 1,147 UI tests. Logs /tmp/xerxes-reaction-handoff-{before,focused,microtask,check,test,build}.log. Build d70d4f3f904f9582. Recovery of vanished source watches remains incomplete; this repair addresses missed wake-ups in a live daemon. Full roadmap remains active.

## Completion reaction capacity release

Repeated-use review found that ReactionMailbox.configure counted exhausted, settled grants until their expiry. Consequently 16 completed one-shot background follow-ups could prevent further watches for up to 24 hours. Admission now counts unspent eligible grants plus any grant with an unresolved executor, including expired/cancelled executors awaiting cleanup. Finished outcomes and usage remain stored; no migration or deletion is needed.

Seven regression cases failed before the fix, then passed: 20 sequential completed/failed/cancelled/interrupted one-shot reactions retain history without consuming capacity, while live/cancelled/expired unresolved executors retain their slot until settlement. Focused mailbox/dispatcher/command/normal-daemon checks passed 33 tests and 228 assertions. Full root check/test/build and git diff --check passed: 3,334 runtime tests, 3 skipped; 1,147 UI tests. Logs /tmp/xerxes-reaction-capacity-{before,focused,check,test,build}.log; build 1e8f07f84cf372dc. Configuration guide documents capacity behavior. The broader roadmap remains active.

## Persistent monitor inspector records

New terminal watches atomically persist their trigger, match and expiry alongside the run in monitor_configurations. The inspector merges attached watches with up to 100 stored watch records; separate reaction execution runs have no watch configuration and are excluded. A stored record is shown as interrupted, archived or detached, with sourceStatus explaining its stored outcome and lack of a local process attachment. Successful stored outcomes are explanatory text, not failure warnings. The latest 20 durable events are restored with the omitted-event count; earlier evidence remains accessible through run.events. Owner-scoped lookups reject other-session access.

Stopping an archived/interrupted entry revokes any remaining reaction grant without rewriting its run outcome. Detached entries may belong to another live daemon and reject stop requests through this host. No process or stream handles are reconstructed. History-read failure retains available in-memory health with an explicit archive-unavailable error, rather than dropping live entries or inventing an empty archive. Older runs lacking configuration remain accessible in Runs. Protocol and configuration docs describe these boundaries.

Tests reopen the durable history under simulated owner death and verify stored configuration, last-20 evidence, omitted count, reaction cancellation, owner isolation and preservation of source control. Another-host test verifies detached state and stop refusal. Renderer tests accept all three stored states and exclude them from the watching count. Full root check/test/build and git diff --check passed: 3,336 runtime tests, 3 skipped; 1,150 UI tests. A final root check after the source-status presentation change also passed. Logs /tmp/xerxes-monitor-archive-{focused,restart,check,final-check,test,build}.log; build a22afd043888c324. Native terminal acceptance and actual process reattachment remain unproven; the full roadmap remains active.

## State-aware monitor controls

Monitor summaries now expose stopAction: stop-watch for attached watchers, cancel-reactions for settled/interrupted entries with waiting or unresolved reaction work, and null for detached or fully settled entries. The TUI labels and enables S from that capability, uses the selected row while detail loads, and preserves cancellation/new-watch discoverability at 40 columns. Older responses without the capability retain a conservative live-watch fallback. Unknown action values are rejected. The slash response reports the current stored watch state and reaction revocation rather than claiming a historical execution was stopped. Protocol and configuration docs are aligned.

Runtime archive tests now verify action changes after cancellation and refusal for another daemon's watch. Daemon contract tests verify list/stop capabilities. Renderer tests exercise enabled cancellation, disabled settled/detached actions and the narrow footer. Full root check/test/build and git diff --check passed: 3,336 runtime tests, 3 skipped; 1,154 UI tests. Logs /tmp/xerxes-monitor-controls-{focused,ui,check,test,build}.log; build eaa44d2f25ff65c2. An initial optional-property type error was corrected before the passing gate. The full roadmap remains active.

## Nested model-budget enforcement

Budget review found that withModelCallBudget replaced the active scope. Nested work could therefore stop charging its parent and bypass an enclosing call limit. Scopes now compose, preflight all enclosing admissions before changing counters, deduplicate reentry of the same budget, and fan usage receipts out to every scope even if one checkpoint fails. Partial admission-checkpoint failure settles earlier admitted scopes conservatively rather than stranding their pending counters. Optional work and final assertions inspect every enclosing scope.

Separately admitted schedule attempts and monitor reactions explicitly use withIndependentModelCallBudget, so asynchronous notification context from an ended run does not block a new authorized host run. This distinction is host-owned; nested work within a run continues to inherit its parents. The configuration guide explains the boundary. This is call-limit enforcement and usage accounting, not a total token or monetary cap.

The parent-limit regression failed before the change. Focused tests cover provider-call counts, parent/child usage, child denial without parent consumption, same-budget reentry, checkpoint failure fan-out, earlier-scope settlement, closed parents, and independent host runs. A first full gate encountered a mixed-module import error because the new export was added after that process loaded the previous module; a fresh full gate without concurrent code edits passed. Final root check/test/build and git diff --check passed: 3,343 runtime tests, 3 skipped; 1,154 UI tests. Logs /tmp/xerxes-nested-budget-{before,final-focused,final-check,final-test,final-build}.log; build eb624887efaef440. Whole-reaction token budgets and the full roadmap remain unfinished.

## Read-only context inspector

/context now opens a restorable TUI inspector for instructions, memory layers, retained conversation and tool schemas; /usage retains its usage/subscription behavior. The turn runner captures named assembly segments and a timestamp with its existing transient request scaffold. context.inspect reads only the connection's active session, returns labelled approximate contributions and provenance, and distinguishes missing scaffold data from empty sections. It makes no provider calls. Named memory layers are identified from actual assembly names, not inferred from transcript text.

Pages contain at most 20 entries with 8000-character excerpts. A content fingerprint covering session/model/scaffold and transcript rejects stale paging, including equal-length transcript edits. Excerpts are rendered only for the selected page. The inspector supports section navigation, paging, scrolling, refresh and Escape; failed requests preserve the last displayed page. Notes explicitly distinguish latest scaffold and retained transcript from the exact live provider payload, and token estimates from billing. Individual retrieval source files/scores, pinning, exclusion, selected-turn branching and compaction-history controls remain open.

Focused tests cover unavailable scaffold, memory separation, bounded pages/excerpts and stale generations. Daemon contract tests verify active-session isolation, read-only slash behavior and stale rejection; the normal CLI completion fixture verifies a real assembled bootstrap layer appears. Renderer tests cover 220x65, 40x18, stale-page preservation and restoration; slash handler opens the overlay without a model turn. Initial implementation/type and asynchronous Escape assertion errors were corrected before final validation. Full root check/test/build and git diff --check passed: 3,346 runtime tests, 3 skipped; 1,158 UI tests. Logs /tmp/xerxes-context-inspector-{focused,contract,ui,final-check,final-test,final-build}.log; build 1625afd6f4e951f4. Full roadmap remains active.

## Bounded MCP configuration reads

Reproduced a settings-reader hang when mcp.json is a FIFO with no writer, and silent replacement of invalid UTF-8 bytes inside a configured command. Both regression cases failed before repair. Added a shared MCP document reader used by startup and the settings store: nonblocking open, regular-file validation, an actual byte-read cap of 1 MiB plus one overflow byte, and strict UTF-8 decoding. The actual read remains bounded if another writer grows the file after stat. Startup keeps its existing linked-file support and warning-based failure behavior; editing still rejects symlinks and multiple hard links. Missing files remain empty configuration. No transport is launched from malformed bytes.

Focused configuration/settings and normal daemon/one-shot/ACP fixtures passed 39 tests. Added explicit startup/editor FIFO subprocess cases with bounded cleanup, exact-limit/over-limit and invalid-encoding checks, and a linked-file compatibility test. Final root check/test/build and git diff --check passed: 3,350 runtime tests, 3 skipped; 1,158 UI tests. Logs /tmp/xerxes-mcp-read-before.log, /tmp/xerxes-mcp-read-final-{focused,check,test,build}.log; build 7bb4b745971db9e7. FIFO acceptance is POSIX-only; Windows behavior is not verified here. The full roadmap remains active and incomplete.

## Persisted compaction history in /context

Main sessions and delegated children now retain the latest 100 successful compaction stamps in metadata while preserving last_compaction. The bounded validator reads legacy stamps without inventing older history and strips unknown fields. /context adds a compaction section with newest-first pages, timestamp/reason, message reduction, token estimates and recorded archive location. History is explicitly outside model context, contributes zero context tokens, and participates in stale-generation detection. The view does not read or verify archive paths and does not claim restoration support.

The first full gate exposed that subagent conversation saves regenerated metadata and dropped compaction stamps after the initial update. The save path now carries validated history into the persisted transcript. Its regression checks a completed child turn's loaded transcript; main daemon tests check the actual session JSON and context.inspect response. Additional tests exercise legacy metadata, retention, invalid persisted values, reload, generation invalidation and keyboard section wrap at 220x65 and 40x18. The initial renderer test used the wrong arrow input helper and was corrected before passing.

Final root check/test/build and git diff --check passed: 3,351 runtime tests, 3 skipped; 1,160 UI tests. Logs /tmp/xerxes-compaction-history-{focused,child,ui,verified-check,verified-test,verified-build}.log; build b128653c3a9d9293. Native terminal acceptance, archive restoration, pin/exclude controls and selected-turn branching remain unverified or unfinished. Full roadmap remains active.

## Context page recovery controls

Reproduced a stale-section cursor bug: after navigation failed, the retained previous page supplied next_offset for the newly selected section. Next/Previous now require a successful page matching both requested section and offset. Failed pages remain readable, section navigation remains available, and Refresh resets the cursor/generation. A response with the wrong section or offset is rejected without replacing valid content.

The regression failed before the repair and passed afterward; an additional renderer test verifies mismatched-response rejection. UI check, all 1,162 UI tests, UI build and git diff --check passed. Logs /tmp/xerxes-context-cursor-{before,focused,check,test,build}.log. Runtime code was unchanged, so runtime tests were not repeated. Full roadmap remains active.

## Durable lifetime schedule attempt limits

Added max_runs (1–10000 or null) and host-owned runs_started to persisted CronJob records, daemon create/update/list payloads, manage_schedule and the /schedules form/inspector. Omitted edits preserve the limit; edits and resume do not reset admissions. Existing records default to zero because prior attempts cannot be reconstructed. Manual and scheduled work share the same admission path, which saves the increment before invoking work. Failed persistence prevents execution; exhausted jobs reject manual admission and pause on scheduled eligibility checks. Raising or removing the limit permits further admissions without erasing used attempts. Failed, cancelled and retried admissions consume capacity.

Runtime tests cover successful/failed execution, mixed manual/scheduled runs, restart, limit exhaustion, increasing the limit and failed counter persistence. Daemon contracts verify create payloads; renderer tests edit and clear lifetime and model-call limits. Full root check/test/build and git diff --check passed: 3,354 runtime tests, 3 skipped; 1,163 UI tests. A subsequent focused cancellation regression passed with all 4 lifetime-limit tests; runtime code was unchanged after the full gate. Logs /tmp/xerxes-schedule-count-{focused,contract,ui,final-check,final-test,final-build,cancel}.log; build e76d57384bed691b. Full session-loop routing, idle wake-up admission, expiry, stop-condition evaluation and total token budgets remain unfinished. Full roadmap remains active.

## Persisted schedule admission expiry

Added expires_at to CronJob persistence, schedule create/update and payloads, manage_schedule, and the TUI form/inspector. Explicit timezone-bearing timestamps use the existing strict calendar parser and normalize to UTC. Omitted edits preserve expiry; null removes it. New values must be future instants, but past persisted values remain inspectable. Automatic eligibility and manual admission reject at or after the cutoff without incrementing attempts; automatic checks pause expired jobs. Restart cannot extend expiry. Already-admitted work retains its own timeout/cancellation controls.

Tests cover the exact subsecond cutoff, restart, manual refusal, clearing expiry, preserved attempts and invalid calendar/typed input. Daemon contracts verify normalized create values and preservation on edit. Renderer tests set and clear expiry through keyboard input; existing exact-payload expectations were updated. Full root check/test/build and git diff --check passed: 3,357 runtime tests, 3 skipped; 1,164 UI tests. Logs /tmp/xerxes-schedule-expiry-{focused,ui,final-check,final-test,final-build}.log; build 45b2d9ab8f0ce829. Same-session routing, stop-condition evaluation and total token budgets remain incomplete; this is admission expiry, not an execution kill deadline. Full roadmap remains active.

## Session-targeted scheduled follow-ups

Schedule create/update and manage_schedule now accept target=session or independent. Session binding comes from the authenticated active session ID, persists as target_session_id, and requires max_runs plus expires_at. Omitted edits retain the binding; explicit session mode on an already-bound schedule does not retarget it from another tab. /schedules exposes the target selector and inspector identity. Follow-ups use the existing per-session operation queue, recheck expiry after waiting, and pass cancellation through the shared scheduler. Bound events stream to clients viewing that conversation without duplicate manual-call emission.

Added expectedSessionId to the internal session-open options: existing or restored identity must match, and missing saved history cannot silently create a replacement chat. Restart reloads a saved target by its persistent ID. The restart test first exposed an incomplete fixture (only an unanswered prompt, intentionally not saved), then a genuine alias comparison failure between macOS temporary paths and canonical workspace paths. Runtime resume now compares canonical workspace directories while preserving different-workspace rejection.

Daemon tests verify bound creation/required limits, serialized follow-ups, cancellation while queued, missing/replaced identity refusal and successful restart with prior conversation context retained. Renderer tests exercise target selection and preserve limit/expiry editing. Final full root check/test/build and git diff --check passed: 3,360 runtime tests, 3 skipped; 1,165 UI tests. Logs /tmp/xerxes-session-schedule-{contract,restart,restart-fixed,ui,complete-check,complete-test,complete-build}.log; build 2826f1450a079ed0. Dedicated /loop, automatic stop-condition evaluation, total token budgets, goal-adjacent status and broader roadmap acceptance remain unfinished. Full roadmap remains active.

## Prevent model-tool self-wait on session follow-ups

The native CLI schedule adapter now marks requests originating from model tools. An immediate run targeting that tool's own active conversation (by persisted ID or legacy workspace/session key) fails before scheduler admission, with guidance to schedule a future wake-up or use external controls after the turn finishes. The origin marker is host-only and cannot be supplied as model arguments. External controls retain queue behavior. This prevents the parent turn waiting for a follow-up that cannot start until that same parent returns.

A daemon regression invokes the host from inside an active runner, verifies an actionable error, successful parent completion and zero consumed attempts. Existing queue/cancellation/restart contracts remain passing. Full root check/test/build and git diff --check passed: 3,361 runtime tests, 3 skipped; 1,165 UI tests. Logs /tmp/xerxes-followup-self-wait-{focused,check,test,build}.log; build 408359f13b2325f9. The broader roadmap remains active, including /loop and stop-condition evaluation.

## Conversation follow-up controls through /loop

Added /loop as a canonical command in the daemon and TUI. Its overlay scopes list and actions to the current conversation, and new drafts start paused with a ten-minute interval, ten-attempt cap and 24-hour expiry. The prompt remains user-supplied; saving does not enable execution. Existing /schedules controls remain available. Server-side scope checks reject attempts to operate another conversation's job. Keyboard tests cover the overlay, new draft, and cancellation; daemon tests cover scoping and command actions.

The full root check/test/build and git diff --check passed: 3,362 runtime tests, 3 skipped; 1,169 UI tests. Logs /tmp/xerxes-loop-command-verified-{check,test,build}.log; build 93acb00cc0492314. Automatic stop-condition evaluation and total token budgets remain unfinished. This records completion of the bounded /loop entry point, not the full roadmap.

## Live goal synchronization after model edits

Fixed the stale goal objective shown in the header and Tasks card while update_goal returned a different objective. Turn-local metadata previously copied the immutable goal change log, hiding updates from the live session until turn completion and risking overwriting an intervening human edit. The turn now shares the authoritative goal-log property with the session, keeping compare-and-set reads and writes current. At tool/status/turn boundaries the daemon publishes changed objective/phase values through status_update; session.goal also notifies attached clients after persistence. The adapter preserves omitted fields on telemetry-only frames and clears the display on explicit null.

Regression tests cover a real goal tool changing live metadata before the next inference, a subsequent human edit surviving provider failure, socket delivery during a still-active turn, clear during that turn, and UI replacement/clear/telemetry preservation. Existing layout tests now replace a visible goal at 240x64, 150x40, 80x24 and 40x16 and assert the old objective disappears. Full root check/test/build and git diff --check passed: 3,364 runtime tests, 3 skipped; 1,170 UI tests. Logs /tmp/xerxes-goal-live-{focused,socket,ui,final-check,final-test,final-build}.log. The running user's daemon was not interrupted; rebuilt code requires a daemon/TUI restart. This does not claim native terminal acceptance or completion of the broader feature roadmap.

## Follow-up status beside the goal

The session header now shows the next eligible follow-up wake, attempts used/remaining, and the latest recorded attempt state. The F10 inspector includes populated follow-up details and L opens the existing /loop controls for pause, cancellation and output history. Active/cancelling work takes precedence over future eligibility; paused, expired, exhausted and recovery-required jobs do not promise a next wake. Status polling uses the existing scoped schedule RPC, invokes no model, does not overlap requests, and discards late responses after a conversation switch. Failed reads remove stale wake promises.

Scoped schedule requests now support an owner_session_id assertion and list responses identify their owner. Mismatches fail before listing or acting, preventing delayed controls from applying in a different conversation. The /loop overlay passes the assertion when the TUI has an active session ID. RunHistory exposes an indexed latest-outcome projection restricted to owner, source and kind; it reads no archived output. The projection survives restart and is exposed on scoped schedule records.

Focused tests cover owner denial before mutation, persisted outcome scoping, populated F10 at 220x65 and 40x18, expired/exhausted/paused/recovery/cancelling states, late responses, mismatched response owners, and keyboard access to controls. Narrow-layout tests caught an empty heading consuming checklist space; rendering now omits unavailable follow-up content and keeps close controls visible. Two type-check failures in new code/tests were corrected before the final gate. Full root check/test/build and git diff --check passed: 3,365 runtime tests, 3 skipped; 1,180 UI tests. Logs /tmp/xerxes-followup-status-verified-{check,test,build}.log; build 3343cec99d01b211. Native terminal acceptance is not claimed. Automatic stop-condition evaluation, whole-follow-up token budgets and other roadmap requirements remain unfinished; the full goal stays active.

## Follow-up stop conditions with recorded model evidence

Session schedules now persist an optional stop_condition, editable through /loop and exposed through the native manage_schedule schema. Each attempt receives the condition and instructions to check current evidence. The complete action is model-only and bound through host AsyncLocalStorage to the active attempt, conversation and schedule. It rejects foreign sessions/jobs, aborted or late attempts, missing evidence and a changed condition. Completion persists a model-reported evidence record, pauses future wakes and is idempotent within the attempt. A compare-and-set write rejects an intervening metadata edit. Up to 20 reports are retained; the inspector shows recent reports, and the conversation status labels the condition model-reported rather than independently certified.

Manual execution is refused after completion until explicit resume. Resume preserves report history and attempts used. Recurring and one-shot completion survives restart and a later provider failure, and completed one-shots are retained for review. The first full gate found that preserving live metadata also preserved the current running receipt on known one-shot failure, blocking retries. Failure handling now removes that receipt while retaining concurrently written completion state. Retry recovery and ordinary one-shot deletion remain verified. The scoped header status request now uses summary=true, excluding evidence/report history and limiting prompt excerpts.

Tests cover condition validation/persistence, automatic success/failure for recurring and one-shot jobs, restart/manual refusal/rearm, active host identity and cancellation, idempotence, late calls, concurrent write refusal, summary projection, and editing/clearing conditions at 220x65 and 40x18. Final full root check/test/build and git diff --check passed: 3,371 runtime tests, 3 skipped; 1,183 UI tests. Logs /tmp/xerxes-stop-condition-{regression,cas-fixed,complete-check,complete-test,complete-build}.log; build 5cfbe066913028b3. No native terminal or live external condition verification is claimed. Total follow-up token budgets and other roadmap requirements remain unfinished; the overall goal is active.

## Lifetime schedule token accounting — integration in progress

Added cumulative attempt accounting with replacement checkpoints, cache-inclusive
input totals, conservative unknown-usage handling, and persisted `max_total_tokens`
admission thresholds. Scheduler admission refuses exhausted or unknown historical
usage before increasing attempts. Native schedule tools and the `/loop` form expose
the threshold, including clear/remove behavior and an in-flight overshoot warning.
Negative attempt contributions are rejected before aggregation; receipt overflow
now remains a terminal accounting error rather than disappearing after a catch.

Focused verification: 27 runtime tests across schedule token usage, model budgets,
native scheduled child execution and schedule tools; a separate scheduler limit
run passed 10 tests (overlapping coverage). Form rendering/edit/clear/validation
passed 19 tests, including 220x65 and 40x18. Logs:
`/tmp/xerxes-total-token-tests.log`, `/tmp/xerxes-token-form-tests.log`.
The latest full production gate has not been rerun for this slice. Cumulative
usage/status presentation, further daemon lifecycle coverage and the complete
roadmap acceptance audit remain pending. This threshold is not a hard provider
billing cap; already admitted calls can overshoot. Overall goal remains active.

## Lifetime schedule accounting and status — full gate verified

The daemon now projects bounded cumulative usage and token admission state in all
schedule payloads, including the lightweight session summary. The schedule
inspector renders lifetime totals separately from last-attempt usage. Expanded
follow-ups show totals, and idle blocked follow-ups display exhausted/incomplete
usage instead of promising a next wake. UI parsing rejects inconsistent budget
payloads; running/cancelling status retains precedence while receipts settle.

A native daemon integration test executes two model attempts, verifies cumulative
cached and uncached input plus output in a freshly reopened JobStore, checks the
projected threshold state, then confirms a third attempt is rejected before any
provider call or attempt increment. UI coverage checks both exhausted and unknown
usage. Full root check/test/build and git diff --check passed: 3,378 runtime tests,
3 skipped; 1,187 UI tests. Build a103ca2ec49c1f23. Logs:
/tmp/xerxes-lifetime-final-{check,test,build}.log. These are offline/native-runtime
and headless-renderer results, not live-provider or native-terminal acceptance.
The measured threshold is not a hard billing cap. Broader monitor budgets and
other roadmap requirements remain unfinished; the overall goal stays active.

## Monitor reaction token admission — full gate verified

Automatic terminal monitors now accept an optional lifetime measured-token
threshold through the native tool, monitor.create RPC and TUI creation form.
Reaction policies persist the threshold in the mailbox schema. Claims include
prior measured consumption, and parent, child and auxiliary provider calls share
that admission scope. Exhausted or unknown usage prevents a subsequent claim;
another eligible watch in the same session can proceed. Idle budget exhaustion
is exposed in reaction health and the inspector with cumulative usage. Spent
token-limited grants release active-policy capacity without deleting their history.

Tests cover persistent exhausted/unknown admission, prior consumption on the next
claim, independent watch eligibility, policy capacity, native child plus auxiliary
spend blocking the next parent call, creation at 220x65 and 40x18, and inspector
status. An initial parameterized-test edit mistakenly let Bun's completion callback
occupy an omitted fourth tuple parameter; explicit undefined entries fixed the test
fixture before validation. Full root check/test/build and git diff --check passed:
3,383 runtime tests, 3 skipped; 1,190 UI tests. Build fb032b4b2d859b07. Logs:
/tmp/xerxes-monitor-budget-final-{check,test,build}.log. In-flight calls can
overshoot this measured threshold; a hard billing cap is not claimed. Policy
editing, additional event sources and other roadmap items remain unfinished.

## Guarded monitor reaction limit editing

E in the monitor inspector opens a limit editor. monitor.update atomically checks
the policy revision and owner, refuses unresolved/cancelled/expired execution,
preserves attempts, usage and consumed evidence, and enforces active-policy
capacity when higher limits reactivate a grant. Successful edits dispatch pending
evidence through normal session admission. Match/source and watch expiry remain
outside this limits editor. The editor retains its draft after an error and returns
to the inspector on Escape.

Mailbox tests verify stale/active/cross-owner rejection, cumulative state and restart;
a socket RPC test verifies guarded saves and cross-session rejection. Wide/narrow
renderer tests cover draft preservation and close behavior. Full root check/test/
build and git diff --check passed. Exact counts and build are recorded below.
3,385 runtime tests passed, 3 skipped; 1,192 UI tests passed. Build
2ebc5e890ab90092. Logs /tmp/xerxes-monitor-policy-final-{check,test,build}.log.
The overall roadmap remains incomplete. A new user-requested model/provider
inventory and routing guidance feature is captured as section 17 of the roadmap;
that feature is planned, not implemented.

## Model-visible inventory and explicit agent selectors

The daemon-native tool registry now provides list_available_models. Without a
profile it returns bounded, credential-free configured profile summaries and
provider/profile counts. A profile query reuses discovery and returns model IDs,
runtime-offered reasoning levels and known context/output capacity with provenance.
Pagination is revision guarded. Discovery failure is distinct from an empty list;
quota remains explicitly unknown. Capability discovery may refresh metadata caches
but does not switch the active provider or the parent conversation model.

AgentTool, TaskCreateTool and SpawnAgents now accept explicit provider_profile and
reasoning_effort alongside model. These cannot be combined with intelligence tiers
or used without a model. Existing native provider routing handles execution;
catalog membership is not a promise of entitlement. Default prompt guidance points
to discovery; details stay in the tool schema to respect the prompt size limit.

Focused tests cover paging, changed catalogs, credential exclusion, unknown quota,
cancellation, discovery errors, tool identity, explicit spawn forwarding and invalid
selector combinations. A local HTTP discovery fixture verifies the daemon returns
provider catalog data without switching parent configuration. The first full gate
caught overlong prompt guidance; shortened text preserves the existing 6,000-byte
ceiling. Final root check/test/build and git diff --check passed: 3,391 runtime
 tests, 3 skipped; 1,192 UI tests. Build 12fa6dd8b4e0801f. Logs:
/tmp/xerxes-inventory-final-{check,test,build}.log.

Section 17 remains incomplete: editable routing notes, authoritative quota adapters,
further selection validation and broader host coverage still need implementation.
No live subscription quota measurement or overall production readiness is claimed.

### Provider and model routing preferences

Implemented `/config` F6 routing-note editor for the selected explicit provider
profile and optional model. Tab switches scopes while retaining drafts; F2 saves
only the selected note, independently of agent-mode settings; Escape returns to
the settings draft. Failed saves retain note text. Notes persist in SQLite with
per-key optimistic revisions, bounded lengths/count, and empty revision tombstones.
The compatible `model.routing_note.get/save` RPCs validate configured profiles.
`list_available_models` exposes provider/model notes as user preferences, and note
changes invalidate pagination revisions. Inventory responses now shrink pages to
60,000 UTF-8 bytes instead of relying on downstream truncation of JSON.

Verified in this worktree: persistence and stale-write rejection; scoped discovery;
RPC save-to-model-inventory projection without switching the active model; draft
retention after conflicts at 220x65 and 40x18; complete pagination with maximum-size
Unicode notes. Full root check/test/build and git diff --check passed: 3394 runtime
passed, 3 skipped; 1194 UI passed. Build dd47ddc3a1af8f9d.
Logs: /tmp/xerxes-routing-bounded-{check,test,build}.log.

Remaining in section 17: authoritative subscription quota adapters, stricter spawn
selection validation, inventory availability outside the native daemon host, and
live/native acceptance. This evidence does not establish completion of the wider
roadmap. The preceding opinion/status turn made no implementation progress; this
turn implemented and verified the next available settings/discovery work.

### Explicit child-provider selection validation

Native daemon child routes now validate configured/discovered model IDs and the
runtime-offered reasoning choices before task allocation and again before model
execution. The host validates explicit provider selections through a new optional
`validateProviderSelection` port; CLI daemon composition wires it to the existing
profile store, model discovery, and reasoning capability lookup. Identity checks
reject profile/host changes during asynchronous validation. Execution checks the
cancellation signal before and after validation. Parent model/profile are unchanged.

Focused tests prove rejection before allocation, execution-time rejection with no
model calls, cancellation during held validation, unknown model/effort rejection,
and daemon inventory/selection without parent mutation. Full root gate passed:
3396 runtime passed, 3 skipped; 1194 UI passed; type checks, build and diff check
passed. Build d6493f89b196c1f8. Logs:
/tmp/xerxes-selection-final-{check,test,build}.log.

Scope limits: these are compatibility checks, not entitlement proof. Provider
fallback catalogs and generic reasoning fallbacks retain their existing semantics.
Inherited-provider selections and non-daemon host wiring still need corresponding
validation. Subscription quota adapters, broader model inventory host wiring,
and the remaining roadmap remain incomplete. Previous goal turn was progress
(routing-note implementation); this turn adds and verifies spawn-path guards.

### Delegation allocation cancellation

Inspection after the explicit-selection guards found that delegation tools did
not pass their cancellation signal into allocation, so cancellation while awaiting
provider validation could still allocate a child. Added a non-persisted allocation
signal to SpawnAgentOptions, forwarded by AgentTool, TaskCreateTool, SpawnAgents,
and HandoffTool. Native selection preflight checks it before and after validation
and passes it to the validator. The tool closes any allocation returned after
cancellation before reporting the abort. The signal is not the child lifetime
signal and is not serialized into task configuration.

Tests prove zero allocation/model calls when cancelled during held preflight,
forwarding through all four tool entry points, cleanup of late allocations from
an injected manager, and rejection before another allocation on an aborted signal.
Full root check/test/build plus git diff --check passed: 3398 runtime passed,
3 skipped; 1194 UI passed. Build 1fcecd02b8bbeeda. Logs:
/tmp/xerxes-spawn-cancel-final-{check,test,build}.log.

Previous goal turn was progress (explicit route validation); this turn fixes and
verifies its cancellation boundary. Remaining roadmap/section 17 gaps listed
above remain open; this is not a full-goal completion claim.

### Reasoning capability provenance

The inventory previously flattened all reasoning lists to runtime_validation,
masking whether choices came from a live provider declaration, bundled catalog,
or provider fallback. Added precise provenance to reasoning capability factories
while retaining their existing source field for UI/protocol compatibility.
Model inventory now exposes reasoning_source, reasoning_shape, and nullable
 default_reasoning_effort. Unsupported spawn routes report unavailable controls.
Tests cover each origin, unknown defaults, and control shapes; existing bounded
Unicode-note pagination continues passing with the additional metadata.

Full root check/test/build and git diff --check passed: 3400 runtime passed,
3 skipped; 1194 UI passed. Build 7dafd63fef64cca5.
Logs: /tmp/xerxes-reasoning-origin-{check,test,build}.log.
Previous goal turn was progress (allocation cancellation); this turn completes
the concrete provenance projection missing from discovery. Broader host wiring,
inherited-route validation, authoritative quota adapters, and the rest of the
roadmap remain incomplete.

### Persisted routing/settings integrity

Routing-note reads now validate persisted profile/model/note fields, canonical
whitespace, length limits, and positive safe-integer revisions. Reads are capped
at 501 rows and reject a collection exceeding the 500-row storage contract.
Updating a corrupted note fails explicitly rather than replacing it as a new
record. Agent settings revisions are checked on read/write and cannot overflow
Number.MAX_SAFE_INTEGER. Revision tombstones remain valid; none are erased.

Verification for this localized persistence change: 11 focused settings/inventory
tests passed and 3 daemon settings/inventory contract tests passed. Runtime type
check, root build and git diff --check passed. Logs:
/tmp/xerxes-settings-integrity-{check,build}.log. The full repository test suite
was not rerun for this localized change; the preceding full-gate result remains
historical, not a claim for the new state.
Previous goal turn was progress (reasoning provenance); this turn implements
persisted-boundary validation required by the repository contract. Broader host,
quota and roadmap work remains open.

### Profile-bound subscription usage discovery

Found existing auth/usage adapters used by /usage, but that collector selects the
first profile per provider or a shared OAuth session. Added a separate profile-bound
adapter for dedicated-key Kimi/Kimi Code and Z.ai/Zhipu profiles on recognized
provider HTTPS endpoints. Model inventory accepts optional include_usage only with
provider_profile; default discovery performs no quota request. Reports carry used
percentage, reset timestamps/durations, observed time, source and credential scope;
remaining_tokens stays null. Unknown endpoints, missing keys and unsupported/error
responses remain unknown. Shared OAuth profiles are deliberately not attributed.
Provider body errors are withheld, environment endpoint overrides are not used for
these credential-bound calls, and a ten-second request signal limits network time.
Caller cancellation propagates. Daemon checks profile identity after usage lookup.

Fixed a preexisting Kimi parser bug converting explicit used_percent values <=1
into fractions (0.5 percent became 50 percent). The explicit percent field now
retains its units. Existing percentage-only compatibility behavior remains.

Offline tests verify key/endpoint binding, no environment/shared credential
fallback, no credential output, reset projection, cancellation, optional lookup,
and sub-one-percent usage. Full root check/test/build and git diff --check passed:
3407 runtime passed, 3 skipped; 1194 UI passed. Build 630d43ea18d1de95.
Logs: /tmp/xerxes-profile-quota-final-{check,test,build}.log.
No live-provider verification was performed. Shared OAuth account attribution,
remaining quota adapters, broader inventory hosts and the wider roadmap remain
open. Previous turn was progress (persisted settings integrity); this turn adds
real provider-specific usage lookup through existing adapters.

### Strict model-facing quota units

Profile-bound usage now requests strict parsing from the existing usage adapters.
Kimi list rows must carry an explicit used_percent value in [0,100]; ambiguous
percentage-only rows and malformed keyed windows fail instead of being guessed.
Z.ai percentages outside [0,100] fail instead of being clamped. Model-facing
reports retain raw provider scope/type labels rather than inferring time windows
from partial type/unit names, and do not include untyped remaining quantities.
Legacy /usage formatting behavior remains unchanged by the strict opt-in path.

21 focused auth/profile-usage/model-inventory tests passed. Runtime type check,
root build and diff check passed; build 0248ef2120e770d7. Logs:
/tmp/xerxes-quota-units-{check,build}.log. The full test suite was not rerun for
this localized parser change. Live provider acceptance remains outstanding.
Previous goal turn was progress (profile quota wiring); this turn removes numeric
and unit guesses that were unsuitable for model routing decisions. The remaining
roadmap is still open.

### Built-in Codex shared-login usage

Verified from current code that the built-in codex profile has no API key and
inference resolves its stored CodexSession. Added usage lookup for that exact
built-in profile/provider/official endpoint combination, using the same credential
source and account-routing header. Reports explicitly identify shared_codex_login
scope rather than implying a private profile allowance; tokens/account IDs are
excluded. Custom profiles, changed endpoints and explicit-key variants remain
unknown. Strict Codex parsing rejects invalid percentages and does not assume a
5-hour/weekly duration absent the provider window length. Cancellation or the
request deadline stops waiting on shared OAuth refresh without cancelling it for
other callers. No live provider call was performed during verification.

Offline fixtures verify credential/header binding, scope, unknown attribution,
invalid data and cancellation during held credential resolution. Full root
check/test/build and git diff --check passed: 3410 runtime passed, 3 skipped;
1194 UI passed. Build e0e4b1604cc338c6. Logs:
/tmp/xerxes-codex-quota-final-{check,test,build}.log.
Previous goal turn was progress (strict quota units). Remaining work includes
live adapter acceptance, additional OAuth attribution, non-daemon inventory
composition, inherited-route validation, and the broader roadmap.

### Embedded inventory and QueryEngine session ownership

Moved model-inventory registration into the common registerCoreTools path with an
optional ModelInventoryHost port, exported through the package tools surface. The
native daemon supplies its existing callback through this path; embedded Xerxes
hosts can pass coreTools.modelInventory. Without a host callback, no tool is
advertised and no desktop credentials are read implicitly.

An actual embedded tool turn exposed that QueryEngine had a stable sessionId but
did not pass it to runTurn, leaving session-aware tools without ownership context.
Fixed that boundary. The embedded regression exercises model-issued discovery,
callback output returning to the model, generated IDs, distinct named sessions,
and reuse of the same named session. Existing QueryEngine tests remain passing.

Full root check/test/build and git diff --check passed: 3411 runtime passed,
3 skipped; 1194 UI passed. Build 37a089e04d36ae46.
Logs: /tmp/xerxes-inventory-host-{check,test,build}.log.
Previous goal turn was progress (Codex quota). Automatic one-shot inventory
composition, inherited-route validation, live provider acceptance, and remaining
roadmap work are still incomplete.

### One-shot and ACP inventory composition

Added a profile-store inventory host for CLI paths without a daemon server and
wired it into one-shot and ACP core tools. It reuses native model discovery,
Codex catalog helpers, reasoning factories, routing-note storage and profile quota
adapters. Catalog failures are explicit and sanitized; profile identity changes
across awaited discovery reject the result. It does not switch the active profile
or persist catalog discoveries. Built-in agent tool catalogs now include inventory
when the host registers it; custom allow/exclude rules remain intact.

The one-shot subprocess regression now actually calls list_available_models and
checks the inventory returned in the next model request, instead of only checking
schema registration. A local HTTP fixture verifies capacities, notes, reasoning
provenance, credential omission, failures and cancellation for standalone discovery.
ACP uses the same composition, but this slice did not add a separate ACP inventory
round-trip fixture. Provider-specific live acceptance remains outstanding.

Full root check/test/build and git diff --check passed: 3412 runtime passed,
3 skipped; 1194 UI passed. Build f067560241b94ca4.
Logs: /tmp/xerxes-standalone-final-{check,test,build}.log.
Previous goal turn was progress (embedded inventory). Remaining work includes
inherited-route validation, provider-specific discovery parity beyond generic and
Codex catalogs, live acceptance, and the wider roadmap.

### ACP inventory subprocess acceptance

Added the previously missing ACP-specific round-trip fixture. It starts the real
CLI acp command with isolated home/workspace directories, opens a session over
NDJSON stdio, submits a prompt, and drives an actual list_available_models call
from a local model endpoint. The saved profile points to a local /models fixture.
The test confirms one catalog request, one tool execution, a second model request
containing discovered context/model data, no profile API key in the tool response,
and clean ACP shutdown. This replaces the prior evidence gap where ACP was wired
but only the one-shot inventory path had an end-to-end test.

9 ACP CLI/runner tests passed; runtime/package type checks and git diff --check
passed. Logs: /tmp/xerxes-acp-inventory-{focused,check}.log. This test-only change
did not rerun the full repository suite or build. No live provider was contacted.
Previous goal turn was progress (standalone inventory); this turn supplies direct
ACP acceptance evidence. Inherited-route validation, provider-specific parity,
live acceptance, and the remaining roadmap still require work.

### Bounded quota HTTP transport

Usage fetchers previously followed fetch redirects and consumed an unbounded JSON
body (including unbounded error bodies before slicing). Requests now refuse
redirects, use a ten-second deadline, enforce a 256 KiB streamed/body-length limit,
strictly decode UTF-8, and cancel stalled readers on caller abort. HTTP failures
report status without returning provider body text. Cleanup preserves the original
failure and releases the reader/listeners. These transport protections apply to
both /usage and model-facing profile quota.

20 auth/profile-usage tests passed, including an actual loopback redirect test
proving the destination receives no request, stalled-reader cancellation, byte
limits with/without Content-Length, and malformed UTF-8. Runtime type check, root
build and diff check passed. Logs: /tmp/xerxes-usage-http-{tests,check,build}.log.
Full repository tests were not rerun for this localized transport hardening.
Previous goal turn was progress (ACP round-trip acceptance); this turn closes the
quota response-size/redirect boundary. Live external provider verification and
remaining roadmap work are still open.

### Standalone spawn selection validation — September 6, 2026

One-shot and ACP native subagent hosts now use the standalone profile inventory's
catalog and reasoning metadata to validate explicit provider/model selections.
Validation runs through the existing pre-allocation and pre-execution hooks.
Configured models remain selectable even if omitted from the catalog, matching
the daemon's policy; unknown models and unsupported efforts fail explicitly.
Cancellation and profile replacement during discovery invalidate the result.

The CLI tier-selection integration fixture now serves catalog requests and
asserts that both validation calls use the child profile credentials before the
child request uses its configured model and reasoning effort. Focused tests also
cover unknown profiles, invalid models/efforts, pre-abort, in-flight cancellation,
and profile replacement. The first full run found the old POST-only fixture;
after correcting that fixture, the complete gate passed: 3418 runtime tests,
3 skipped, 1195 UI tests, root checks/build and git diff --check. Build:
28fab54aba3c68cf. Logs: /tmp/xerxes-selection-final-{check,test,build}.log.

The routing-note editor retry/saved-text correction from the preceding progress
turn is included in this full gate. Inherited-provider validation, specialized
provider catalog parity, native UI/live provider acceptance and the broader
roadmap remain unfinished. This evidence proves this integration slice, not
overall production readiness.

### Standalone Copilot and Radius discovery — September 6, 2026

Standalone inventory and explicit spawn selection now use native subscription
discovery for GitHub Copilot and /v1/config for Radius, instead of treating both
as generic OpenAI model endpoints. Copilot uses the native OAuth credential
exchange and catalog client, with cancellation forwarded. Raw OAuth adapter
errors are replaced with an actionable login/connection error because the
profile API key cannot redact subscription tokens. Radius uses the selected
profile key and reports valid positive integer context/output limits from its
gateway configuration. Reasoning provenance remains independently labeled.

28 focused profile-inventory, Radius gateway and Copilot auth tests passed;
runtime/UI/desktop type checks, root build and diff check passed. Tests include
a loopback Radius gateway with credential/path assertions and an injected
Copilot catalog shared by inventory and spawn validation. Logs:
/tmp/xerxes-provider-parity-{tests,check,build}.log. This localized adapter change
did not rerun the full repository suite; the preceding full gate is recorded
above. Live subscription verification and inherited-provider selection remain
open. The preceding goal turn was progress (standalone spawn validation), and
this turn closes two concrete standalone discovery mismatches.

### Inherited agent selection validation — September 6, 2026

Daemon, one-shot and ACP hosts now validate inherited model/effort overrides
against a snapshot of the parent's resolved connection. The validator does not
look up the mutable active profile or change the child's transport. An unchanged
parent model with no explicit effort requires no catalog fetch. Overrides use
the same discovery and reasoning validation as explicit profile selections,
before allocation and again before execution, with the host generation check
and cancellation barriers preserved.

The first focused run exposed an omitted effort being converted to an empty
string at execution. Both explicit and inherited validation now receive undefined
for that case. Tests cover captured parent credentials despite later mutation,
alternate model discovery, rejected model/effort choices, cancellation, rejection
before allocation and rejection before a model call, plus real CLI detached-child
completion. 76 focused tests passed. The full root check/test/build/diff gate then
passed: 3422 runtime tests, 3 skipped, 1195 UI tests. Build be1c6ae1d1348b25.
Logs: /tmp/xerxes-inherited-final-{check,test,build}.log.

This turn is progress; it also verifies the preceding Copilot/Radius integration
under the full gate. Live provider/native UI acceptance and the remaining
workflow roadmap are still open; full production readiness is not established.

### Copilot catalog cancellation — September 6, 2026

Catalog retries now reject promptly on caller cancellation rather than waiting
out Retry-After. Native retry timers are cleared, injected waits are detached
without issuing another request, and 429 response bodies are cancelled before
retry. The request deadline remains active through body parsing instead of
ending after headers. Request cleanup removes its caller abort listener.

24 focused Copilot auth/profile inventory tests passed, including pre-cancelled
requests, cancellation during an unresolved injected retry wait, and cancellation
during body consumption. Runtime/UI/desktop type checks, root build and diff
check passed. Build ce5da8de31b1db65. Logs:
/tmp/xerxes-copilot-cancel-{tests,check,build}.log. The full suite was not rerun
for this localized transport fix; the prior full gate is recorded above.
Previous goal turn was progress (inherited routing validation); this turn closes
a cancellation gap on that discovery path. Broader roadmap and live acceptance
remain open.

### Inventory pagination capability revisions — September 6, 2026

Model inventory now resolves reasoning metadata before calculating its revision.
The hash includes effort options, defaults, control shape, provenance and catalog
source/warning, along with model capacities and routing notes. A changed earlier
page therefore prevents a caller from continuing with an obsolete token. Quota
observations remain outside this catalog revision because they are independently
timestamped account usage, not model capabilities.

13 focused inventory/profile tests, type checks, root build and diff check passed.
Logs: /tmp/xerxes-inventory-revision-{tests,check,build}.log. The full suite was
not repeated for this localized change. This resolves capability changes supplied
by adapters; it does not establish cache freshness. Resolving all matching entries
also makes adapter batching/cache behavior relevant to discovery latency, which
remains to be audited. Previous goal turn was progress (Copilot cancellation).
Overall production readiness and the remaining roadmap are still unproven.

### Daemon Codex inventory snapshot — September 6, 2026

Each daemon model-inventory call now gets Codex model IDs, capacities and reasoning
from one fresh catalog. Inventory does not reuse the separate UI reasoning cache
or refetch the catalog for each model. Caller cancellation is passed through the
catalog host port to native authentication/fetch. Codex discovery failures are
actionable errors rather than raw OAuth adapter text. The inventory captures
profile configuration and rejects results if the selected profile changes while
the request is pending.

157 daemon-server and inventory/profile regression tests passed, along with type
checks, root build and diff check. A deterministic daemon integration verifies
one catalog call, live effort changes invalidating pagination, and profile-change
rejection. Logs: /tmp/xerxes-inventory-snapshot-{tests,regression,check,build}.log.
The full repository suite was not repeated for this adapter change. Previous
goal turn was progress (capability-aware pagination); this turn closes its Codex
request amplification and stale reasoning-cache path. Other roadmap requirements
and live acceptance remain open.

### Inventory capacity override precedence — September 6, 2026

Standalone inventory and the daemon Codex snapshot now resolve capacity fields
through the runtime's shared profile precedence: user override, provider metadata,
bundled catalog, then unknown. Live metadata is overlaid in memory without
rewriting profiles or replacing user overrides. Missing live fields retain cached
metadata, including the existing qualified-model fallback. Context/output source
labels distinguish overrides from provider and bundled catalog values.

13 inventory/profile tests and 3 selected daemon inventory tests passed, including
setting and clearing user overrides and a daemon Codex override assertion. Type
checks, root build and diff check passed; build 6863a0ded0b98d60. Logs:
/tmp/xerxes-inventory-capacity-{tests,daemon,check,build}.log. The full suite was
not repeated for this localized capability projection. Previous goal turn was
progress (daemon Codex snapshot); this turn fixes the capacity precedence gap.
Live acceptance and remaining roadmap requirements are still unfinished.

### Context inspector tool evidence — September 6, 2026

The tools section now includes retained tool results before tool schemas, with
original transcript positions and tool names when available. Evidence can be
inspected without an assembled scaffold. Conversation contents are preserved;
the scope note explicitly warns that the two sections overlap and their token
estimates must not be summed. Standard stale-generation, excerpt and pagination
rules apply to results. PROTOCOL.md documents the expanded section contents.

Focused context-inspection and existing narrow/wide context-overlay tests,
type checks, root build and diff check passed. Logs:
/tmp/xerxes-context-evidence-{tests,ui,check,build}.log. This change does not
implement context pins, exclusions or selected-turn branching; those roadmap
requirements remain open. The full repository suite and live TUI were not rerun.
Previous goal turn was progress (inventory capacity precedence); this turn fills
the missing retained tool evidence in the context inspector.

### Independent session branch state — September 6, 2026

Branching now deep-copies its source state before asynchronous destination
allocation. Nested metadata and extra state no longer share references with the
source. Branches retain session reasoning/permission settings and their pin
flags, plus the known API-call counter. Source lineage and existing transcript,
thinking, tool and token history copying remain intact.

145 daemon-server tests passed, including a real slash/RPC branch operation that
checks settings and source independence after changing branch metadata and
messages. Type checks, root build and diff check passed; build 2ccbd9116a056c9d.
Logs: /tmp/xerxes-branch-copy-{tests,check,build}.log. The full repository gate
was not repeated for this localized branch correction. This is a prerequisite
repair, not selected-turn branching: selecting a historical turn and handling
in-flight boundaries remain open, along with pins/exclusions and the broader
roadmap. Previous goal turn was progress (tool evidence inspection).

### Native TUI branch wiring and running-turn guard — September 6, 2026

The TUI branch command called an unimplemented session.branch RPC and locally
switched IDs without loading the branched transcript. It now sends the supported
native slash command and resumes the returned branch with keepCurrent, preserving
the source session. Rejections do not switch or close either session. Both the
TUI busy guard and daemon turn/session-operation guard prevent partial in-flight
copies. Design guidance documents the behavior and the absence of historical
turn selection.

The initial full gate exposed a memory-lock test that assumed worker startup
within 50ms. The test now waits for a lock-acquired signal and checks the holder
completion marker before accepting the competing save, instead of measuring a
minimum wall-clock wait. Its isolated check passed; the subsequent full gate
passed with 3429 runtime tests, 3 skipped, 1197 UI tests, checks/build/diff check.
Build 8a9981b41adaf9f7. Logs: /tmp/xerxes-branch-wire-final-{check,test,build}.log.
Previous goal turn was progress (independent branch state). Historical branching,
pins/exclusions, other roadmap features and live acceptance remain unfinished.

### Branch usage certainty and restoration — September 6, 2026

Branches now preserve incomplete or unknown token/API-call accounting instead of
inheriting a fresh session's complete-accounting defaults. A fresh runtime reload
test verifies persisted branch messages, reasoning and permission choices, counters
and incomplete flags. 153 daemon/session-history/telemetry tests, type checks,
root build and diff check passed. Logs: /tmp/xerxes-branch-restore-{tests,regression,check,build}.log.
The full repository gate was not repeated for this localized correction.

### Historical retained-turn branching — September 6, 2026

`/branch --through-turn N [title]` now selects a completed retained user-turn
prefix. `/context` labels those turn numbers. The selector validates tool-call
and result pairing and requires an assistant response after the last tool round.
The source remains intact; the branch retains current model and permission
settings but omits later derived metadata and marks aggregate usage unknown.
This does not recover compacted history or restore workspace files.

A Luna agent implemented the selector and focused tests while the parent wired
the daemon, command catalog, TUI and documentation. Parent review caught a
completion-state bug, which was corrected with regression coverage. A second
Luna agent completed a read-only context-controls audit; pins and exclusions
remain unimplemented.

Full root check, test, build and diff check passed: 3438 runtime tests passed,
3 skipped, and 1197 UI tests passed. Logs:
`/tmp/xerxes-historical-final-{check,test,build}.log`.
Native terminal acceptance and the remaining roadmap are still unfinished.

### Session context pins and exclusions — September 6, 2026

The Memory section in `/context` now exposes bounded optional-source snapshots.
J/K selects a source, I pins/unpins, and X excludes/includes automatic recall.
Controls persist per session and apply to the next daemon turn. The daemon uses
captured source contents, validates source identity plus inspector/control
revisions, and rejects active-turn changes. Mandatory assembly guidance and
self-memory cannot be excluded through this surface. Explicit memory tools can
still read excluded sources.

Pins retain scanned/fenced snapshots across source edits and suppress current
retrieval of the same scope/path. Exclusions filter before ranking. Pins reserve
their rendered context budget rather than disappearing under ordinary memory
clipping. Source previews are separate from the last assembled reference and do
not inflate section token totals. Controls remain inspectable after restart even
before a new scaffold is assembled; compaction preserves their metadata.

Persistence writes only the target session and rolls back in-memory changes on
failure. Context saves and session deletion are mutually exclusive. Transcript
storage rejects stale writers with conflicting context controls, preventing an
independent runtime's ordinary flush from silently overwriting newer settings.
Deterministic tests cover both deletion orderings, failed saves, stale writers,
restart, compaction, provider assembly, source filtering and TUI errors.

Two Luna agents supplied bounded implementation and independent review; parent
integration addressed byte-budget and persistence race findings. Full root check,
test, build and diff check passed: 3454 runtime tests passed, 3 skipped, and 1198
UI tests passed. Build b01727b075ffde0e. Logs:
`/tmp/xerxes-context-controls-final-{check,test,build}.log`.

The previous goal turn delivered verified historical branching; this turn
delivered context controls. The full roadmap remains incomplete. The next
goal-workflow audit confirms that structured completion criteria, checked evidence
records, token/time budgets and their F10 projections remain absent; existing
goal lifecycle/CAS/round-cap behavior does not prove those features complete.
Native terminal acceptance is also still outstanding.

### Goal criteria, evidence and F10 inspection — September 6, 2026

Goal create/edit tools now accept bounded completion criteria. Each criterion can
record a same-session tool-call reference and the model's explanation of its
relevance. The host rejects missing, ambiguous, denied, failed or pending calls,
including attempts to cite the goal tools themselves. This checks execution
outcome; it does not independently certify the model's relevance claim. Goals
with declared criteria require evidence for each criterion before completion.
Goals without criteria retain their prior completion behavior.

Criteria and evidence survive persistence and strict replay. Changed requirements
invalidate affected evidence, objective edits invalidate all evidence, and CAS
rejects stale updates. Aggregate criteria/evidence data is bounded to 32 KiB.
Duplicate evidence is idempotent. Candidate histories are validated before
mutation, so rejected changes cannot corrupt the current goal. Legacy goals can
add pending criteria without changing their objective.

F10 reads the active session through the read-only `goal.inspect` RPC and shows
criteria, evidence, rounds and blockers. Polling skips overlapping requests,
rejects stale session responses, and retains the last valid view after errors.
The managed turn runner now publishes completed tool records before the next
provider inference, allowing evidence from the same running turn to resolve.

Two Luna agents implemented the domain and inspector in parallel. Parent review
and integration found and corrected live evidence timing and legacy-goal replay
bugs. Focused tests include a real managed turn followed by session reload,
malformed replay, failed evidence, transactional rejection and inspector polling.
Full root check, test, build and diff check passed: 3467 runtime tests passed,
3 skipped, and 1203 UI tests passed. Build d35a5a67b6d5a634.
Logs: `/tmp/xerxes-goal-criteria-final-{check,test,build}.log`.

Explicit user-decision evidence, goal token/time budgets, durable continuation
scheduling, native terminal acceptance and the remaining roadmap are unfinished.
This slice does not establish overall production readiness.

### Goal wall-time limits and cancellation — September 6, 2026

`/goal --duration 30m` configures a total wall-time limit from original goal
creation. Whole seconds/minutes/hours are accepted. The native goal create/edit
tools expose `max_duration_ms`; automatic rounds cannot raise their own limit.
Duration is persisted and replay-validated, includes pauses and process downtime,
and preserves criteria/evidence when edited. Expired goals cannot resume until
their limit is raised. Existing goals without a configured duration remain uncapped.

The daemon observes goals created during an already-running human turn, aborts
active work at the deadline, blocks further rounds with `time-limit`, and saves
the blocker. Expiry before provider launch emits an explicit unstarted terminal
event. Goal cancellation also reaches pre-turn compaction; external cancellation
rejects promptly even if a provider ignores its signal and prevents retries.
F10 shows elapsed/remaining time and expired states, including narrow layouts.

Two Luna agents implemented the domain/UI and compaction cancellation; parent
integration wired commands, tools, runtime guards, persistence and socket tests.
Tests cover mid-turn creation/expiry, no-provider expiry, durable blocker state,
duration edits, expired resume rejection, malformed replay, legacy behavior and
compaction cancellation. Full root check, test, build and diff check passed:
3480 runtime tests passed, 3 skipped, and 1205 UI tests passed.
Build d5318413f2004628. Logs:
`/tmp/xerxes-goal-time-final-{check,test,build}.log`.

The next token-budget slice needs a goal-ID-scoped durable usage ledger, not
session lifetime counters. Existing `ModelCallBudget` is a useful admission
primitive, but goals need checkpointed call admissions/settlements, explicit
ownership for detached descendants and auxiliary calls, and dynamic activation
when a tool creates a goal mid-turn. Unknown usage or crash-pending admissions
must not reset to zero on resume; changing a cap must preserve prior spend.
Already-admitted concurrent calls can overshoot an admission cap and must remain
accounted for. Token budgets, durable goal continuation scheduling, explicit
user-decision evidence, native terminal acceptance and other roadmap items are
still unfinished. The previous goal turn delivered criteria/evidence; this turn
delivered time limits and verified cancellation. Overall completion is unproven.

### Durable goal token budgets — September 6, 2026

`/goal --tokens 100000` sets a persisted goal-specific admission limit. Native
goal tools expose `max_total_tokens`; automatic rounds cannot raise their own
cap. F10 and `goal.inspect` expose measured input/output usage, pending calls
and accounting completeness. Changing the limit preserves recorded spend.

The daemon uses a SQLite ledger at `runs/goal-tokens.sqlite` beneath Xerxes home.
Admissions and settlements are transactional and include reported cache tokens.
Queued and retried in-process subagents retain the original goal scope, and a
replacement goal cannot absorb their spend. Goals created by a tool begin
accounting with subsequent provider calls; the call emitting creation is excluded.
Already-admitted concurrent calls may overshoot the cap. This is a local usage
limit, not a provider subscription quota or an exact token reservation.

Capped goals reject further admissions and resume when spend reaches the cap,
when usage is unknown, or when pending admissions belong to another owner after
restart. Legacy goals without a ledger retain an unknown baseline. Invalid usage
and numeric overflow settle as unknown before propagating an error. Unknown
accounting is never replaced by zero. Uncapped goals can continue with uncertainty
visible. Complete cross-process recovery of detached agent scopes remains open.

Two Luna agents implemented the ledger and domain/inspector changes. Parent
review and integration covered provider calls, command/tool wiring, subagent
scope capture, persistence and socket-level behavior. Tests include cached usage,
detached retries, replacement goals, same-millisecond goal creation, malformed
receipts, durable cap rejection and raising a cap without resetting spend.

Full root check, test, build and diff check passed: 3502 runtime tests passed,
3 skipped, and 1208 UI tests passed. Build 9535669f0773d597.
Logs: `/tmp/xerxes-goal-token-final-{check,test,build}.log`.

Durable goal continuation scheduling, explicit user-decision evidence, native
terminal acceptance and the remaining roadmap are unfinished. This verification
establishes this slice's automated checks, not overall production readiness.

### User decision evidence and goal-only durability — September 6, 2026

F10 now lets the user select a criterion with Tab, open its acceptance-note
editor with A, and explicitly accept with Enter. Esc cancels. Ctrl+R retains the
note while refreshing the goal for review and a deliberate new submission.
The editor remains visible below long checklists. Typing shortcut letters or
pasting multiline text cannot accept a criterion or navigate away.

`goal.decision` binds the note to the connection's active session, goal ID,
criterion and revision. It refuses a running turn or pending session operation,
uses the shared session-operation gate, and persists before acknowledgement.
The host creates the decision ID and timestamp. The model's goal tools expose
the resulting `user-decision` evidence through `get_goal` but cannot create it.
Trusted daemon clients can invoke the RPC; transport authentication does not
attest physical human input. Evidence is labelled distinctly from model-assessed
tool relevance. Recording it does not resume or complete the goal.

The domain preserves legacy tool evidence, validates both evidence variants,
retains history, rejects stale revisions and mixed identities, and invalidates
affected evidence after requirement edits. Human decisions may be recorded on
active, paused or blocked goals while preserving phase, blocker and budgets.
Completed goals remain immutable. The inspector also now accepts ordinary goals
that carry createdAt without a configured time limit.

Socket validation uncovered a durability bug: sessions with a goal but no
conversation exchange were omitted by empty-chat filtering. Validated goal
history now qualifies for persistence, listing and reload, including history
whose current goal has been cleared. Untouched empty sessions remain omitted.
Malformed recognized goal logs are rejected during transcript normalization.

Luna agents implemented domain, inspector and persistence work; parent review
corrected keyboard/paste handling, stale-response guards, evidence normalization
and test synchronization, and added native-tool and socket integration coverage.
Focused runtime checks passed 46 tests; focused inspector checks passed 29.
Full root checks passed and the runtime suite passed 3510 tests, with 3 skips.
The initial UI gate found one narrow-footer regression. After correction, the
affected layout/inspector tests, UI typecheck, all 1214 UI tests, build and diff
check passed. Build d8db1d6e0bfdea5b. Logs:
`/tmp/xerxes-goal-decision-final-{check,test,build}.log`,
`/tmp/xerxes-goal-decision-ui-final.log`, and
`/tmp/xerxes-goal-decision-layout.log`.

Read-only continuation audit: goal rounds still execute in submitTrackedTurn's
in-process loop; a durable queued wake does not exist. Loops already use the
CronJob/JobStore/CronScheduler receipt, lease and session-operation path. The next
implementation must share session wakeup admission, retain goal-round authority,
revalidate goal identity/phase/budgets before one round, and fence duplicate or
crash-interrupted work. Persisted pending work must remain distinct from the
current process-local authorization to resume a goal. Routing through the cron
runner unchanged would use the wrong origin and tool authority. Human-work
precedence, restart receipts, goal edits, cancellation and cap exhaustion need
socket-level coverage. Durable continuation and the remaining roadmap, including
native terminal acceptance, are unfinished; overall production readiness is not
established by these automated checks.

### Durable queued goal continuation — September 6, 2026

The daemon now stages a durable, session-scoped `goal_wake` receipt instead of
continuing through an in-process loop. Goals, session follow-ups and monitor
reactions share per-session operation admission. Already queued human work has
priority; active work is not preempted. Each goal admission revalidates identity,
revision, activation and budgets, regenerates the prompt, reserves its round and
persists the claim before compaction or provider work begins.

Queued receipts survive reload but restart does not restore authorization:
`/goal resume` is required. A foreign-owner running receipt becomes interrupted,
and the reserved round is never replayed. Branches strip the source receipt.
The F10 inspector and inspection/decision RPCs expose the current receipt,
including queued, running, settled, interrupted and cancelled outcomes. Receipts
are a single current slot, not an archive of every prior wake.

Parent review fixed an ownership gap while saving a claim, cancellation and
disconnect before provider launch, incorrect settled labels on failed rounds,
and eviction of goal-only sessions on disconnect. Socket tests deliberately
hold the claim save while submitting, cancelling or disconnecting, and prove
that no cancelled provider work starts. Restart tests inspect the saved claim
from inside the runner before work and prove that interrupted rounds are not
replayed. Ordinary chats with no goal bypass continuation persistence.

Three Luna agents implemented the queue, receipt/persistence and UI projection,
then independently reviewed recovery and ran UI checks; the parent integrated
and corrected daemon behavior. An initial full run exposed an obsolete wiring
assertion and an unnecessary post-turn save that held ownership after ordinary
chats. After correction, focused regression checks passed 50 tests, then the
full root check, test, build and diff check passed: 3528 runtime tests passed,
3 skipped, and 1220 UI tests passed. Build b045dab138769eb1. Evidence:
`/tmp/xerxes-goal-wake-final-{check,test,build}.log`,
`/tmp/xerxes-goal-wake-regressions.log` and
`/tmp/xerxes-goal-wake-socket-current.log`.

The next section 11 gap is an explicit current milestone, with durable revision
and history semantics rather than relabelling a possibly stale todo. Detached
agent recovery/accounting, broader monitor and PR/CI workflows, plugin lifecycle,
authenticated remote continuation, native terminal acceptance and live-provider
acceptance remain part of the unfinished roadmap. These automated checks do not
establish overall production readiness.

### Durable current milestones — September 6, 2026

Goals now store optional `currentMilestone` in their revisioned change history.
The milestone is bounded to 1000 characters, survives session reload, and is
separate from objective, acceptance evidence, phase and budgets. Objective edits
clear stale milestone text unless the edit supplies a replacement. Completed
goals reject milestone setters and explicit milestone edits; F10 labels their
retained value as the last milestone. Strict replay rejects invalid milestone
mutations without reinterpreting legacy goals that lack this field.

Humans can use `/goal milestone <text>`, `/goal milestone` and
`/goal milestone clear`. The native TUI command retains the transcript and
forwards these to the same daemon/domain path. Model tools expose the value in
`get_goal`, accept it during human-authorized create/edit, and offer
`update_goal` action `milestone` with current_milestone string or null. The
current goal round can update progress context, but cannot smuggle objective or
budget edits through that action. Subagents, unrelated background work and stale
rounds lack authority. Continuation prompts include the current value as quoted
progress context, not completion proof. F10 displays it beside the other goal
state and discards stale values when switching sessions.

Luna agents implemented the domain, command/docs and UI projection; the parent
wired native tools, continuation prompts and socket persistence tests, reviewed
completed-goal restrictions and corrected TUI help and authority documentation.
Focused integration checks passed 48 tests; domain/tool/command checks passed
60; focused UI checks passed 110. Full root checks and all 3541 runtime tests
passed, with 3 skips. UI runs exposed older context/schedule tests that asserted
or edited before asynchronous results had rendered. Those tests now wait for
the observable error or ready state without weakening assertions. Their focused
27 tests, UI typecheck, all 1225 UI tests, build and diff check then passed.
Build 2668e1cf379fb53f. Logs:
`/tmp/xerxes-goal-milestone-final-{check,test,build}.log`,
`/tmp/xerxes-goal-milestone-ui-final.log`,
`/tmp/xerxes-goal-milestone-ui-synchronization.log`.

Next verified gap: recovered detached subagents retain task/history identity
but do not persist their parent goal's model-call budget binding. After restart,
the daemon subagent.retry path invokes runtime.retrySubagent outside the parent
turn's GoalTokenBudget; respawnRecovered does not reconstruct the scope. This
can bypass descendant goal accounting. The next slice needs a durable binding,
live-owner and goal-identity validation, restored call accounting before launch,
and restart/retry tests proving cap enforcement and rejection of stale bindings.
The broader unfinished roadmap and native/live-provider acceptance remain open.

### Recovered subagent goal accounting — September 6, 2026

Native delegation now captures a serializable model-call ownership binding at
spawn, before asynchronous setup. The task snapshot, persisted manifest and
tool/retry wire projections preserve it. Recovery distinguishes explicitly
unbound work from legacy records with unknown ownership; malformed bindings
retain an unrecoverable marker instead of becoming unrestricted work.

The native recovery boundary reconstructs the original goal scope through an
injected daemon host port before spawning. It requires the original source
session and goal, active/armed lifecycle state, durable ledger and current daemon
owner. Provider admission continues to enforce the ledger cap and pending-owner
fences. Replaced goals cannot absorb old work's charges; ownership changes during
asynchronous provider validation are checked again by scope capture. Legacy
records in sessions with goal history require newly dispatched work. Other
non-restorable enclosing scopes refuse recovered execution rather than losing
their limits. This does not implement resumable scheduled-run call scopes.

Luna agents implemented the binding/persistence layer, recovery helper and
integration fixtures. Root reviewed and corrected validation bounds, strict
TypeScript fixtures and actual successful-retry coverage, and integrated native
manager/host capture, the CLI host port, and tool wire projections. Focused
existing/new recovery checks passed 92 tests; the final focused integration set
passed 15. The full root check, test, build and diff check passed: 3556 runtime
tests, 3 skipped, and 1225 UI tests. Build ce03854b91141c49. Evidence:
`/tmp/xerxes-recovery-focused.log`, `/tmp/xerxes-recovery-integration.log`, and
`/tmp/xerxes-recovery-final-{check,test,build}.log`.

Next confirmed ownership gap: the daemon forwards sessionKey to subagent retry,
but the CLI callback discards it. Shared native-host id/name lookup does not
filter the requesting session, allowing cross-session selection (including
ambiguous recovered names). Recovery also uses the host's current execution
configuration rather than reconstructing the owning session's full workspace
and provider context. Fix owner-scoped lookup and verify execution context with
two-session/restart tests. Broader monitoring/PR-CI workflows, plugin lifecycle,
authenticated remote continuation, native terminal and live-provider acceptance,
and the final requirement-by-requirement roadmap audit remain unfinished.

### Owner-scoped retries and captured workspaces — September 6, 2026

The daemon CLI now resolves the requesting session key to a live parent identity
before invoking retry. Native retry lookup checks exact IDs before names,
refuses other owners' IDs, scopes names to the requested owner and rejects
ambiguous names. The lookup includes retained archived task identities, without
reactivating them, and deduplicates recovered/live representations of one task.

Native spawns capture the source session's validated absolute workspace before
asynchronous selection/setup. Task config, snapshots, persisted manifests and
tool/retry wire results preserve the project root. Runner tools, active-session
context, conversation projectRoot, history loading and worktree factories use
that captured workspace. Legacy records resolve the owning session's workspace;
invalid saved paths remain explicitly invalid and cannot fall back to the host
root. A fixed worktree adapter cannot silently allocate an alternate project;
the production daemon uses the workspace-aware factory.

Review found two additional budget/workspace paths that the preceding recovery
slice had not covered: archived task rebuild dropped serialized ownership labels,
and recovered resume/sendInput reset called fresh spawn without restoring its
scope. Rebuild now retains both labels and workspace. Recovered reset shares
respawn recovery while preserving its new-task/new-history semantics; in-process
reset carries the original runtime scopes through its replacement spawn. Tests
cover charging and denial at the cap, not only snapshot fields.

Luna agents supplied the owner resolver, persistence, core workspace wiring and
regression fixtures; root integrated the CLI/tool paths, reviewed and corrected
archive/reset coverage, centralized the worktree guard, and replaced indirect
workspace assertions with actual file reads. The final 26-test focused set
passed. It reads distinct project sentinel files before/after restart and checks
an isolated child reads its worktree file while retaining the source project
root. The full root check, test, build and diff check passed: 3574 runtime tests,
3 skips, 1225 UI tests. Build 97e247033d7b72ce. Logs:
`/tmp/xerxes-retry-workspace-integration.log` and
`/tmp/xerxes-retry-workspace-final-{check,test,build}.log`.

Next confirmed gap: inherited provider routing is not durable across recovery.
Explicit profile/model/effort fields survive, but a changed active connection or
a same-name profile with a changed provider/base URL can reroute recovered work.
Capture and validate nonsecret routing identity without persisting credentials
or overriding inline runtime settings with an unrelated profile. Provider
configuration-change/restart tests are required. Native recursive delegation is
currently blocked by the child tool filter; no nested-worktree inheritance bug
was established on that execution path. Monitoring/PR-CI, plugin lifecycle,
authenticated remote continuation, native terminal/live-provider acceptance and
the final full-roadmap completion audit remain open.

### Durable subagent provider routing — September 6, 2026

Native agent creation now captures a SHA-256 routing fingerprint before async
selection. Task config, archived rebuilds, snapshots, persisted manifests and
agent/retry wire results retain it. Production daemon, one-shot and ACP native
hosts supply inherited connection identity and explicit profile route resolvers.
The fingerprint covers configured provider, endpoint, transport and known
noncredential deployment selectors; API keys are never stored. It is not proof
of authenticated account identity. Inline connection overrides stay authoritative.

Recovered retry/reset requires the original route in production route-aware
hosts, rejects missing/malformed legacy identity and refuses changed inherited
or same-name profile routes. Validation happens before and after asynchronous
selection and again before child execution; explicit profile client creation
checks the same profile snapshot atomically. Credential rotation on an unchanged
route remains allowed. Archive review additionally found that provider profile
and explicit reasoning effort were absent from task records/fallback snapshots;
both now survive rebuild. Root also corrected CLI Responses API factory overrides
from the ignored responsesApi key to the supported responses_api key.

Luna agents implemented core host, persistence and initial fixtures. Root wired
all production CLI host constructions, implemented the routing resolver, reviewed
archive selectors, and corrected test fixtures that did not actually exercise
inherited routing or configuration races. Added reset, actual provider-request
model/effort, late-validation and credential-rotation checks. Focused integration:
30 passed. Full root check/test/build/diff gate: 3586 runtime passed, 3 skipped,
1225 UI passed. Build 100ad1866ae476d6. Logs:
`/tmp/xerxes-provider-recovery-focused.log` and
`/tmp/xerxes-provider-recovery-final-{check,test,build}.log`.

Next confirmed scheduling gap: CLI runScheduleCommand creates enabled legacy
Scheduler triggers, while production daemon startup runs CronScheduler; legacy
evaluate/markFired have no production caller. Trace the CLI into the unified
execution backend and give existing triggers explicit migration/status handling;
do not silently activate old jobs. Broader file/WebSocket/webhook monitoring,
PR/CI workflows, plugin lifecycle, authenticated remote continuation, native
terminal/live-provider acceptance and full-roadmap completion audit remain open.

### CLI schedules use the executing daemon — September 6, 2026

The native `xerxes schedule` command no longer creates enabled records in the
legacy Scheduler store. It sends project-scoped schedule RPCs to the existing
daemon, using the same JobStore, CronScheduler lease/admission, run, archive and
delivery paths as `/schedules`. Create returns the generated job ID; list,
inspect, pause/resume (`disable`/`enable`), manual `fire`, cancel and remove are
available. Connection project assertions reject mismatched explicit sockets.
No new daemon is launched implicitly and mutations are not retried. Transport
close/timeout after transmission reports an unknown outcome. Current daemon
availability is required; legacy records stay in the explicit paused migration
flow and are not activated automatically.

Intervals and standard five-field cron are supported. Ambiguous legacy cron
slash-step and combined restricted day constraints are rejected, as are event
and webhook schedules and legacy owner/store/delivery-id CLI options. Selected
unambiguous slash-separated time expressions convert to standard cron.
`schedule.remove` is workspace-scoped, rejects active/unreconciled execution,
and checks its inspected revision under the store writer lock before deletion.
Archived output and delivery history are preserved. CLI help, configuration
examples and protocol documentation describe the behavior.

Luna agents supplied the control client, initial adapter and real-daemon CLI
fixture. Root completed adapter validation/output, corrected fixture lifecycle
and archive timing, wired CLI/server/project assertions, added revision-guarded
removal and replaced legacy-only tests. The actual CLI subprocess test creates
an enabled interval job, observes automatic execution and archived output,
pauses it, manually executes while paused, restarts the daemon, resumes, observes
another automatic run, and rejects a mismatched project socket. No external
provider is used. Focused checks: 20 passed. Full root check/test/build/diff gate:
3594 runtime passed, 3 skipped, 1225 UI passed. Build 3a767c1d2e4aeb08. Logs:
`/tmp/xerxes-schedule-cli-focused.log` and
`/tmp/xerxes-schedule-cli-final-{check,test,build}.log`.

Next source extension is file-change monitoring. Current MonitorSummary/Watch,
RunHistory monitor configuration, monitor tools/RPC and TUI creation/display
assume terminal sources. Extend source identity and persisted configuration,
workspace-bound path resolution, bounded deterministic event delivery and
subscription cleanup together. Restart must explicitly report gaps and must not
pretend terminal handles can be reconstructed. WebSocket/webhook sources,
PR/CI workflows, plugin lifecycle, authenticated remote continuation, native
terminal/live-provider acceptance and the full-roadmap completion audit remain
unfinished.

### File-change monitors through native tools, RPC and TUI — September 6, 2026

File watches now use the existing monitor domain, durable Runs history and
bounded reaction mailbox. The native `monitor_file` tool and `monitor.create`
file source resolve the authenticated owning session workspace. `/monitors`
creation puts Source first and File path next, hides terminal-only fields and
explains metadata-only/coalesced observations. Inspection shows the file source,
resolved path, retained evidence and attachment health. Legacy terminal monitor
rows and responses remain compatible; source JSON is an additive SQLite migration.

The Bun-native adapter watches one existing regular file. It validates workspace
containment, resolves an allowed internal symlink once to its target, detects
metadata changes/deletion/recreation and atomic replacement, and observes parent
replacement as a visible failure. Initial subscription rechecking closes the
baseline-to-attachment gap. Bounded debounce and serialized state checks avoid
indefinite postponement and out-of-order reads. Event identities include the
change kind, preventing deletion from being discarded as a duplicate change.
No contents are read, and no model turn runs merely because a watch is idle.

Pending attachments count toward host/session admission limits and abort on
owner disposal. Stop, expiry, failure and shutdown detach observers. Late source
errors cannot rewrite a completed watch. Notification-only events create no
reaction claims; explicit reaction grants obey attempt limits and stop revokes
queued work. File watches are interrupted on shutdown; restored inspection
explicitly says downtime changes were not observed and requires a new baseline.
This is not automatic reattachment or a lossless filesystem event journal.

Luna agents implemented the source adapter, persistence and UI/RPC fixtures.
Root wired the domain/tools/daemon/CLI, reviewed and corrected callback lifecycle,
symlink attachment, setup races, source parsing, form field routing and test
cleanup, and added domain/tool/reaction regression coverage. A real daemon socket
test creates a watch, writes an actual temporary file, retrieves the event,
rejects foreign-session inspection/stop and outside paths, then proves stop
suppresses later events. Native source tests include atomic replacement and
internal/external symlink behavior; persistence tests migrate an old schema.

The frozen full root check/test/build/diff gate passed: 3610 runtime tests,
3 skipped, 1231 UI tests. Build 7f81619fa7d815e8. Evidence:
`/tmp/xerxes-file-monitor-final-{check,test,build}.log`. Native interactive visual
acceptance and live-provider acceptance were not performed by this gate.

Next: WebSocket and authenticated webhook event adapters, including bounded
payloads, deterministic matching/deduplication, source cleanup and explicit
reconnect gaps or cursor recovery. PR/CI workflows, plugin lifecycle,
authenticated remote continuation, native terminal/live-provider acceptance and
the final requirement-by-requirement roadmap audit remain unfinished.

### WebSocket watches and Git checkpoint — September 6, 2026

WebSocket server-push monitoring now connects the native source adapter, durable
source configuration, monitor domain, registered model tool, daemon RPC and
`/monitors` creation/inspection. The form exposes URL, literal match and common
reaction settings. Plaintext URLs are loopback-only; credentials, query parameters
and fragments are rejected. Authentication headers and application subscription
messages are not supported by this generic adapter. Existing terminal and file
sources remain compatible.

Only text messages of at most 64 KiB are accepted into the monitor domain.
Literal matching precedes model wakeups and a bounded content-identity set
suppresses duplicates. Gap events explicitly report disconnected observation;
three lifetime retries use 250 ms, 1 s and 4 s delays with five-second connection
timeouts. Initial failures reject creation; exhausted retries and unsupported
frames fail visibly. Stop, expiry and shutdown release sockets and timers.
Shutdown records interruption rather than implying messages during downtime
were recovered. The 64 KiB check applies after Bun delivers a complete message:
Bun's public client options do not expose a configurable inbound parser limit.
A strict pre-allocation transport bound remains a hardening item before claiming
readiness for arbitrary hostile feeds.

Luna agents supplied the native source, persistence, UI and real-daemon tests.
Root integrated production CLI/RPC/tools and shared monitor lifecycle, reviewed
and corrected refused-reconnect failure delivery, stale callbacks, cleanup after
callback errors, stop-during-gap behavior and source-switch form semantics.
Focused domain tests prove 5000 irrelevant messages create no reaction claim,
duplicate matching content produces one event, gap evidence remains visible,
owner isolation, bounded excerpts and reaction cancellation. Native tests use
local Bun WebSocket servers for message identity, reconnects, exhausted refused
reconnects and invalid frames. RPC tests cover actual socket delivery, foreign
access rejection, stopping and retained interruption inspection.

The frozen checkpoint gate passed: root check, test, build and diff check;
3624 runtime tests passed, 3 skipped, and 1236 UI tests passed. Build
1dae863a8921b69a. Logs: `/tmp/xerxes-checkpoint-final-{check,test,build}.log`.
No live-provider or native interactive visual acceptance was performed.

This is an implementation checkpoint, not completion of the production-readiness
goal. Remaining work includes authenticated webhook monitoring, strict inbound
WebSocket transport bounds and broader feed integrations, PR/CI watch-to-review,
transactional plugin lifecycle, authenticated remote continuation, native
terminal/live-provider acceptance and the full requirement-by-requirement audit.

## Attachment ownership and bounded WebSocket checkpoint (2026-09-06)

Submitting an image during a live turn no longer sends its accompanying text as
an image-free steer. The complete message queues with its own attachments and an
explicit explanation; queue previews show the attachment count. Busy retries,
queue editing and interpolation preserve message attachment ownership, and older
queued messages do not consume later draft attachments. Live image steering is
still unsupported; an empty-composer Enter can interrupt and dispatch the queue.
Completed goals no longer occupy the live plan card; F10 retains their durable
record, and unfinished tasks remain visible.

WebSocket monitoring now uses a bounded incremental frame decoder over Bun's
supported TCP/TLS socket APIs. It validates advertised lengths before payload
buffering, bounds fragmented text to 64 KiB and HTTP upgrade headers to 16 KiB,
and handles masked control replies with bounded output buffering. Protocol
violations fail the watch without reconnecting. Cancellation destroys pending
connections, and TLS verification is explicitly enabled. This supersedes the
previous checkpoint's post-assembly inbound-limit caveat. Tests use local servers
and a documented public self-signed test certificate, not live feeds.

This remains a checkpoint, not completion of the wider production-readiness
roadmap. Authenticated webhooks, broader feed integrations, PR/CI workflows,
plugin lifecycle, remote continuation and native/live-provider acceptance remain.

Checkpoint validation completed on the frozen worktree: root check/test/build
and `git diff --check` passed; 3,636 runtime tests passed, 3 skipped, and 1,240 UI
tests passed. Build `26455ab11693b37f`. Logs:
`/tmp/xerxes-image-checkpoint-{check,test,build}.log`. No live-provider or native
interactive visual acceptance was performed in this checkpoint.

## Authenticated webhook monitors (2026-09-06)

Named webhook monitoring is wired through production daemon startup/shutdown,
`runtime.monitor_webhooks` host configuration, environment-resolved secrets,
`monitor.sources` discovery, `monitor.create`, model tools `list_monitor_sources`
and `monitor_webhook`, Runs history and `/monitors` creation/inspection. No HTTP
listener starts without explicit configuration. Source names, not secrets, are
exposed to clients. Existing matching, expiry, ownership, event retention and
reaction budgets are reused. Source data is explicitly untrusted evidence in
reaction prompts. Stop detaches; shutdown interrupts; restart does not invent
recovered deliveries.

The generic receiver authenticates timestamp/delivery-ID/raw-body HMAC-SHA256,
checks timestamp freshness before and after reading, caps bodies at 64 KiB,
concurrent reads at 32 and body duration at five seconds. Replay IDs are retained
for the signing window with a bounded 4096-entry cache per source; capacity
refuses new admission instead of forgetting valid IDs. Secrets remain in host
memory/environment. Cache persistence across process restart, native third-party
signature protocols, public hosting and external exactly-once effects are not
claimed. The configuration guide documents the sender contract and limitations.

Luna agents implemented source, domain/persistence and TUI work in parallel.
Root integrated configuration, CLI lifecycle, RPC, model registration, docs and
real signed HTTP-to-daemon event tests; review corrected streaming-reader timer
and abort cleanup, partial-body timeout admission, callback failure visibility,
removed-watcher delivery, constructor bounds and initial cancellation behavior.
Tests include 4096 admitted IDs with overflow/replay rejection, 32 stalled readers
and shutdown cancellation, a real body deadline, concurrent duplicates, unsigned
rejection, source isolation, cross-session denial, interrupted persistence,
secret-free discovery and wide/narrow TUI form behavior/error preservation.

Final root check/test/build, docs:build and diff checks passed on the worktree:
3,654 runtime tests passed, 3 skipped; 1,244 UI tests passed. Build
`dd4c61ccbaa49f44`. Logs `/tmp/xerxes-webhook-final-{check,test,build}.log` and
`/tmp/xerxes-webhook-docs.log`. Native interactive and live-provider acceptance
were not performed. This slice is local/uncommitted after checkpoint 9a57ab12.

Remaining production-readiness work includes broader feed/PR/CI integrations,
transactional plugin lifecycle, authenticated remote continuation, native
terminal/live-provider acceptance and the full requirement-by-requirement audit.
The overall goal remains incomplete.

## Plugin registration isolation and production-wiring audit (2026-09-06)

Async native plugin discovery previously registered into the live indexes before
`register()` settled. Partial tools/hooks/providers/channels were observable, and
rollback based on old keys could delete unrelated host registrations made during
the await. Discovery now gives each module an isolated registration view, checks
live name conflicts again at commit, and publishes all added indexes without an
await. Failure discards only that view. Retained readers forward to the committed
live registry; late mutations are rejected. Staging copies are released after
settlement so retained plugin closures do not retain every preceding registry
snapshot. Module side effects outside capability registration are not rolled back.

Focused regression tests cover pending visibility, concurrent host changes on
failure, commit-time conflicts, late mutations, provider/channel/hook publication,
reader freshness and duplicate discovery. Existing extension hardening and native
provider tests remain applicable.

A fresh production-wiring audit found a larger gap than a settings toggle:
`runtime/features.ts` composes PluginRegistry discovery/hooks for embedding hosts,
but normal CLI startup does not invoke that composition or provide a native
plugin registry to the daemon. `/plugins` correctly reports an unconfigured host.
Plugin providers have a client-factory port but production client creation does
not supply it; PluginTool has no schema/capability definition for the production
tool executor; plugin channel objects have no validated adapter lifecycle there.
Slash plugins are a separate registry. Thus module registration tests do not prove
production plugin execution or installation.

Remaining plugin work must include one host-owned registry generation shared by
inventory and execution, explicit persisted trust/enablement, typed tool/channel
contracts with existing permission and cancellation boundaries, all provider
creation paths, actual turn-hook composition, dependency-checked generation
replacement, persisted install/update recovery, and daemon/TUI management with
real CLI integration tests. This registration repair is a prerequisite, not a
claim that the plugin lifecycle requirement is complete.

### Daemon startup and shutdown repair

Reproduced a failed Unix socket bind followed by `stop()` replacing the original
startup error with `ERR_SERVER_NOT_RUNNING`. Socket binding now participates in
startup cleanup. Shutdown shares one completion promise, tolerates an unbound
listener, and attempts the remaining resource cleanup after a component fails.
The CLI retains the original startup exception while reporting cleanup failures.
Regression cases cover failed binding, concurrent/repeated shutdown, and a later
startup failure combined with failing cleanup; the latter also checks socket
removal and runtime shutdown.

Verification for this worktree: root `bun run check`, `bun run test`, and
`bun run build` passed, with 3,660 runtime tests passing (3 skipped) and 1,244 UI
tests passing. The rebuilt `dist/cli.js` accepted an isolated socket connection
and exited cleanly on SIGTERM; an overlong socket path retained its original
bind error without `ERR_SERVER_NOT_RUNNING`. Logs are under
`/tmp/xerxes-startup-{check,test,build}.log`.
