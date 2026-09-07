# Terminal design guidelines

Xerxes' terminal client is a React application rendered by OpenTUI from `xerxes/src/ui/`.
It should feel calm and fast: the prompt is the focus, session context is compact, and tools or
approvals appear only when needed.

## Visual principles

Session branching uses `/branch [title]` (or `/fork`) on an idle conversation.
The TUI sends the native branch command, then loads the new session through its
normal resume flow while keeping the source session open. Branch errors leave
the current conversation visible. Running turns and active session operations
must finish or be stopped before branching; copying a partial tool exchange is
not a valid branch. `/branch --through-turn N [title]` copies through completed
retained user turn N. `/context` labels these turn numbers in Conversation; they
refer to the retained transcript, not compacted-away history. Historical branches
use the current session model/reasoning/permission choices but omit current derived
metadata and tool/thinking logs that may describe later work. Historical aggregate
usage is unknown. This branches conversation history; it does not restore files.

In `/context`, Memory exposes optional source snapshots separately from assembled
instructions. J/K selects a source, I pins/unpins its snapshot, and X toggles
exclusion from automatic recall. Changes save for the next turn and require an
idle conversation. Keep success and failure visible without replacing the chat.
Pins and exclusions survive restart and compaction; explicit memory-read tools
can still read excluded sources. Mandatory guidance remains outside these controls.

F10 separates completion criteria from the task checklist. Show each criterion as
pending or linked to an execution record, with its explanation and tool-call ID.
Evidence relevance is model-assessed, not automatic certification. Keep rounds,
blockers and refresh failures visible. Poll only while the inspector is mounted,
retain the last valid view on failure, and reject responses for a prior session.

- Use a quiet dark canvas with a readable light theme fallback.
- Keep workspace, model, mode, and session context in a thin header or footer rather than a large
  dashboard.
- Make the existing `❯` prompt visually dominant and keep keyboard hints concise.
- Use the code-native Xerxes/Derafsh Kaviani mark consistently. Do not import or imitate third
  party logos, artwork, or proprietary product text.
- Prefer spacing, divider rules, typography, and color contrast over decorative chrome.

## Interaction principles

- Keyboard operation is complete without a pointer; mouse behavior is additive.
- Preserve transcript virtualization, streaming states, approvals, clarification, and session
  overlays when changing layout.
- Ensure narrow terminals collapse nonessential metadata before hiding input or active turn status.
- Screen-reader/plain-text output must retain the same information without depending on color or
  glyph-only state.
- A visual refactor must not change daemon v35 payloads or gateway behavior.

## Verification

Use component tests for mark rendering, theme tokens, compact/wide decisions, and status labels.
Then run:

```sh
bun run typecheck
bun run test:ui
bun run build:ui
bun run --cwd xerxes smoke:ui
```

## Compact conversation layout

The workspace header contains the project and panel keys; the session row contains its title and
context usage. Mode, model, and effective write policy stay visible beside the composer while a
turn runs. Detailed session telemetry appears with expanded details. The welcome uses the existing
Derafsh Kaviani artwork with a compact wordmark; short terminals omit artwork before sacrificing
input or useful starting actions.

Successful sequences of three or more tool calls fold into a one-line summary. Click a summary or
press **F9** to expand/collapse tool groups. Failed and unfinished calls remain visible. The agent
rail appears only at 120 columns or wider, retains at least 80 columns for conversation after the
gutter, and folds completed agents. **Ctrl+F6** toggles the rail; **F6** still opens the complete
agent inspector at every width. **F7** and **F8** retain the diff and terminal panels.

Approval details scroll independently with **PgUp/PgDn**, keeping the choices reachable on short
screens. The full command is available in that scroll area. **Y** approves once, **a** approves for
the session, **3** selects permanent approval only when offered, and **N/Esc** denies. Opening or
closing a prompt preserves the draft.

Goals and todos open with F10 (or a goal/todo click). The inspector is a bounded,
scrollable overlay: arrows, Page Up/Down, and Home/End navigate; Escape or F10
closes it. A completed turn preserves this user-opened inspector. Short agent
panels show only the todo count so the checklist cannot consume the agent list.
The welcome uses a centered 120-cell column; conversation and composer fill the available pane with two-cell gutters. The startup screen uses bounded top spacing. Distinct layout keys prevent native padding from leaking across the first-message transition; the composer rule is a container-sized border.

The original interactive design is now the layout reference: a branded workspace row, a ruled session row, a vertical welcome hierarchy, separated starter labels and consequences, padded user bands, an assistant author row, and a composer anchored below the content in both welcome and conversation states. At 30 rows or more, the composer gains vertical padding and a lower boundary above the separate shortcut row; shorter terminals retain the compact variant. The Derafsh art remains the project asset and is hidden below 40 rows.

Task progress uses a padded card with up to three unfinished tasks, with F10 opening the full ordered plan. The goal inspector separates the objective from the checklist and highlights the active task. Agent rows show status and resource counts; detailed tool output is available in the inspector. The rail shows a task count and F10 link rather than a clipped duplicate checklist.

The changes viewer uses a wider file index and a horizontally scrollable code area. Left/Right scroll code, brackets jump between files, and clicking a file selects its section.

An empty Agent View is a centered panel capped at 64 columns and 11 rows, with only the close shortcut. Populated views are capped at 48 rows. The list and inspector split depends on the actual resized panel width; narrow panels open details in place. Enter pins the inspector so navigation scrolls its details, and Escape returns to the list. Action hints reflect the selected agent's available retry or cancel action.

The independent-session Agent View is a separate surface from the F6 child-agent inspector.
It uses a centered workspace capped at 160 columns and 44 rows, growing with its chat list
and preview rather than filling a large terminal. Roomy layouts separate rows, align saved
chat metadata without dotted leaders, and keep a delimiter between the attached title and
activity status. Short terminals retain compact rows and reachable dispatch controls.

### Runs inspector

Bare `/runs` opens a persistent user-controlled overlay. The list keeps
selection by run identity as results refresh, and detail responses are ignored
when a newer selection replaces them. Acknowledgement applies to the inspected
revision, never whatever revision happens to arrive later. Running output can
refresh without resetting the user's scroll position. Unavailable history is
shown as an error, not presented as a successful empty inbox.

Use side-by-side list/detail at 100 columns of panel width or more; stack at
smaller widths. Compact hints and omit duplicate headings in narrow layouts.
Keep Escape and output scrolling reachable in a 40-column terminal. Retain the
overlay on turn completion and clear it only on explicit dismissal or session
teardown.

## Operational dialogs

Use `ui/opentui/dialogChrome.tsx` for operational settings and inspectors. A dialog has a title and short purpose, one content region, and a separate keyboard footer. Settings separate labels from values; selected fields use the selection surface. Group long details under short section labels and put actionable failures before routine metadata.

Empty schedules, monitors, workspaces, snapshots, custom agents and server settings use a bounded card rather than an empty master/detail split. Show one useful next step. A creation button must open the same editor as its keyboard shortcut; never decorate an inert command as a clickable control. Hide actions that require a selected item when the list is empty.

On terminals shorter than 26 rows, drop decorative header/footer spacing and put field labels and values on one line. Keep errors, cancellation and the selected input visible. MCP/LSP editors scroll the selected field into view and edit next to its label. Their saved launch values and credentials remain hidden.

The style applies to runs, schedules/follow-ups, monitors and reaction limits, context, snapshots, workspaces/recovery, deliveries, agent modes and routing notes, custom agents, and MCP/LSP settings. Skill/plugin guides use scrollable cards with Home/End, arrows and page navigation. Model/reasoning pickers share the title surface; terminal and agent inspectors share the empty-state treatment. Preserve the welcome layout, full-width conversation, and dedicated diff/output rendering.
