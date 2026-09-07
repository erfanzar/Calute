# TUI feature wiring audit

Updated 2026-09-07 against the source checkout. Backend tests, TUI reachability
and live external-service acceptance are separate checks.

## Connected paths

| Feature | Entry point | Behavior |
| --- | --- | --- |
| Background shells | `/terminals`, F8 | Inspect output, send input, stop existing processes |
| Runs | `/runs` | History, unread results and supported cancellation |
| Monitors | `/monitors` | Create, inspect, stop and edit reactions |
| Schedules and follow-ups | `/schedules`, `/loop` | Create/edit, pause/resume, run and history |
| Goals | `/goal`, F10 | Inspect goal/todos; unlimited defaults; `/goal unlimited` removes saved caps |
| Agent tiers | `/config` | Provider, model, reasoning and usage notes |
| Project specialists | `/custom-agents`, `/agents edit` | Create/edit validated Markdown, reject conflicting saves, reload definitions |
| Remote workspaces | `/machine` | Saved SSH targets, local renderer over an SSH-forwarded remote daemon, return to local chat |
| Skills | `/skills` | List, search, inspect, trust, diagnostics and local bundle installation |
| Plugins | `/plugins` | Local native tool-module installation, persisted enable/disable and inspection |
| MCP/LSP | `/config mcp`, `/config lsp` | Settings and daemon configuration calls |
| Review | `/diff`, `/snapshots`, `/workspaces`, `/context` | Changes, restore previews, retained workspaces and context |
| Discovery | `/features`, `/help` | Entry points and setup requirements |

Remote handoff runs the remote installation in its own remote session; it does
not migrate a local chat or synchronize files. SSH and provider credentials
remain host-owned. Plugin installation supports local native tool modules;
provider/channel/hook plugins need a separate embedding host. Skill search uses
the admitted local and bundled catalog; no network marketplace is connected.

## Remaining separate workflows

- `/tools` is inventory; independent per-tool policy toggles are not implemented.
- Per-agent pause needs checkpoint semantics; retry is available for dead agents.
- Saved spawn-tree replay uses trees held by the current TUI process.
- Voice recording needs an actual capture/transcription host.
- Remote daemon transport and cross-machine conversation migration are separate
  from the working SSH terminal handoff.

Tests cover daemon routing, persisted settings, safe file updates, local
extension execution, keyboard controls, narrow layouts, and preserving chat
state through overlays. Live SSH acceptance requires a configured remote host;
offline SSH tests inject the process port and verify arguments, errors,
cancellation and terminal restoration.
