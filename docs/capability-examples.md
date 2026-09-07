# Capability examples

Type slash commands in the TUI. Quoted messages below are prompts to send after closing the relevant picker. External services require your own configuration.

## Plugin Creator

Run `/plugin-creator`, then send:

> Build a text-statistics plugin that counts Unicode characters, words and lines. Include Bun tests for empty and invalid input and a README with usage examples.

This opens a new session with the plugin authoring preset. Your prior chat stays saved in Agent View. Ordinary new sessions keep their configured default preset. `/creator` remains the separate agent-preset authoring mode.

A working reference is [text-stats.ts](../examples/plugins/text-stats.ts). Verify its actual installation and execution path offline, from the repository root:

```sh
bun test xerxes/test/pluginCreator.test.ts
```

To install the reference in the TUI, substitute your absolute checkout path:

```text
/plugins install "/absolute/path/to/Xerxes-Agents/examples/plugins/text-stats.ts"
/plugins inspect text-statistics
```

Ask “Call plugin_text_stats with args [\"Hello world\\n🙂\"].” Expected result: `{"characters":13,"words":3,"lines":2}`. Model calls pass positional arguments using `{"args":[...]}`.

```text
/plugins disable text-statistics
/plugins enable text-statistics
```

Disabled plugins reject new calls. Keep installed source files in place. Restart Xerxes after editing installed module code. Authoring does not automatically install a plugin. Managed installation supports native tool modules; hooks, providers and channels need an embedding host.

## Specialists and skills

Run `/custom-agents`, press **N**, enter this example, then **Ctrl+S**:

```markdown
---
name: reviewer
description: Review changes for reproducible correctness bugs without editing files.
tools: ReadFile, GlobTool, GrepTool, ListDir
---
Read relevant code and report concrete findings with file paths, triggers and impact.
Do not edit files. Say when evidence is insufficient.
```

Ask “Use the reviewer specialist on my current changes.” Inspect delegation with **F6**. Existing `.xerxes/agents` definitions load automatically. `/init` can inspect the repository and generate missing setup while preserving existing files.

For a local skill, save this as `review-checklist/SKILL.md`:

```markdown
---
name: review-checklist
description: Review input validation, error propagation and state restoration.
---
Inspect changed code. Check invalid inputs, cancellation and persistence.
Report confirmed bugs with a reproduction, or say none were found.
```

Run `/skills install "/absolute/path/to/review-checklist"`, then `/skills search review-checklist`. Ask “Use review-checklist on my changes.” Referenced assets belong inside the skill directory. Shell preprocessing requires explicit skill trust.

For a complete agent preset, run `/creator` and ask “Create a testing-focused coding preset with examples of when to use it.” `/preset list` lists presets; `/preset use <id>` selects one before the first turn. `/reload` refreshes externally edited project definitions. `/` completion and `/help` show commands actually available in your session.

## Models and intelligence

Open `/config`, then choose provider, model and reasoning effort for **light**, **balanced** and **smart**. Ask “List available models and reasoning levels; use a light agent to locate tests and a smart agent to review concurrency.” Choose configured models, not guessed names.

`/provider` and `/reasoning` change the current conversation. The composer displays reasoning effort. Subscription quota is only available when the provider exposes it; unknown usage is not a guaranteed remaining allowance.

## Terminals, agents and runs

Ask “Start a background shell that prints progress once per second for five seconds, then exits.” Open `/terminals` (**F8**) to inspect output and exit status. Use its controls to send input or stop a long-running command.

Ask “Use two agents: one maps source modules, the other locates tests; each should give a short summary.” `/agents` (**F6**) inspects them. `/runs` shows history and unread results. `/workspaces` inspects retained agent workspaces.

## Goals

Ask “Create a goal to audit the parser with three todos: invalid input, cancellation and persistence; verify each with tests.” `/goal` (**F10**) shows progress. New goals have no default token or duration cap. `/goal unlimited` removes caps from an existing goal while preserving progress. `/goal resume` resumes a paused or blocked goal when eligible. Completed todos alone do not establish that the goal is finished.

## Schedules, monitors and loops

- `/schedules`, **N**: prompt “Review the working tree without editing”; recurring cron `0 9 * * *`; choose your timezone. Save paused, review the next-run preview, then **P** to enable. Check results in `/runs`.
- `/monitors`, **N**: choose a file source and an existing absolute build-log path. Fill the shown fields and save. Append a line yourself to verify detection. Terminal sources observe sessions started in Xerxes; network sources need configuration.
- `/loop`: follow the setup for a bounded follow-up such as “Check the build output and report when it finishes.” Review timing and limits before starting.

The owning daemon must remain running. Pause or remove your test schedules and monitors afterward.

## Machines and browser

Open `/machine` and press **N** (or **Enter** when empty) to add a workspace using the form. **Tab** moves between name, SSH host and project folder; **Enter** saves and **Esc** returns without saving. On **SSH host**, press **F2** to choose an alias from `~/.ssh/config` and its `Include` files. On **Project folder**, press **F2** to browse that host (starting at its home directory when blank): **Enter** opens a folder, **Left/Backspace** goes up, and **Space** selects the current folder. **Esc** returns with your draft intact; manual entry also works. Browsing requires an already trusted SSH host and noninteractive authentication (such as your SSH agent). If needed, first connect using `ssh <alias>` in a terminal. Saving does not connect automatically. The equivalent command is:

```text
/machine add compute my-ssh-alias "/home/me/My Project"
/machine
```

Select `compute`, press **Enter** and authenticate through your SSH configuration. Connection automatically installs Bun if missing or too old, and builds the latest published GitHub main revision under ~/.xerxes/remote-runtime. Unchanged builds are reused; older releases and existing installations are preserved. Setup requires Git (and curl, bash, unzip if Bun needs installing). Providers still need remote authentication; local credentials are never copied. Setup errors are logged in ~/.xerxes/remote-runtime/setup.log. The TUI renders locally while an SSH tunnel carries daemon RPC. Exit the remote workspace view to return to your previous local chat. It does not migrate conversations or synchronize files. `/machine remove compute` removes the saved entry.

For browser tools, supply an already-running Chromium-compatible browser endpoint with `/browser connect <endpoint>`. Ask “Read the current page title.” Xerxes attaches to the supplied CDP endpoint; it does not launch a browser.

## Review, recovery and integrations

| Entry point | Try it |
| --- | --- |
| `/diff` or F7 | Make a small local edit; inspect its patch and file entry. |
| `/snapshots` | Browse a saved snapshot and preview a restore before applying it. |
| `/context` | Inspect context and memory for the current conversation. |
| `/config mcp` | Configure a server you control, then inspect its exposed tools using `/tools`. |
| `/config lsp` | Configure your project's language server; check diagnostics on a known syntax error. |
| `/features` | Discover capability entry points. |
| `/help` | Find commands and keyboard shortcuts. |

These examples are walkthroughs, not proof of live external connectivity. The plugin reference test uses a temporary manifest, makes no network calls and leaves real plugin configuration untouched.

### Run a shell command and discuss the result

Type `!ls` (or another `!command`) to execute it in the project directory.
Xerxes displays stdout/stderr and nonzero exit codes, then asks the model to
review the result without rerunning the command. The submitted context retains
the command and output for later turns and session restoration. `{!pwd}` inside
a normal prompt substitutes command output without creating a separate follow-up.
If the model is already working, a new `!command` waits in the queue so its
follow-up does not race the active turn.
