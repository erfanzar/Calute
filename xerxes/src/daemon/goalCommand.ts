// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The human half of the goal subsystem: `/goal`.
 *
 * The model drives a goal through typed tool calls; a person drives the same
 * durable state through this command. Both go through `goalDomain`, so there is
 * exactly one definition of what a goal is and what transitions are legal —
 * a second, parallel notion living in the UI is how the two-engine drift this
 * codebase already suffers from starts.
 *
 * Parsing and rendering live here rather than in the TUI so every surface
 * (terminal, bridge, channels) shows the same words for the same state.
 *
 * Modelled on DeepSeek Harness's `/goal`
 * (github.com/deepseek-ai/deepseek-harness, MIT); no source is reproduced.
 */

import {
  DEFAULT_MAX_GOAL_ROUNDS,
  clearGoal,
  createGoal,
  editGoal,
  getGoal,
  GoalError,
  pauseGoal,
  resumeGoal,
  setGoalMilestone,
  type GoalPhase,
  type GoalRef,
  type GoalView,
} from "../runtime/goalDomain.js";

export const GOAL_USAGE =
  "Usage: /goal [<objective>|clear|edit <objective>|pause|resume|unlimited|milestone [<text>|clear]|--duration <Ns|Nm|Nh>|--tokens <count>]";

export type GoalCommand =
  | { readonly kind: "show" }
  | { readonly kind: "create"; readonly objective: string }
  | { readonly kind: "edit"; readonly objective: string }
  | { readonly kind: "invalid-edit" }
  | { readonly kind: "duration"; readonly value: string }
  | { readonly kind: "tokens"; readonly value: string }
  | { readonly kind: "unlimited" }
  | { readonly kind: "pause" }
  | { readonly kind: "resume" }
  | { readonly kind: "milestone-show" }
  | { readonly kind: "milestone"; readonly value: string | null }
  | { readonly kind: "clear" };

export interface GoalCommandResult {
  readonly ok: boolean;
  readonly text: string;
}

/**
 * Parse only the grammar `/goal` owns; anything else is an objective.
 *
 * Deliberately not a flag parser. A goal objective is prose, and prose starting
 * with a word this command happens to know is far more likely to be an
 * objective than a mistyped subcommand — so only the exact control words, and
 * only as the whole input, are treated as commands.
 */
export function parseGoalCommand(rawInput: string): GoalCommand {
  const input = rawInput.trim();
  if (!input) return { kind: "show" };
  const control = input.toLowerCase();
  if (control === "clear") return { kind: "clear" };
  if (control === "pause") return { kind: "pause" };
  if (control === "unlimited") return { kind: "unlimited" };
  if (control === "resume") return { kind: "resume" };
  if (control === "milestone") return { kind: "milestone-show" };
  if (/^milestone\s+clear$/iu.test(input)) return { kind: "milestone", value: null };
  if (/^milestone\s/iu.test(input)) return { kind: "milestone", value: input.slice(9).trim() };
  if (control === "edit") return { kind: "invalid-edit" };
  if (/^--duration(?:\s|$)/u.test(input)) return { kind: "duration", value: input.slice(10).trim() };
  if (/^--tokens(?:\s|$)/u.test(input)) return { kind: "tokens", value: input.slice(8).trim() };
  if (/^edit\s/iu.test(input)) return { kind: "edit", objective: input.slice(4).trim() };
  return { kind: "create", objective: input };
}

/** Commands that mean something from this exact live state. */
function commandHint(goal: GoalView): string {
  if (goal.phase === "active") {
    return goal.activation === "armed"
      ? "/goal edit <objective>, /goal milestone, /goal pause, /goal clear"
      : "/goal edit <objective>, /goal milestone, /goal resume, /goal clear";
  }
  if (goal.phase === "complete") return "/goal <objective>, /goal milestone, /goal clear";
  return "/goal edit <objective>, /goal milestone, /goal resume, /goal clear";
}

/**
 * Render a goal for a person.
 *
 * Compare-and-set internals (the revision the tools echo back) are deliberately
 * absent: a person has no use for them, and printing them invites hand-editing
 * of state whose whole purpose is to detect concurrent edits.
 */
function renderGoal(title: string, goal: GoalView): GoalCommandResult {
  const blocker = goal.blockedReason
    ? [`Blocker: ${goal.blockedReason.code}: ${goal.blockedReason.message}`]
    : [];
  return {
    ok: true,
    text: [
      title,
      `Status: ${goal.phase satisfies GoalPhase}`,
      ...blocker,
      `Objective: ${goal.objective}`,
      ...(goal.currentMilestone === undefined ? [] : [`Milestone: ${goal.currentMilestone}`]),
      `Rounds: ${goal.roundsStarted}/${goal.maxGoalRounds === DEFAULT_MAX_GOAL_ROUNDS ? 'unlimited' : goal.maxGoalRounds}`,
      `Total token admission cap: ${goal.maxTotalTokens ?? 'unlimited'}`,
      ...(goal.maxDurationMs === undefined ? [] : [`Wall-time limit: ${goal.maxDurationMs}ms from creation (includes pauses)`]),
      `Activation: ${goal.activation}`,
      "",
      `Commands: ${commandHint(goal)}`,
    ].join("\n"),
  };
}

const refOf = (goal: GoalView): GoalRef => ({ id: goal.id, revision: goal.revision });

const missingGoal = (action: string): GoalCommandResult => ({
  ok: false,
  text: `No goal is currently set; /goal ${action} requires one. ${GOAL_USAGE}`,
});

/**
 * Execute one `/goal` invocation against a session's durable metadata.
 *
 * Returns text rather than printing or emitting: the caller owns how it reaches
 * the person, and a pure function is testable without a socket.
 */
export function runGoalCommand(
  metadata: Record<string, unknown>,
  sessionId: string,
  rawInput: string,
  now: number = Date.now(),
): GoalCommandResult {
  const command = parseGoalCommand(rawInput);
  try {
    const current = getGoal(metadata, sessionId);
    switch (command.kind) {
      case "show":
        return current
          ? renderGoal("Goal", current)
          : { ok: true, text: `No goal is currently set.\n${GOAL_USAGE}` };
      case "invalid-edit":
        return { ok: false, text: `Goal editing requires a replacement objective.\n${GOAL_USAGE}` };
      case "create": {
        if (current && current.phase !== "complete") {
          return {
            ok: false,
            text:
              `A goal is already ${current.phase}. Use /goal edit <objective> to change it, `
              + "or /goal clear before replacing it.",
          };
        }
        return renderGoal("Goal created", createGoal(metadata, sessionId, { objective: command.objective }, now));
      }
      case "edit": {
        if (!current) return missingGoal("edit");
        // A completed goal is history. Editing it would rewrite what was
        // achieved; the honest reading of "edit" here is "start the next one".
        if (current.phase === "complete") {
          return renderGoal("Goal created", createGoal(metadata, sessionId, { objective: command.objective }, now));
        }
        return renderGoal(
          "Goal updated",
          editGoal(metadata, sessionId, refOf(current), { objective: command.objective }, now),
        );
      }
      case "duration": {
        if (!current) return missingGoal("--duration");
        const match = /^(\d+)(s|m|h)$/u.exec(command.value);
        const duration = match ? Number(match[1]) * ({ s: 1000, m: 60_000, h: 3_600_000 }[match[2]!] ?? 0) : 0;
        if (!Number.isSafeInteger(duration) || duration < 1) return { ok: false, text: `Use /goal --duration 30m (positive whole seconds, minutes or hours).` };
        return renderGoal("Goal time limit updated", editGoal(metadata, sessionId, refOf(current), { maxDurationMs: duration }, now));
      }
      case "tokens": {
        if (!current) return missingGoal("--tokens");
        const count = /^\d+$/u.test(command.value) ? Number(command.value) : 0;
        if (!Number.isSafeInteger(count) || count < 1) return { ok: false, text: 'Use /goal --tokens 100000 (a positive whole token count).' };
        return renderGoal('Goal token cap updated', editGoal(metadata, sessionId, refOf(current), { maxTotalTokens: count }, now));
      }
      case "unlimited": {
        if (!current) return missingGoal("unlimited");
        return renderGoal("Goal limits removed. Use /goal resume if blocked or paused.", editGoal(metadata, sessionId, refOf(current), { maxGoalRounds: DEFAULT_MAX_GOAL_ROUNDS, maxDurationMs: null, maxTotalTokens: null }, now));
      }
      case "pause":
        if (!current) return missingGoal("pause");
        return renderGoal("Goal paused", pauseGoal(metadata, sessionId, refOf(current), now));
      case "resume":
        if (!current) return missingGoal("resume");
        return renderGoal("Goal resumed", resumeGoal(metadata, sessionId, refOf(current), now));
      case "milestone-show":
        if (!current) return missingGoal("milestone");
        return renderGoal("Goal", current);
      case "milestone":
        if (!current) return missingGoal("milestone");
        return renderGoal(
          command.value === null ? "Goal milestone cleared" : "Goal milestone updated",
          setGoalMilestone(metadata, sessionId, refOf(current), command.value, now),
        );
      case "clear":
        if (!current) return { ok: true, text: "No goal to clear." };
        clearGoal(metadata, sessionId, refOf(current), now);
        return { ok: true, text: "Goal cleared." };
    }
  } catch (error) {
    if (error instanceof GoalError) {
      // The domain's own message names the exact transition it refused, which
      // is more useful than a generic "invalid for the current state".
      return { ok: false, text: `${error.message}\nRun /goal to see the available commands.` };
    }
    throw error;
  }
}
