// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";

import { parseGoalCommand, runGoalCommand } from "../src/daemon/goalCommand.js";
import { blockGoal, completeGoal, getGoal, resetGoalActivations } from "../src/runtime/goalDomain.js";

function fresh(): Record<string, unknown> {
  resetGoalActivations();
  return {};
}

test('goal token command validates a positive whole cap and preserves the goal identity', () => {
  const metadata = fresh()
  runGoalCommand(metadata, 'token-command', 'finish work', 1000)
  const id = getGoal(metadata, 'token-command')!.id
  expect(runGoalCommand(metadata, 'token-command', '--tokens 100000', 2000).ok).toBe(true)
  expect(getGoal(metadata, 'token-command')).toMatchObject({ id, maxTotalTokens: 100000 })
  const before = JSON.stringify(metadata)
  expect(runGoalCommand(metadata, 'token-command', '--tokens -1', 2000).ok).toBe(false)
  expect(runGoalCommand(metadata, 'token-command', '--tokens 1.2', 2000).ok).toBe(false)
  expect(JSON.stringify(metadata)).toBe(before)
})

test('goal duration sets an explicit wall-time limit and rejects malformed values without mutation', () => {
  const metadata = fresh()
  runGoalCommand(metadata, 'duration', 'finish work', 1000)
  expect(runGoalCommand(metadata, 'duration', '--duration 30m', 2000).ok).toBe(true)
  expect(getGoal(metadata, 'duration')?.maxDurationMs).toBe(1_800_000)
  const before = JSON.stringify(metadata)
  expect(runGoalCommand(metadata, 'duration', '--duration 0s', 2000).ok).toBe(false)
  expect(runGoalCommand(metadata, 'duration', '--duration later', 2000).ok).toBe(false)
  expect(JSON.stringify(metadata)).toBe(before)
})

test("only exact control words are subcommands; everything else is an objective", () => {
  expect(parseGoalCommand("")).toEqual({ kind: "show" });
  expect(parseGoalCommand("  ")).toEqual({ kind: "show" });
  expect(parseGoalCommand("pause")).toEqual({ kind: "pause" });
  expect(parseGoalCommand("RESUME")).toEqual({ kind: "resume" });
  expect(parseGoalCommand("edit")).toEqual({ kind: "invalid-edit" });
  expect(parseGoalCommand("edit ship the release")).toEqual({
    kind: "edit",
    objective: "ship the release",
  });
  // Prose that merely begins with a control word is an objective. A goal is
  // written in English, and "clear the backlog" is not a request to discard
  // the goal.
  expect(parseGoalCommand("clear the backlog")).toEqual({
    kind: "create",
    objective: "clear the backlog",
  });
  expect(parseGoalCommand("pause the ingestion job")).toEqual({
    kind: "create",
    objective: "pause the ingestion job",
  });
  expect(parseGoalCommand("milestone")).toEqual({ kind: "milestone-show" });
  expect(parseGoalCommand("milestone ship the migration")).toEqual({ kind: "milestone", value: "ship the migration" });
  expect(parseGoalCommand("milestone clear")).toEqual({ kind: "milestone", value: null });
});

test('milestone command shows, sets and clears the current value', () => {
  const metadata = fresh()
  runGoalCommand(metadata, 'milestone-session', 'ship the feature', 1)
  expect(runGoalCommand(metadata, 'milestone-session', 'milestone', 2).text).not.toContain('Milestone:')
  const set = runGoalCommand(metadata, 'milestone-session', 'milestone implement the API', 3)
  expect(set.ok).toBe(true)
  expect(set.text).toContain('Milestone: implement the API')
  const shown = runGoalCommand(metadata, 'milestone-session', 'milestone', 4)
  expect(shown.text).toContain('Milestone: implement the API')
  const cleared = runGoalCommand(metadata, 'milestone-session', 'milestone clear', 5)
  expect(cleared.ok).toBe(true)
  expect(cleared.text).not.toContain('Milestone: implement the API')
})

test('milestone remains visible as the last milestone after completion and cannot be changed', () => {
  const metadata = fresh()
  runGoalCommand(metadata, 'milestone-complete', 'ship the feature', 1)
  runGoalCommand(metadata, 'milestone-complete', 'milestone release candidate built', 2)
  runGoalCommand(metadata, 'milestone-complete', 'pause', 3)
  const goal = getGoal(metadata, 'milestone-complete')!
  completeGoal(metadata, 'milestone-complete', { id: goal.id, revision: goal.revision }, 4)
  expect(runGoalCommand(metadata, 'milestone-complete', '', 5).text).toContain('Milestone: release candidate built')
  const rejected = runGoalCommand(metadata, 'milestone-complete', 'milestone changed after completion', 6)
  expect(rejected.ok).toBe(false)
})

test("the full human lifecycle runs through the same domain the tools use", () => {
  const metadata = fresh();
  expect(runGoalCommand(metadata, "s1", "", 1).text).toContain("No goal is currently set");

  const created = runGoalCommand(metadata, "s1", "migrate the store", 2);
  expect(created.ok).toBe(true);
  expect(created.text).toContain("Goal created");
  expect(created.text).toContain("Objective: migrate the store");
  expect(created.text).toContain("Activation: armed");
  expect(getGoal(metadata, "s1")?.phase).toBe("active");

  // A second create must not silently discard work in flight.
  const second = runGoalCommand(metadata, "s1", "something else", 3);
  expect(second.ok).toBe(false);
  expect(second.text).toContain("already active");

  const edited = runGoalCommand(metadata, "s1", "edit migrate the store safely", 4);
  expect(edited.text).toContain("Objective: migrate the store safely");

  const paused = runGoalCommand(metadata, "s1", "pause", 5);
  expect(paused.text).toContain("Status: paused");
  expect(getGoal(metadata, "s1")?.activation).toBe("disarmed");

  const resumed = runGoalCommand(metadata, "s1", "resume", 6);
  expect(resumed.text).toContain("Status: active");
  expect(getGoal(metadata, "s1")?.activation).toBe("armed");

  expect(runGoalCommand(metadata, "s1", "clear", 7).text).toBe("Goal cleared.");
  expect(getGoal(metadata, "s1")).toBeUndefined();
});

test("operations that need a goal say so instead of failing opaquely", () => {
  const metadata = fresh();
  for (const action of ["pause", "resume", "edit new objective"]) {
    const result = runGoalCommand(metadata, "s1", action, 1);
    expect(result.ok).toBe(false);
    expect(result.text).toContain("No goal is currently set");
  }
  const invalid = runGoalCommand(metadata, "s1", "edit", 1);
  expect(invalid.ok).toBe(false);
  expect(invalid.text).toContain("replacement objective");
});

test("a refused transition reports the domain's own reason, not a generic error", () => {
  const metadata = fresh();
  runGoalCommand(metadata, "s1", "ship it", 1);
  // Already active and armed: resume has nothing to do, and says which goal.
  const resumed = runGoalCommand(metadata, "s1", "resume", 2);
  expect(resumed.ok).toBe(false);
  expect(resumed.text).toContain("already active and armed");
  expect(resumed.text).toContain("Run /goal");
});

test("editing a completed goal starts the next one rather than rewriting history", () => {
  const metadata = fresh();
  runGoalCommand(metadata, "s1", "first objective", 1);
  runGoalCommand(metadata, "s1", "pause", 2);
  const before = getGoal(metadata, "s1")!;
  // Complete it the way the tools would, then edit.
  completeGoal(metadata, "s1", { id: before.id, revision: before.revision }, 3);

  const edited = runGoalCommand(metadata, "s1", "edit second objective", 4);
  expect(edited.text).toContain("Goal created");
  expect(edited.text).toContain("Objective: second objective");
  expect(getGoal(metadata, "s1")?.phase).toBe("active");
});

test('unlimited removes persisted caps without losing progress and allows a blocked goal to resume', () => {
  const metadata = fresh()
  runGoalCommand(metadata, 'uncapped', 'finish the work', 1000)
  runGoalCommand(metadata, 'uncapped', '--tokens 2000000', 1001)
  runGoalCommand(metadata, 'uncapped', '--duration 30m', 1002)
  runGoalCommand(metadata, 'uncapped', 'milestone verify results', 1003)
  blockGoal(metadata, 'uncapped', getGoal(metadata, 'uncapped')!, { code: 'token-budget', message: 'Goal token budget exhausted (2090726/2000000)' }, 1004)
  const before = getGoal(metadata, 'uncapped')!
  const result = runGoalCommand(metadata, 'uncapped', 'unlimited', 1005)
  expect(result.ok).toBe(true)
  expect(result.text).toContain('Rounds: 0/unlimited')
  expect(result.text).toContain('Total token admission cap: unlimited')
  const restored = JSON.parse(JSON.stringify(metadata))
  expect(getGoal(restored, 'uncapped')).toMatchObject({ id: before.id, objective: before.objective, currentMilestone: 'verify results', phase: 'blocked', roundsStarted: 0 })
  expect(getGoal(restored, 'uncapped')?.maxTotalTokens).toBeUndefined()
  expect(getGoal(restored, 'uncapped')?.maxDurationMs).toBeUndefined()
  expect(runGoalCommand(restored, 'uncapped', 'resume', 1006).ok).toBe(true)
})
