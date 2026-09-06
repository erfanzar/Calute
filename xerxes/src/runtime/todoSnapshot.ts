// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export function parseTodoList(text: string): readonly Record<string, unknown>[] {
  const items: Record<string, unknown>[] = []
  for (const line of text.split('\n')) {
    const match = /^\s*(\d+)\.\s+\[([ x~])\]\s+(.*\S)\s*$/.exec(line)
    if (!match) continue
    const [, index, mark, content] = match
    items.push({
      content,
      id: `todo-${index}`,
      status: mark === 'x' ? 'completed' : mark === '~' ? 'in_progress' : 'pending',
    })
  }
  return items
}


export function todosFromExecutions(executions: readonly unknown[]): readonly Record<string, unknown>[] {
  for (let i = executions.length - 1; i >= 0; i--) {
    const value = executions[i];
    if (!value || typeof value !== "object") continue;
    const row = value as Record<string, unknown>;
    if (row.name !== "TodoWriteTool" || row.permitted === false || row.error) continue;
    const result = row.result ?? row.return_value;
    if (typeof result === "string") return parseTodoList(result);
  }
  return [];
}
