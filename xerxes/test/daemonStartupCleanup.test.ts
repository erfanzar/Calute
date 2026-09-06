// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";
import { mkdtemp, rm, stat } from "node:fs/promises";
import { join } from "node:path";
import { InMemoryDaemonRuntime } from "../src/daemon/runtime.js";
import { DaemonServer } from "../src/daemon/server.js";

test("failed listen preserves its error and can be stopped repeatedly", async () => {
  const directory = await mkdtemp("/tmp/xerxes-startup-");
  let shutdowns = 0;
  const runtime = new InMemoryDaemonRuntime();
  runtime.shutdown = async () => { shutdowns++; };
  const server = new DaemonServer({
    // Exceeds the Unix-domain socket path limit without exceeding NAME_MAX.
    socketPath: join(directory, "s".repeat(150)), runtime,
  });
  try {
    let failure: unknown;
    try { await server.start(); } catch (error) { failure = error; }
    expect(failure).toBeInstanceOf(Error);
    expect(String(failure)).toMatch(/listen|ENAMETOOLONG|EINVAL/);
    expect(String(failure)).not.toContain("Server is not running");
    await Promise.all([server.stop(), server.stop()]);
    expect(shutdowns).toBe(1);
  } finally {
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("concurrent shutdown drains and closes a listening daemon once", async () => {
  const directory = await mkdtemp("/tmp/xerxes-startup-");
  const socketPath = join(directory, "daemon.sock");
  const runtime = new InMemoryDaemonRuntime();
  let flushes = 0;
  let shutdowns = 0;
  runtime.flushSessions = async () => { flushes++; await Bun.sleep(10); };
  runtime.shutdown = async () => { shutdowns++; };
  const server = new DaemonServer({ socketPath, runtime });
  try {
    await server.start();
    await Promise.all([server.stop(), server.stop(), server.stop()]);
    await server.stop();
    expect(flushes).toBe(1);
    expect(shutdowns).toBe(1);
    expect(await stat(socketPath).catch(() => null)).toBeNull();
  } finally {
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("later startup failure retains the original error when cleanup also fails", async () => {
  const directory = await mkdtemp("/tmp/xerxes-startup-");
  const failure = new Error("webhook listener could not bind");
  const cleanupFailure = new Error("webhook cleanup failed");
  const socketPath = join(directory, "daemon.sock");
  let shutdowns = 0;
  const runtime = new InMemoryDaemonRuntime();
  runtime.shutdown = async () => { shutdowns++; };
  const server = new DaemonServer({
    socketPath,
    runtime,
    monitorWebhookServer: {
      start() { throw failure; },
      async stop() { throw cleanupFailure; },
    },
  });
  try {
    await expect(server.start()).rejects.toBe(failure);
    expect(await stat(socketPath).catch(() => null)).toBeNull();
    expect(shutdowns).toBe(1);
  } finally {
    await server.stop().catch(() => {});
    await rm(directory, { recursive: true, force: true });
  }
});
