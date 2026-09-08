// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** HTML proxy/challenge pages are not useful model errors and may contain client IPs. */
export function httpErrorBody(body: string): string {
  if (/<!doctype\s+html|<html[\s>]/i.test(body)) {
    return /cloudflare/i.test(body)
      ? 'Request blocked by Cloudflare. Check this provider endpoint and network access.'
      : 'Provider returned an HTML error page. Check the configured endpoint and network access.'
  }
  return body.slice(0, 4096)
}
