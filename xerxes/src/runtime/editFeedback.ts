// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { relative, resolve } from 'node:path'
import type { ToolExecutor } from '../executors/toolRegistry.js'
import type { LspAdapter } from '../tools/claudeTools/search.js'
import { EditDiagnostics, diagnosticKey, formatEditDiagnostics, type Diagnostic } from './editDiagnostics.js'

const writes = new Set(['AppendFile', 'Edit', 'FileEditTool', 'NotebookEditTool', 'Write', 'WriteFile', 'append_file', 'edit_file', 'write_file'])
const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value)

/** Turn-owned baselines. A server failure never suppresses the independent checker report. */
export class EditFeedback {
  private readonly before = new Map<string, readonly Diagnostic[] | undefined>()
  private readonly changed = new Set<string>()
  constructor(private readonly cwd: string, private readonly checker = new EditDiagnostics(cwd), private readonly lsp?: LspAdapter) {}

  wrap(executor: ToolExecutor): ToolExecutor {
    return { execute: async (call, context, signal) => {
      let path: string | undefined
      if (writes.has(call.function.name)) {
        const inputs = call.function.arguments
        if (record(inputs)) {
          const candidate = inputs.file_path ?? inputs.path ?? inputs.notebook_path
          if (typeof candidate === 'string' && candidate.trim()) path = resolve(this.cwd, candidate)
        }
      }
      if (path) {
        await this.checker.noteFileWillChange(path)
        if (this.lsp && !this.before.has(path) && this.before.size < 32) {
          // Reserve before awaiting so repeated edits do not overwrite the original baseline.
          this.before.set(path, undefined)
          this.before.set(path, await this.collect(path, signal))
        }
        if (signal?.aborted) throw new DOMException('Edit cancelled', 'AbortError')
      }
      const result = await executor.execute(call, context, signal)
      if (path) this.changed.add(path)
      return result
    } }
  }

  async report(signal?: AbortSignal): Promise<string> {
    if (!this.changed.size) return ''
    const checker = await this.checker.report().catch(() => undefined)
    const reports = checker?.text ? [checker.text] : []
    if (!this.lsp || signal?.aborted) return reports.join('\n\n')
    let unconfirmed = 0
    for (const path of [...this.changed].slice(0, 8)) {
      if (signal?.aborted) break
      const after = await this.collect(path, signal)
      if (!after) { unconfirmed++; continue }
      const baseline = this.before.get(path)
      const keys = new Set(baseline?.map(diagnosticKey))
      const added = after.filter(item => !keys.has(diagnosticKey(item)))
      const formatted = formatEditDiagnostics([{ path, diagnostics: added }].filter(file => file.diagnostics.length), { cwd: this.cwd, source: 'lsp', maxChars: 1500 }).text
      if (formatted) reports.push(baseline ? formatted : formatted.replace('new problem(s)', 'current problem(s) (pre-edit baseline unconfirmed)'))
    }
    if (unconfirmed) reports.push(`[lsp] Diagnostics unconfirmed for ${unconfirmed} edited file(s); server unavailable, unsupported, or no matching-version publication. Checker results above are independent.`)
    if (this.changed.size > 8) reports.push('[lsp] Diagnostic report limited to 8 edited files.')
    return reports.join('\n\n').slice(0, 6000)
  }

  private async collect(path: string, signal?: AbortSignal): Promise<readonly Diagnostic[] | undefined> {
    try {
      const abort = AbortSignal.any([AbortSignal.timeout(1500), ...(signal ? [signal] : [])])
      const result = await this.lsp!.execute({ action: 'diagnostics', filePath: relative(this.cwd, path), line: 0, character: 0, diagnosticsWaitMs: 1000 }, abort)
      if (!record(result) || result.fresh !== true || !Array.isArray(result.diagnostics)) return undefined
      const diagnostics: Diagnostic[] = []
      for (const item of result.diagnostics) {
        if (!record(item) || typeof item.message !== 'string' || !record(item.range) || !record(item.range.start) || !record(item.range.end)) return undefined
        const start = item.range.start, end = item.range.end
        if (![start.line, start.character, end.line, end.character].every(value => Number.isSafeInteger(value) && Number(value) >= 0)) return undefined
        if (item.severity === 3 || item.severity === 4) continue
        diagnostics.push({ source: 'lsp', code: '', severity: item.severity === 2 ? 'warning' : 'error', message: item.message,
          range: { startLine: Number(start.line) + 1, startColumn: Number(start.character) + 1, endLine: Number(end.line) + 1, endColumn: Number(end.character) + 1 } })
      }
      return diagnostics
    } catch { return undefined }
  }
}
