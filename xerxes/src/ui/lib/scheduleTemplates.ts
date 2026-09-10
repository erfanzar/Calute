// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
export const SCHEDULE_TEMPLATES = [
  { name: 'Morning briefing', schedule: '0 9 * * 1-5', prompt: 'Summarize recent changes in this repository, open questions and the next useful tasks. Read only; do not modify files.' },
  { name: 'Weekly review', schedule: '0 16 * * 5', prompt: 'Review this week’s repository changes. Summarize completed work, risks, missing tests and priorities for next week. Read only.' },
  { name: 'Daily test report', schedule: '0 9 * * *', prompt: 'Read this repository’s instructions, run its documented test command and report failures with useful output. Do not change code or install dependencies.' },
  { name: 'Dependency review', schedule: '0 10 * * 1', prompt: 'Inspect dependency manifests and lockfiles. Report outdated or inconsistent dependencies with evidence. Do not install, upgrade or edit anything.' },
  { name: 'Documentation check', schedule: '0 14 * * 3', prompt: 'Compare public interfaces and examples with the repository documentation. Report concrete discrepancies and their file locations. Read only.' },
  { name: 'Workday reminder', schedule: '0 9 * * 1-5', prompt: 'Remind me to review my current project priorities and choose the next task. Keep the reminder brief.' },
] as const
export type ScheduleTemplate = { readonly name: string; readonly prompt: string; readonly schedule: string }
export function scheduleDescription(cron: string, timezone: string, interval?: number, at?: string): string {
  if (interval !== undefined) return Number.isSafeInteger(interval) && interval > 0 ? `Every ${interval % 3600 === 0 ? `${interval / 3600} hours` : interval % 60 === 0 ? `${interval / 60} minutes` : `${interval} seconds`}` : 'Enter a positive interval'
  if (at !== undefined) return at ? `Once at ${at}` : 'Choose a future date and time'
  const fields = cron.trim().split(/\s+/)
  if (fields.length !== 5) return 'Enter five cron fields'
  const [minute, hour, day, month, weekday] = fields as [string, string, string, string, string]
  const zone = timezone || 'UTC'
  if (day === '*' && month === '*') {
    if (/^\d+$/.test(minute) && Number(minute) < 60 && /^\d+$/.test(hour) && Number(hour) < 24) {
      const time = `${hour.padStart(2, '0')}:${minute.padStart(2, '0')}`
      if (weekday === '*') return `Every day at ${time} (${zone})`
      if (weekday === '1-5') return `Weekdays at ${time} (${zone})`
      const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
      if (/^[0-7]$/.test(weekday)) return `Every ${days[Number(weekday)]} at ${time} (${zone})`
    }
    if (hour === '*' && weekday === '*' && /^\*\/\d+$/.test(minute) && Number(minute.slice(2)) > 0 && Number(minute.slice(2)) < 60) return `Every ${minute.slice(2)} minutes within each hour (${zone})`
  }
  return `Custom cron: ${cron} (${zone})`
}
