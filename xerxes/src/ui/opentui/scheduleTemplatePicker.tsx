// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useKeyboard } from '@opentui/react'
import { useState } from 'react'
import type { Theme } from '../theme.js'
import { SCHEDULE_TEMPLATES, scheduleDescription, type ScheduleTemplate } from '../lib/scheduleTemplates.js'
import { Box, Text } from './primitives.js'
import { DialogHeader, DialogFooter } from './dialogChrome.js'
export function ScheduleTemplatePicker({t, onSelect, onClose}: {t: Theme; onSelect: (template: ScheduleTemplate) => void; onClose: () => void}) {
  const [selected, setSelected] = useState(0)
  const template = SCHEDULE_TEMPLATES[selected]!
  useKeyboard(key => {
    if (key.eventType === 'release' || !['escape','up','down','return'].includes(key.name)) return
    key.preventDefault(); key.stopPropagation()
    if (key.name === 'escape') onClose()
    else if (key.name === 'return') onSelect(template)
    else setSelected(value => (value + (key.name === 'up' ? SCHEDULE_TEMPLATES.length - 1 : 1)) % SCHEDULE_TEMPLATES.length)
  })
  return <Box flexDirection="column" flexGrow={1} minHeight={0}>
    <DialogHeader t={t} title="Schedule templates" subtitle="Choose a starting point. Review the prompt and timing before saving." />
    {SCHEDULE_TEMPLATES.map((row, index) => <Box key={row.name} backgroundColor={selected === index ? t.color.completionCurrentBg : undefined} onMouseDown={() => setSelected(index)}><Text color={selected === index ? t.color.accent : t.color.text}>{selected === index ? '› ' : '  '}{row.name}</Text></Box>)}
    <scrollbox flexGrow={1} minHeight={0} contentOptions={{ flexDirection: 'column' }}><Text color={t.color.accent} wrap="wrap">{scheduleDescription(template.schedule, 'UTC')}</Text><Text wrap="wrap">{template.prompt}</Text><Text color={t.ds.meta}>Creates an editable draft, initially paused.</Text></scrollbox>
    <DialogFooter t={t}><Text color={t.ds.secondary}>↑↓ choose · Enter use template · Esc back</Text></DialogFooter>
  </Box>
}
