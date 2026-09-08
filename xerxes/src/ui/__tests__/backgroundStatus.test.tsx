// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { EventEmitter } from 'node:events'
import { testRender } from '@opentui/react/test-utils'
import { act, useState } from 'react'
import { afterEach, expect, it, vi } from 'vitest'
import { BackgroundStatus } from '../opentui/backgroundStatus.js'
import { ActivityOverlay, activityAction } from '../opentui/activityOverlay.js'
import { parseActivity, recentOutcomes, type ActivityRow } from '../opentui/useActivity.js'
import { DEFAULT_THEME } from '../theme.js'
import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
const request = vi.fn()
vi.mock('../app/gatewayContext.js', () => ({ useOptionalGateway: () => gateway }))
const gw = Object.assign(new EventEmitter(), { request })
const gateway = { gw }
const shell: ActivityRow = { id:'shell-a',kind:'shell',title:'bun test',detail:'/repo',state:'running',startedAt:Date.now()-1000,endedAt:null,action:'stop',scope:'session' }
const watcher: ActivityRow = { ...shell,id:'watch-a',kind:'watcher',state:'watching',title:'Build completion' }
const reply = (rows: ActivityRow[]) => ({ ok:true, rows, omitted:0 })
afterEach(() => { request.mockReset(); gw.removeAllListeners(); resetOverlayState() })
it.each([40,220])('shows live counts at %s columns and opens unified activity', async width => {
  request.mockResolvedValue(reply([shell,{...shell,id:'b'},watcher]))
  const setup=await testRender(<BackgroundStatus sessionId="chat" t={DEFAULT_THEME}/>,{width,height:8})
  try {
    await act(async()=>{}); await setup.flush()
    expect(setup.captureCharFrame()).toContain('2 shells running')
    expect(setup.captureCharFrame()).toContain('1 watcher active')
    await setup.mockMouse.click(5,0); await setup.flush()
    expect(getOverlayState().activity).toBe(true)
    expect(request).toHaveBeenCalledTimes(1)
  } finally { act(()=>setup.renderer.destroy()) }
  expect(gw.listenerCount('background_changed')).toBe(0)
})
it('updates on lifecycle events while idle and clears stale counts on disconnect',async()=>{
  request.mockResolvedValueOnce(reply([shell])).mockResolvedValue(reply([{...shell,state:'failed',endedAt:Date.now(),action:null}]))
  const setup=await testRender(<BackgroundStatus sessionId="chat" t={DEFAULT_THEME}/>,{width:80,height:6})
  try {
    await act(async()=>{}); await setup.flush()
    expect(setup.captureCharFrame()).toContain('1 shell running')
    await act(async()=>{gw.emit('background_changed')}); await setup.flush()
    expect(setup.captureCharFrame()).toContain('1 failed')
    expect(setup.captureCharFrame()).not.toContain('shell running')
    await act(async()=>{gw.emit('close')}); await setup.flush()
    expect(setup.captureCharFrame()).toContain('status unavailable')
    expect(setup.captureCharFrame()).not.toContain('failed')
  } finally { act(()=>setup.renderer.destroy()) }
})
it('ignores old replies after switching chats',async()=>{
  let resolveOld!:(value:unknown)=>void, switchSession!:(value:string)=>void
  request.mockReturnValueOnce(new Promise(resolve=>{resolveOld=resolve})).mockResolvedValue(reply([]))
  function Harness(){const [sid,setSid]=useState('old');switchSession=setSid;return <BackgroundStatus sessionId={sid} t={DEFAULT_THEME}/>}
  const setup=await testRender(<Harness/>,{width:80,height:6})
  try {
    await act(async()=>{});await setup.flush()
    await act(async()=>{switchSession('new')});await setup.flush()
    await act(async()=>{resolveOld(reply([shell]))});await setup.flush()
    expect(setup.captureCharFrame().trim()).toBe('')
    expect(request).toHaveBeenLastCalledWith('background.activity',{session_id:'new'})
  }finally{act(()=>setup.renderer.destroy())}
})
it('validates input and limits outcome badges to thirty seconds',()=>{
  expect(()=>parseActivity({ok:true,rows:[{}]})).toThrow()
  expect(recentOutcomes([{...shell,endedAt:100}],30100)).toHaveLength(0)
  expect(recentOutcomes([{...shell,endedAt:100}],200)).toHaveLength(1)
  expect(activityAction(watcher)).toEqual({method:'monitor.stop',params:{monitor_id:'watch-a'}})
})
it('shows command, elapsed time and sends a session-scoped stop from the activity panel',async()=>{
  request.mockImplementation((method:string)=>Promise.resolve(method==='terminal.control'?{ok:true}:method==='terminal.inspect'?{ok:true,terminal:{output:'12 tests passed'}}:reply([shell])))
  const setup=await testRender(<ActivityOverlay sessionId="chat" t={DEFAULT_THEME}/>,{width:100,height:35})
  try{
    await act(async()=>{});await setup.flush()
    expect(setup.captureCharFrame()).toContain('bun test')
    expect(setup.captureCharFrame()).toContain('shell · running')
    await act(async()=>setup.mockInput.pressEnter());await setup.flush();await act(async()=>{});await setup.flush()
    expect(setup.captureCharFrame()).toContain('12 tests passed')
    await act(async()=>setup.mockInput.pressKey('ESCAPE'));await setup.flush()
    await setup.mockInput.pressKey('s');await setup.flush()
    expect(request).toHaveBeenCalledWith('terminal.control',{terminal_id:'shell-a',action:'kill',session_id:'chat'})
  }finally{act(()=>setup.renderer.destroy())}
})
