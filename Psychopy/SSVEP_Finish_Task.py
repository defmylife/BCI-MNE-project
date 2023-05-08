#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2022.2.5),
    on April 29, 2023, at 15:18
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# #Global Variable
# HzFrequenceleftrectangle = 12
# HzFrequencerightrectangle = 20
# Display_Hz = win.getActualFrameRate()
# num_repeat_trails = 16.0

# #Duration_of_time
# start_time_Instruction = 0.0
# end_time_Instruction = 2.0
# start_time_Cue = 0.0
# end_time_Cue = 0.5
# start_time_Stimulit = 0.0
# end_time_Stimulit = 5.0
# start_time_Blank = 0.0
# end_time_Blank = 0.5


#Red Block Append 
#outlet.push_sample([1]) = red block append left
#outlet.push_sample([2]) = red block append right

###### Set up LabStreamingLayer stream #####
from pylsl import StreamInfo, StreamOutlet

info = StreamInfo(name='psychopy_marker_stream', type='Markers', channel_count=1,
    channel_format='int32', source_id='myuniqueid1234')
outlet = StreamOutlet(info) 
#############################################

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2022.2.5'
expName = 'SSVEP_Finish_Task'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
}
# --- Show participant info dialog --
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='C:\OpenBCI_LSL\Psychopy\Finish_code_psychopy\SSVEP_Finish_Task.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# --- Setup the Window ---
win = visual.Window(
    size=(1024, 768), fullscr= True, screen=0, 
    winType='pyglet', allowStencil=False,
    monitor='testMonitor', color= 'black' , colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
win.mouseVisible = False
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
Display_Hz = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess
# --- Setup input devices ---
ioConfig = {}

# Setup iohub keyboard
ioConfig['Keyboard'] = dict(use_keymap='psychopy')

ioSession = '1'
if 'session' in expInfo:
    ioSession = str(expInfo['session'])
ioServer = io.launchHubServer(window=win, **ioConfig)
eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard(backend='iohub')

#Global Variab
L_blink = Display_Hz/6.0
R_blink = Display_Hz/10.0
num_repeat_trails = 8.0
squareSize = 0.5
position_of_left_square_x = 0.35
position_of_left_square_y = 0.0

#Duration_of_time
start_time_Instruction = 0.0
end_time_Instruction = 10.0
start_time_Cue = 0.0
end_time_Cue = 1.5
start_time_Stimulit = 0.0
end_time_Stimulit = 10.0
start_time_Blank = 0.0
end_time_Blank = 0.5

# --- Initialize components for Routine "Instruction" ---
text = visual.TextStim(win=win, name='text',
    text='Ready and Prepare \nfor the test',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# --- Initialize components for Routine "Cue" ---
Red_rectangle = visual.Rect(
    win=win, name='Red_rectangle',
    width=(squareSize, squareSize)[0], height=(squareSize, squareSize)[1],
    ori=0.0, pos=(-position_of_left_square_x, position_of_left_square_y), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor=[1.0000, -1.0000, -1.0000], fillColor=[1.0000, -1.0000, -1.0000],
    opacity=None, depth=0.0, interpolate=True)
white_rectangle_right = visual.Rect(
    win=win, name='white_rectangle_right',
    width=(squareSize, squareSize)[0], height=(squareSize, squareSize)[1],
    ori=0.0, pos=(position_of_left_square_x, position_of_left_square_y), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-2.0, interpolate=True)

# --- Initialize components for Routine "Stimulit" ---
Rectangle_12_Hz = visual.Rect(
    win=win, name='Rectangle_12_Hz',
    width=(squareSize, squareSize)[0], height=(squareSize, squareSize)[1],
    ori=0.0, pos=(-0.35, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=1.0, depth=0.0, interpolate=True)
Rectangle_20_Hz = visual.Rect(
    win=win, name='Rectangle_20_Hz',
    width=(squareSize, squareSize)[0], height=(squareSize, squareSize)[1],
    ori=0.0, pos=(0.35, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=1.0, depth=-1.0, interpolate=True)

# --- Initialize components for Routine "Blank" ---
End_repeat = visual.TextStim(win=win, name='End_repeat',
    text=None,
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine 

# --- Prepare to start Routine "Instruction" ---
continueRoutine = True
routineForceEnded = False
# update component parameters for each repeat
# keep track of which components have finished
InstructionComponents = [text]
for thisComponent in InstructionComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "Instruction" ---
while continueRoutine and routineTimer.getTime() < end_time_Instruction:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text* updates
    if text.status == NOT_STARTED and tThisFlip >= start_time_Instruction-frameTolerance:
        # keep track of start time/frame for later
        text.frameNStart = frameN  # exact frame index
        text.tStart = t  # local t and not account for scr refresh
        text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'text.started')
        text.setAutoDraw(True)
    
    # ####### Tell psychopy to send event marker ##########
    # if frameN ==1:
    #     outlet.push_sample([1])
    # ######################################################   
    
    if text.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > text.tStartRefresh + end_time_Instruction-frameTolerance:
            # keep track of stop time/frame for later
            text.tStop = t  # not accounting for scr refresh
            text.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.stopped')
            text.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in InstructionComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "Instruction" ---
for thisComponent in InstructionComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
if routineForceEnded:
    routineTimer.reset()
else:
    routineTimer.addTime(-end_time_Instruction)

# set up handler to look after randomisation of conditions etc
Repeat_32_Trials = data.TrialHandler(nReps= num_repeat_trails, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('position_of_red_rectangle.xlsx'),
    seed=None, name='Repeat_32_Trials')
thisExp.addLoop(Repeat_32_Trials)  # add the loop to the experiment
thisRepeat_32_Trial = Repeat_32_Trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisRepeat_32_Trial.rgb)
if thisRepeat_32_Trial != None:
    for paramName in thisRepeat_32_Trial:
        exec('{} = thisRepeat_32_Trial[paramName]'.format(paramName))

for thisRepeat_32_Trial in Repeat_32_Trials:
    currentLoop = Repeat_32_Trials
    # abbreviate parameter names if possible (e.g. rgb = thisRepeat_32_Trial.rgb)
    if thisRepeat_32_Trial != None:
        for paramName in thisRepeat_32_Trial:
            exec('{} = thisRepeat_32_Trial[paramName]'.format(paramName))
    
    # --- Prepare to start Routine "Cue" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    
    # text.setText('Turn left')
    white_rectangle_right.setPos((position_white_x, position_white_y))
    Red_rectangle.setPos((position_red_x, position_red_y))
    
    # keep track of which components have finished
    CueComponents = [Red_rectangle, white_rectangle_right]
    for thisComponent in CueComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Cue" ---
    while continueRoutine and routineTimer.getTime() < end_time_Cue:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Red_rectangle* updates
        if Red_rectangle.status == NOT_STARTED and tThisFlip >= start_time_Cue-frameTolerance:
            # keep track of start time/frame for later
            Red_rectangle.frameNStart = frameN  # exact frame index
            Red_rectangle.tStart = t  # local t and not account for scr refresh
            Red_rectangle.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Red_rectangle, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Red_rectangle.started')
            Red_rectangle.setAutoDraw(True)  
        
        ####### Tell psychopy to send event marker ##########
        if frameN ==1:
            if position_red_x == -position_of_left_square_x and position_red_y == position_of_left_square_y:
                outlet.push_sample([1]) # left cue = marker no.1
            elif position_red_x == position_of_left_square_x and position_red_y == position_of_left_square_y:
                outlet.push_sample([4]) # right cue = marker no.4
        ######################################################   
        
        if Red_rectangle.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Red_rectangle.tStartRefresh + end_time_Cue-frameTolerance:
                # keep track of stop time/frame for later
                Red_rectangle.tStop = t  # not accounting for scr refresh
                Red_rectangle.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Red_rectangle.stopped')
                Red_rectangle.setAutoDraw(False)
        
        # *white_rectangle_right* updates
        if white_rectangle_right.status == NOT_STARTED and tThisFlip >= start_time_Cue-frameTolerance:
            # keep track of start time/frame for later
            white_rectangle_right.frameNStart = frameN  # exact frame index
            white_rectangle_right.tStart = t  # local t and not account for scr refresh
            white_rectangle_right.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(white_rectangle_right, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'white_rectangle_right.started')
            white_rectangle_right.setAutoDraw(True)
        if white_rectangle_right.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > white_rectangle_right.tStartRefresh + end_time_Cue-frameTolerance:
                # keep track of stop time/frame for later
                white_rectangle_right.tStop = t  # not accounting for scr refresh
                white_rectangle_right.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'white_rectangle_right.stopped')
                white_rectangle_right.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in CueComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Cue" ---
    for thisComponent in CueComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-end_time_Cue)
    
    # --- Prepare to start Routine "Stimulit" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    # keep track of which components have finished
    StimulitComponents = [Rectangle_12_Hz, Rectangle_20_Hz]
    for thisComponent in StimulitComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Stimulit" ---
    while continueRoutine and routineTimer.getTime() < end_time_Stimulit:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame 
        
        # *Rectangle_12_Hz* updates
        if Rectangle_12_Hz.status == NOT_STARTED and tThisFlip >= start_time_Stimulit-frameTolerance:
            # keep track of start time/frame for later
            Rectangle_12_Hz.frameNStart = frameN  # exact frame index
            Rectangle_12_Hz.tStart = t  # local t and not account for scr refresh
            Rectangle_12_Hz.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Rectangle_12_Hz, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Rectangle_12_Hz.started')
            Rectangle_12_Hz.setAutoDraw(True)
            
        ####### Tell psychopy to send event marker ##########
        if frameN ==1:
            if position_red_x == -position_of_left_square_x and position_red_y == position_of_left_square_y:
                outlet.push_sample([2]) # left stimuli = marker no.2
            elif position_red_x == position_of_left_square_x and position_red_y == position_of_left_square_y:
                outlet.push_sample([5]) # right stimuli = marker no.5
        ###################################################### 
            
        if Rectangle_12_Hz.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Rectangle_12_Hz.tStartRefresh + end_time_Stimulit-frameTolerance:
                # keep track of stop time/frame for later
                Rectangle_12_Hz.tStop = t  # not accounting for scr refresh
                Rectangle_12_Hz.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Rectangle_12_Hz.stopped')
                Rectangle_12_Hz.setAutoDraw(False)
        if Rectangle_12_Hz.status == STARTED:  # only update if drawing
            Rectangle_12_Hz.setOpacity((frameN % (L_blink)*2) >= (L_blink), log=False)
        
        # *Rectangle_20_Hz* updates
        if Rectangle_20_Hz.status == NOT_STARTED and tThisFlip >= start_time_Stimulit-frameTolerance:
            # keep track of start time/frame for later
            Rectangle_20_Hz.frameNStart = frameN  # exact frame index
            Rectangle_20_Hz.tStart = t  # local t and not account for scr refresh
            Rectangle_20_Hz.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Rectangle_20_Hz, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Rectangle_20_Hz.started')
            Rectangle_20_Hz.setAutoDraw(True)
        if Rectangle_20_Hz.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Rectangle_20_Hz.tStartRefresh + end_time_Stimulit-frameTolerance:
                # keep track of stop time/frame for later
                Rectangle_20_Hz.tStop = t  # not accounting for scr refresh
                Rectangle_20_Hz.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Rectangle_20_Hz.stopped')
                Rectangle_20_Hz.setAutoDraw(False)
        if Rectangle_20_Hz.status == STARTED:  # only update if drawing
            Rectangle_20_Hz.setOpacity((frameN % (R_blink)*2) >= (R_blink), log=False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in StimulitComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Stimulit" ---
    for thisComponent in StimulitComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-end_time_Stimulit)
    
    # --- Prepare to start Routine "Blank" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    # keep track of which components have finished
    BlankComponents = [End_repeat]
    for thisComponent in BlankComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Blank" ---
    while continueRoutine and routineTimer.getTime() < end_time_Blank:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *End_repeat* updates
        if End_repeat.status == NOT_STARTED and tThisFlip >= start_time_Blank-frameTolerance:
            # keep track of start time/frame for later
            End_repeat.frameNStart = frameN  # exact frame index
            End_repeat.tStart = t  # local t and not account for scr refresh
            End_repeat.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(End_repeat, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'End_repeat.started')
            End_repeat.setAutoDraw(True)
            
        ####### Tell psychopy to send event marker ##########
        if frameN ==1:
            if position_red_x == -position_of_left_square_x and position_red_y == position_of_left_square_y:
                outlet.push_sample([3]) # left blank = marker no.3
            elif position_red_x == position_of_left_square_x and position_red_y == position_of_left_square_y:
                outlet.push_sample([6]) # right blank = marker no.6
        ######################################################  
        
        if End_repeat.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > End_repeat.tStartRefresh + end_time_Blank-frameTolerance:
                # keep track of stop time/frame for later
                End_repeat.tStop = t  # not accounting for scr refresh
                End_repeat.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'End_repeat.stopped')
                End_repeat.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in BlankComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Blank" ---
    for thisComponent in BlankComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-end_time_Blank)
    thisExp.nextEntry()
    
# completed 32.0 repeats of 'Repeat_32_Trials'


# --- End experiment ---
# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
if eyetracker:
    eyetracker.setConnectionState(False)
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()

