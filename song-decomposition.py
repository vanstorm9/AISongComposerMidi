import midi
import numpy as np

#path = 'example.mid'
path = 'Songs/my-heart-will-go-on-titanic.mid'
#path = 'Songs/Suteki-Da-Ne.mid'
#path = 'Songs/10_little_indians.mid'
#path = 'training-ground/twinkle_twinkle.mid'
#path = 'result.mid'
#path = 'result-sdn-4-2333-d10.mid'
#path = 'Songs/1-2-3_ngoi_sao.mid'
#path = 'Songs/twinkle_twinkle.mid'
#path = 'Songs/waldstein_2.mid'
#path = 'Songs/elise.mid'
#path = 'Songs/Mozart-Movement.mid'
#path = 'Songs/london-bridges.mid'
#path = 'Songs/grenade.mid'
pattern = midi.read_midifile(path)
#print pattern


# Goes through extracted song and reconstruct them (pattern[1])
# Generic

#tr = 0
limit = 200
tr = 0
start_val = 1
i = 1

# Grenade sample window
'''
tr = 5
start_val = 80
i = 80
'''

# Suteki Da Ne sample window
'''
tr = 2
start_val = 1
i = 1
'''

#print pattern[1]

print 'Extracting all of pattern[1]'

# Instantiate a MIDI Pattern (contains a list of tracks)
pat = midi.Pattern()
note_on = 0
note_off = 0

# Instantiate a MIDI Track (contains a list of MIDI events)
track = midi.Track()

# Append the track to the pattern
pat.append(track)


while True:
    
    
    print 'tr: ', tr

    if tr > len(pattern) - 1:
    #if tr > 0:
        print 'breaking. . .'
        break
        


    '''
    while True:
        if i > len(pattern[tr]) - 2:
            break
        
        track.append(pattern[tr][i])

        i = i + 1

    '''
    i = 0
    ii = 0
    tick_ar = np.array([])
    pitch_ar = np.array([])
    velocity_ar = np.array([])
    
    
    while True:
        #print pattern[tr][i]
        note_type = pattern[tr][i].name
        tick = pattern[tr][i].tick
        if len(pattern[tr][i].data) == 0:
            print 'skipped'
            ii = ii + 1
            break
        pitch = pattern[tr][i].data[0]
        if note_type == 'Note On' or note_type == 'Note Off':
            #print i
            if ii > len(pattern[tr])-5:
            #if i > len(pattern[tr]) - 2:
                break

            
            
            #print tick
            # Because some pattern[][].data does not have a second array element
            if len(pattern[tr][i].data) == 2:
                velocity = pattern[tr][i].data[1]
            else:
                velocity = 0




            # Place all of tick, pitch, and velocity values in indiviudal vectors
            #tick = np.array([tick])
            #pitch = np.array([pitch])
            #velocity = np.array([velocity])

            
            if i == start_val:
                tick_ar = np.array([tick])
                pitch_ar = np.array([pitch])
                velocity_ar = np.array([velocity])
            else:
                tick_ar = np.concatenate((tick_ar, np.array([tick])))
                pitch_ar = np.concatenate((pitch_ar, np.array([pitch])))
                velocity_ar = np.concatenate((velocity_ar, np.array([velocity])))

            ii = ii + 1
            
        #tick = pattern[tr][i].tick
        
        # To reconstruct the entire song in its (piano-like) original form
        
        if note_type == 'Note On':
            channel = pattern[tr][i].channel
            data = pattern[tr][i].data
            #track.append(midi.NoteOnEvent(tick= tick, channel=channel, data=[np.array(pitch), velocity]))
            track.append(midi.NoteOnEvent(tick= tick, channel=channel, data=data))
            note_on = note_on + 1
        elif note_type == 'Note Off':
            channel = pattern[tr][i].channel
            data = pattern[tr][i].data
            #track.append(midi.NoteOffEvent(tick= tick, channel=channel, data=[np.array(pitch), velocity]))
            track.append(midi.NoteOffEvent(tick= tick, channel=channel, data=data))
            note_off = note_off + 1
        '''
        elif note_type == 'Program Change':
            channel = pattern[tr][i].channel
            track.append(midi.ProgramChangeEvent(tick= tick, channel=channel, data=[np.array(pitch)]))
        
        elif note_type == 'Control Change':
            channel = pattern[tr][i].channel
            track.append(midi.ControlChangeEvent(tick= tick, channel=channel, data=[np.array(pitch)]))
        
        elif note_type == 'Track Name':
            text = pattern[tr][i].text
            data = pattern[tr][i].data
            track.append(midi.TrackNameEvent(tick= tick, text=text, data=data))
        
        elif note_type == 'Pitch Wheel':
            tick = pattern[tr][i].tick
            channel = pattern[tr][i].channel
            data = pattern[tr][i].data
            track.append(midi.TrackNameEvent(tick= tick, channel=text, data=data))
        elif note_type == 'Set Tempo':
            tick = pattern[tr][i].tick
            data = pattern[tr][i].data
            track.append(midi.TrackNameEvent(tick= tick, data=data))
        elif note_type == 'MIDI Port/Cable':
            tick = pattern[tr][i].tick
            data = pattern[tr][i].data
            track.append(midi.TrackNameEvent(tick= tick, data=data))
        else:
            print note_type
            print pattern[tr][i]
        '''

        i = i + 1
        
        #track.append(midi.NoteOnEvent(tick= tick, channel=1, data=[np.array(pitch), velocity]))
        
        
    #print len(pat[tr-1])

    # Emergency break if we want to just decompose one channel
    # break
    tr = tr + 1
#print pat

eot = midi.EndOfTrackEvent(tick=1)
track.append(eot)

file_name = "example.mid"

midi.write_midifile(file_name, pat)

print 'note_on: ', note_on
print 'note_off: ', note_off
print ''
print 'Midi file was written for ', path, ' in ', file_name
