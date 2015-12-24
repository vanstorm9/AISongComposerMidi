import mido
from mido import MidiFile
from mido.midifiles import MidiTrack
from mido import Message
from mido import MetaMessage

pattern = MidiFile('Songs/Suteki-Da-Ne.mid')
#pattern = MidiFile('Songs/twinkle_twinkle.mid')
mid = MidiFile()


tracks = MidiTrack()
#tracks.append(tracks)
'''
for message in pattern:
    
    
    if message.type == 'note_on' or message.type == 'note_off':
        #print message
        mid.tracks.append(mid.Message(message.type, note=message.note, velocity=message.velocity, time=message.time))
    #elif message.type == 'control_change':
    #    mid.tracks.append(Message(message.type, control=message.control, value=message.value, time=message.time))
    
    #else:
    #    print message
    #    print message.type
    
    
    #tracks.append(Message(message.type, note=message.note, velocity=message.velocity, time=message.time))
    #tracks.append(message)

'''

for message in pattern:
    
    
    if message.type == 'note_on' or message.type == 'note_off':
        tracks.append(Message(message.type, note=message.note, velocity=message.velocity, time=message.time))

    elif message.type == 'pitchwheel':
        tracks.append(Message(message.type, channel=message.channel, pitch=message.pitch, time=message.time))
    elif message.type == 'control_change':
        tracks.append(Message(message.type, channel=message.channel,control=message.control, value = message.value, time=message.time))
    elif message.type == 'program_change':
        tracks.append(Message(message.type, channel=message.channel, program=message.program, time=message.time))
    elif message.type == 'set_tempo':
        MetaMessage('set_tempo', tempo = message.tempo, time = message.time)
    elif message.type == 'midi_port':
        MetaMessage('midi_port', port = message.port, time = message.time)
    elif message.type == 'end_of_track':
        MetaMessage('end_of_track', time = message.time)
    else:
        print message
        print message.type


mid.save('example.mid')
print 'New midi file saved!'

