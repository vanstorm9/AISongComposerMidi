import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import midi
from midiutil.MidiFile import MIDIFile
import os
import os.path



#path = 'example.mid'
#path = 'Songs/Suteki-Da-Ne.mid'
#path = 'Songs/Mozart-Movement.mid'
#path = 'Songs/beethoven_ode_to_joy.mid'
#path = 'Songs/twinkle_twinkle.mid'
#path = 'Songs/grenade.mid

slash = '/'

print 'Extracting all of pattern[1]'
# Instantiate a MIDI Pattern (contains a list of tracks)
pat = midi.Pattern()

#folder_trans = 'training-songs'
#folder_trans = 'instruments'
folder_trans = np.array(['instruments/piano','instruments/guitar'])
#folder_trans = 'training-video-test'
#folder_trans = 'training-kid-songs'
#folder_trans = 'training-classical-songs'
'''
num_files = len([f for f in os.listdir(folder_trans)
                    if os.path.isfile(os.path.join(folder_trans, f))])
'''
#path_ar = ['Songs/twinkle_twinkle.mid', 'Songs/Suteki-Da-Ne.mid']

def tick_to_time(tick):

    if tick !=0:
        time = 60000/(tick*192)
    else:
        time = 0
    return time

def pitch_prev_array_add(pitch, pitch_ar):
    if pitch_ar == None:
        pitch_ar = np.array([pitch])
    else:
        pitch_ar = np.concatenate((pitch_ar, np.array([pitch])))
    return pitch_ar

def tranverse_all_folders(folder_trans):
    j = 0
    k = 0
    while k < folder_trans.size:
        
        for path in os.listdir(folder_trans[k]):
            pattern = midi.read_midifile(folder_trans[k] + slash + path)
            print folder_trans[k] + slash + path
            # Instantiate a MIDI Track (contains a list of MIDI events)
            track = midi.Track()
            # Append the track to the pattern
            pat.append(track)
            # Goes through extracted song and reconstruct them (pattern[1])
            '''
            tr = 1
            start_val = 1
            i = 1
            '''
            # Grenade sample window
            '''
            tr = 5
            start_val = 80
            i = 80
            '''
            # Suteki Da Ne sample window
            tr = 1
            start_val = 1
            i = 1
            
            while True:

                # This is the if statement to break out of loop
                # Iterates to end of song or at a set number
                #if i > len(pattern[tr]) - 2:
                if i > 1200:
                    break
                tick = pattern[tr][i].tick
                pitch = pattern[tr][i].data[0]

                # Because some pattern[][].data does not have a second array element
                if len(pattern[tr][i].data) == 2:
                    velocity = pattern[tr][i].data[1]
                else:
                    velocity = 0
                # Place all of tick, pitch, and velocity values in indiviudal vectors
                
                tick = np.array([tick])
                pitch = np.array([pitch])
                velocity = np.array([velocity])
                if i == start_val:
                    tick_ar = tick
                    pitch_ar = pitch
                    velocity_ar = velocity
                else:
                    tick_ar = np.concatenate((tick_ar, tick))
                    pitch_ar = np.concatenate((pitch_ar, pitch))
                    velocity_ar = np.concatenate((velocity_ar, velocity))
                # To reconstruct the entire song in its (piano-like) original form
                #track.append(midi.NoteOnEvent(tick= tick, channel=1, data=[pitch, velocity]))
                i = i + 1
            
            print tick_ar.shape
            if j == 0:
                tick_u_ar = tick_ar
                velocity_u_ar = velocity_ar
                pitch_u_ar = pitch_ar

                tick_u_ar = np.array([tick_u_ar])
                velocity_u_ar = np.array([velocity_u_ar])
                pitch_u_ar = np.array([pitch_u_ar])
            else:
                
                tick_u_ar = np.concatenate((tick_u_ar, tick_ar[None,:]))
                pitch_u_ar = np.concatenate((pitch_u_ar, pitch_ar[None,:]))
                velocity_u_ar = np.concatenate((velocity_u_ar, velocity_ar[None,:]))
            j = j + 1
            k = k + 1
    return pattern, tick_u_ar, velocity_u_ar, pitch_u_ar




# Go through all folders and form the matrix
pattern, tick_ar, velocity_ar, pitch_ar = tranverse_all_folders(folder_trans)


print tick_ar.shape
print pitch_ar.shape

'''
print tick_ar.shape
pitch_ar = np.array([pitch_ar])
tick_ar = np.array([tick_ar])

uni_ar = np.array([[]])
uni_ar = np.concatenate((tick_ar, pitch_ar))
#uni_ar = uni_ar.T
print uni_ar.shape
'''


#### Clustering Code ###


from mpl_toolkits.mplot3d import Axes3D


from sklearn.cluster import KMeans
np.random.seed(42)

centers = [[1, 1], [-1, -1], [1, -1]]
X = tick_ar
#X = pitch_ar
y = np.array([0,1])

estimators = {'k_means_iris_3': KMeans(n_clusters=2),
              'k_means_iris_8': KMeans(n_clusters=2),
              'k_means_iris_bad_init': KMeans(n_clusters=2, n_init=1,
                                              init='random')}


fignum = 1
for name, est in estimators.items():
    fig = plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
'''
for name, label in [('Setosa', 0),
                    ('Versicolour', 1),
                    ('Virginica', 2)]:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
'''
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
plt.show()


