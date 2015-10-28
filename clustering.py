import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from time import time

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import midi
from midiutil.MidiFile import MIDIFile
import os
import os.path



slash = '/'

print 'Extracting all of pattern[1]'
# Instantiate a MIDI Pattern (contains a list of tracks)
pat = midi.Pattern()


# Array of directories to transfers through
folder_trans = np.array(['instruments/guitar',
                         'instruments/piano',
                         'instruments/violin'])

# Set number of clusters
num_of_clusters = 3

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
    skip = False
    label_ar = np.array([])
    while k < folder_trans.size:
        
        for path in os.listdir(folder_trans[k]):
            #print path
            pattern = midi.read_midifile(folder_trans[k] + slash + path)
            print folder_trans[k] + slash + path
            # Instantiate a MIDI Track (contains a list of MIDI events)
            track = midi.Track()
            # Append the track to the pattern
            pat.append(track)
            # Goes through extracted song and reconstruct them (pattern[1])


            temp = np.array([k])
            label_ar = np.concatenate((label_ar, temp))


            # Midi file track information
            tr = 0
            start_val = 1
            i = 1
            limit = 200



            # To choose track that has enough notes
            p = 0
            exc = True

            
            while p < len(pattern):
                if len(pattern[tr]) >= limit:
                    exc = False
                    break
                tr = tr + 1
                p = p + 1

            if exc:
                print "All of this song's tracks does not have enough notes"
                quit()

                
            while True:

                # This is the if statement to break out of loop
                # Iterates to end of song or at a set number
                #if i > len(pattern[tr]) - 2:
                if i > limit:
                    break
                #print pattern[tr][i]
         
                
                tick = pattern[tr][i].tick

                #print pattern[tr][i].data
                #print len(pattern[tr][i].data)
                
                
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
            
            #print tick_ar.shape
            if skip:
                j= j + 1
                continue

                
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
            
    return pattern, tick_u_ar, velocity_u_ar, pitch_u_ar, label_ar




def threeD_plot(X,y, label_ar,num_of_clusters):
    np.random.seed(42)

    #digits = load_digits()
    data = scale(X)

    n_samples, n_features = data.shape
    n_digits = num_of_clusters
    labels = y

    sample_size = 300

    print("n_digits: %d, \t n_samples %d, \t n_features %d"
          % (n_digits, n_samples, n_features))


    print(79 * '_')
    print('% 9s' % 'init'
          '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')

    def bench_k_means(estimator, name, data):
        t0 = time()
        estimator.fit(data)
        print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
              % (name, (time() - t0), estimator.inertia_,
                 metrics.homogeneity_score(labels, estimator.labels_),
                 metrics.completeness_score(labels, estimator.labels_),
                 metrics.v_measure_score(labels, estimator.labels_),
                 metrics.adjusted_rand_score(labels, estimator.labels_),
                 metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
                 metrics.silhouette_score(data, estimator.labels_,
                                          metric='euclidean',
                                          sample_size=sample_size)))


    bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
                  name="k-means++", data=data)

    bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
                  name="random", data=data)

    # in this case the seeding of the centers is deterministic, hence we run the
    # kmeans algorithm only once with n_init=1
    pca = PCA(n_components=n_digits).fit(data)
    bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
                  name="PCA-based",
                  data=data)
    print(79 * '_')

    ###############################################################################
    # Visualize the results on PCA-reduced data

    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
    y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=8)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('Clustering pitches between Guitar, Piano, and Violin')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()



# Go through all folders and form the matrix
pattern, tick_ar, velocity_ar, pitch_ar, label_ar = tranverse_all_folders(folder_trans)


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


#X = tick_ar
X = pitch_ar
y = label_ar


threeD_plot(X, y, label_ar,num_of_clusters)





