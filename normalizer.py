#converts all midi files in the current folder

import glob
import os
import music21

def transpose_cmajor_aminor(path):
    #converting everything into the key of C major or A minor

    # major conversions
    majors = dict([("A-", 4),("A", 3),("B-", 2),("B", 1),("C", 0),("D-", -1),("D", -2),("E-", -3),("E", -4),("F", -5),("G-", 6),("G", 5)])
    minors = dict([("A-", 1),("A", 0),("B-", -1),("B", -2),("C", -3),("D-", -4),("D", -5),("E-", 6),("E", 5),("F", 4),("G-", 3),("G", 2)])


    #os.chdir("./")
    for file in glob.glob(path):
        print 'Parsing file. . .'
        score = music21.converter.parse(file)
        print 'score: ', score
        key = score.analyze('key')
        print 'first key: ', key.tonic.name, key.mode
        if key.mode == "major":
            print 'major'
            halfSteps = majors[key.tonic.name]
            
        elif key.mode == "minor":
            print 'minor'
            halfSteps = minors[key.tonic.name]
        print 'half steps: ', halfSteps
        newscore = score.transpose(halfSteps)
        print 'newscore: ', newscore
        key = newscore.analyze('key')
        print key
        print key.tonic.name, key.mode
        newFileName = "C_" + file
        newscore.write('midi',newFileName)
