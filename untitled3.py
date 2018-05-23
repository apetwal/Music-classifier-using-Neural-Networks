import numpy as np
from keras.models import model_from_json
from music21 import *
from os import listdir
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from os.path import isfile, join
batch_size=50;
global length
global listl
global fnotes
global data
data=np.zeros((1,batch_size))
listl=[]
def data_appender(i_fnotes):
    global data
    
#   print("data_shape")
#   print(np.shape(data))
#   print("f_notes")
#   print(np.shape(i_fnotes))
    data= np.concatenate((data,i_fnotes),axis=0)
    return data
def split (renotes):
    global fnotes
    fnotes=np.array([])
#    print ("renotes",len(renotes))
    l=int(len(renotes)/batch_size)
    global listl
#    print("lennotes  ",lennotes)
    listl.append(l)
#    print("listl",listl)
#    print (l)
    for i in range(0,(batch_size*l)):
        fnotes=np.append(fnotes,[renotes[i]],axis=0)
#    print("lol")
    fnotes=fnotes.reshape(l,batch_size)
#    print("fnotes",np.size(fnotes))
    return fnotes


def noteextractor(filelocation):
    keyboard_nstrument = ["KeyboardInstrument", "Piano", "Harpsichord", "Clavichord", "Celesta", ]
    midi = converter.parse(filelocation)
    notes_to_parse = None
    notes=[]
    renotes=[]
    lnotes=[]
    notes_to_parse = None
    try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
    except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
    offset = 0

#print("\nnotes\n")
##print(notes)
    for i in range (0,len(notes)):
        if (notes[i][0].isdigit()):
            lul=0;
#        renotes.append(notes[i])
        else:
#        print(notes[i])
            p1 = pitch.Pitch(notes[i])
            lnotes.append(notes[i])
            renotes.append(str(p1.midi)) 
        l=len(renotes)
#    print("renotes",np.shape(renotes))
    return (renotes)


#seed = 7
#np.random.seed(seed)
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


song=0
Xtestr=np.random.randint(20,high=100,size=(100000,batch_size))

#print (Xtest)
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
if (song==1):
    
    mypath='E:\work\loll'
    midlen=13
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for i in range (0,len(onlyfiles)):
        lp=mypath+'\*'+onlyfiles[i]
        print("file     ",i)
        newstr = lp[:midlen] + lp[midlen+1:]
        temp=data_appender(split(noteextractor(newstr)))
#    print(np.shape(temp))
    Xtest=temp
    Xtest = np.delete(Xtest, (0), axis=0)
    predictions=loaded_model.predict(Xtest)
    predictions=np.around(predictions)
    mlist=[]


    for z in range (0,len(onlyfiles)):
        if (z==0):        
            mlist.append(listl[z])
        else:
            mlist.append(listl[z]+listl[z-1])
    print (mlist)
    for i in range (0,len(mlist)):
        if (i==0):
            print("prediction for Song " ,i+1,"    ",predictions[0:mlist[i]])
        else:
            print("prediction for Song " ,i+1,"    ",predictions[mlist[i-1]:mlist[i]])
    counter=0
    for i in range (1,len(predictions)):
        if (predictions[i]==0):
            counter=counter+1
    print("total sample cases ",len(predictions))
    print("wrongly classified cases ",counter)
    
if (song==0):
    counter=0
    predictions=loaded_model.predict(Xtestr)
    predictions=np.around(predictions)
    print("random Sequence prediction   ",predictions)
    for i in range (1,len(predictions)):
        if (predictions[i]==1):
            counter=counter+1
    print("wrongly classified cases ",counter)
