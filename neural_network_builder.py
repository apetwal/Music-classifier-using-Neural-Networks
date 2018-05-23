import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import h5py

#import tensorflow as tf
pickle_in = open(r"E:\work\datafpickle2.pickle","rb")
fdata = pickle.load(pickle_in)
Y=fdata[:,0]
X=np.delete(fdata,0,1)
model = Sequential()
model.add(Dense(30, input_dim=50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=150, batch_size=10)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("model saved")
score = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
#print(rounded)
