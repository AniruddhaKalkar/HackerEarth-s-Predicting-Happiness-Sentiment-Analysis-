from keras.optimizers import SGD,Adam
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import LSTM
import math
import json
import os
import numpy as np

training_data=np.load("training_data1_100.npy")
print("loaded 1st")
training_data=np.concatenate((training_data,np.load("training_data2_100.npy")))
print("loaded 2nd")
training_data=np.concatenate((training_data,np.load("training_data3_100.npy")))
print("loaded 3rd")
training_data=np.concatenate((training_data,np.load("training_data4_100.npy")))
print("loaded 4th")
training_data=np.concatenate((training_data,np.load("training_data5_100.npy")))
print("loaded 5th")
training_data=np.concatenate((training_data,np.load("training_data6_100.npy")))
print("loaded 6th")
training_data=np.concatenate((training_data,np.load("training_data7_100.npy")))
print("loaded 7th")
training_data=np.concatenate((training_data,np.load("training_data8_100.npy")))
print("loaded")

X=np.array([i[0] for i in training_data])

#Y=np.array([i[1].astype(np.int32) for i in training_data])

Y=np.reshape(np.array([i[1].astype(np.int32) for i in training_data]).flatten(),(len(X),2))

X_train=X[:-1000]
# for i in Y:
#     print(i)
X_test=X[-1000:]
y_train=Y[:-1000]
y_test=Y[-1000:]

n_nodes_output=2
batch_size=128
total_batches=int(math.ceil(len(X)/batch_size))
epochs=10

model=Sequential()
model.add(LSTM(162,input_shape=(81,100)))
model.add(Dropout(0.8))
model.add(Dense(units=n_nodes_output,activation="softmax"))
sgd=SGD(lr=0.01,decay=1e-6,nesterov=True)
ad=Adam(lr=0.01,decay=1e-6)
model.compile(optimizer=ad,loss="binary_crossentropy",metrics=["accuracy"])
print(model.summary())
model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(X_test,y_test))
score,acc=model.evaluate(X_test,y_test,batch_size=500)
print(score,acc)


expnense_model=model.to_json()
with open("Model2_100.json","w") as json_file:
    json_file.write(expnense_model)
model.save_weights("weights2_100.h5")

