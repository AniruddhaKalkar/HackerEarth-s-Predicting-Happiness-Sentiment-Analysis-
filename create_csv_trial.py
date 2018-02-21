from keras.optimizers import SGD
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout
from keras.optimizers import Adam
import math
import json
import os
import numpy as np
from keras.models import model_from_json

test_data=np.load("test_data1_100.npy")
print("loaded 1st")
test_data=np.concatenate((test_data,np.load("test_data2_100.npy")))
print("loaded 2nd")
test_data=np.concatenate((test_data,np.load("test_data3_100.npy")))
print("loaded 3rd")
test_data=np.concatenate((test_data,np.load("test_data4_100.npy")))
print("loaded 4th")
test_data=np.concatenate((test_data,np.load("test_data5_100.npy")))
print("loaded 5th")
test_data=np.concatenate((test_data,np.load("test_data6_100.npy")))
print("loaded 6th")



json_file = open("Model2_100.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("weights2_100.h5")
ad=Adam(lr=0.01,decay=1e-6)
model.compile(optimizer=ad,loss="categorical_crossentropy",metrics=["accuracy"])

labels=['happy','not_happy']
count=1
with open('submission_100.csv', 'w') as f:
    f.write('User_ID,Is_Response\n')
    for tup in test_data:

        features=tup[0]
        user_id=tup[1]
        pred=model.predict(np.array([features]))
        lbl = np.argmax(pred[0,:])
        f.write('{},{}\n'.format(user_id,labels[lbl]))
        if count %1000 is 0:
            print(count)
        count+=1