import numpy as np
import pandas as pd
import pickle
import sklearn
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import string
import re
import tflearn
import nltk
import tensorflow as tf

from nltk.tokenize import word_tokenize
from collections import Counter
from random import shuffle
from  matplotlib import pyplot

#get train And Test Data Set
train_file='D:\\Predict Happiness HEML\\train.csv'
test_file='D:\\Predict Happiness HEML\\test.csv'

stop_words=['about','above','after','again','against','all','am','an','and','any','are','as','at','be','because','been','before','being','below','between','both','by','can','could','did','do','does','doing','down','during','each','few','for','from','further','had','has','have','having','he','ll','her','here','hers','herself','him','himself','his','how','how','i','ve','if','in','into','is','it','its','itself','let','me','more','most','must','my','myself','of','off','on','once','only','or','other','ought','our','ours','ourselves','out','over','own','same','shall','she','should','so','some','such','than','that','the','their','theirs','them','themselves','then','there','these','they','re','this','those','through','to','too','under','until','up','very','was','we','were','what','when','where','which','while','who','whom','why','with','would','you','your','yours','yourself', 'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

#stop_words=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
def remove_punctuation(snt):
    return re.sub(r'[\W]',' ',snt.lower())

def remove_stop_words(sent):
    words = sent.split()
    resultwords=[word.lower() for word in words if word.lower() not in stop_words]
    return resultwords



train_df=pd.read_csv(train_file)
test_df=pd.read_csv(test_file)

vocab=np.load("D:\\wordlist100d.npy")
vocab=np.array([word.decode('utf-8') for word in vocab])
vocablist=vocab.tolist()
vocab_size=len(vocab)
embedding=np.load("D:\\vectorlist100d.npy")
emd_list=embedding.tolist()
emd_dim=len(embedding[0])

print(vocab_size,emd_dim)

#Clean Data and Extract Features
label_df=train_df['Is_Response']
labels=list(set(label_df))
np.save("labels.npy",labels)
lblencdr=LabelEncoder()
intencdng=lblencdr.fit_transform(labels)
intencdng=intencdng.reshape(len(intencdng),1)
onehot=OneHotEncoder(sparse=False)
ht=onehot.fit_transform(intencdng)
dscrptn_df=train_df['Description']
dscrptn_df=[remove_punctuation(str(i).lower()) for i in dscrptn_df]
dscrptn_df=[remove_stop_words(str(i).lower()) for i in dscrptn_df]
dscrptn_ln=pd.DataFrame([len(i) for i in dscrptn_df])
test_dscrptin_df=test_df['Description']
user_df=test_df['User_ID']

print(dscrptn_ln.mean())
def getLabel(i):
    lbl=onehot.transform(lblencdr.transform(label_df.loc[[i]]))
    return lbl

def create_feature_set(exmple):

    feature_set=np.zeros(shape=(81,100))
    i=0
    #words=exmple.strip().split()
    for word in exmple:
        if i >=81:
            break
        if word.lower() in vocablist:
            index=vocablist.index(word.lower())
            feature_set[i]=embedding[index]
            i+=1
        else:
            feature_set[i]=embedding[-1]
            i+=1
    return np.array(feature_set)

def create_train_data():
    training_data=[]
    i=0
    for example in dscrptn_df:
        #print(example)
        features=create_feature_set(example)
        if i%100 is 0:
            print(i)
        label=getLabel(i)
        i+=1
        training_data.append([features,label])

    shuffle(training_data)
    batch_size=5000
    np.save("training_data1_100.npy",training_data[0:batch_size])
    np.save("training_data2_100.npy", training_data[batch_size:2*batch_size])
    np.save("training_data3_100.npy", training_data[2*batch_size:3*batch_size])
    np.save("training_data4_100.npy", training_data[3*batch_size:4*batch_size])
    np.save("training_data5_100.npy", training_data[4*batch_size:5*batch_size])
    np.save("training_data6_100.npy", training_data[5*batch_size:6*batch_size])
    np.save("training_data7_100.npy", training_data[6*batch_size:7*batch_size])
    np.save("training_data8_100.npy", training_data[7*batch_size:])

def create_test_data():
    test_data=[]
    i=0
    for example in test_dscrptin_df:
        #print(example)
        if i%1000 is 0:
            print(i)
        exmple=remove_punctuation(example)
        exmple=remove_stop_words(exmple)
        features=create_feature_set(exmple)
        user_id=user_df[i]
        i+=1
        test_data.append([features,user_id])
    test_data=np.array(test_data)
    batch_size=5000
    np.save("test_data1_100.npy", test_data[0:batch_size])

    np.save("test_data2_100.npy", test_data[batch_size:2 * batch_size])
    np.save("test_data3_100.npy", test_data[2 * batch_size:3 * batch_size])
    np.save("test_data4_100.npy", test_data[3 * batch_size:4 * batch_size])
    np.save("test_data5_100.npy", test_data[4 * batch_size:5 * batch_size])
    np.save("test_data6_100.npy", test_data[5 * batch_size:])

create_train_data()
create_test_data()
