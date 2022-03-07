import numpy as np
from keras.datasets import mnist
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def getClass(classes: list = [0,1], amount = 100):
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    indices = []
    for i in classes:
        ind = np.array(list(np.argwhere(train_y == i).flatten()))    # indices which belong to class 1
        # np.random.shuffle(ind)
        ind = list(ind)[:amount]
        indices += ind
    return train_X[indices],train_y[indices]

def giveShape(X,y):
    y = y.reshape(y.shape[0],1)
    X = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
    X = np.concatenate((X,y),axis = 1)
    np.random.shuffle(X)
    return np.hsplit(X,np.array([X.shape[1]-1]))

def getRandom(classes: list = [0,1], amount = 100):
    X,y = getClass(classes,amount)
    X,y = giveShape(X,y)
    y = y.flatten()
    return X,y

def getEyeData(amount = 1000):
    df = pd.read_csv("datasets/EEG Eye State.csv")
    X = df.to_numpy()
    np.random.shuffle(X)
    # print(X.shape)
    X,y = np.hsplit(X,np.array([X.shape[1]-1]))
    y= y.flatten()

    # print(y.shape)
    ind0 = list(np.where(y==0)[0][:amount])
    ind1 = list(np.where(y==1)[0][:amount])

    # print(len(ind0),len(ind1))
    ind = ind0 + ind1
    # print(len(ind))
    y= y.reshape((y.shape[0],1))
    X = np.concatenate((X,y),axis = 1)

    X = X[ind]
    X,y = np.hsplit(X,np.array([X.shape[1]-1]))
    y= y.flatten()
    return X,y

def getAdultData(amount = 1000):
    df = pd.read_csv("datasets/re_adult.csv")
    R = df[['fnlwgt','Capital-gain','Capital-loss']].to_numpy()
    I = df[['Age','Education-num','Hours-per-week']].to_numpy()
    C = df[['Workclass','Education','Marital-status','Occupation','Relationship','Race','Sex','Native-country']].to_numpy()

    y = df['Class'].to_numpy()
    y=y.flatten()
    y0 = np.where(y==' <=50K')
    y1 = np.where(y==' >50K')
    # print(y)

    np.put(y,y0,0)
    np.put(y,y1,1)
    # print(y)
    y0 = np.array(y0).flatten()[:amount]
    y1 = np.array(y1).flatten()[:amount]

    ind = np.append(y0,y1)
    # np.random.shuffle(ind)

    R = R[ind]
    I = I[ind]
    C = C[ind]
    y = y[ind]
    # print(y.shape)

    # print(np.unique(C[:,0]))
    C1 = np.zeros((C.shape[0],1))
    for i in range(C.shape[1]):
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(C[:,i])
        # print(integer_encoded)
        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        C1 = np.concatenate((C1,onehot_encoded),axis = 1)

    # print(y.shape)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)
    # print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    y1 = onehot_encoded.flatten()

    return R,I,C1,y

def getAdultData2(amount = 1000, col = 'Sex', set = 'Male'):
    df = pd.read_csv("datasets/re_adult.csv")
    df = df[df[col] == set]
    R = df[['fnlwgt','Capital-gain','Capital-loss']].to_numpy()
    I = df[['Age','Education-num','Hours-per-week']].to_numpy()
    C = df[['Workclass','Education','Marital-status','Occupation','Relationship','Race','Sex','Native-country']].to_numpy()

    y = df['Class'].to_numpy()
    y=y.flatten()
    y0 = np.where(y==' <=50K')
    y1 = np.where(y==' >50K')
    # print(y)

    np.put(y,y0,0)
    np.put(y,y1,1)
    # print(y)
    y0 = np.array(y0).flatten()[:amount]
    y1 = np.array(y1).flatten()[:amount]

    ind = np.append(y0,y1)
    # np.random.shuffle(ind)

    R = R[ind]
    I = I[ind]
    C = C[ind]
    y = y[ind]

    return R,I,C,y

def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)