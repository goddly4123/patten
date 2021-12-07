import os
import numpy as np
import cv2
import tensorflow as tf
import pickle, gzip

from random import shuffle
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

from show_img import return_label

tf.autograph.experimental.do_not_convert(func=None)

TRAIN_DIR = 'dataset'
IMG_SIZE = 180
IMG_SIZE1 = 180

label_original = return_label()
final_label = os.listdir(TRAIN_DIR)
final_label.pop(final_label.index('.DS_Store'))

def create_train_data():
    train = []
    test = []
    global final_label

    count = 0
    count2 = 0
    for label in final_label:
        dir = os.listdir(os.path.join(TRAIN_DIR,label))
        #print(label)
        #print(dir,'>>>>>>>>>>>>>>>>\n')#/Users/papa/Desktop/Project/dataset/NG'
        if len(dir)>10:
            train_limit = int(len(dir)*.8)
            num = 0
            shuffle(dir)
            for pick in tqdm(dir):
                if pick.split('.')[1] == 'pickle':                           #for avoid hidden system file
                    path = TRAIN_DIR + '/' + label + '/' + pick #'/Users/papa/Desktop/Project/dataset/NG/0000.pickle'
                    #print(path)
                    img = load_data(path)
                    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
                    L = make_label(label=label)

                    if num == 0:
                        print(label, L)
                    num += 1

                    if num <= train_limit:
                        train.append([np.array(img).astype('float32')/255, np.array(L).astype('float32')])
                        count += 1
                    else:
                        test.append([np.array(img).astype('float32') / 255, np.array(L).astype('float32')])
                        count2 += 1

                else:
                    print(pick,'\n\n\n\n')
    print('count of data : 80:20={}ea:{}ea'.format(count,count2))



    #shuffle(train)
    #shuffle(test)

    #np.save('train_data.npy', training_data)
    
    #train = training_data[:-1*int(len(training_data)*.2)]
    #test = training_data[-1*int(len(training_data)*.2):]
    
    X = np.array([i[0] for i in train])
    Y = np.array([i[1] for i in train])

    X_test = np.array([i[0] for i in test])
    Y_test = np.array([i[1] for i in test])

    print(len(X))
    print(len(Y))
    print(len(X_test))
    print(len(Y_test))
    
    return X, Y, X_test, Y_test


def make_label(label):
    global label_original
    final = [0] * len(label_original)

    for i in label.split('_'):
        final[label_original.index(i)] = 1

    return final
    

def load_data(file_name):
    data = None
    # load and uncompress.
    with gzip.open(file_name,'rb') as f:
        data = pickle.load(f)
        
    return np.array(data)

def make_network(IMG_SIZE1, IMG_SIZE2):
    classifier = Sequential()

    # Step 1 - Convolution
    classifier.add(Conv2D(32, (2, 2), input_shape=(IMG_SIZE1, IMG_SIZE2, 5), activation='relu'))

    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.5))  # 임의로 30% 데이터 삭제로 과대학습 예방

    # Adding next convolutional layer
    classifier.add(Conv2D(64, (2, 2), activation='relu'))
    classifier.add(Dropout(0.5))  # 임의로 30% 데이터 삭제로 과대학습 예방

    classifier.add(Conv2D(64, (2, 2), activation='relu'))
    #classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.5))  # 임의로 30% 데이터 삭제로 과대학습 예방


    # Adding fully connected layers
    classifier.add(Flatten())  # 펼쳐라

    #classifier.add(Dense(units=128,activation='relu'))
    #classifier.add(Dropout(0.5))  # 임의로 30% 데이터 삭제로 과대학습 예방
    #classifier.add(Dense(units=64, activation='relu'))
    #classifier.add(Dropout(0.5))  # 임의로 30% 데이터 삭제로 과대학습 예방

    classifier.add(Dense(units=4, activation='sigmoid'))  # 결과물이 2개이면 sigmoid, 3개이상이면 softmax

    print(classifier.summary())
    return classifier


def load_data2(dir, i):
    data = None
    name = os.path.join(dir,i)
    
    # load and uncompress.
    with gzip.open(name,'rb') as f:
        data = pickle.load(f)
        
    return data


def concat_img(data):
    img = None
    
    (e,d,c,b,a) = cv2.split(data)
    img = cv2.hconcat([a,b])
    img = cv2.hconcat([img,c])
    img = cv2.hconcat([img,d])
    img = cv2.hconcat([img,e])
    
    return img


if __name__ == '__main__':
    
    classifier = make_network(IMG_SIZE, IMG_SIZE)

    print('==================================================')
    print('학습 = 1,   검사 = 2 를 누르세요')
    print('==================================================')

    if input() == '1':
        X, Y, x, y = create_train_data()
        #X = tf.data.Dataset.from_tensor_slices((X,Y)).batch(32)
        #Y = tf.data.Dataset.from_tensor_slices(Y).batch(24)
        #x = tf.data.Dataset.from_tensor_slices((x,y)).batch(8)
        #y = tf.data.Dataset.from_tensor_slices(y).batch(8)

        #print(X)
        
        # 2개이면 binary_crossentropy, categorical_crossentropy
        classifier.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["acc"])

        #classifier.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        mc = ModelCheckpoint('best_model_xx.h5', monitor='val_loss', mode='min', save_best_only=True)
        
        classifier.fit(x=X, y=Y,
                       epochs=100,
                       validation_data=(x,y),
                       callbacks=[es, mc])

    else:
        
        classifier.load_weights("best_model_xx.h5")

        dir_ = 'dataset/NG_BK'
        
        for i in os.listdir(dir_):
            img_final = None
            
            if True: 
                data = load_data2(dir_, i)
                frame = concat_img(data)
                data = np.array(cv2.resize(data, (IMG_SIZE, IMG_SIZE)))

                #학습이 가능토록 데이터 변환
                img_array = np.array(data).astype('float32') / 255
                img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 5)

                A = classifier.predict(img_array)
                np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
                #label_list = ['NG', 'OK', 'BACK']
                print(label_original)
                print(A[0])
                #print(label_list[np.argmax(A[0])])
                #print('NG일 확률 : {}%\n'.format(round(A[0][0]*100,2)))

                cv2.imshow('Image', frame)
                
                # Press Q on keyboard to quit program
                key = cv2.waitKey()
                if key & 0xFF == ord('q'):
                    break
            
            
        # Closes all the frames
        cv2.destroyAllWindows() 
