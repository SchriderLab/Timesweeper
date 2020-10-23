import matplotlib

matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pylab as plt
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (Conv2D, Dense, Dropout,
                                     Flatten, Input, MaxPooling2D, concatenate)
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import np_utils

numSubWins = 11

def train(trainingDir, testingDir, epochOption, outputModel):
    hard = np.loadtxt(trainingDir+"hard.fvec",skiprows=1)
    nDims = int(hard.shape[1] / numSubWins)
    h1 = np.reshape(hard,(hard.shape[0],nDims,numSubWins))
    neut = np.loadtxt(trainingDir+"neut.fvec",skiprows=1)
    n1 = np.reshape(neut,(neut.shape[0],nDims,numSubWins))
    soft = np.loadtxt(trainingDir+"soft.fvec",skiprows=1)
    s1 = np.reshape(soft,(soft.shape[0],nDims,numSubWins))

    both=np.concatenate((h1,n1,s1))
    y=np.concatenate((np.repeat(0,len(h1)),
                      np.repeat(1,len(n1)), 
                      np.repeat(2,len(s1))))

    #reshape both to explicitly set depth image. need for theanno not sure with tensorflow
    both = both.reshape(both.shape[0],nDims,numSubWins,1)
    if (trainingDir==testingDir):
        (X_train, X_test,
         y_train, y_test) = train_test_split(both, y, test_size=0.2)
         
    else:
        X_train = both
        y_train = y
        #testing data
        hard = np.loadtxt(testingDir+"hard.fvec",skiprows=1)
        h1 = np.reshape(hard,(hard.shape[0],nDims,numSubWins))
        neut = np.loadtxt(testingDir+"neut.fvec",skiprows=1)
        n1 = np.reshape(neut,(neut.shape[0],nDims,numSubWins))
        soft = np.loadtxt(testingDir+"soft.fvec",skiprows=1)
        s1 = np.reshape(soft,(soft.shape[0],nDims,numSubWins))

        both2=np.concatenate((h1,n1,s1))
        X_test = both2.reshape(both2.shape[0],nDims,numSubWins,1)
        y_test=np.concatenate((np.repeat(0,len(h1)),
                               np.repeat(1,len(n1)), 
                               np.repeat(2,len(s1))))


    Y_train = np_utils.to_categorical(y_train, 5)
    Y_test = np_utils.to_categorical(y_test, 5)
    (X_valid, X_test,
     Y_valid, Y_test) = train_test_split(X_test, Y_test, test_size=0.5)

    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        horizontal_flip=True)

    validation_gen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        horizontal_flip=False)

    test_gen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        horizontal_flip=False)

    #https://machinelearningmastery.com/cnn-long-short-term-memory-networks/

    #print(X_train.shape)
    print("training set has %d examples" % X_train.shape[0])
    print("validation set has %d examples" % X_valid.shape[0])
    print("test set has %d examples" % X_test.shape[0])
    
    model_in = Input(X_train.shape[1:])
    h = Conv2D(128, 3, activation='relu',padding="same", name='conv1_1')(model_in)
    h = Conv2D(64, 3, activation='relu',padding="same", name='conv1_2')(h)
    h = MaxPooling2D(pool_size=3, name='pool1',padding="same")(h)
    h = Dropout(0.15, name='drop1')(h)
    h = Flatten(name='flaten1')(h)

    dh = Conv2D(128, 2, activation='relu',dilation_rate=[1,3],padding="same", name='dconv1_1')(model_in)
    dh = Conv2D(64, 2, activation='relu',dilation_rate=[1,3],padding="same", name='dconv1_2')(dh)
    dh = MaxPooling2D(pool_size=2, name='dpool1')(dh)
    dh = Dropout(0.15, name='ddrop1')(dh)
    dh = Flatten(name='dflaten1')(dh)

    dh1 = Conv2D(128, 2, activation='relu',dilation_rate=[1,4],padding="same", name='dconv4_1')(model_in)
    dh1 = Conv2D(64, 2, activation='relu',dilation_rate=[1,4],padding="same", name='dconv4_2')(dh1)
    dh1 = MaxPooling2D(pool_size=2, name='d1pool1')(dh1)
    dh1 = Dropout(0.15, name='d1drop1')(dh1)
    dh1 = Flatten(name='d1flaten1')(dh1)

    h =  concatenate([h,dh,dh1])
    h = Dense(512,name="512dense",activation='relu')(h)
    h = Dropout(0.2, name='drop7')(h)
    h = Dense(128,name="last_dense",activation='relu')(h)
    h = Dropout(0.1, name='drop8')(h)
    output = Dense(5,name="out_dense",activation='softmax')(h)
    model = Model(inputs=[model_in], outputs=[output])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # define early stopping callback
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=5, \
                              verbose=1, mode='auto')
    
    model_json = model.to_json()
    with open(outputModel+".json", "w") as json_file:
        json_file.write(model_json)
    modWeightsFilepath=outputModel+".weights.hdf5"
    checkpoint = ModelCheckpoint(modWeightsFilepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')

    callbacks_list = [earlystop,checkpoint]
    #callbacks_list = [earlystop] #turning off checkpointing-- just want accuracy assessment

    datagen.fit(X_train)
    validation_gen.fit(X_valid)
    test_gen.fit(X_test)
    start = time.time()
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32), \
                        steps_per_epoch=len(X_train) / 32, epochs=epochOption,verbose=1, \
                        callbacks=callbacks_list, \
                        validation_data=validation_gen.flow(X_valid,Y_valid, batch_size=32), \
                        validation_steps=len(X_test)/32)
    #model.fit(X_train, Y_train, batch_size=32, epochs=100,validation_data=(X_test,Y_test),callbacks=callbacks_list, verbose=1)
    score = model.evaluate_generator(test_gen.flow(X_test,Y_test, batch_size=32),len(Y_test)/32)
    sys.stderr.write("total time spent fitting and evaluating: %f secs\n" %(time.time()-start))

    print("evaluation on test set:")
    print("diploSHIC loss: %f" % score[0])
    print("diploSHIC accuracy: %f" % score[1])

def predict():


    #import data from predictFile
    x_df=pd.read_table(argsDict['predictFile'])
    if argsDict['simData']:
        testX = x_df[list(x_df)[:]].as_matrix()
    else:
        testX = x_df[list(x_df)[4:]].as_matrix()
    nDims = int(testX.shape[1]/numSubWins)
    np.reshape(testX,(testX.shape[0],nDims,numSubWins))
    #add channels
    testX = testX.reshape(testX.shape[0],nDims,numSubWins,1)
    #set up generator for normalization 
    validation_gen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        horizontal_flip=False)
    validation_gen.fit(testX)
    
    #import model
    json_file = open(argsDict['modelStructure'], 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(argsDict['modelWeights'])
    print("Loaded model from disk")
    
    #get predictions
    preds = model.predict(validation_gen.standardize(testX))
    predictions = np.argmax(preds,axis=1)
    
    #np.repeat(0,len(h1)),np.repeat(1,len(n1)), np.repeat(2,len(s1)), np.repeat(3,len(ls1)), np.repeat(4,len(lh1)
    classDict = {0:'hard',1:'neutral',2:'soft',3:'linkedSoft',4:'linkedHard'}
    
    #output the predictions
    outputFile = open(argsDict['predictFileOutput'],'w')
    outputFile.write('chrom\tclassifiedWinStart\tclassifiedWinEnd\tbigWinRange\tpredClass\tprob(neutral)\tprob(likedSoft)\tprob(linkedHard)\tprob(soft)\tprob(hard)\n')
    for index, row in x_df.iterrows():
        if argsDict['simData']:
            outputFile.write('{}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\n'.format(classDict[predictions[index]],preds[index][1],preds[index][3],preds[index][4], \
                preds[index][2],preds[index][0]))
        else:
            outputFile.write('{}\t{}\t{}\t{}\t{}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\n'.format( row['chrom'],row['classifiedWinStart'],row['classifiedWinEnd'],row['bigWinRange'], \
                classDict[predictions[index]],preds[index][1],preds[index][3],preds[index][4],preds[index][2],preds[index][0]))
    outputFile.close
    print("{} predictions complete".format(index+1))
