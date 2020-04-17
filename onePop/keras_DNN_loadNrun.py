#!/usr/bin/env python3
#import tensorflow as tf
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
import sys, argparse, os, time
import numpy as np
from sklearn.metrics import confusion_matrix
import keras
from keras.utils import plot_model, to_categorical
from keras.models import Model
from keras.layers import Masking, Conv1D, AveragePooling1D, Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
print('Done with imports')

def build_DNN(X_train, Y_train, X_valid, Y_valid, batch_sizes, lr, checkpoint_file_name, nClasses):
    n_examples, ali_len =X_train.shape

    sys.stderr.write("Building network; input shape: %s\n" %(str(X_train.shape)))
    sys.stderr.write("Building network; validation shape: %s\n" %(str(X_valid.shape)))
    #Arhitecture
    dropout_rate=0.25
    dropout_rate2=0.1
    l2_lambda = 0.0001

    input1 = Input(shape=(ali_len,))
    dense1 = Dense(128, kernel_regularizer=keras.regularizers.l2(l2_lambda), activation='relu')(input1)
    dod1 = Dropout(dropout_rate)(dense1)
    dense2 = Dense(128, kernel_regularizer=keras.regularizers.l2(l2_lambda), activation='relu')(dod1)
    dod2 = Dropout(dropout_rate)(dense2)
    output = Dense(nClasses, kernel_initializer='normal', activation='softmax')(dod2)

    print("lr: %g" %(lr))
    model_cnn = Model(inputs=input1, outputs=output)
    optimizer = Adam(lr=lr)
    model_cnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model_cnn.summary())
    #Model stopping criteria
    callback1=EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='auto')
    callback2=ModelCheckpoint(checkpoint_file_name, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    
    sys.stderr.write("Ready to train on %d training examples with %d validation examples\n" %(len(X_train), len(X_valid)))
    #Run
    print(X_train)
    print(X_train.shape)
    print(Y_train)
    print(Y_train.shape)
    model_cnn.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid), batch_size=batch_sizes, callbacks=[callback1, callback2], epochs=50, verbose=1, shuffle=True)

    return(model_cnn)

def writeTestFile(testFileName, testX, testPosX, testy):
    np.savez_compressed(testFileName, X=testX, posX=testPosX, y=testy)

def main():
    parser = argparse.ArgumentParser(description='Keras training run')
    parser.add_argument( '-i', help = "File with input data in NPZ format",dest='infile')
    parser.add_argument( '-c', help = "Path/name of file in which the best network will be saved",dest='netfile')
    parser.add_argument( '-l', help = "Learning rate", type=float, dest='lr', default=0.001)
    args = parser.parse_args()
    
    print("Reading input")
    u = np.load(args.infile)
    trainX, testX, valX = u['trainX'], u['testX'], u['valX']
    trainy, testy, valy = u['trainy'], u['testy'], u['valy']
    nClasses = len(np.unique(trainy))
    trainy, testy, valy = to_categorical(trainy, num_classes=nClasses), to_categorical(testy, num_classes=nClasses), to_categorical(valy, num_classes=nClasses)
    print('Done.')
    
    model_cnn=build_DNN(X_train=trainX, Y_train=trainy, X_valid=valX, Y_valid=valy, batch_sizes=256, lr=args.lr, checkpoint_file_name=args.netfile, nClasses=nClasses)

    #Load best model
    model_cnn.load_weights(args.netfile)
    model_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('Evaluate with best weights')
    evals = model_cnn.evaluate(testX, testy, batch_size=32, verbose=0, steps=None)
    print(evals)
    predict = model_cnn.predict(testX)
    testy=[np.argmax(y, axis=None, out=None) for y in testy]
    predict=[np.argmax(y, axis=None, out=None) for y in predict]
    print("Confusion Matrix")
    print(confusion_matrix(testy, predict))

if __name__ == "__main__":
    startTime = time.clock()
    main()
    print('Total clock time elapsed: %g seconds' %(time.clock()-startTime))
