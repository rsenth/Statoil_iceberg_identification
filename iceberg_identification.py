
#Identify if an object is a ship or an iceberg from satellite data.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from os.path import join as opj
import keras
import sys
import gc
from scipy import interpolate
#Import Keras.
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras.preprocessing import image
from inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing.image import ImageDataGenerator




batch_size=32
train = pd.read_json("../input/train.json")
target_train=train['is_iceberg']
test = pd.read_json("../input/test.json")

train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors='coerce')#We have only 133 NAs.
train['inc_angle'] = train['inc_angle'].fillna(method='pad')
test['inc_angle'] = pd.to_numeric(test['inc_angle'], errors='coerce')

X_angle = train['inc_angle']
X_test_angle = test['inc_angle']

    #Generate the training data
# Define the image transformations here
gen = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         width_shift_range = 0.0,
                         height_shift_range = 0.0,
                         channel_shift_range=0,
                         zoom_range = 0.5,
                         rotation_range = 15)


def transform_train_test(): 

    #Transform the 2-channel input data to 3-channel data used by InceptionResnetV2


    #Get the 2 channels of training data first
    X_band_small1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
    X_band_small2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])

    #Input data has dimensions 75 X 75 X 2
    #First interpolate the input data to 149 X 149 X 2 grid so that InceptionResnetV2 can be used 
    xcoord = np.arange(0, 75)
    ycoord = np.arange(0, 75)
    dx = 74.0/148.

    xcoord_new = np.arange(0, 74.1, dx)
    ycoord_new = np.arange(0, 74.1, dx)

    X_band_1 = np.zeros((X_band_small1.shape[0], xcoord_new.shape[0], ycoord_new.shape[0]))
    X_band_2 = np.zeros((X_band_small1.shape[0], xcoord_new.shape[0], ycoord_new.shape[0]))

    #Interpolation 
    for i in range(X_band_1.shape[0]):
        f = interpolate.interp2d(xcoord, ycoord, X_band_small1[i,:,:])
        X_band_1[i,:,:] = f(xcoord_new, ycoord_new)

        f = interpolate.interp2d(xcoord, ycoord, X_band_small2[i,:,:])
        X_band_2[i,:,:] = f(xcoord_new, ycoord_new)

    del X_band_small1, X_band_small2

    #Construct 3 channel data from the 2-channel input data
    X_band_3=np.fabs(np.subtract(X_band_1,X_band_2))
    X_band_4=np.maximum(X_band_1,X_band_2)
    X_band_5=np.minimum(X_band_1,X_band_2)

    X_train = np.concatenate([X_band_3[:, :, :, np.newaxis],
                              X_band_4[:, :, :, np.newaxis],
                              X_band_5[:, :, :, np.newaxis]], axis=-1)

    del X_band_1, X_band_2, X_band_3, X_band_4, X_band_5


    X_band_smalltest_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
    X_band_smalltest_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])

    X_band_test_1 = np.zeros((X_band_smalltest_1.shape[0], xcoord_new.shape[0], ycoord_new.shape[0]))
    X_band_test_2 = np.zeros((X_band_smalltest_2.shape[0], xcoord_new.shape[0], ycoord_new.shape[0]))

    #Interpolate the input data to 149 X 149 X 2 grid
    for i in range(X_band_test_1.shape[0]):
        f = interpolate.interp2d(xcoord, ycoord, X_band_smalltest_1[i,:,:])
        X_band_test_1[i,:,:] = f(xcoord_new, ycoord_new)

        f = interpolate.interp2d(xcoord, ycoord, X_band_smalltest_2[i,:,:])
        X_band_test_2[i,:,:] = f(xcoord_new, ycoord_new)

    del X_band_smalltest_1, X_band_smalltest_2

    #Construct 3 channel data from the 2-channel input data
    X_band_test_3=np.fabs(np.subtract(X_band_test_1,X_band_test_2))
    X_band_test_4=np.maximum(X_band_test_1,X_band_test_2)
    X_band_test_5=np.minimum(X_band_test_1,X_band_test_2)

    X_test = np.concatenate([X_band_test_3[:, :, :, np.newaxis], 
                             X_band_test_4[:, :, :, np.newaxis],
                            X_band_test_5[:, :, :, np.newaxis]],axis=-1)

    del X_band_test_1, X_band_test_2, X_band_test_3, X_band_test_4, X_band_test_5

    return X_train, X_test


# Merge the generator for image input with angle input 
def gen_flow_for_two_inputs(X1, X2, y):
    genX1 = gen.flow(X1,y,  batch_size=batch_size,seed=55)
    genX2 = gen.flow(X1,X2, batch_size=batch_size,seed=55)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            #np.testing.assert_array_equal(X1i[0],X2i[0])
            yield [X1i[0], X2i[1]], X1i[1]

#Construct a model by concatentating image and angle as inputs
#Image is fed to InceptionResnetV2, and the angle is fed to dense layer
def angle_image_model(X_train):
    input_2 = Input(shape=[1], name="angle")
    angle_layer = Dense(1, )(input_2)
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, 
                 input_shape=X_train.shape[1:], classes=1)
    x = base_model.output
    x = GlobalMaxPooling2D()(x)

    merge_one = concatenate([x, angle_layer])

    merge_one = Dropout(0.6)(merge_one)
    merge_one = Dense(50, activation='relu', name='fc1',kernel_initializer='he_normal')(merge_one)
    merge_one = Dropout(0.5)(merge_one)
    predictions = Dense(1, activation='sigmoid',kernel_initializer='he_normal')(merge_one)
    
    model = Model(input=[base_model.input, input_2], output=predictions)
    
    sgd = Adam(lr=1e-4) 
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model


#Use K-fold Cross Validation with Data Augmentation.
def kfoldcv_prediction(X_train, X_angle, X_test):
    K=20
    folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=16).split(X_train, target_train))
    y_test_pred_log = 0
    y_train_pred_log=0
    y_valid_pred_log = 0.0*target_train

    for j, (train_idx, test_idx) in enumerate(folds):
        print('\n===================FOLD=',j)
        X_train_cv = X_train[train_idx]
        y_train_cv = target_train[train_idx]
        X_holdout = X_train[test_idx]
        Y_holdout= target_train[test_idx]
        
        #Angle
        X_angle_cv=X_angle[train_idx]
        X_angle_hold=X_angle[test_idx]

        #define file path and get callbacks
        file_path = "%s_aug_model_weights.hdf5"%j
        es = EarlyStopping('val_loss', patience=30, mode="min")
        msave = ModelCheckpoint(file_path, save_best_only=True)
        callbacks = [es, msave]

        gen_flow = gen_flow_for_two_inputs(X_train_cv, X_angle_cv, y_train_cv)
        satimage_model = angle_image_model(X_train)
        satimage_model.fit_generator(
                gen_flow,
                steps_per_epoch=48,
                epochs=200,
                shuffle=True,
                verbose=2,
                validation_data=([X_holdout,X_angle_hold], Y_holdout),
                callbacks=callbacks)

        #Getting the Best Model
        satimage_model.load_weights(filepath=file_path)
        #Getting Training Score
        score = satimage_model.evaluate([X_train_cv,X_angle_cv], y_train_cv, verbose=0)
        print('Train loss:', score[0])
        print('Train accuracy:', score[1])
        #Getting Test Score
        score = satimage_model.evaluate([X_holdout,X_angle_hold], Y_holdout, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        #Getting validation Score.
        pred_valid = satimage_model.predict([X_holdout,X_angle_hold])
        y_valid_pred_log[test_idx] = pred_valid.reshape(pred_valid.shape[0])

        #Getting Test Scores
        temp_test = satimage_model.predict([X_test, X_test_angle])
        y_test_pred_log += temp_test.reshape(temp_test.shape[0])

        #Getting Train Scores
        temp_train = satimage_model.predict([X_train, X_angle])
        y_train_pred_log += temp_train.reshape(temp_train.shape[0])

        del satimage_model
        gc.collect()
        keras.backend.clear_session() 


    y_test_pred_log = y_test_pred_log/K
    y_train_pred_log = y_train_pred_log/K

    print('\n Train Log Loss Validation= ',log_loss(target_train, y_train_pred_log))
    print(' Test Log Loss Validation= ',log_loss(target_train, y_valid_pred_log))
    return y_test_pred_log



def main():
    keras.backend.clear_session() 
    X_train, X_test = transform_train_test() 
    preds = kfoldcv_prediction(X_train, X_angle, X_test)

    submission = pd.DataFrame()
    submission['id']=test['id']
    submission['is_iceberg']=preds
    submission.to_csv('sub.csv', index=False)


if __name__ == "__main__":
    main()
