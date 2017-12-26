from __future__ import print_function
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical, np_utils
import numpy as np
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="test/train")
a = ap.parse_args()
mode = a.mode 

"""
There are 5 stages in a keras program: 
1) create model
2) compile 
3) fit(train)
4) evaluate(accuracy,loss)
5) test(predict)
"""

# For reproducibility
np.random.seed(1000)

def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()

def accuracy(test_x,test_y,model):
    """
    Compute test accuracy
    Input: test images, test labels and model 
    """
    result = model.predict(test_x)
    predicted_class = np.argmax(result,axis = 1)
    true_class = np.argmax(test_y,axis = 1)
    num_correct = np.sum(predicted_class == true_class)     # Number of entries where prediction is right
    accuracy = float(num_correct)/result.shape[0]           # result.shape[0] is the number of examples(entries)
    return (accuracy * 100)

# Load the dataset
# CIFAR-10 contains 60000 32x32x3 images with 10 classes
# There are 10000 training images and 10000 test images
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Pre-processing
X_train = X_train.astype('float32')/255
X_test =  X_test.astype('float32')/255

# convert class labels(0-9) to one-hot class labels
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

if mode == "train":
    # Compile the model. We start with a learning rate of 0.0001 and use learning rate decay
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])

    # create checkpoint
    filepath = "cifar10_weights_10epoch.hdf5"
    checkpoint = ModelCheckpoint(filepath,monitor='val_acc',verbose=0,save_best_only=True,mode='max')
    callbacks_list = [checkpoint]

    # Train the model
    model_info = model.fit(X_train, Y_train,batch_size=128,shuffle=True,epochs=10,
                        validation_data=(X_test,Y_test),callbacks=callbacks_list)
    print("Model saved")

    # model_info = model.fit(X_train, to_categorical(Y_train),batch_size=128,shuffle=True,epochs=20,
    #             validation_data=(X_test,to_categorical(Y_test)),callbacks=[EarlyStopping(min_delta=0.001,patience=3)])

    plot_model_history(model_info)

    # Evaluate the model
    print("Evaluating model...")
    scores = model.evaluate(X_test,Y_test)

    print("\n Loss: {a:8.3f}, Accuracy: {b:8.3f}".format(a=scores[0],b=scores[1]))
    print("Accuracy on test data : ",accuracy(X_test,Y_test,model))

if mode == "test":
    # Load pre-trained weights
    model.load_weights('./cifar10_weights_10ep.hdf5')

    # Compile the model. We start with a learning rate of 0.0001 and use learning rate decay
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])

    print("Accuracy on test data : ",accuracy(X_test,Y_test,model))
