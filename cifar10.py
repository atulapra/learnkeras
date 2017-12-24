import os 
import time
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras.datasets import cifar10

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

"""
There are 5 stages in a keras program: 
1) create model
2) compile 
3) fit(train)
4) evaluate(accuracy,loss)
5) test(predict)
"""

np.random.seed(2017) 

if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")

# Load input
# CIFAR-10 contains 32x32x3 images with 10 classes
(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
num_train, img_channels, img_rows, img_cols =  train_features.shape
num_test, _, _, _ =  test_features.shape
num_classes = len(np.unique(train_labels))

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Show example images
fig = plt.figure(figsize=(8,3))
for i in range(num_classes):
    ax = fig.add_subplot(2,5,1+i,xticks = [],yticks = [])
    idx = np.where(train_labels[:]==i)[0]
    features_idx = train_features[idx,::]
    img_num = np.random.randint(features_idx.shape[0])
    im = np.transpose(features_idx[img_num,::], (1, 2, 0))
    ax.set_title(class_names[i])
    plt.imshow(im)
plt.show()

# Pre-processing
train_features = train_features.astype('float32')/255
test_features = test_features.astype('float32')/255

# convert class labels(0-9) to one-hot class labels
train_labels = np_utils.to_categorical(train_labels, num_classes)
test_labels = np_utils.to_categorical(test_labels, num_classes)

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
    plt.show()

def accuracy(test_x,test_y,model):
    """
    Compute test accuracy
    """
    result = model.predict(test_x)
    predicted_class = np.argmax(result,axis = 1)
    true_class = np.argmax(test_y,axis = 1)
    num_correct = np.sum(predicted_class == true_class)     # Number of entries where prediction is right
    accuracy = float(num_correct)/result.shape[0]           # result.shape[0] is the number of examples(entries)

    return (accuracy * 100)

# Model definition

model = Sequential()
# 48 filters with a kernel size of 3x3
# 'same' padding is to ensure that output size is same as input size 
model.add(Conv2D(48,kernel_size=(3,3),padding='same',activation='relu',input_shape=(3, 32, 32)))
# Shape is 48x3x32x32
model.add(Activation('relu'))
model.add(Conv2D(48,kernel_size=(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Now, shape is 48x3x16x16
model.add(Dropout(0.25))
model.add(Conv2D(96,kernel_size=(3,3),padding='same'))
# Now : 96x3x16x16
model.add(Activation('relu'))
model.add(Conv2D(96,kernel_size=(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Now : 96x3x8x8 
model.add(Dropout(0.25))
model.add(Conv2D(192,kernel_size=(3,3),padding='same'))
# Now : 192x3x8x8
model.add(Activation('relu'))
model.add(Conv2D(192,kernel_size=(3,3),padding='same'))
# Now : 192x3x8x8
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Now : 192x3x4x4
model.add(Dropout(0.25))
model.add(Flatten())
# Now : 9216
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Now compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Train(fit) the model
start = time.time()
model_info = model.fit(train_features,train_labels,batch_size=128,epochs=20,validation_data=(test_features,test_labels),verbose=1)
end = time.time()

# Plot model history
plot_model_history(model_info)
print("Model took ",end - start," seconds to train")

# Compute test accuracy
print("Accuracy on test data : ",accuracy(test_features,test_labels,model))


