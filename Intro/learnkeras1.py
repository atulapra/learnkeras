# Train and save weights
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense              # for fully connected layers
import numpy

# Five stages: create model, compile, fit(train), evaluate(print accuracy,loss), test(predict)

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]                            # the last column is output

# create model
model = Sequential()
model.add(Dense(12, input_dim=8,activation='relu',kernel_initializer='uniform'))
model.add(Dense(8,activation='relu',kernel_initializer='uniform'))
model.add(Dense(1,activation='sigmoid',kernel_initializer='uniform'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# checkpoint
filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max')
callbacks_list = [checkpoint]

# Fit the model
model.fit(X, Y, epochs=150,validation_split=0.33,batch_size=10,callbacks=callbacks_list, verbose=2)

# Evaluate the model (training accuracy)
# scores = model.evaluate(X, Y)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)

# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)