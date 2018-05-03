
# libraries required to run this script
import os
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras import optimizers
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# fix random seed for reproducibility
np.random.seed(21)


# function to extract MFCC features
def vectorize_audio(audiopath):
    (sig,rate) = sf.read(audiopath)
    mfcc_feat = mfcc(sig,rate)
    d_mfcc_feat = delta(mfcc_feat,2)
    num_feat = len(d_mfcc_feat)
    return np.mean(d_mfcc_feat[int(num_feat / 10):int(num_feat * 9 / 10)], axis=0)


#data labelling
data_dir = "D:\Users\shubham\Desktop\Speech_Recognition and emotion detection\\bank\\NewSamples"
audio_files = os.listdir(data_dir)
# (0:angry, 1:sad, 2:neutral, 3:happy)
dict_to_map = {'1' : 2, '3' : 3, '4' : 1, '5' : 0}
Y_array = []
X_array = []

for fil in audio_files:
    if fil[7] == '1' or fil[7] == '3' or fil[7]=='4' or fil[7] == '5' :
        x = vectorize_audio(os.path.join(data_dir,fil))
        X_array.append(list(x))
        Y_array.append(dict_to_map[fil[7]])

X_array = np.array(X_array)
Y_array = np.array(Y_array)


# splitting the dataset for training and test part
X_train, X_test, Y_train, Y_test = train_test_split(X_array, Y_array, test_size=0.3, random_state=21)


"""
from sklearn.preprocessing import StandardScaler

# Define the scaler
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)
"""


# encode classes with one output per neuron
Y_trainn = np_utils.to_categorical(Y_train,num_classes=4)
Y_testt = np_utils.to_categorical(Y_test,num_classes=4)


#libraries to build the model
from keras.models import Sequential
from keras.layers import Dense

#create model
model = Sequential()
model.add(Dense(22,input_dim=13,activation='relu'))
model.add(Dense(4,activation='softmax'))

# compile model
adam = optimizers.adam(lr=0.0005,decay=0.000001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics= ['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Fit the model or train the model
model.fit(X_train,Y_trainn, epochs=1500, batch_size=10,callbacks=[early_stopping], validation_data=(X_test,Y_testt))

# evaluate the model using test dataset
score = model.evaluate(X_test, Y_testt,verbose=1)
print(score)

"""
from keras.models import load_model
# Creates a HDF5 file to save the model
model.save('my_model.h5')

# Returns a compiled model identical to the previous one
#model = load_model('my_model.h5')
"""

# calculate predictions
y_pred = model.predict(X_test)
print(y_pred.argmax(1))

# Confusion matrix
confusion_matrix(Y_test, y_pred.argmax(1))

#classification report
print(classification_report(Y_test, y_pred.argmax((1))))
