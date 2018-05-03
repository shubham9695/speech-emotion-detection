
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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# fix random seed for reproducibility
np.random.seed(7)


# function to extract MFCC features
def vectorize_audio(audiopath):
    (sig,rate) = sf.read(audiopath)
    mfcc_feat = mfcc(sig,rate)
    d_mfcc_feat = delta(mfcc_feat,2)
    num_feat = len(d_mfcc_feat)
    return np.mean(d_mfcc_feat[int(num_feat / 10):int(num_feat * 9 / 10)], axis=0)


#data labelling
data_dir = "D:\Users\850034074\Desktop\Speech_Recognition and emotion detection\\bank\\audio"
audio_files = os.listdir(data_dir)

# (0:angry, 1:sad, 2:neutral)
dict_to_map = {'a' : 0, 's' : 1, 'n' : 2}
Y_array = []
X_array = []

for fil in audio_files:
        x = vectorize_audio(os.path.join(data_dir,fil))
        X_array.append(list(x))
        Y_array.append(dict_to_map[fil[0]])

X_array = np.array(X_array)
Y_array = np.array(Y_array)


# splitting the dataset for training and test part
X_train, X_test, Y_train, Y_test = train_test_split(X_array, Y_array, test_size=0.25, random_state=7)


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
Y_trainn = np_utils.to_categorical(Y_train,num_classes=3)
Y_testt = np_utils.to_categorical(Y_test,num_classes=3)


#libraries to build the model
from keras.models import Sequential
from keras.layers import Dense

#create model
model = Sequential()
model.add(Dense(20,input_dim=13,activation='relu'))
model.add(Dense(3,activation='softmax'))

# compile model
adam = optimizers.adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics= ['accuracy'])

# Fit the model or train the model
model.fit(X_train,Y_trainn, epochs=2500, batch_size=10, validation_data=(X_test,Y_testt))

# evaluate the model using test dataset
score = model.evaluate(X_test, Y_testt,verbose=1)
print(score)


"""
from keras.models import load_model
# Creates a HDF5 file
#model.save('my_model.h5')

# Returns a compiled model identical to the previous one
#model = load_model('my_model.h5')
"""

# calculate predictions
y_pred = model.predict(X_test)
print(y_pred.argmax(1))
print(Y_test)

# Confusion matrix
confusion_matrix(Y_test, y_pred.argmax(1))

#classification report
print(classification_report(Y_test, y_pred.argmax((1))))
