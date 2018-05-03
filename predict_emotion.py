
# libraries required to run this script

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import numpy as np
import soundfile as sf
from keras.models import load_model

# fix random seed for reproducibility
np.random.seed(7)


# function to extract MFCC features
def vectorize_audio(audiopath):
    (sig,rate) = sf.read(audiopath)
    mfcc_feat = mfcc(sig,rate)
    d_mfcc_feat = delta(mfcc_feat,2)
    num_feat = len(d_mfcc_feat)
    return np.mean(d_mfcc_feat[int(num_feat / 10):int(num_feat * 9 / 10)], axis=0)


# function to predict the emotion of teh audio file
# accepts audiopath and modelpath as input and return emotion class(0,1,2,3)
# (0:angry, 1:sad, 2:neutral, 3:happy)
def predict_emotion(audiopath,modelpath):
    X = vectorize_audio(audiopath)
    X = X.reshape(1,-1)
    model = load_model(modelpath)
    return np.argmax(model.predict(X))


#audiopath = "D:\Users\shubham\Desktop\Speech_Recognition and emotion detection\\bank\\audio\\a15.wav"
#modelpath = "D:\\Users\\shubham\\New_model.h5"
