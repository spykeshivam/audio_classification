import librosa
from pydub import AudioSegment
from IPython.display import Audio
import soundfile as sf
import sklearn
import sklearn.preprocessing
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa.display

MP3FILE='Shepard.mp3'
WAVEFILE='test.wav'

def convert_mp3_to_wav(mp3):
    samples,sr=librosa.load(mp3,sr=None)
    #Audio(data=samples,rate=sr) #To listen to the audio on a jupyter notebook
    sf.write('test.wav', samples,sr,'PCM_24')
    return samples,sr


def MFCC(samples,sr):
    mfcc=librosa.feature.mfcc(y=samples,sr=sr)
    mfcc=sklearn.preprocessing.scale(mfcc,axis=1)
    plt.figure(figsize=(15,10))
    librosa.display.specshow(mfcc,x_axis='time',sr=sr)
    plt.show()

def display_wav(samples,sr):
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(samples, sr=sr)

#AudioSegment.from_mp3('Shepard.mp3')
#convert_mp3_to_wav('Shepard.mp3')

if __name__=="__main__":
    samples,sr=convert_mp3_to_wav(MP3FILE)
    samples, sample_rate = librosa.load(WAVEFILE, sr=None)


    display_wav(samples,sample_rate)
    MFCC(samples,sr)
    
