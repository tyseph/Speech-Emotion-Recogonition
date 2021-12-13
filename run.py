from librosa.filters import chroma
from pydub.playback import play
import sounddevice
from scipy.io.wavfile import read, write


def record():
    fs = 44100
    #second =  int(input("Enter time duration in seconds: "))
    second = 5
    print("Recording.....n")
    record_voice = sounddevice.rec(int(second * fs), samplerate=fs, channels=1)
    sounddevice.wait()
    write("./uploaded/00-00-00-00-00-00-00.wav", fs, record_voice)
    file_name = "D:\\4th year\\IWS\\project\\uploaded\\00-00-00-00-00-00-00.wav"
    #file_name = "./03-01-01-01-01-01-02.wav"
    return file_name


def convert(audio):
    # Create an AudioSegment instance
    wav_file = audio.split_to_mono()
    return wav_file[0]

# new  = load.extract_feature(record(), mfcc=True, chroma=True, mel=True)
# print(new)

# print(load.load_file(record()))