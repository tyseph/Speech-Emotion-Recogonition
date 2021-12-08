import pickle
import librosa
import soundfile
import numpy as np

emotions = {
    '00': 'test',
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

observed_emotions = ['calm', 'happy', 'fearful', 'disgust', 'test']

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(
                S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(
                X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result


def load_it(feature):
    filename = 'finalized_model.sav'
    model = pickle.load(open(filename, 'rb'))
    x = []
    x.append(feature)
    return model.predict(x)


def load_file(file):
    file_name = file
    # file_name = os.path.basename(
    #     "D:\\4th year\\IWS\\project\\speech-emotion-recognition-ravdess-data\\Actor_01\\03-01-01-01-01-01-01.wav")
    emotion = emotions[file_name.split("-")[2]]
    print(emotion)
    return load_it(extract_feature(file_name, mfcc=True, chroma=True, mel=True))
