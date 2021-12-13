from flask import *
import librosa
from scipy.signal.signaltools import residue
import load
import speech_recognition as sr
import run
import os

UPLOAD_FOLDER = './uploaded'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route("/", methods=["GET"])
def home():
    final_pred = "Click the icon to start Recording!"
    return render_template('index.html', final_pred=final_pred)


@app.route("/result")
def result():
    emotion = load.load_file()


@app.route("/loading", methods=['POST'])
def move_forward():
    file_name = run.record()
    return render_template("index.html", final_pred="You seem " + (load.load_it(load.extract_feature(file_name, mfcc=True, chroma=True, mel=True)))[0])

@app.route("/upload")
def test():
    final_pred = "Upload a .wav file to predict emotion!"
    return render_template("upload.html", final_pred=final_pred)

@app.route('/uploader', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        print(f)
        f.save(f.filename)
        return render_template("upload.html", final_pred="You seem " + (load.load_file(f.filename))[0])


if __name__ == '__main__':
    app.run(debug=True)
