from flask import *
from scipy.signal.signaltools import residue
import load
import speech_recognition as sr
import run

UPLOAD_FOLDER = './uploaded'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/result")
def result():
    emotion = load.load_file()


@app.route("/loading", methods=['POST'])
def move_forward():
    file_name = run.record()
    return render_template('index.html', final_pred=load.load_it(load.extract_feature(file_name, mfcc=True, chroma=True, mel=True)))


@app.route('/', methods=['POST'])
def test():
    if request.method == 'POST':
        f = request.files['file']
        print(f)
        f.save(f.filename)
        return render_template("index.html", final_pred=load.load_file(f.filename))


if __name__ == '__main__':
    app.run(debug=True)
