from flask import Flask
from flask import render_template, request
app=Flask(__name__)

from werkzeug.utils import secure_filename

import os
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
from keras.preprocessing.image import load_img
import librosa.display
import scipy.io.wavfile as wavfile
import numpy
import os.path
from os import walk
from scipy import stats
import numpy as np
import librosa 
import numpy as np
from scipy.stats import norm
import pickle
import matplotlib.pyplot as plt
# Import the libraries
import matplotlib.pyplot as plt
from sklearn import svm
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input

UPLOAD_FOLDER = '/app/uploads'
ALLOWED_EXTENSIONS = {'wav', 'png', 'jpg', 'jpeg'}
#pp = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def home():
    return "Hello, Flask!"
    
@app.route('/upload')
def upload():
   return render_template("form.html")



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS 
           


@app.route('/uploader', methods = ['GET','POST'])
def upload_file():
   if request.method == 'GET':
       return "hello"
   if request.method == 'POST':
         file = request.files['file']
         if file.filename == '':
              return 'No file selected'
         if file and allowed_file(file.filename):
          filename = secure_filename(file.filename)
          file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                   
          signal, rate = librosa.load(UPLOAD_FOLDER+'/'+filename)  
           #The Mel Spectrogram
          S = librosa.feature.melspectrogram(signal, sr=rate, n_fft=2048,    hop_length=512, n_mels=128)
          S_DB = librosa.power_to_db(S, ref=np.max)
          S_DB = S_DB.flatten()[:1200]
          clf = pickle.load(open('SVM.pkl' , 'rb'))
          ans = clf.predict([S_DB])[0]
          music_class = str(ans)
          print(music_class)
          return music_class
          
          
    
               
      
		
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  
   