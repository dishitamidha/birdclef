import os
from glob import glob
from flask import Flask, flash, request, redirect, send_from_directory, url_for, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import pandas as pd
import librosa as lb
import glob
import librosa.display
import warnings
warnings.filterwarnings('ignore')
from IPython.display import Image
from PIL import Image
import statistics as st
from sklearn.preprocessing import LabelEncoder
from numpy import asarray
from sklearn.model_selection import train_test_split
# from torch.utils.data import Dataset, DataLoader
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import efficientnet.tfkeras as efn 
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
logger = tf.get_logger()
logger.setLevel(logging.ERROR) 


from numpy.random import seed
seed(42)
tf.random.set_seed(42)


birds_name = pd.read_csv('birds_names.csv')
common_name_dict = {}
image_name_dict = {}

for idx, r in birds_name.iterrows():
  common_name_dict[r['primary_label_encoded']] = r['common_name']
  image_name_dict[r['common_name']] = r['image_path']


class config:
  N_FFT = 2048
  HOP_LEN = 1024
  SAMPLE_RATE = 22050
  DURATION = 5

def saveMel(signal, sr, directory):
  
    N_FFT = 1024         
    HOP_SIZE = 1024      
    N_MELS = 128          
    WIN_SIZE = 1024      
    WINDOW_TYPE = 'hann' 
    FEATURE = 'mel'      
    FMIN = 1400

    fig = plt.figure(1,frameon=False)
    fig.set_size_inches(6,6)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    S = librosa.feature.melspectrogram(y=signal, sr=sr,
                                        n_fft=N_FFT,
                                        hop_length=HOP_SIZE, 
                                        n_mels=N_MELS, 
                                        htk=True, 
                                        fmin=FMIN, 
                                        fmax=sr/2) 
    librosa.display.specshow(librosa.power_to_db(S**2,ref=np.max), fmin=FMIN) 

    fig.savefig(directory)
    plt.ioff()
    fig.clf()
    ax.cla()
    plt.clf()
    plt.close('all')


size = {'desired': 5,
        'minimum': 4,
        'stride' : 0
}
step=1

def audio_proc(audiopath, dest):
  signal, sr = librosa.load(audiopath)

  step = (size['desired']-size['stride'])*sr
  
  nr=0;
  melpaths = []

  for start, end in zip(range(0,len(signal),step),range(size['desired']*sr,len(signal),step)):
      nr=nr+1
    
      if end-start > size['minimum']*sr:

        correction = "_" + str(nr) + ".png"
        
        melpath = dest + correction
       
        print(melpath + '---')

        saveMel(signal[start:end], sr ,melpath)
        melpaths.append(melpath)

  return melpaths

'''
Function for saving melspectograms as images

'''

def create_image(audiopath):
    components = audiopath.split('/')
    components = components[-1].split('.')
    filename = components[0]
    dest = 'content/test_images/' + filename
    final_dests = audio_proc(audiopath, dest)
    return final_dests



def make_prediction(test_df, model):


  if model == 'effnet':
    from efficientnet.keras import center_crop_and_resize, preprocess_input
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator=test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=None,
    x_col="image_path",
    y_col=None,
    batch_size=16,
    shuffle=False,
    class_mode=None,
    target_size=(224,224))

    model = load_model('models/effnet-with-new-data.h5')

  
  elif model == 'xception':
    from tensorflow.keras.applications.xception import Xception, preprocess_input
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator= test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=None,
    x_col="image_path",
    y_col=None,
    batch_size=16,
    shuffle=False,
    class_mode=None,
    target_size=(299,299))

    model = load_model('models/xception-with-new-data.h5')

  test_generator.reset()
  pred = model.predict_generator(test_generator,verbose=1)
  predicted_class_indices=np.argmax(pred,axis=1)

  del model

  final_predicted_index = st.mode(predicted_class_indices) 
  
  print(predicted_class_indices)

  return (common_name_dict[final_predicted_index])


 

ALLOWED_EXTENSIONS = {'wav', 'ogg', 'mp3'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'uploads')

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])

def index():
    res = ''
    img = ''
    filename = None
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            audiopath = 'static/uploads/' + filename
            final_dests = create_image(audiopath)

            test_df = pd.DataFrame()
            test_df = test_df.assign(image_path = final_dests)

            res = make_prediction(test_df, 'xception')
            img = image_name_dict[res]
            print(img)
            
    return render_template('index.html', filename=filename, prediction = res, im=img)

if __name__ == '__main__':
    app.run(port=3000, debug=True)