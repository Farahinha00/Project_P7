!pip install joblib
!pip install pandas
!pip install lime
!pip install gdown

import tensorflow as tf
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, LSTM, Dropout, Flatten, SimpleRNN
import joblib
import lime
from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

import gdown

url_model = 'https://drive.google.com/file/d/1xLUbmweUrzLHSeD-oA6JCn0p_7MMG1rw/view?usp=sharing'
url_tokenizer = 'https://drive.google.com/file/d/1T1TyJIXPg_JIiiGW6wSZId2rbqQVFtfr/view?usp=sharing'

gdown.download(url_model, './Common/mon_best_model.h5', quiet=False)
gdown.download(url_tokenizer, './Common/tokenizer.pkl', quiet=False)

# Charger le modèle pré-entraîné
best_model = tf.keras.models.load_model('./Common/mon_best_model.h5')
tokenizer = joblib.load('./Common/tokenizer.pkl')

def tokenize_data(data,tokenizer,maxlen=None):
    X = tokenizer.texts_to_sequences(data)
    return pad_sequences(X,maxlen=maxlen)

def model_predict(texts):
    # Assure-toi que 'texts' est prétraité de la même manière que les données d'entraînement
    processed_texts = tokenize_data(texts,tokenizer,maxlen=50)
    return best_model.predict(processed_texts)

def make_inference(input_data):
    # Prédiction à l'aide du modèle
    processed_data = tokenize_data([input_data],tokenizer,maxlen=50)
    prediction = best_model.predict(processed_data)

    # Affichage des résultats de la prédiction
    print("Prédiction brute du modèle : ", prediction)
    
    explainer = LimeTextExplainer(class_names=['bad', 'good'])
    exp = explainer.explain_instance(input_data, model_predict, num_features=10)
    exp.show_in_notebook(text=True,show_table=True)
    
    return prediction, exp


