import pandas as pd
import numpy as np
import json
import spacy
import time
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def save_model(model_name, model, history):
    assert type(history) is dict, 'history needs to be a dict'
    model.save('savedmodels/models/' + model_name)
    with open('savedmodels/model_info/{}.json'.format(model_name), 'w') as f:
        json.dump(history, f)
        f.close()

df = pd.read_csv('data/WELFake_Dataset.csv')
# df = pd.read_csv('data/kaggle.csv')

# df = df.fillna('')
df = df.dropna(subset='title') # Remove NaN
df = df.drop_duplicates(subset='title') # Remove duplicates
df = df.reset_index(drop=True) # Reset index


train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
train_df, dev_df = train_test_split(train_df, test_size=0.2, shuffle=False)
train_df, dev_df, test_df = train_df.reset_index(), dev_df.reset_index(), test_df.reset_index()

nlp = spacy.load('en_core_web_sm')

def preprocess(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

print('Processing text...')
x_train = [' '.join(preprocess(sent)) for sent in train_df['title']]
x_dev = [' '.join(preprocess(sent)) for sent in dev_df['title']]
x_test = [' '.join(preprocess(sent)) for sent in test_df['title']]
print('Done!')

print('Tokenizing...')
# Tokenizing
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(x_train)
token_train = tokenizer.texts_to_sequences(x_train)
token_dev = tokenizer.texts_to_sequences(x_dev)
token_test = tokenizer.texts_to_sequences(x_test)
print('Done!')

# Padding 
feature_size = 100
token_train = tf.keras.preprocessing.sequence.pad_sequences(token_train, feature_size)
token_dev = tf.keras.preprocessing.sequence.pad_sequences(token_dev, feature_size)
token_test = tf.keras.preprocessing.sequence.pad_sequences(token_test, feature_size)

print('Training RNNs...')
dropout_rates = [0.3, 0.5]
learning_rates = [1e-3, 1e-4]
index = 0
for dropout in dropout_rates:
    for lr in learning_rates:
        print('Dropout={} lr={}\n'.format(dropout, lr))
        vocab_size = len(tokenizer.word_index)
        embedding_vector_features = 100
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(name='inputs', shape=[feature_size]),
            tf.keras.layers.Embedding(vocab_size+1, embedding_vector_features, input_length=feature_size),
            tf.keras.layers.SimpleRNN(embedding_vector_features),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                    metrics=['accuracy'])
        
        history = model.fit(token_train, train_df['label'], validation_data=(token_dev, dev_df['label']), 
                        epochs=15, 
                        batch_size=64)
        
        hist = history.history
        hist['lr'] = lr
        hist['dropout'] = dropout

        y_preds = model.predict(token_test)
        y_preds = np.where(y_preds >= 0.5, 1, 0)
        report = classification_report(test_df['label'], y_preds, output_dict=True)
        hist['report'] = report
        print(classification_report(test_df['label'], y_preds))

        save_model('rnn_{}'.format(index), model, hist)
        index += 1


print('Training LSTMSs...')
dropout_rates = [0.3, 0.5]
learning_rates = [1e-3, 1e-4]
index = 0
for dropout in dropout_rates:
    for lr in learning_rates:
        print('Dropout={} lr={}\n'.format(dropout, lr))
        vocab_size = len(tokenizer.word_index)
        embedding_vector_features = 100
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(name='inputs', shape=[feature_size]),
            tf.keras.layers.Embedding(vocab_size+1, embedding_vector_features, input_length=feature_size),
            tf.keras.layers.LSTM(embedding_vector_features),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                    metrics=['accuracy'])
        
        history = model.fit(token_train, train_df['label'], validation_data=(token_dev, dev_df['label']), 
                        epochs=15, 
                        batch_size=64)
        
        hist = history.history
        hist['lr'] = lr
        hist['dropout'] = dropout

        y_preds = model.predict(token_test)
        y_preds = np.where(y_preds >= 0.5, 1, 0)
        report = classification_report(test_df['label'], y_preds, output_dict=True)
        hist['report'] = report
        print(classification_report(test_df['label'], y_preds))

        save_model('lstm_{}'.format(index), model, hist)
        index += 1


print('Training BiLSTMs...')
dropout_rates = [0.3, 0.5]
learning_rates = [1e-3, 1e-4]
index = 0
for dropout in dropout_rates:
    for lr in learning_rates:
        print('Dropout={} lr={}\n'.format(dropout, lr))
        vocab_size = len(tokenizer.word_index)
        embedding_vector_features = 100
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(name='inputs', shape=[feature_size]),
            tf.keras.layers.Embedding(vocab_size+1, embedding_vector_features, input_length=feature_size),
            Bidirectional(tf.keras.layers.LSTM(embedding_vector_features)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                    metrics=['accuracy'])
        
        history = model.fit(token_train, train_df['label'], validation_data=(token_dev, dev_df['label']), 
                        epochs=15, 
                        batch_size=64)
        
        hist = history.history
        hist['lr'] = lr
        hist['dropout'] = dropout

        y_preds = model.predict(token_test)
        y_preds = np.where(y_preds >= 0.5, 1, 0)
        report = classification_report(test_df['label'], y_preds, output_dict=True)
        hist['report'] = report
        print(classification_report(test_df['label'], y_preds))

        save_model('bilstm_{}'.format(index), model, hist)
        index += 1

print('Done!')