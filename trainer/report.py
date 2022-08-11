#%%
import data_class
from sklearn.metrics import accuracy_score, precision_score,f1_score,recall_score,classification_report,confusion_matrix,ConfusionMatrixDisplay
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as PathEffects
import pandas as pd
import os
import argparse
import params

import tensorflow as tf

def reverse_one_hot(y):
    y_transf = np.zeros(len(y))
    for indx1,image in enumerate(y):
        y_transf[indx1]= np.argmax(image)
    return y_transf

def test_model(hparams, model):
    test_ds = data_class.get_test_ds(hparams)
    #y_pred = model.predict(test_ds)
    score = model.evaluate(test_ds, verbose = 0) 
    print('\nTest loss:', score[0]) 
    print('Test accuracy:', score[1],'\n')


def tsne_visualize_raw_data(hparams):
    print(hparams.tsne_ds)
    ds = data_class.get_train_ds(hparams)
    #features_raw = np.concatenate(list(ds.map(lambda x, y : x)))
    #labels = np.concatenate(list(ds.map(lambda x, y : y)))
    m = list(ds.map(lambda x, y :[x, y]))
    features_raw = []
    labels = []
    for feature,label in m:
        features_raw.append(feature)
        labels.append(label)
    features_raw = np.concatenate(features_raw)
    labels = np.concatenate(labels)
    labels = reverse_one_hot(labels)
    features_raw = features_raw.reshape(len(labels), 299*299*3)
    tsne = TSNE(n_components=2, random_state=123)
    x = tsne.fit_transform(features_raw) 
    plot_tsne(x,'Raw_Data',ds,labels,hparams.model_dir)

def tsne_visualize_model(hparams, model):
    hparams.tsne_ds = None
    ds = data_class.get_train_ds(hparams)
    model2 = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    train_ds = np.concatenate(list(ds.take(5).map(lambda x, y : x)))
    features = model2(train_ds)
    labels = np.argmax(model(train_ds), axis=-1)
    #labels = reverse_one_hot(labels)
    tsne = TSNE(n_components=2, random_state=123)
    x = tsne.fit_transform(features) 
    plot_tsne(x,'model',ds,labels,hparams.model_dir)

def plot_tsne(x,title,ds,labels,model_dir):
    tx = x[:, 0]
    ty = x[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    colors = ['red', 'blue', 'green', 'brown', 'yellow', 'orange', 'pink', 'black']
    classes = ds.class_names
    print(classes)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for idx, c in enumerate(colors):
        indices = [i for i, l in enumerate(labels) if idx == l]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        ax.scatter(current_tx, current_ty, c=c, label=classes[idx])
    ax.legend(loc='upper left', bbox_to_anchor=(0.8, 1.15))
    plt.title(title)
    if not os.path.exists(model_dir+'report'):
        os.mkdir(model_dir+'report')
    plt.savefig(f'{model_dir}report/{title}.png')
    #plt.savefig(f'{title}.png')

def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range
#%%
def main():

    hparams = params.get_hparams()
    test_ds = data_class.get_test_ds(hparams)
    model = tf.keras.models.load_model(hparams.model_dir+hparams.model_type.lower()+'_unfrozen')
    tsne_visualize_raw_data(hparams)
    tsne_visualize_model(hparams,model)
    model.summary()
    m = list(test_ds.map(lambda x, y :[x, y]))
    
    features = []
    labels = []
    for feature,label in m:
        features.append(feature)
        labels.append(label)
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    y_pred = model.predict(features,batch_size =32)
    labels = np.argmax(labels, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)

    cm = classification_report(labels, y_pred)
    print(cm)
    disp = ConfusionMatrixDisplay.from_predictions(labels,y_pred,
                               display_labels=test_ds.class_names)
    disp.plot()
    plt.title(hparams.model_type)
    plt.savefig(f'{hparams.model_dir}report/{hparams.model_type}_confusion_matrix.png')
    plt.savefig('test.png')

    

if __name__ == "__main__":
    main()
# %%
