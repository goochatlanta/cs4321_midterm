import data_class
from sklearn.metrics import classification_report,ConfusionMatrixDisplay
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as PathEffects
import pandas as pd
import os
import params
import tensorflow as tf
"""
report_module
Description: Contains functions to make a report of the model. The plots of the report are saved
             in the model's directory under the report folder
Functions:  test_model
            tsne_visualize_raw_data
            tsne_visualize_model
            plot_tsne
            scale_to_01_range
"""

def test_model(hparams, model):
    """
    Inputs: 
      hparams: <arg object> the arguments
      model: <Keras.Model> the type of the dataset to load
    Outputs:
      It prints the accuracy and the loss of the model in the trin dataset
    """
    test_ds = data_class.get_test_ds(hparams)
    score = model.evaluate(test_ds, verbose = 0) 
    print('\nTest loss:', score[0]) 
    print('Test accuracy:', score[1],'\n')


def tsne_visualize_raw_data(hparams):
    """
    Inputs: 
        hparams: <arg object> the arguments
    Description:
        extract the 2 bigest components of the raw data
    """
    ds = data_class.get_train_ds(hparams)
    
    # map the features and the labels
    m = list(ds.map(lambda x, y :[x, y]))

    # prepare the data for the tsne
    features_raw = []
    labels = []
    for feature,label in m:
        features_raw.append(feature)
        labels.append(label)
    features_raw = np.concatenate(features_raw)
    labels = np.concatenate(labels)
    labels = np.argmax(labels, axis=-1)
    features_raw = features_raw.reshape(len(labels), 299*299*3)
    
    # create the tsne model to get the 2 bigest components
    tsne = TSNE(n_components=2, random_state=123)
    
    # extract the components for each image
    x = tsne.fit_transform(features_raw) 
    
    # call the plot function
    plot_tsne(x,'Raw_Data',ds,labels,hparams.model_dir)

def tsne_visualize_model(hparams, model):
    """
    Inputs: 
        hparams: <arg object> the arguments
        model: the model that we want to extract the features
    Description:
        extract the 2 bigest components of the model from the preivious layer of the classifier
    """
    # set the argument to none to get the augmented data    
    hparams.tsne_ds = None
    # prepare the train data for the tsne
    ds = data_class.get_train_ds(hparams)
    model2 = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    train_ds = np.concatenate(list(ds.take(5).map(lambda x, y : x)))
    features = model2(train_ds)
    labels = np.argmax(model(train_ds), axis=-1)

    # extract the components
    tsne = TSNE(n_components=2, random_state=123)
    x = tsne.fit_transform(features)

    # call the plot function 
    plot_tsne(x,'model',ds,labels,hparams.model_dir)

def plot_tsne(x,title,ds,labels,model_dir):
    """
    Inputs: 
      x: <List> The list with the values of the 2 strongest components for each image
      title: <String> the title of the plot and the file
      ds: <dataset> the dataset with the initial images and labels
      labels: the labels for each image. The list must be 1 dimenssion
      model_dir: The path of the model that we want to tsne
    Outputs:
      Save the tsne plot for the raw train dataset
    """
    
    # prepare data for plotting
    tx = x[:, 0]
    ty = x[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    colors = ['red', 'blue', 'green', 'brown', 'yellow', 'orange', 'pink', 'black']
    classes = ds.class_names
    
    # plot the data 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for idx, c in enumerate(colors):
        indices = [i for i, l in enumerate(labels) if idx == l]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        ax.scatter(current_tx, current_ty, c=c, label=classes[idx])
    ax.legend(loc='upper left', bbox_to_anchor=(0.8, 1.15))
    plt.title(title)

    # save plot to dir
    if not os.path.exists(model_dir+'report'):
        os.mkdir(model_dir+'report')
    plt.savefig(f'{model_dir}report/{title}.png')

def scale_to_01_range(x):
    """
    Inputs: 
        x: <np.array> the components that we want to scale
    Description:
        scale the components from 0 to 1 
    """
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

def main():

    hparams = params.get_hparams()
    # get the train ds
    test_ds = data_class.get_test_ds(hparams)
    
    # load the model fron the report
    model = tf.keras.models.load_model(hparams.model_dir+hparams.model_type.lower()+'_unfrozen')
    
    # create the tsne plots
    tsne_visualize_raw_data(hparams)
    tsne_visualize_model(hparams,model)
    
    # print the models architecture
    model.summary()

    #prepare data for confusion matris report
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

    # print the classification report
    cm = classification_report(labels, y_pred)
    print(cm)

    # create and save the confusion matrix report
    disp = ConfusionMatrixDisplay.from_predictions(labels,y_pred,
                               display_labels=test_ds.class_names)
    disp.plot()
    plt.title(hparams.model_type)
    plt.savefig(f'{hparams.model_dir}report/{hparams.model_type}_confusion_matrix.png')

if __name__ == "__main__":
    main()

