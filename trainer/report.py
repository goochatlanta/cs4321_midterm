import data_class
from sklearn.metrics import accuracy_score, precision_score,f1_score,recall_score,classification_report,confusion_matrix
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as PathEffects

def reverse_one_hot(y):
    y_transf = np.zeros(len(y))
    for indx1,image in enumerate(y):
        y_transf[indx1]= np.argmax(image)
    return y_transf

def test_model(hparams, model):
    test_ds = data_class.get_test_ds(hparams)
    y_pred = model.predict(test_ds)
    test_ds = test_ds.unbatch()
    images = list(test_ds.map(lambda x, y: x))
    labels = list(test_ds.map(lambda x, y: y))
    
    y_pred = reverse_one_hot(y_pred)
    labels = reverse_one_hot(labels)
    
    print("-----------------------")
    print('prediction',y_pred[50])
    print('labels',labels[50])
    print("-----------------------")

    print(confusion_matrix(labels,y_pred))
    print(classification_report(labels,y_pred))

    acc=accuracy_score(labels,y_pred)
    prec=precision_score(labels,y_pred,average="micro")
    f1 = f1_score(labels,y_pred, average="micro")
    rec = recall_score(labels,y_pred,average="micro")
    print("accuracy =", acc,"\nprecision =", prec, "\nrecall =", rec, "\nf1=", f1)

def tsne_visualize(x,y):
    mnist_tsne_train = TSNE(learning_rate = 'auto' ,random_state=123, init='pca').fit_transform(x)
    tsne_scatter(mnist_tsne_train, y, "test_tsne")


def tsne_scatter(x, colors, name="test"):
     # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(12, 12))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    #save fig
    plt.savefig(name)

    return f, ax, sc, txts
