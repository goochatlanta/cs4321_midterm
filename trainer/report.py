import data_class
from sklearn.metrics import accuracy_score, precision_score,f1_score,recall_score,classification_report,confusion_matrix
import numpy as np

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
