
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from matplotlib import pyplot as plt



if __name__ == '__main__':



    

    cm = confusion_matrix(labels_test, lenet_test_predictions)
    target_names = ('blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names)
    disp.plot()
    plt.show()