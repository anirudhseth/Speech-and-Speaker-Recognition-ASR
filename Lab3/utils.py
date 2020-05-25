''' plot functions '''
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
sn.set()

def load_data_npz(fpath, feature, dynamic=False):
    data = np.load(fpath, allow_pickle=True)['data'].item()
    if dynamic:
        print('using dynamic features')
        features = data[feature+'_dynamic']
    else:
        print('using ordinary (not dynamic) features')
        features = data[feature]
    targets = data['targets']
    return features, targets


def plot_train_val_curves(history, model_name):
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Accuracies')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'results/' + model_name + '/acc.png')
    plt.show()

    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.savefig(f'results/' + model_name + '/loss.png')
    plt.show()
    plt.clf()

    
def plot_confusion_matrix(cm, n_classes, model_name, title):
    
    class_labels = [i for i in range(n_classes)]# TODO replace with actual class names

    sn.heatmap(cm, norm=LogNorm(cm.min(),cm.max()), cmap='Blues',
                # cbar_kws={"ticks":[0,1,10,1e2,1e3,1e4,1e5]},
                vmin = 0.001, vmax=10000
    )
    plt.title(title)
    plt.ylabel('Ground truth')
    plt.xlabel('Predicted utterance')

    plt.savefig(f'results/{model_name}/{title}.png')
    plt.show()


   