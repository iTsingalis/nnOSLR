from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from utils import get_loadings, plot_patches
from sklearn.metrics import roc_curve, auc
from numpy.random import RandomState
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mnist import MNIST
import numpy as np
import argparse

def log_class(Xs_tr_pca, Xs_tst_pca, lb_tr, lb_tst):

    param_grid = {"estimator__C": np.logspace(-3, 3, 6)}
    logistic = LogisticRegression(solver='lbfgs', max_iter=10000, n_jobs=3)
    classifier = OneVsRestClassifier(logistic)
    search = GridSearchCV(classifier, param_grid=param_grid, cv=5, return_train_score=False, n_jobs=-1, verbose=1)
    search.fit(Xs_tr_pca, lb_tr)

    y_score = search.predict_proba(Xs_tst_pca)

    y_pred = search.best_estimator_.predict(Xs_tst_pca)

    acc = np.mean(y_pred == lb_tst)
    print('Acc. {0}'.format(acc))

    return y_score


def main():

    n_comp = 6
    method = "nnOSLR"
    plot_components = True

    file_name = 'nnsOSLR_2020-12-26_16-33-18_c_0_to_6.pickle'
    W = get_loadings(file_name=file_name, n_components=n_comp)

    if plot_components:
        plot_patches(W, 'nnOSLR', image_shape, nCols=3, nRows=2)

    # Embed data
    X_tr_pca = np.dot(X_tr, W)
    X_tst_pca = np.dot(X_tst, W)

    # Classify data
    score_tst = log_class(X_tr_pca, X_tst_pca, labels_tr, labels_tst)

    # Plot performance data
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.set_window_title("Classification comp {}".format(n_comp))
    ax.set_aspect('auto')

    # Compute ROC curve and ROC area for each class
    fpr, tpr, roc_auc = dict(), dict(), dict()

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels_tst.ravel(), score_tst.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve
    ax.plot(fpr["micro"],
            tpr["micro"],
            marker      =  None,
            linestyle   =   '-',
            color       =   'b',
            markersize  =   1.5,
            label       =   '{0} (J: {1}, Area: {2:0.2f})'.format(method, n_comp, roc_auc["micro"]))

    ax.plot([0, 1], [0, 1], linestyle=(0, (5, 10)), color='k')

    ax.set_xlabel('False Positive Rate', fontsize=18)
    ax.set_ylabel('True Positive Rate', fontsize=18)
    lgd = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2,
                    mode="expand", borderaxespad=0., fontsize=13)
    plt.grid(True)

    plt.show(block=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # #  Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train")

    FLAGS = parser.parse_args()

    if 'mnist' in FLAGS.task_name:

        DATASET = 'mnist'
        image_shape = (28, 28)
        mndata = MNIST('../dataset/mnist')

        images_tr, labels_tr = mndata.load_training()
        images_tst, labels_tst = mndata.load_testing()

        X_tr = np.asarray(images_tr).astype(np.float32)
        labels_tr = np.asarray(labels_tr).astype(np.int32)

        X_tst = np.asarray(images_tst).astype(np.float32)
        labels_tst = np.asarray(labels_tst).astype(np.int32)

        np.random.seed(int(9001))
        idx = np.random.choice(len(images_tr), 10000, replace=False)

        X_tr = X_tr[idx, :]
        labels_tr = labels_tr[idx]

    else:
        raise ValueError('task_name is not specified.')

    print('Center Data...')
    scaler = StandardScaler(with_std=False)

    X_tr = scaler.fit_transform(X_tr)
    X_tst = scaler.transform(X_tst)

    # Binarize the output for multiclass roc evaluation
    classes = np.unique(labels_tr)
    label_binarize = preprocessing.LabelBinarizer()
    labels_tr = label_binarize.fit_transform(labels_tr)
    labels_tst = label_binarize.transform(labels_tst)

    n_classes = labels_tr.shape[1]

    main()