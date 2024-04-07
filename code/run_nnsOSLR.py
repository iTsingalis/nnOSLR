from sklearn.preprocessing import StandardScaler
from nnsOSLR import NNOSLR
from mnist import MNIST
import numpy as np
import argparse
import time
import os

cur_path = os.path.dirname(__file__)


def main():

    print ('Run nnOSLR.')

    fit_start_time = time.time()

    # # Fit model
    NNOSLR(**alg_settings).fit(Xs.T)

    fit_time = (time.time() - fit_start_time)
    hours, rem = divmod(fit_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # # # Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train")

    FLAGS = parser.parse_args()

    if 'mnist' in FLAGS.task_name:

        datasetName = FLAGS.task_name
        image_shape = (28, 28)
        mndata = MNIST('../dataset/mnist')
        images_tr, labels_tr = mndata.load_training()
        images_tst, labels_tst = mndata.load_testing()
        X = np.asarray(images_tr).astype(np.float64)

        n_components = 16
        alg_settings = {"store_model"       : True,
                        "cache"             : os.path.join(cur_path, 'cache', datasetName),
                        "n_components"      : n_components,
                        "active_tolerance"  : 1e-3,
                        "learning_rate"     : 1e-9,
                        "l2_break_tolerance": 1e-3,
                        "n_epochs"          : 500,
                        "seed"              : 1000,
                        "verbose"           : True}

    else:
        raise ValueError('task_name is not specified.')

    print('Dataset selected {}'.format(datasetName))

    scaler = StandardScaler(with_std=False)
    Xs = scaler.fit_transform(X)

    n_features = X.shape[1]
    n_samples = X.shape[0]

    print('n_components: {}, n_features: {}, n_samples: {}'.format(n_components, n_features, n_samples))

    main()

    print("---------------------")
    print("-------Finish--------")
    print("---------------------")