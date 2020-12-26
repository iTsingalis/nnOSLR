import sys
sys.path.append('../../..')
from scipy.stats import halfnorm
from tabulate import tabulate
from utils import *
import numpy as np
import datetime
import logging
import pickle
import json
import time
import os

class FoundViolation(Exception): pass
class EarlyStop(Exception): pass

class NNOSLR(object):
    def __init__(self,
                 n_components       =   1,
                 learning_rate      =   1e-5,
                 l2_break_tolerance =   1e-3,
                 active_tolerance   =   1e-4,
                 seed               =   1000,
                 n_epochs           =   500,
                 verbose            =   False,
                 store_model        =   False,
                 cache              =   './cache'):

        self.seed                       =   seed
        self.cache                      =   cache
        self.active_tolerance           =   active_tolerance
        self.l2_break_tolerance         =   l2_break_tolerance
        self.store_model                =   store_model
        self.n_components               =   n_components
        self.n_epochs                   =   n_epochs
        self.verbose                    =   verbose
        self.learning_rate              =   learning_rate

    def __halfNorm(self):

        n_components=1 # One component at a time.
        random_state = np.random.RandomState(seed=self.seed)
        fan_in, fan_out = (self.n_features, n_components)
        variance = 2. / (fan_in + fan_out)
        std = 1 * np.sqrt(variance)

        W = halfnorm.rvs(loc=0, scale=std, size=(self.n_features, n_components), random_state=random_state)

        return W

    def __check_possitiveness(self):

        poss_bool = self._W > -self.active_tolerance
        if not np.all(poss_bool):
            raise FoundViolation(
                'Violation Found Stop (non-Negativity) : {0} - Epochs: {1} '.format(self.l2_w_error,self.running_epochs))

    def __check_convergence(self):

        if (self.l2_w_error <= self.l2_break_tolerance) or self.running_epochs > self.n_epochs:
            self.__check_possitiveness()
            raise EarlyStop('Early Stop (l2 break): {0} - Epochs: {1}'.format(self.l2_w_error, self.running_epochs))

    def fit(self, X):

        fit_start_time = time.time()
        self.n_features, self.n_samples = X.shape[0], X.shape[1]

        precomputed_n_components = 0
        self._components = np.empty((self.n_features, 0), dtype=np.float)

        self._deflation_iter = 1
        for i in range(self.n_components):
            try:
                self.active_idx = np.zeros((self.n_features, 1), dtype=bool)
                self._W = self.__halfNorm()
                self.running_epochs = 1
                self.n_Iteration = 1

                while True: # Epoch loop
                    try:
                        np.random.shuffle(np.transpose(X))
                        for index_sample in range(self.n_samples):
                            sample = X[:, index_sample][:, None]

                            self.partial_fit(sample)

                            self.__check_convergence()

                            self.n_Iteration = self.n_Iteration + 1

                        if self.verbose:
                            self.__printData()

                        self.__check_possitiveness()
                        self.running_epochs = self.running_epochs + 1
                    except EarlyStop as e:
                        logging.warning(e)
                        self.__printData()
                        time.sleep(2)
                        break
            except FoundViolation as e:
                logging.error(e)
                self.__printData()
                break

            self._components = np.hstack((self._components, self._W))

            # Projection Deflation
            X = self.__projectionDeflation(X)

            self._deflation_iter = self._deflation_iter + 1

            if self.verbose:
                self.__printData()

            if self.store_model:
                self.__store_model()

        return self

    def partial_fit(self, X):

        # Define value close to zero
        close_to_zero = 0.0

        self.grad_Y = self.transform(X) * X

        # Check values close to zero (active_tolerance)
        active_idx_local = (self._W) <= self.active_tolerance

        # Keep history of active possitions. Once active forever active!!
        self.active_idx[active_idx_local] = True

        # Active constraint
        self._W[self.active_idx == True] = close_to_zero
        self.grad_Y[self.active_idx == True] = close_to_zero

        inactive_s = np.sum(np.dot(self.grad_Y.T, self._W))

        # Lagrangian gradient
        grad_Q = self.grad_Y - self._W * inactive_s

        self._W += self.learning_rate * grad_Q
        self.l2_w_error = abs(np.linalg.norm(self._W) - 1.)

        return self

    def __store_model(self):

        if not os.path.exists(self.cache):
            os.makedirs(self.cache)

        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
        md = {"components" :self._components,
              "running_epochs": self.running_epochs,
              "n_Iteration": self.n_Iteration, "learning_rate": self.learning_rate}

        file_path_model = os.path.join(self.cache, 'nnsOSLR_{0}_c_{1}_to_{2}'.format(st, 0, self._components.shape[1]))
        with open(file_path_model + '.pickle', 'wb') as handle:
            pickle.dump(md, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __printData(self):

        poss_check = np.where(self._W < -self.active_tolerance)[0]
        negative_percentage = float(len(poss_check)) / len(self._W)
        print(tabulate([['nnOSLR',
                         "{0}/{1}".format(self._deflation_iter, self.n_components),
                         "{:.30f}".format(self.learning_rate),
                         "{0}/{1}".format(self.running_epochs, self.n_Iteration),
                         "{:.5f}".format(np.linalg.norm(self._W)),
                         "{0}".format(np.sum(self.active_idx)),
                         "{0}".format(np.sum(self.n_features)),
                         "{0}".format(float(np.sum(self.active_idx)) / self.n_features),
                         "{0:.5f}({1}/{2})".format(float(negative_percentage), len(poss_check), len(self._W)),
                         ]],
                       headers=['Alg.','n_comp', 'learning-rate', 'Epoch/Iter', 'l2_w', 'n_active', 'n_feature',
                                'sparsity(%)', 'negative W(%)'],
                       tablefmt='orgtbl')+'\n')

    def get_params(self):

        return deepcopy(self._components)

    def __projectionDeflation(self, X):

        X = np.dot(X.T, np.eye(self.n_features) - np.dot(self._W, self._W.T)).T
        return deepcopy(X)

    def transform(self, X):

        Y = np.dot(self._W.T, X)
        return deepcopy(Y)
