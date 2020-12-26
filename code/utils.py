import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import pickle
import os

def get_loadings(file_name, n_components):

    file_path = os.path.join('cache', 'mnist', file_name)

    with open(file_path, "rb") as input_model:
        precomputed_components = pickle.load(input_model)

    precomputed_components = precomputed_components["components"][:, :n_components]

    print('get_nnsOSLR_loadings: {}'.format(precomputed_components.shape[1]))

    return precomputed_components

def plot_patches(evecs_sparse, alf_name, image_shape, nCols=10, nRows=10):

    order = 'C'

    heights = [image_shape[0]] * nRows
    widths = [image_shape[1]] * nCols

    fig_width = 8.
    fig_height = fig_width * sum(heights) / sum(widths)

    fig, axarr = plt.subplots(nRows, nCols, figsize=(fig_width, fig_height), gridspec_kw={'height_ratios': heights})
    fig.canvas.set_window_title(alf_name)
    cnt=0
    for i in range(nRows):
        for j in range(nCols):

            img = evecs_sparse[:, cnt]
            axarr[i, j].imshow(img.reshape(image_shape, order=order), cmap=plt.cm.gray)
            axarr[i, j].axis('off')
            cnt=cnt+1

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()
