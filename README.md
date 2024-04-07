# Non-Negative Oja’s Subspace Learning Rule (NN-OSLR)
This repository contains an implementation of manuscript "[Non-Negative Sparse PCA: An Intelligible Exact Approach](https://ieeexplore.ieee.org/document/9305265)".


## Usage
### 1. Requirements
The requirements are in the requirements.txt file. 


### 2. Download Dataset
You can download the dataset from [here](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) and extract them to the folder dataset (see directory structure below).


### 3. Train and Test

To train the algorithm, you can run

```angular2
run_nnsOSLR.py --task_name mnist
```

To test the algorithm, you can run

```angular2
run_classification.py --task_name mnist
```


### Directory structure

```
.
├── code
│   ├── nnsOSLR.py
│   ├── run_classification.py
│   ├── run_nnsOSLR.py
│   └── utils.py
├── dataset
│   └── mnist
│       ├── t10k-images-idx3-ubyte
│       ├── t10k-images-idx3-ubyte.gz
│       ├── t10k-labels-idx1-ubyte
│       ├── t10k-labels-idx1-ubyte.gz
│       ├── train-images-idx3-ubyte
│       ├── train-images-idx3-ubyte.gz
│       ├── train-labels-idx1-ubyte
│       └── train-labels-idx1-ubyte.gz
├── README.md
└── requirements.txt
```

## Reference
If you use this code in your experiments please cite this work by using the following bibtex entry:

```
@ARTICLE{9305265,
  author={I. {Tsingalis} and C. {Kotropoulos} and A. {Drosou} and D. {Tzovaras}},
  journal={IEEE Transactions on Emerging Topics in Computational Intelligence}, 
  title={Non-Negative Sparse PCA: An Intelligible Exact Approach}, 
  year={2020},
  pages={1-13},
  doi={10.1109/TETCI.2020.3042268}}

```
