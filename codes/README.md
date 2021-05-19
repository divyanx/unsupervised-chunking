# Team 37 - NLP Project (Unsupervised Chunking)

## Structure

```
.
├── implementation - Chunking Algos
│  ├── hierarchical
│  │  ├── Agglomerative.py
│  │  └── Preprocess.py
│  └── kMeans.py
├── original - Clustering of numbers
│  ├── hierarchical
│  │  ├── Agglomerative.py
│  │  └── Preprocess.py
│  └── kMeans.py
├── README.md
└── requirements.txt

```
   
   
## Running

Firstly install the dependencies using:

```sh
$ pip install -r requirements.txt
```

### Original

The original folder contains the clustering for simple numbers and the data is hardcoded in them

These can be run in the following ways:

#### K-Means

```sh
$ python3 kMeans.py
```

#### Hierarchical

```sh
$ python3 Preprocess.py # This will generate distance matrix
$ python3 Agglomerative.py
```

### Implementations

#### K-Means

```sh
$ python3 kMeans.py <path_to_data_file> <path_to_output_file>
```

#### Hierarchical

```sh
$ python3 Preprocess.py <path_to_data_file> # This will generate distance matrix
$ python3 Agglomerative.py
```

Currently, Hierarchical is just training a model, and evaluating through the model is not implementated. It displays a graph at the end of execution.

## Hyperparameters

Both the models have a hyperparameter which is the number of lines to read from the data

The kMeans.py implementation has another hyperparameter called scaling_factor which should be kept > 1
