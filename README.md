[![](https://img.shields.io/badge/license-GPL--3.0-blue)](https://www.gnu.org/licenses/)
[![](https://img.shields.io/badge/Python-3.8-green)](https://www.python.org/)

# Characterizing the Efficiency of Graph Neural Network Frameworks with a Magnifying Glass

<!---This is a Python3 implementation of our benchmark experiments for graph neural network frameworks, as described in our paper.--> 

## Overview

Here we provide the benchmark code used in our paper. 
The repository is organised as follows:
* ```requirements.txt```: All required Python libraries to run the code.
* ```Code/.py```: Our implementation.


## Requirements
<!--OGB=1.3.3--> 

[![](https://img.shields.io/badge/numpy-1.19.5-green)](https://numpy.org/devdocs/index.html)


[CUDA 11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive) has been used.

We use [pyinstrument](https://github.com/joerick/pyinstrument) to measure the runtime of each key function of GNNs and that of each GNN model along with its breakdown results.

We use [CodeCarbon](https://github.com/mlco2/codecarbon) to measure power andenergy consumption.


## Dataset
We use 6 real-world network datasets:
| | # Nodes | # Edges | # Features | # Classes | Train / Val / Test |
| ---------- | :-----------:  | :-----------: | :-----------: | :-----------:  | :-----------: |
|PPI| 14,755 | 225,270 | 50 | 121 | 0.66 / 0.12 / 0.22 |
|Flickr| 89,250 | 899,756 | 500 | 7 | 0.50 / 0.25 / 0.25 |
|ogbn-Arxiv| 169,343 | 1,166,243 | 128 | 40 | 0.54 / 0.29 / 0.17 |
|Reddit| 232,965 | 114,615,892 | 602 | 41 | 0.66 / 0.10 / 0.24 |
|Yelp| 716,847 | 13,954,819 |300 | 100 | 0.75 / 0.10 / 0.15 |
|ogbn-Products| 2,449,029 | 61,859,140 | 100 | 47 | 0.08 / 0.02 / 0.90 |



The datasets are downloaded form:
1. [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT)
2. [DGL](https://www.dgl.ai/)
3. [PyG](https://www.pyg.org/)
4. [OGB](https://ogb.stanford.edu/)


## Running the code
1. Download the data as described above.
2. Run ```Code/.py``` to 
3. Run ```Code/.py``` to 


<!---## Cite--> 

<!---Please cite our paper if you use this code in your own work:--> 
