[![](https://img.shields.io/badge/license-GPL--3.0-blue)](https://www.gnu.org/licenses/)
[![](https://img.shields.io/badge/Python-3.8-green)](https://www.python.org/)

# Characterizing the Efficiency of Graph Neural Network Frameworks with a Magnifying Glass

<!---This is a Python3 implementation of our benchmark experiments for graph neural network frameworks, as described in our paper.--> 

## Overview

This repository contains the benchmark code used in our paper. 


## Requirements

<!--PyTorch v1.11.0--> 
<!--OGB=1.3.3--> 
<!--DGL=0.8.2--> 
<!--PyG=2.0.4--> 
<!--CUDA=11.3--> 

[![](https://img.shields.io/badge/PyTorch-1.11.0-blueviolet)](https://pytorch.org/get-started/previous-versions/)
[![](https://img.shields.io/badge/OGB-1.3.3-orange)](https://ogb.stanford.edu/docs/home/)
[![](https://img.shields.io/badge/DGL-0.8.2-blue)](https://www.dgl.ai/pages/start.html)
[![](https://img.shields.io/badge/PyG-2.0.4-yellow)](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
[![](https://img.shields.io/badge/CUDA-11.3-green)](https://developer.nvidia.com/cuda-11.3.0-download-archive)

We use [pyinstrument](https://github.com/joerick/pyinstrument) to profile the code. We use [CodeCarbon](https://github.com/mlco2/codecarbon) to measure power/energy consumption.


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


The datasets can be downloaded from [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT), [DGL](https://www.dgl.ai/), [PyG](https://www.pyg.org/), and [OGB](https://ogb.stanford.edu/).


## Running the code
1. ```code/Functional``` includes implementations for functional tests: data loader, sampler, and graph convolutional layers.
2. ```code/GraphSAGE``` includes implementations of GraphSAGE.
3. ```code/ClusterGCN``` includes implementations of ClusterGCN. 
4. ```code/GraphSAINT``` includes implementations of GraphSAINT. 

```bash
$ python3 -m pyinstrument -o [**].html \--html [**].py
```
Replace [\*\*] by file name and run the command will profile the code and save the result in an html file. See an example below.
![alt text](https://github.com/xhuang2016/GNN-Benchmark/blob/main/pyinstrument.png?raw=true)

Please refer to our paper for more details.


<!---## Cite--> 

<!---Please cite our paper if you use this code in your own work:--> 
