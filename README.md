# Posterior Label Smoothing for Node Classification
This repository is the official implementation of ["Posterior Label Smoothing for Node Classification"](https://arxiv.org/abs/2406.00410) accepted by AAAI 2026.

## Abstract
Label smoothing is a widely studied regularization technique in machine learning. However, its potential for node classification in graph-structured data, spanning homophilic to heterophilic graphs, remains largely unexplored. We introduce posterior label smoothing, a novel method for transductive node classification that derives soft labels from a posterior distribution conditioned on neighborhood labels. The likelihood and prior distributions are estimated from the global statistics of the graph structure, allowing our approach to adapt naturally to various graph properties. We evaluate our method on 10 benchmark datasets using eight baseline models, demonstrating consistent improvements in classification accuracy. The following analysis demonstrates that soft labels mitigate overfitting during training, leading to better generalization performance, and that pseudo-labeling effectively refines the global label statistics of the graph.


## Environment Setup
To set up the environment, use the configuration specified in the `environment.yml` file. Run the following command:
```
conda env create --file environment.yml 
```

## Training GNNs with PosteL
To train GNNs using the PosteL, use the following command:
```
python training.py --labeling_method=postel --soft_label_ratio=$ALPHA --smoothing_ratio=$BETA --dataset=$DATASET --net=$GNN
```

## Citation
Please cite our paper if you use the model or this code in your own work:
```
@article{heo2024posterior,
  title={Posterior Label Smoothing for Node Classification},
  author={Heo, Jaeseung and Park, Moonjeong and Kim, Dongwoo},
  journal={arXiv preprint arXiv:2406.00410},
  year={2024}
}
```