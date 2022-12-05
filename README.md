# DSC180 - Graph Neural Network

Author: Barry Xue

This repository contains the materials and codes from the exploration of the topic of Graph Neural Network.

## Options to Run the `run.py`
`run.py` can run node classification task and edge prediction task with GCN Models (Multi-layer GCN for node classification, and GCN-AE for edge prediction).
* Use the `--task` flag to choose between the two tasks.
    - Example: python run.py --name 'expt_test' --dataset 'Cora' --task 'Edge Prediction'

There are also two dataset to run either task upon: Cora, CiteSeer, and PubMed.
* Use the `--dataset` flag to choose between the two datasets.
    - Example: python run.py --name 'expt_test' --dataset 'CiteSeer' --task 'Node Classification'

Other options:
1. `--epochs`: change the number of training epochs.
2. `--hidden_size`: change the first layer hidden layer size.
3. `--encode_size`: change the encoding size/final hidden layer size.
4. `--train`/`--validation`/`--test`: specify number of training sample per classes, validation size, and testing data size.