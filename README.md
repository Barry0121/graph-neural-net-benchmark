# DSC180 - Graph Neural Network

Author: Barry Xue. Jianming Geng, Winston Yu

This repository contains training code for "Investigating New Adversarial Attack on Deep Graph Neural Networks: Creating Perturbation from Latent Space" report and the materials and codes from the exploration of the topic of Graph Neural Network.

# Option 1: Train WGAN network with GraphRNN, Discriminator, and Inverter with `main.py`
`main.py` will call the training function, and then it will call the visualization function to generate the graphs.
* All user defined parameters can be modified in `src/models/args.py` file. 

# Option 2: Test GCN and GCN-AE with `run.py`
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

## Note on installing `pytorch_geometric`
* Normally, the command: `conda install -c pyg pyg` will work on MacOS, Windows, and any Linux distro with anaconda/miniconda installed.
* There are two known cases where this wouldn't work:
1. If the pytorch isn't installed, or the installed version in the environment is <1.12.x, y9ou might need to look into alternative installation method (ex: building from source, use pip, etc).
2. The newest MacOS Ventura (v13.0.1) has installation issue, due to the lack of support for M1 Macbooks (pytorch scatter doesn't support 'mps' device yet). One way to get around this issue is to follow this post: https://github.com/rusty1s/pytorch_scatter/issues/241.
