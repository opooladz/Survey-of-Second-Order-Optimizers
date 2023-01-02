# Code for NeurIPS 2022 Higher Order Optimization: Improving Levenberg-Marquardt Algorithm for Neural Networks

Paper can be found here  https://arxiv.org/abs/2212.08769
### Dataset

indicate in the argparser `--dataset`

* mnist (from pytorch, 28 x 28 resolution, 60000 samples)
* mnist_small (from sklearn, 8 x 8 resolution, flattened to a 64-vector, ~2000 samples)
* regression 

### Network Architecture

indicate in the argparser `--net_type`

* MLP (fully connected layers and ReLU only) 
* CNN (LeNet-like CNN with Conv2d, Maxpool)

### Optimizer

indicate in the argparser `--optimizer`

* Adam
* SGD
* LM
* HF 
* KFAC
* EKFAC
* KFAC-Adam
* EKFAC-Adam

### run the experiment

```python

python main.py --dataset 'mnist' --net_tye 'cnn' --optimizer 'SGD'

```

the training loss and testing accuracy of **each iteration** will be saved as `.npy` file into the `loss_acc_timing` folder after the training. load them and plot in the `plot.ipynb`


# If Used Please Cite 
```
@article{pooladzandi2022improving,
  title={Improving Levenberg-Marquardt Algorithm for Neural Networks},
  author={Pooladzandi, Omead and Zhou, Yiming},
  journal={arXiv preprint arXiv:2212.08769},
  year={2022}
}
```
