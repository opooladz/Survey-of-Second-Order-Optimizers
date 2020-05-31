# Term Project for ECE236C: Comparing 2nd-Order Optimizers

### Dataset

indicate in the argparser `--dataset`

* mnist (from pytorch, 28 x 28 resolution, 60000 samples)
* mnist_small (from sklearn, 8 x 8 resolution, flattened to a 64-vector, ~2000 samples)
* regression (**to be added**)

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
