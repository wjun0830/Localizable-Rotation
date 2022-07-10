### Dependency
The code is built with following libraries:
- [PyTorch](https://pytorch.org/) 1.2
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [scikit-learn](https://scikit-learn.org/stable/)

### Dataset
- Imbalanced [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html). The original data will be downloaded and converted by `imbalancec_cifar.py`.

### Training

- To train the baseline+LoRot on long-tailed imbalance with ratio of 100

```bash
python cifar_train_lorot-E.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None
python cifar_train_lorot-I.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None
```

- To train the LDAM-DRW+LoRot training on long-tailed imbalance with ratio of 100

```bash
python cifar_train_lorot-E.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type LDAM --train_rule DRW
python cifar_train_lorot-I.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type LDAM --train_rule DRW
```

### Acknowledgement

Implementations for Imbalanced Classification of LoRot is based on [LDAM-DRW](https://github.com/kaidic/LDAM-DRW)
