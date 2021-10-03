# HypMix for Vision

### Requirements
This code has been tested with  
python 3.6.8
torch 1.0.0  
torchvision 0.2.1
### Additioanal packages required
matplotlib==3.0.2  
numpy==1.15.4  
pandas==0.23.4  
Pillow==5.4.1  
scipy==1.1.0  
seaborn==0.9.0  
six==1.12.0  

### Important :Running each of the following commands will automatically create a subdirectory containing the output of that particular expeiment in the manifold_mixup/supervised/experiments directory

### How to run experiments for CIFAR10
```
python main.py --dataset cifar10 --data_dir data/cifar10/ --root_dir experiments/ --labels_per_class 10 --arch preactresnet18  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train mixup_hidden --mixup_alpha 2.0
```

### How to run experiments for CIFAR100
```
python main.py --dataset cifar100 --data_dir data/cifar100/ --root_dir experiments/ --labels_per_class 10 --arch preactresnet18  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train mixup_hidden --mixup_alpha 2.0
```

### Acknowledgement

Repository forked from [Manifold_mixup](https://github.com/vikasverma1077/manifold_mixup).