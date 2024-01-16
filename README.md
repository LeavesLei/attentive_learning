# attentive_learning

This repository contains the code for the paper "Attentive Learning Facilitates Generalization of Neural Networks" by Shiye Lei, Fengxiang He, Haowen Chen, and Dacheng Tao.

## Dependencies

- Python 3.6
- Pytorch 1.10


## Run

### Dataset-distraction Stability and Generalization (Fig.3)

##### 1. Model training

   
[DATASET] ∈ {'cifar10', 'cifar100'},

[NET] ∈ {'resnet', 'vgg', 'wrn', 'vit'}, 

[OPTIMIZER] ∈ {'adam',  'rmsprop',  'sgd',  'sgdm'}


For `INDEX=[1-10]`, `RATIO=[0.1,0.2,...,1.0]`, run
   ```bash
   python train.py --dataset [DATASET] --net_type [NET]  --trainset_mode former --optimizer_name [OPIMIZER] --former_sample_ratio RATIO --index INDEX
   python train.py --dataset [DATASET] --net_type [NET]  --trainset_mode latter --optimizer_name [OPIMIZER] --latter_sample_ratio RATIO --index INDEX
   ```

##### 2. Evaluate the dataset-distraction stability

```bash
python test_single.py --dataset [DATASET] --net_type [NET]  --optimizer_name [OPIMIZER] 
```

### Distraction stability and Source Sample Size (Fig. 4)


```bash
python test_locally_and_sample_size.py --dataset [DATASET] --net_type [NET]
```

### Distraction stability and label noise (Fig. 5)

##### 1. Model training

[DATASET] ∈ {'cifar10', 'cifar100'},

[NET] ∈ {'resnet', 'vgg', 'wrn'}, 



For `INDEX=[1-10]`, `NOISE_RATIO=[0.05, 0.1, 0.15, 0.2, 0.25]`, run
   ```bash
python train_label_noise.py --dataset [DATASET] --net_type [NET]  --noise_set former --label_noise_ratio NOISE RATIO --index INDEX
python train_label_noise.py --dataset [DATASET] --net_type [NET]  --noise_set latter --label_noise_ratio NOISE RATIO --index INDEX
```

##### 2. Evaluate the dataset-distraction stability

```bash
python test_label_noise.py --dataset [DATASET]
```

### Distraction stability and similarity (Fig. 6)

##### 1. Model training

[DATASET] ∈ {'cifar10', 'cifar100'},

[NET] ∈ {'resnet', 'vgg', 'wrn'}, 

[MODE] ∈ {'green', 'gray', 'red'}, 



For `INDEX=[1-10]`, `ALPHA=[0.1,0.2,...,0.9]`, run
   ```bash
python train_dist_shift.py --dataset [DATASET] --net_type [NET]  --mode [MODE] --alpha ALPHA --index INDEX
python train_dist_shift.py --dataset [DATASET] --net_type [NET]  --mode full --index INDEX
python train_dist_shift.py --dataset [DATASET] --net_type [NET]  --mode sub --index INDEX
```

##### 2. Evaluate the dataset-distraction stability
```bash
python test_dist_shift.py --dataset [DATASET] --net_type [NET]
```

## Citation
```
@article{lei2024attentive,
  title={Attentive Learning Facilitates Generalization of Neural Networks},
  author={Lei, Shiye and He, Fengxiang and Chen, Haowen and Tao, Dacheng},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  year={2024}
}
```

## Contact

For any issue, please kindly contact

Shiye Lei: [leishiye@gmail.com](mailto:leishiye@gmail.com)  
Fengxiang He: [F.He@ed.ac.uk](mailto:F.He@ed.ac.uk)  
Hoawen Chen: [haowchen@student.ethz.ch](mailto:haowchen@student.ethz.ch)  
Dacheng Tao: [dacheng.tao@ntu.edu.sgu](mailto:dacheng.tao@ntu.edu.sg)

---

Last update: Tue 16 Jan 2024
