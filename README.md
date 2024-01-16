# attentive_learning

This repository contains the code for the paper "[Attentive Learning Facilitates Generalization of Neural Networks](https://arxiv.org/abs/2112.03467)" by Shiye Lei, Fengxiang He, Haowen Chen, and Dacheng Tao.

## Dependencies

- Python 3.6
- Pytorch 1.10


## Run

1. setting the data path of TinyImageNet in `main.py`:

   `tinyimagenet_path = [YOUR DATA PATH]`

2. setting the path of result folder in `main.py`:

   `figure_save_path = [YOUR FIGURE PATH]`

3. run `main.py`

   ```bash
   python main.py --dataset [YOUR DATASET NAME]
   ```

   [YOUR DATASET NAME] ∈ {'mnist', 'fashionmnist', 'cifar10', 'cifar100', 'imdb', 'tinyimagenet'}


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
Dacheng Tao: [dacheng.tao@sydney.edu.au](mailto:dacheng.tao@ntu.edu.sg)

---

Last update: Tue 16 Jan 2024
