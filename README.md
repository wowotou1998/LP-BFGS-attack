# LP-BFGS attack: An adversarial attack based on Hessian with limited pixels

Official implementation for

- LP-BFGS attack: An adversarial attack based on Hessian with limited pixels.

For any questions, contact (zhangjiebao2014@mail.ynu.edu.cn).

## Requirements

1. [Python](https://www.python.org/)
2. [Pytorch](https://pytorch.org/)
3. [Torattacks >= 3.2.6](https://github.com/Harry24k/adversarial-attacks-pytorch)
4. [Torchvision](https://pytorch.org/vision/stable/index.html)
5. [Pytorchcv](https://github.com/osmr/imgclsmob)
6. [pytorch-minimize](https://github.com/rfeinman/pytorch-minimize)
7. [captum](https://github.com/pytorch/captum), IntegratedGradient implementation by Pytorch

## Preparations
- some datasets will be downloaded.


## Attack

```
python attack_models.py
```



## Citation

If you find this repo useful for your research, please consider citing the paper

```
@misc{lp-bfgs-attack,
  doi = {10.48550/arXiv.2210.15446},
  url = {https://arxiv.org/abs/2210.15446},
  author = {Zhang, Jiebao and Qian, Wenhua and Nie, Rencan and Cao, Jinde and Xu, Dan},
  title = {LP-BFGS attack: An adversarial attack based on the Hessian with limited pixels},
  publisher = {arXiv},
  year = {2022},
}
```
