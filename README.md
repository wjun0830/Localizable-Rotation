# Tailoring Self-Supervision for Supervised Learning
[[Arxiv](https://arxiv.org/abs/2207.10023)] | [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850342.pdf)]
[]()

Official PyTorch Repository of "Tailoring Self-Supervision for Supervised Learning" (ECCV 2022 Paper)
<p align="center">
    <img src=figures/LoRot-I.png width="45%"> 
    <img src=figures/LoRot-E.png width="45%"> 
</p>

### ImageNet Pretrained Weight
[link](https://www.dropbox.com/scl/fo/buex1pu63rhsw9lccjap2/h?dl=0&rlkey=77s9a8gzbyx2azlyktllf98fg)
 Method | Top-1 Error (best) | Top-5 Error (best) | Model file
 -- | -- | -- | --
 ResNet-50 | 23.68 | 7.05 | [model](https://www.dropbox.com/s/2sn5t0rt3wndshv/vanilla.pth.tar?dl=0)
 ResNet-50 + LoRot-I| 22.29 | 6.40 | [model](https://www.dropbox.com/s/xgysvba08weabeh/LoRot-I.pth.tar?dl=0)
 ResNet-50 + LoRot-E | 22.28 | 6.34 | [model](https://www.dropbox.com/s/629fco9inypgx4x/LoRot-E.pth.tar?dl=0)

##  Cite LoRot (Tailoring Self-Supervision for Supervised Learning)

If you find this repository useful, please use the following entry for citation.
```
@inproceedings{moon2022tailoring,
  title={Tailoring Self-Supervision for Supervised Learning},
  author={Moon, WonJun and Kim, Ji-Hwan and Heo, Jae-Pil},
  booktitle={European Conference on Computer Vision},
  year={2022},
  organization={Springer}
}
```

## Contributors and Contact

If there are any questions, feel free to contact with the authors: WonJun Moon (wjun0830@gmail.com), Ji-Hwan Kim (damien911224@gmail.com).

