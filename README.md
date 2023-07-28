# Tailoring Self-Supervision for Supervised Learning
[[Arxiv](https://arxiv.org/abs/2207.10023)] | [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850342.pdf)] |
[[Video](https://www.youtube.com/watch?v=H4fX0KQfp2s&t=136s&ab_channel=WonJunMoon)]

Official PyTorch Repository of "Tailoring Self-Supervision for Supervised Learning" (ECCV 2022 Paper)
<p align="center">
    <img src=figures/LoRot-I.png width="45%"> 
    <img src=figures/LoRot-E.png width="45%"> 
</p>

**Abstract.**
Recently, it is shown that deploying a proper self-supervision is a prospective way to enhance the performance of supervised learning. Yet, the benefits of self-supervision are not fully exploited as previous pretext tasks are specialized for unsupervised representation learning. To this end, we begin by presenting three desirable properties for such auxiliary tasks to assist the supervised objective. First, the tasks need to guide the model to learn rich features. Second, the transformations involved in the self-supervision should not significantly alter the training distribution. Third, the tasks are preferred to be light and generic for high applicability to prior arts. Subsequently, to show how existing pretext tasks can fulfill these and be tailored for supervised learning, we propose a simple auxiliary self-supervision task, predicting localizable rotation (LoRot). Our exhaustive experiments validate the merits of LoRot as a pretext task tailored for supervised learning in terms of robustness and generalization capability.


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

