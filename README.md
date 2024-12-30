<h1 align="center"> <p>Dynamic-Tuning</p></h1>



<p align="center">
  <picture>
    <img width="20%" alt="Dynamic-Tuning" src="./logo.png">
  </picture>
</p>


The official implementation of "2024NeurIPS Dynamic Tuning Towards Parameter and Inference Efficiency for ViT Adaptation".

> Wangbo Zhao<sup>1</sup>, Jiasheng Tang<sup>2,3</sup>,  Yizeng Han<sup>4</sup>, Yibing Song<sup>2,3</sup>, Kai Wang<sup>1</sup>, Gao Huang<sup>4</sup>, Fan Wang<sup>2</sup>, Yang You<sup>1</sup>
>
> <sup>1</sup>[National University of Singapore](https://www.nus.edu.sg/), <sup>2</sup>[DAMO Academy, Alibaba Group](https://damo.alibaba.com/?language=zh), <sup>3</sup>Hupan Lab, <sup>4</sup>[Tsinghua University](https://www.tsinghua.edu.cn/)
>
>  [Paper](https://arxiv.org/abs/2403.11808)


## News 🚀🚀🚀
- `2024.10.16`: We update the code: add a distillation technique (our paper in NeurIPS 2024 verision), support actually efficient inference, support semantic segmentation. Our paper in NeurIPS 2024 verision will be released soon.
- `2024.09.26`: DyT is accepted by NeurIPS 2024. We will update the code and paper soon.
- `2024.03.23`: The code is released.

## Abstract
Existing parameter-efficient fine-tuning (PEFT) methods have achieved significant success on vision transformers (ViTs) adaptation by improving parameter efficiency. However, the exploration of enhancing inference efficiency during adaptation remains underexplored. This limits the broader application of pre-trained ViT models, especially when the model is computationally extensive. In this paper, we propose Dynamic Tuning (DyT), a novel approach to improve both parameter and inference efficiency for ViT adaptation. Specifically, besides using the lightweight adapter modules, we propose a token dispatcher to distinguish informative tokens from less important ones, allowing the latter to dynamically skip the original block, thereby reducing the redundant computation during inference. Additionally, we explore multiple design variants to find the best practice of DyT. Finally, inspired by the mixture-of-experts (MoE) mechanism, we introduce an enhanced adapter to further boost the adaptation performance. We validate DyT across various tasks, including image/video recognition and semantic segmentation. For instance, DyT achieves comparable or even superior performance compared to existing PEFT methods while evoking only 71%-85% of their FLOPs on the VTAB-1K benchmark.
<p align="center">
<img src="https://github.com/NUS-HPC-AI-Lab/Dynamic-Tuning/assets/56866854/b957598b-1e22-438d-9fe0-4b1317501c61" width=80% height=45%
class="center">

## 🛠 Dataset Prepare
- For VTAB-1K, we recommend to adopt the split provided by [SSF](https://github.com/dongzelian/SSF). You can directly download the VTAB-1K from their repo.
- For other image datasets, they will be automatically downloaded when you first run our code.
- For video datasets (K400 and SSv2), you can download them from [OpenDataLab](https://opendatalab.org.cn/OpenMMLab/Kinetics-400) or their offical websites.

## 🛠 Installation
```
pip install -r requirements.txt # install torch, timm, torchvision, etc.
wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth # download the ckpt from timm
```

## ⚙️ Fine-tuning
```
bash ./train_IN21K.sh  # training on complete datasets
bash ./train_vtab.sh # training on vtab benchmark
bash ./train_video.sh # training on video datasets
```

## ⚙️ Measure Inference Speed
```
bash ./measure_speed.sh
```

## Citation
If you found our work useful, please consider citing us.
```
@article{zhao2024dynamic,
  title={Dynamic tuning towards parameter and inference efficiency for vit adaptation},
  author={Zhao, Wangbo and Tang, Jiasheng and Han, Yizeng and Song, Yibing and Wang, Kai and Huang, Gao and Wang, Fan and You, Yang},
  journal={arXiv preprint arXiv:2403.11808},
  year={2024}
}
```


## Acknowledge
The repo is partly built based on [AdaptFormer](https://github.com/ShoufaChen/AdaptFormer), [AdViT](https://github.com/MengLcool/AdaViT), and [PETL-ViT](https://github.com/JieShibo/PETL-ViT). We are grateful for their generous contribution to open source.


## Contact
🔥🔥🔥 If you are interested in this work and hope to cooperate with us, please drop an email to wangbo.zhao96@gmail.com 🔥🔥🔥
