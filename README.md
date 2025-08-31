# DS-VTON: High-Quality Virtual Try-On via Disentangled Dual-Scale Generation

[📚 Paper](https://arxiv.org/abs/2506.00908) - [🤖 Code](https://github.com/Fu61/DS-VTON)


## Overall
A more effective coarse-to-fine dual-scale diffusion framework for try-on tasks.
<div align="center">
  <img src="assets/teaser.png" width="100%" height="100%"/>
</div>

## Abstract
Despite recent progress, most existing virtual try-on methods still struggle to simultaneously address two core challenges: accurately aligning the garment image with the target human body, and preserving fine-grained garment textures and patterns. In this paper, we propose DS-VTON, a dual-scale virtual try-on framework that explicitly disentangles these objectives for more effective modeling. DS-VTON consists of two stages: the first stage generates a low-resolution try-on result to capture the semantic correspondence between garment and body, where reduced detail facilitates robust structural alignment. The second stage introduces a residual-guided diffusion process that reconstructs high-resolution outputs by refining the residual between the two scales, focusing on texture fidelity. In addition, our method adopts a fully mask-free generation paradigm, eliminating reliance on human parsing maps or segmentation masks. By leveraging the semantic priors embedded in pretrained diffusion models, this design more effectively preserves the person's appearance and geometric consistency. Extensive experiments demonstrate that DS-VTON achieves state-of-the-art performance in both structural alignment and texture preservation across multiple standard virtual try-on benchmarks.

## Method
Overview of our proposed method. The overall pipeline consists of a two-scale generation process: the low-resolution stage produces a coarse try-on result, which is then refined by the high-resolution stage. Both stages share the same network architecture.
<div align="center">
  <img src="assets/DS-VTON.png" width="100%" height="100%"/>
</div>

## Visualization
Qualitative visual results comparison with other methods.
<div align="center">
  <img src="assets/compare.png" width="100%" height="100%"/>
</div>

## Installation
```shell
git clone https://github.com/Fu61/DS-VTON
cd DS-VTON
conda env create -f env.yaml
conda activate ds-vton
pip install -r requirements.txt
```

## Inference
```shell
## low-resolution result
python ./inference/low_res_infer.py --config ./configs/low_res_infer.yaml

## high-resolution result
python ./inference/high_res_infer.py --config ./configs/high_res_infer.yaml
```

## Acknowledgement
We highly reference the network architecture of the [Moore-AnimateAnyone
](https://github.com/MooreThreads/Moore-AnimateAnyone) project.

## Citation
If you find our work helpful or inspiring, please feel free to cite it.
```
@misc{sun2025dsvtonhighqualityvirtualtryon,
      title={DS-VTON: High-Quality Virtual Try-on via Disentangled Dual-Scale Generation}, 
      author={Xianbing Sun and Yan Hong and Jiahui Zhan and Jun Lan and Huijia Zhu and Weiqiang Wang and Liqing Zhang and Jianfu Zhang},
      year={2025},
      eprint={2506.00908},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.00908}, 
}
```
