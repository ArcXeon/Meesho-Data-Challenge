---
license: apache-2.0
---
# PVTv2

This is the Hugging Face PyTorch implementation of the [PVTv2](https://arxiv.org/abs/2106.13797) model.

## Model Description

The Pyramid Vision Transformer v2 (PVTv2) is a powerful, lightweight hierarchical transformer backbone for vision tasks. PVTv2 infuses convolution operations into its transformer layers to infuse properties of CNNs that enable them to learn image data efficiently. This mix transformer architecture requires no added positional embeddings, and produces multi-scale feature maps which are known to be beneficial for dense and fine-grained prediction tasks.

Vision models using PVTv2 for a backbone:
1. [Segformer](https://arxiv.org/abs/2105.15203) for Semantic Segmentation.
2. [GLPN](https://arxiv.org/abs/2201.07436) for Monocular Depth.
3. [Deformable DETR](https://arxiv.org/abs/2010.04159) for 2D Object Detection.
4. [Panoptic Segformer](https://arxiv.org/abs/2109.03814) for Panoptic Segmentation.