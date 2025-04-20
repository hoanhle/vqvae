## PyTorch VQVAEs

# PyTorch VQVAEs

This repository contains PyTorch implementations of various Vector Quantized Variational Autoencoders (VQVAEs).

I created this project to explore and better understand VQVAEs through hands-on implementation. My journey began with curiosity about the NeurIPS 2024 best paper: [Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://arxiv.org/abs/2404.02905), which inspired me to dive deeper into the underlying VQVAE architectures.

![vqvae](images/vqvae.png)

## Table of Contents
  * [What are VQVAEs?](#what-are-vqvaes)
  * [Implementations](#implementations)
    + [VQ-VAE](#vq-vae)
    + [VQ-VAE-2](#vq-vae-2)
    + [VQ-GAN](#vq-gan)
    + [RVQ-VAE](#rvq-vae)
    + [DALL-E](#dall-e)
    + [MaskGIT](#maskgit)
    + [Stable Diffusion](#stable-diffusion)


## Implementations

## VQ-VAE

Implementation based on the [VQ-VAE paper](https://arxiv.org/abs/1711.00937).


## Citations

```bibtex
@misc{oord2018neural,
    title   = {Neural Discrete Representation Learning},
    author  = {Aaron van den Oord and Oriol Vinyals and Koray Kavukcuoglu},
    year    = {2018},
    eprint  = {1711.00937},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```