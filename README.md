# Setup steps
```bash
# we use python 3.10
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

# Project Overview

This project implements an end-to-end pipeline for raw image denoising using a deep learning approach that combines the Anscombe transform with the NAFNet architecture. The system addresses the challenge of Poisson-Gaussian noise removal in raw image data while preserving image quality during the demosaicing process.
Key Components
1. Dataset

    Source: RGB2RAW Dataset on Hugging Face

    Content: RAW sensor data paired with corresponding sRGB images and camera metadata

    Characteristics: Multi-camera data with varying sensor profiles and ISP pipelines

2. Architecture

    Backbone: Modified NAFNet (Non-linear Activation Free Network)

    Reference: NAFNet: Nonlinear Activation Free Network for Image Restoration

    Adaptation: Architecture tailored for raw image processing and demosaicing tasks

3. Noise Augmentation

    Noise Model: Poisson-Gaussian noise simulation applied directly to RAW data

    Purpose: Realistic noise modeling that accounts for both shot noise (Poisson) and read noise (Gaussian)

    Implementation: Parameterized noise injection during training

4. Core Innovation: Anscombe Transform

    Transform: Application of the Anscombe variance-stabilizing transform

    Mathematical Basis: Freeman-Tukey transformation generalized for Poisson-Gaussian distributions

    Benefits: Converts signal-dependent noise to approximately Gaussian noise with constant variance

    Reference: Related work on noise stabilization

5. Training Pipeline

    Input: RAW sensor data with camera metadata

    Preprocessing: Anscombe transform for noise variance stabilization

    Processing: Modified NAFNet for joint denoising and demosaicing

    Output: sRGB image reconstruction

    Comparison: Evaluation against ground truth sRGB images

    Loss Function: Combined L1 + L2 loss optimization

### Technical Approach

The pipeline addresses the fundamental challenge of raw image processing where noise characteristics are signal-dependent. By applying the Anscombe transform before neural network processing, we convert Poisson-Gaussian noise into approximately Gaussian noise with unit variance, simplifying the denoising problem. The modified NAFNet architecture then performs simultaneous denoising and demosaicing in this transformed space, with an inverse transform applied to obtain the final sRGB output.
Key Features

    End-to-end raw image processing from sensor data to sRGB

    Signal-aware noise handling through Anscombe transformation

    Joint denoising and demosaicing in a single network

    Camera-agnostic processing leveraging metadata information

    Realistic noise simulation for robust training

### Training Configuration

    Loss Function: Combined L1 (MAE) and L2 (MSE) losses

    Noise Augmentation: Parameterized Poisson-Gaussian noise injection

    Evaluation: PSNR, SSIM metrics on sRGB outputs

    Optimization: Adam optimizer with standard training protocols

### Applications

    Mobile computational photography

    Low-light image enhancement

    Scientific imaging pipelines

    Camera ISP improvement

### References

    NAFNet: Nonlinear Activation Free Network for Image Restoration

    Anscombe Transform and variance stabilization techniques

    Poisson-Gaussian noise modeling in computational photography

    Deep learning for raw image processing