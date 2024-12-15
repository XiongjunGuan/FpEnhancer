<!--
 * @Description: 
 * @Author: Xiongjun Guan
 * @Date: 2024-12-09 15:39:59
 * @version: 0.0.1
 * @LastEditors: Xiongjun Guan
 * @LastEditTime: 2024-12-15 11:53:08
 * 
 * Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
-->
# FpEnhancer
This is a simple baseline for fingerprint enhancement.

## Introduction
This is a `UNet` structured network for fingerprint enhancement. Input a fingerprint image, and the network will output the enhancement result in binary image format. Because a fully convolutional structure is used, there is no requirement for input size.


The basic block  comes from 
> Chen L, Chu X, Zhang X, et al. Simple baselines for image restoration[C]//European conference on computer vision. Cham: Springer Nature Switzerland, 2022: 17-33.

The overall flowchart of our proposed algorithm is shown as follows.
<br>
<p align="center">
    <img src="./images/flowchart_Unet.png"/ width=90%> <br />
</p>
<br>

We also explored the `VQVAE` form. Specifically, the codebook part is added between the original encoder and decoder, and the loss of quantization is additionally supervised.
The codebook block comes from
> Esser P, Rombach R, Ommer B. Taming transformers for high-resolution image synthesis[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021: 12873-12883.

The overall flowchart of our new algorithm is shown as follows.
<br>
<p align="center">
    <img src="./images/flowchart_VQVAE.png"/ width=90%> <br />
</p>
<br>


We use about `800` high-quality rolled fingerprints and binary image extracted by VeriFinger as dataset. During training, `128x128` image patches are randomly sampled from the original complete image. The image patches are then added with some random noise as augmentation. The methods and examples of augmentation can refer to 
> Guan X, Pan Z, Feng J, et al. Joint Identity Verification and Pose Alignment for Partial Fingerprints[J]. arXiv preprint arXiv:2405.03959, 2024.

Examples of image augmentation are shown as follows.
<br>
<p align="center">
    <img src="./images/augmentation.png"/ width=90%> <br />
</p>
<br>

## Run
* **train Unet**
    ```shell
    python train_enhancer.py
    ```
* **train VQVAE**
    ```shell
    python train_VQenhancer.py
    ```

* **test Unet**
    ```shell
    python inference_enhancer.py
    ```
* **test VQVAE**
    ```shell
    python inference_VQenhancer.py
    ```

## Notice :exclamation:
Due to the fact that we only add some simple modal noise during training, there are still challenges in difficult scenarios such as latent fingerprints, highly blurry/incomplete images or complex backgrounds.
Below are examples before and after fingerprint enhancement.

- example 1
<p align="center">
    <img src="./images/ex-1.png"/ width=90%> <br />
</p>
<br>

- example 2
<p align="center">
    <img src="./images/ex-2.png"/ width=90%> <br />
</p>
<br>

- example 3
<p align="center">
    <img src="./images/ex-3.png"/ width=90%> <br />
</p>
<br>

- example 4
<p align="center">
    <img src="./images/ex-4.png"/ width=90%> <br />
</p>
<br>

- example 5
<p align="center">
    <img src="./images/ex-5.png"/ width=90%> <br />
</p>
<br>

## License
This project is released under the MIT license. Please see the LICENSE file for more information.

## Contact me
If you have any questions about the code, please contact Xiongjun Guan gxj21@mails.tsinghua.edu.cn
