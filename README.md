# SIMECO: SIM(3)-Equivariant Shape Completion

**NeurIPS 2025**

[![Website](https://img.shields.io/badge/%F0%9F%A4%8D%20Project%20-Website-blue)](https://sime-completion.github.io)
[![arXiv](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2509.26631)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Model-yellow)](https://huggingface.co/)
[![Colab Demo](https://img.shields.io/badge/Colab-Demo-FF6F00?logo=googlecolab&logoColor=yellow)](https://colab.research.google.com/github/complete3d/simeco/blob/main/demo/demo.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://raw.githubusercontent.com/complete3d/simeco/main/LICENSE)

3D shape completion methods typically assume scans are pre-aligned to a canonical frame. When such alignment is absent in real data, performance collapses. SIMECO is a SIM(3)-equivariant network that delivers generalizable shape completion.

<p align="center">
  <img src="assets/video.gif" alt="teaser" width="650px">
</p>

## Getting Started with the Demo 🤹‍♂️

You can run the demo in one of two ways:

- **Google Colab** (Recommended): No installation needed; get started instantly in the cloud.
- **Local Linux**: Set up the environment locally for GPU-accelerated inference.

### Google Colab ☁️

[<img src="https://colab.research.google.com/assets/colab-badge.svg" height="32"/>](https://colab.research.google.com/github/complete3d/simeco/blob/main/demo/demo.ipynb)

To run the demo on Google Colab, simply click the badge above. Follow the instructions and execute the cells sequentially with a GPU instance. Please note that setting up the environment and installing dependencies will take about 1-2 minutes.

### Local Linux 🖥️

> [!NOTE]
A CUDA-enabled GPU is required for local inference.

1. Clone the repository:

   ```bash
   git clone https://github.com/complete3d/simeco.git && cd simeco && git lfs pull
   ```

2. Create a conda environment with all dependencies (this will take approximately 5 minutes):

   ```bash
   . install.sh
   ```

3. Run the inference code and choose the desired transformation mode (e.g., `sim3`, `translation`, `scale` or `rotation`). The results will be saved in the `data/result` directory.

   ```bash
   python inference.py --pc_file data/pc/2a05d684eeb9c1cfae2ca9bb680dd18b.npy --aug_mode sim3
   ```
---

## 🚀 Usage:

### Training 🏋️‍♀️

* Start training using one of the two parallelization:

   **Distributed Data Parallel (DDP):**
  
    ```bash
    # Replace device IDs with your own
    CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_ddp.sh
    ```

   **Data Parallel (DP):**
  
    ```bash
    # Replace device IDs with your own
    CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_dp.sh
    ```

* Monitor training progress using TensorBoard:
  
  ```bash
  # Replace ${exp_name} with your experiment name (e.g., default)
  # Board typically available at http://localhost:6006
  tensorboard --logdir './output/${exp_name}/tensorboard'
  ```

### Evaluation 📊

* To evaluate a [pre-trained SIMECO model](./ckpt) using a single GPU:

  ```bash
  # Replace device IDs with your own
  CUDA_VISIBLE_DEVICES=0 ./scripts/test.sh
  ```

### Dataset 📂

We use the official PCN dataset. Please download the dataset from [PCN](https://gateway.infinitescript.com/s/ShapeNetCompletion) and place it under the `data/` directory. The expected directory structure is:
```
│PCN/
├──train/
│  ├── complete
│  │   ├── 02691156
│  │   │   ├── 1a04e3eab45ca15dd86060f189eb133.pcd
│  │   │   ├── .......
│  │   ├── .......
│  ├── partial
│  │   ├── 02691156
│  │   │   ├── 1a04e3eab45ca15dd86060f189eb133
│  │   │   │   ├── 00.pcd
│  │   │   │   ├── 01.pcd
│  │   │   │   ├── .......
│  │   │   │   └── 07.pcd
│  │   │   ├── .......
│  │   ├── .......
├──test/
│  ├── complete
│  │   ├── .......
│  ├── partial
│  │   ├── .......
├──val/
│  ├── complete
│  │   ├── .......
│  ├── partial
│  │   ├── .......
├──PCN.json
└──category.txt
```

## 🚧 TODOs

- [x] Code release
- [ ] Debiased protocol
- [ ] HF release

## 🎓 Citation

If you use SIMECO in a scientific work, please consider citing the paper:

<a href="https://arxiv.org/pdf/2509.26631"><img class="image" align="left" width="190px" src="./assets/paper_thumbnail.png"></a>
<a href="https://arxiv.org/pdf/2509.26631">[paper]</a>&nbsp;&nbsp;<a href="https://arxiv.org/abs/2509.26631">[arxiv]</a>&nbsp;&nbsp;<a href="./CITATION.bib">[bibtex]</a><br>
```bibtex
@article{wang2025simeco,
    title={Learning Generalizable Shape Completion with SIM(3) Equivariance}, 
    author={Yuqing Wang and Zhaiyu Chen and Xiao Xiang Zhu},
    journal={arXiv preprint arXiv:2509.26631},
    year={2025}
}
```
<br clear="left"/>

## 🙏 Acknowledgements

Part of our implementation is based on the [PoinTr](https://github.com/yuxumin/PoinTr) repository. We thank the authors for open-sourcing their great work.