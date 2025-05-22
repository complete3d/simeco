# SIMECO

This repository provides the demo and code of SIMECO.

## Getting Started with the Demo 🤹‍♂️

You can run the demo in one of two ways:

- **Google Colab** (Recommended): No installation needed; get started instantly in the cloud.
- **Local Linux**: Set up the environment locally for GPU-accelerated inference.

### Google Colab ☁️

[<img src="https://colab.research.google.com/assets/colab-badge.svg" height="32"/>](https://colab.research.google.com/github/submission13448/simeco/blob/main/demo.ipynb)

To run the demo on Google Colab, simply click the badge above ([anonymous link](https://colab.research.google.com/github/submission13448/simeco/blob/main/demo.ipynb)). Follow the instructions and execute the cells sequentially. Please note that setting up the environment and installing dependencies will take about 1-2 minutes.

### Local Linux 🖥️

> [!NOTE]
A CUDA-enabled GPU is required for local inference.

1. Clone the repository:

   ```bash
   git clone https://github.com/submission13448/simeco.git && cd simeco && git lfs pull
   ```

2. Create a conda environment with all dependencies (this will take approximately 5 minutes):

   ```bash
   conda env create -f build/environment.yml && conda activate simeco
   ```

3. Run the inference code and choose the desired transformation mode (e.g., `sim3`, `translation`, `scale` or `rotation`). The results will be saved in the `data/result` directory.

   ```bash
   python inference.py --pc_file data/pc/2a05d684eeb9c1cfae2ca9bb680dd18b.npy --aug_mode sim3
   ```
---

## Full Runs 🏃

If you have sufficient resources and want to train and evaluate SIMECO end-to-end, follow these steps:

### Extra Dependencies ⚙️

Install the dependencies:

   ```bash
   cd extensions/chamfer_dist
   python setup.py install
   ```

### Dataset 📂

We use the official PCN dataset. The directory structure should be:
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

### Evaluation 📊

To evaluate a [pre-trained SIMECO model](./ckpt) using a single GPU:

```bash
bash ./scripts/test.sh <GPU_IDS> \
    --ckpts <path_to_checkpoint> \
    --config <path_to_config.yaml> \
    --exp_name <experiment_name>
```

Example:
```bash
bash ./scripts/test.sh 0 \
    --ckpts ckpt/checkpoint.pth \
    --config cfgs/SIMECO.yaml \
    --exp_name SIMECO
```

### Training 🏋️‍♀️

To train SIMECO from scratch, run with DDP or DP:

**DistributedDataParallel (DDP)** 
```bash
bash ./scripts/dist_train.sh <NUM_GPU> <port> \
    --config <config> \
    --exp_name <name> \
    [--resume] \
    [--start_ckpts <path>] \
```

Example:
```bash
bash ./scripts/dist_train.sh 2 12345 \
    --config cfgs/SIMECO.yaml \
    --exp_name SIMECO 
```

**DataParallel (DP)** 
```bash
bash ./scripts/train.sh <GPUIDS> \
    --config <config> \
    --exp_name <name> \
    [--resume] \
    [--start_ckpts <path>] \
```

Example:
```bash
bash ./scripts/train.sh 0 \
    --config cfgs/SIMECO.yaml \
    --exp_name SIMECO 
```
