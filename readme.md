# Outfit Transformer: Outfit Representations for Fashion Recommendation

<div align="center"> <img src="https://github.com/owj0421/outfit-transformer/assets/98876272/fc39d1c7-b076-495d-8213-3b98ef038b64" width="512"> </div>

## 📌 Introduction

This repository provides the implementation of **Outfit Transformer**, a model designed for fashion recommendation, inspired by:

> Rohan Sarkar et al. [Outfit Transformer: Outfit Representations for Fashion Recommendation](https://arxiv.org/abs/2204.04812). CVPR 2023.

Checkpoints: https://drive.google.com/drive/folders/18H8l90C0_vSqcdXn4S2rK_MHK4N2DECn?usp=drive_link
## 🛠️ Installation

```bash
conda create -n outfit-transformer python=3.12.4
conda activate outfit-transformer
conda env update -f environment.yml
```

## 📥 Download Datasets & Checkpoints

```bash
mkdir -p datasets
gdown --id 1ox8GFHG8iMs64iiwITQhJ47dkQ0Q7SBu -O polyvore.zip
unzip polyvore.zip -d ./datasets/polyvore
rm polyvore.zip

mkdir -p checkpoints
gdown --id 1mzNqGBmd8UjVJjKwVa5GdGYHKutZKSSi -O checkpoints.zip
unzip checkpoints.zip -d ./checkpoints
rm checkpoints.zip
```

## 🏋️ Training & Evaluation

### Step 1: Precompute CILP Embeddings
Before proceeding with training, make sure to precompute the CLIP embeddings, as all subsequent steps rely on these precomputed features.

```bash
python -m src.run.1_generate_clip_embeddings
```

### Step 2: Compatibility Prediction
Train the model for the Compatibility Prediction (CP) task.

#### 🔥 Train
```bash
python -m src.run.2_train_compatibility \
--wandb_key $YOUR/WANDB/API/KEY
```


## 📝 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

If you use this repository, please mention the original GitHub repository by linking to [outfit-transformer](https://github.com/owj0421/outfit-transformer). This helps support the project and acknowledges the contributors.
