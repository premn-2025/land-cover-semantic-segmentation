# 🛣️ Land-Cover Semantic Segmentation (PyTorch)

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.9+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An end-to-end **Computer Vision project** for **semantic segmentation** using PyTorch. The project is built on the **LandCover.ai dataset**, but it can be applied to any semantic segmentation dataset with customizable settings.

## 📌 About the Project

This project focuses on **semantic segmentation** — classifying each pixel of an image into a particular land cover category (e.g., building, water, woodland, forest, agriculture).

### ✨ Key Features

- 🎯 Train on **customizable classes** from any dataset
- ⚙️ Modify **architecture, optimizer, learning rate**, and other parameters via config file
- 🔍 Perform **selective inference** using `test_classes` to filter only desired classes in output mask
- 📊 Comprehensive evaluation metrics and visualization
- 🚀 Easy-to-use training, testing, and inference pipeline

### 🎨 Selective Inference Example

If the model is trained on multiple classes, but you only want predictions for specific classes:

```python
test_classes = ['building', 'water']
```

The model will output segmentation maps containing only those classes, making it perfect for targeted analysis.

## ⚙️ How It Works

### 🏋️ Training
- **Input:** Images + masks (ground truth)
- **Output:** Trained segmentation model stored in `/models`
- **Process:** Model learns to classify each pixel into land cover categories

### 🧪 Testing  
- **Input:** Test images + masks
- **Output:** Evaluation results and predicted segmentation masks
- **Metrics:** IoU, mIoU, pixel accuracy, and class-wise performance

### 🔮 Inference
- **Input:** Only images (no masks required)
- **Output:** Predicted segmentation masks for specified `test_classes`
- **Use Case:** Production deployment and real-world application

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/premn-2025/land-cover-semantic-segmentation.git
cd Land-Cover-Semantic-Segmentation

# 2. Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# 3. Train the model
cd src
python train.py

# 4. Run testing
python test.py

# 5. Run inference
python inference.py
```

## 📂 Project Structure

```
Land-Cover-Semantic-Segmentation-PyTorch/
├── src/
│   ├── train.py              # Training script
│   ├── test.py               # Testing script
│   ├── inference.py          # Inference script
│   ├── config.py             # Configuration settings
│   ├── model.py              # Model architecture
│   ├── dataset.py            # Dataset loader
│   └── utils.py              # Utility functions
├── data/
│   ├── train/
│   │   ├── images/           # Training images
│   │   └── masks/            # Training masks
│   ├── test/
│   │   ├── images/           # Test images
│   │   └── masks/            # Test masks
│   └── inference/
│       └── images/           # Images for inference
├── models/                   # Saved models
├── results/                  # Output results
├── requirements.txt          # Dependencies
└── README.md
```

## 📊 Dataset

### LandCover.ai Dataset
Download the dataset from:
- 📁 [Kaggle](https://www.kaggle.com/datasets/adrienboulet/landcoverai)

### Dataset Setup
1. Download and extract the dataset
2. Place the `images/` and `masks/` folders inside the `data/train/` directory
3. Organize test data similarly in `data/test/`

### Land Cover Classes
- **Building** - Urban structures and buildings
- **Woodland** - Forested areas and trees  
- **Water** - Rivers, lakes, and water bodies
- **Road** - Roads, pathways, and transportation infrastructure

## 🔧 Configuration

Modify `src/config.py` to customize:

```python
# Model settings
MODEL_NAME = 'unet'  # or 'deeplabv3', 'fcn'
NUM_CLASSES = 4
INPUT_SIZE = (256, 256)

# Training settings
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 100

# Classes for selective inference
TEST_CLASSES = ['building', 'water']  # Filter output classes
```

## 📈 Results

The model achieves competitive performance on the LandCover.ai dataset:

- **mIoU:** ~75-80%
- **Pixel Accuracy:** ~85-90%
- **Training Time:** ~2-3 hours on RTX 3080

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Computer Vision community for inspiration and resources


---

⭐ **Star this repository if you find it helpful!**
