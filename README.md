# 🛣 Land-Cover Semantic Segmentation (PyTorch)

An end-to-end **Computer Vision project** for **semantic segmentation** using PyTorch.  
The project is built on the **LandCover.ai dataset**, but it can be applied to any semantic segmentation dataset with customizable settings.

---

## 📌 About the Project
This project focuses on **semantic segmentation** — classifying each pixel of an image into a particular land cover category (e.g., building, water, woodland).  

### Key Features
- Train on **customizable classes** from any dataset.  
- Modify **architecture, optimizer, learning rate**, and other parameters via a config file.  
- Perform **selective inference** using `test_classes` to filter only the classes you want in the output mask.  

**Example:**  
If the model is trained on multiple classes, but you only want predictions for `"building"` and `"water"`, you can set:  
```python
test_classes = ['building', 'water']
The model will output segmentation maps containing only those classes.

⚙️ How It Works
Training
Input: images + masks (ground truth).

Output: trained segmentation model stored in /models.

Testing
Input: test images + masks.

Output: evaluation results and predicted segmentation masks.

Inference
Input: only images (no masks required).

Output: predicted segmentation masks for the specified test_classes.

🚀 Quick Start
bash
Copy code
# 1. Clone the repo
git clone https://github.com/souvikmajumder26/Land-Cover-Semantic-Segmentation-PyTorch.git
cd Land-Cover-Semantic-Segmentation-PyTorch

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
cd src
python train.py

# 4. Run testing
python test.py

# 5. Run inference
python inference.py
📂 Dataset
Download the dataset from:

Official Site

Kaggle

Place the images/ and masks/ folders inside the train/ directory.

🛡️ License
This project is licensed under the MIT License.

pgsql
Copy code

Do you want me to make this **clean and minimal** (just text), or should I also **add demo images** (training/test