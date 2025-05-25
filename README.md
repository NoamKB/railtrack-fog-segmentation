# Foggy Railway Segmentation Project

## Overview

This project focuses on segmenting railway tracks in foggy conditions using the BiSeNetV2 model. It covers preprocessing, pseudo-labeling, targeted fine-tuning, and result analysis.

---

## 📁 Project Structure

```rail_marking/
├── cfg/ # Model configuration files
├── data/                   # Input datasets (clear/fog/hard_example)
├── model/                  # Trained checkpoints (.pth)
├── my_scripts/ # Pseudo-labels, fine-tuning, comparison tools
├── notebooks/ # Evaluation notebooks
├── output/ # Inference results (before/after)
├── rail_marking/ # Core model, training, data loaders
├── scripts/ # Main training/inference scripts
├── tests/ # Unit tests
├── requirements.txt
├── environment.yml
├── setup.py
├── pyproject.toml
├── README.md
├── Railtrack Segmentation Project Report.pdf
```

---

📂 External Resources (Data & Outputs)
Due to size constraints, datasets, model checkpoints and inference results are hosted externally.

👉 Access all project resources here (https://drive.google.com/drive/folders/13orpV4tsqUL9mubSU6Mx9xsqm8MOvBhg?usp=drive_link)

Contents:
/data/clear/, /data/fog/, /data/hard_example/ – Input datasets and pseudo-labels

/model_checkpoints/ – All .pth files (pretrained and fine-tuned)

/output/ – Inference results and visual comparisons

---

## 📂 Downloaded Resources – Where to Place Them

After downloading the project files from Google Drive, place the folders **in the root directory of the project**, at the same level as `README.md`.

---

## ⚙️ Setup Instructions

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📊 Evaluation Methods

- Visual overlays & inspection
- Entropy, skewness, Bhattacharyya distance
- Tracking performance on hard examples

---

## 🎯 Evaluation Access

All results and models are fully reproducible.  
Please refer to [`fog_segmentation_evaluation.ipynb`](notebooks/fog_segmentation_evaluation.ipynb) for step-by-step visualization and comparison.

For further questions or full-resolution visual results, feel free to reach out.

---

## 📎 Notes

- See `report.pdf` for detailed write-up
- Pseudo-labeling and correction were essential for fog adaptation

---

### 🔗 Based on
Original baseline from https://github.com/xmba15/rail_marking (BiSeNetV2)
