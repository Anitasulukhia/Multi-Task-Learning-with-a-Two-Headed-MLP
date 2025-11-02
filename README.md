#tudent Performance Multi-Task Learning 

This project demonstrates **multi-task learning** using PyTorch, where a single neural network predicts:

1. 🎯 **Final Grade (G3)** — a *regression* task  
2. ❤️ **Romantic Relationship Status** — a *binary classification* task  

By sharing a common representation (the "shared body"), the model learns how academic, demographic, and social factors jointly influence both outcomes.

---

## Dataset

**Source:** [UCI Machine Learning Repository — Student Performance Dataset (ID 320)](https://archive.ics.uci.edu/ml/datasets/Student+Performance)

The dataset contains attributes describing students’ demographic, social, and school-related information.

**Key attributes:**
- `sex`, `age`, `address`, `famsize`, `Pstatus`
- `studytime`, `failures`, `absences`, `schoolsup`, `G1`, `G2`, `G3`
- `Mjob`, `Fjob`, `reason`, `guardian`, `internet`, `romantic`

---

## Data Pre-Processing

Steps applied before training:

1. **Feature & Target separation**
   - Features → `X`
   - Regression target → `G3`
   - Classification target → `romantic`

2. **Encoding categorical data**
   - Binary columns (`yes`/`no`, `F`/`M`, etc.) → `0`/`1`
   - Multi-category columns (`Mjob`, `Fjob`, `guardian`, etc.) → one-hot encoding with `pd.get_dummies()`

3. **Standardization**
   - All numerical columns scaled with `StandardScaler` (zero mean, unit variance)

4. **Splitting**
   - 70 % train / 15 % validation / 15 % test using `train_test_split`

5. **Tensor Conversion**
   - Converted all splits into PyTorch tensors  
   - Custom `StudentPerformanceDataset` implemented for multi-output samples

---

## Model Architecture

The network has a **shared body** (common feature extractor) and two **separate heads**.

### Shared Body
Learns general student representations.

### Head 1
predicts the final grade with additional 2 layer training

### Head 2
predicts the romantic status with additional 2 layer training

---
## Requirements
torch
torchvision
pandas
numpy
scikit-learn
matplotlib
ucimlrepo

