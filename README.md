# AMLS_23-24_SN23047130

## Introduction
The project aims to categorize medical images using machine learning methods. It focuses on two datasets: PneumoniaMNIST and PathMNIST, used for training, validation, and testing.

The project is divided into two tasks:
- Task A: Binary classification using Random Forest
- Task B: Multi-class classification using Conventional Neural Network (CNN)

The machine learning code and results are stored in the 'A' and 'B' folders. The code was programmed in Jupyter Notebook, and to reproduce the results, ".py" files were created with the same outputs as the ".ipynb" files. Task A requires approximately 5 minutes to reproduce, and Task B's full training model takes about 30 minutes. A pre-trained model is also provided for immediate use, though it won't generate the accuracy and loss images for each epoch.

## File Structure

### A/
- `A_ipynb.ipynb`: Jupyter Notebook code for Task A, used to view and generate results directly.
- `A_py.py`: Python code for Task A.
- `Top 10 Feature Importances.png`: Result showing the top 10 feature importances for Task A.
- `Validation Accuracy.png`: Result showing validation accuracy for Task A.

### B/
- `B_ipynb.ipynb`: Jupyter Notebook code for Task B, used to view and generate results directly.
- `B_py.py`: Python code for Task B.
- `B_py_pretrain.py`: Python code for Task B with a pre-trained model.
- `CNN_model.h5`: The pre-trained model file for Task B.
- `Confusion Matrix Heatmap.png`: Heatmap of the confusion matrix for Task B.
- `Model Accuracy in each epoch.png`: Model accuracy for each epoch (produce only during full model training).
- `Model Loss in each epoch.png`: Model loss for each epoch (produce only during full model training).

### Datasets/
- Empty folder. Please place `pneumoniamnist.npz` and `pathmnist.npz` here.

### `main.py`:
- Script to reproduce results. Default run includes `A_py.py` and `B_py_pretrain.py`. For full training, please uncomment the middle part of t code.

### `README.md`

## How to Run

1. Please place `pneumoniamnist.npz` and `pathmnist.npz` in the Datasets folder, which can be downloaded from https://medmnist.com/
2. Ensure `Python 3.11.6` is installed along with the following packages:
   - `scikit-learn==1.3.2`
   - `seaborn==0.13.0`
   - `tensorflow==2.15.0`
   - `numpy==1.26.2`
   - `matplotlib==3.8.2`
3. Run `main.py` to reproduce the results. By default, `A_py.py` and `B_py_pretrain.py` are executed. Uncomment related code in `main.py` for full training.
4. Also, you can run Jupyter Notebook files directly to view the code and generate results.








