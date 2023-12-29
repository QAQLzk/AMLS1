# AMLS_23-24_SN23047130

## Introduction
The project is to use machine learning methods to categorize medical images. There are 2 datasets, PneumoniaMNIST and PathMNIST, that were used for training/validation/testing.

The project is divided into 2 tasks:
 Task A: Binary classification --Random Forest 
 Task B: Multi-class classification --Conventional Neural Network(CNN)

The machine learning code and results are stored in folders 'A' and 'B' of the directory.The code was programmed by jupyter notebook. To reproduce the results, copies of the .py files were created with the same results as the .ipynb file results. For Task A, it needs to take about 5 minutes to reproduce.For Task B, the full training model takes about 30 minutes to run.At the same time, the pre-trained model is saved for direct use. However, when run the pre-trained model, the image of model accuracy and loss in each epoch will not be generated.



The files structure is shown below：
## File Structure
- 'A/':
 - 'A_ipynb.ipynb': the code for task A in jupyter notebook, used for view and generate the results directly
 - 'A_py.py': the code for task A in python, used to reproduce the results
 - 'Top 10 Feature Importance... .png': the result of top 10 feature importance for task A
 - 'Validation Accuracy.png': the result of validation accuracy for task A

- 'B/': 
 - 'B_ipynb.ipynb': the code for task B in jupyter notebook, used for view and generate the results
 - 'B_py.py': the code for task B in python, used to reproduce the results
 - 'B_py_pretrain.py': the code for task B in python, used to reproduce the results with pre-trained model
 - 'CNN_model.h5': the pre-trained model for task B
 - 'Confusion Matrix Heatmap.png': the Heat of confusion matrix for task B
 - 'Model Accuracy in each epoch.png': the result of model accuracy in each epoch (Only be produced when run the full training model)
 - 'Model Loss in each epoch.png': the result of model loss in each epoch (Only be produced when run the full training model)

- "Datasets/": Empty, Please must put pneumoniamnist.npz and pathmnist.npz in this folder

- 'main.py': Used to reproduce the results. Default Running: A_py.py and B_py_pretrained.py . If want to run the full training model, please uncomment the code in it.

- 'README.md':  the README file

## How to run
To run the code, please put the 'pneumoniamnist.npz' and 'pathmnist.npz' in the 'Dataset' folder first. It can be download from https://medmnist.com/

The project is based on python 3.11.6 and the following packages are required:

scikit-learn==1.3.2
seaborn==0.13.0
tensorflow==2.15.0
numpy==1.26.2
matplotlib==3.8.2

Then, please run the 'main.py' file to reproduce the results. The default running is A_py.py and B_py_pretrain.py . If want to run the full training model, please uncomment the code in it.
Also you can run Jupyter Notebook file to view the code and generate the results directly.









