import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns


def load_data():
    # Get the path of current script
    current_script_path = os.path.abspath(__file__)
    folder_path = os.path.dirname(current_script_path)
    project_root = os.path.dirname(folder_path)
    # Create the path of data
    data_path = os.path.join(project_root, 'Datasets', 'pathmnist.npz')
    pretrained_model_path = os.path.join(folder_path, 'CNN_task2.h5')

    return data_path, pretrained_model_path
 

def run_b_pretrained_task():
    # Define the seed value for reproducibility
    seed_value = 42 
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    
    # Load the path of data needed
    data_path, pretrained_model_path = load_data()

    # Load the pre-trained model
    model = load_model(pretrained_model_path)

    # Define the data path and Load dataset
    pneumoniamnist = np.load(data_path)

    # Seperate the dataset
    test_images = pneumoniamnist['test_images']
    test_labels = pneumoniamnist['test_labels']

    # Pre-process
    # Normalisation
    test_images = test_images.astype('float32') / 255

    # Convert labels to the categorical
    test_labels = to_categorical(test_labels)

    # Evaluate the test result
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f"Test Accuracy: {test_accuracy * 100}%")

    # Predict for test dataset
    predictions = model.predict(test_images)

    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels, axis=1)

    # Calculate the confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(9), yticklabels=range(9))
    plt.title('Confusion Matrix Heatmap')
    plt.ylabel('True Classes')
    plt.xlabel('Predicted Classes')
    plt.savefig("Confusion Matrix Heatmap.png")
    plt.show()
   
if __name__ == "__main__":
    run_b_pretrained_task()
