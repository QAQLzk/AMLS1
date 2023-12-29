import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

def load_data():
    # Get the path of the script
    current_script_path = os.path.abspath(__file__)
    folder_path = os.path.dirname(current_script_path)
    project_root = os.path.dirname(folder_path)
    # Combine the path of the dataset
    data_path = os.path.join(project_root, 'Datasets', 'pneumoniamnist.npz')

    return data_path

def run_a_task():

    # Define the data path and Load dataset
    data_path = load_data()
    pneumoniamnist = np.load(data_path)

    # Seperate the dataset
    train_images = pneumoniamnist['train_images']
    train_labels = pneumoniamnist['train_labels']

    val_images = pneumoniamnist['val_images']
    val_labels = pneumoniamnist['val_labels']

    test_images = pneumoniamnist['test_images']
    test_labels = pneumoniamnist['test_labels']

    def preprocess_dataset(images, labels):
        #Flatten the image array
        flattened_images = images.reshape(images.shape[0], -1)
        # Conver to 1-dimension
        labels = labels.ravel()
        return flattened_images, labels

    train_images, train_labels = preprocess_dataset(train_images, train_labels)
    val_images, val_labels = preprocess_dataset(val_images, val_labels)
    test_images, test_labels = preprocess_dataset(test_images, test_labels)

    # Set value of n_estimators(number of tree) from 70 to 120, gap:10
    n_estimators_range = range(70, 131, 10)

    # store each n accuracy
    val_accuracies = []

    # Iteration
    for n_estimators in n_estimators_range:
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        # Train
        clf.fit(train_images, train_labels)
        
        # Iteration
        predicted_val_labels = clf.predict(val_images)
        val_accuracy = accuracy_score(val_labels, predicted_val_labels)

        val_accuracies.append(val_accuracy)

        print(f"n_estimators: {n_estimators}, Validation Accuracy: {val_accuracy}")

    # Plot the change of validation accuracy in different tree numbers
    plt.figure() 
    plt.plot(n_estimators_range, val_accuracies, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs. Number of Estimators')
    plt.savefig("Validation Accuracy vs. Number of Estimators.png")


    # Choose 100 as the final number of estimators
    clf_final = RandomForestClassifier(n_estimators = 100, random_state=42)

    # Train the model
    clf_final.fit(train_images, train_labels)

    # Use Cross-validation to evaluate the model
    scores = cross_val_score(clf, train_images, train_labels, cv=5)
    print("Average cross-validation score: ", scores.mean())

    # Predict the Test dataset
    predicted_labels = clf_final.predict(test_images)
    print("The Accuracy of test dataset:", accuracy_score(test_labels, predicted_labels))


    # Display the top 10 features importance
    importances = clf_final.feature_importances_
    # display the top 10 features
    top_features = 10  

    # Get the index of the most 10 important features
    indices = np.argsort(importances)[-top_features:]

    plt.figure() 
    plt.title('Top 10 Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), ['Pixel' + str(i) for i in indices])
    plt.xlabel('Relative Importance')
    plt.savefig("Top 10 Feature Importances.png")
    plt.show()

if __name__ == "__main__":
    run_a_task()
