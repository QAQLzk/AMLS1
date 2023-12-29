import os
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
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

    return data_path

def run_b_task():

    # Define the seed value for reproducibility
    seed_value = 42 
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    data_path = load_data()
    # Load dataset
    pneumoniamnist = np.load(data_path)


    # Seperate the dataset
    train_images = pneumoniamnist['train_images']
    train_labels = pneumoniamnist['train_labels']

    val_images = pneumoniamnist['val_images']
    val_labels = pneumoniamnist['val_labels']

    test_images = pneumoniamnist['test_images']
    test_labels = pneumoniamnist['test_labels']


    # Pre-process
    # Normalisation
    train_images = train_images.astype('float32') / 255
    val_images = val_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    # Convet labels to the categorical
    train_labels = to_categorical(train_labels)
    val_labels = to_categorical(val_labels)
    test_labels = to_categorical(test_labels)



    # Define the number of classes need to classification
    num_classes = 9

    # Build the CNN Model
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 3)),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(64, (3, 3), padding='same', activation='relu'),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


    # Trainning
    history = model.fit(train_images, train_labels,
                        batch_size=64,
                        epochs=16,
                        validation_data=(val_images, val_labels))



    # Evaluate the test result
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f"Test Accuracy: {test_accuracy * 100}%")


    # Plot the learning process (Accuracy)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()


    # Plot the learning process (Loss)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.savefig("Model loss in each epoch.png")
    plt.show()

    # Predict for test dataset
    predictions = model.predict(test_images)

    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels, axis=1)

    # calculate the confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)

    # plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.title('Confusion Matrix Heatmap')
    plt.ylabel('True Classes')
    plt.xlabel('Predicted Classes')
    plt.savefig("Confusion Matrix Heatmap.png")
    plt.show()


    # Save the model
    model.save('CNN_task2.h5')

if __name__ == "__main__":
    run_b_task()
