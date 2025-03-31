import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator # type: ignore

# Constants
EPOCHS = 20  # Training iterations
IMG_SIZE = (30, 30)
NUM_CLASSES = 43
TEST_RATIO = 0.4

def main():
    # Ensure correct usage
    if len(sys.argv) not in [2, 3, 4]:
        sys.exit("Usage: python traffic.py data_directory [best_model.h5] [test]")
    
    # Load dataset
    image_data, class_labels = load_dataset(sys.argv[1])
    class_labels = tf.keras.utils.to_categorical(class_labels)
    
    # Split into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(image_data), np.array(class_labels), test_size=TEST_RATIO
    )
    
    # Normalize inputs
    x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
    
    # Check if a model should be loaded
    if len(sys.argv) >= 3:
        try:
            model = tf.keras.models.load_model(sys.argv[2])
            print(f"Model loaded from {sys.argv[2]}")
        except Exception as error:
            sys.exit(f"Failed to load model: {error}")
        
        # Evaluate model if "test" argument is given
        if len(sys.argv) == 4 and sys.argv[3] == "test":
            print("Evaluating model...")
            loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
            print(f"Accuracy: {accuracy * 100:.2f}%")
            return
        
        # Make predictions otherwise
        print("Generating predictions...")
        predictions = model.predict(x_test)
        correct_count = sum(np.argmax(pred) == np.argmax(actual) for pred, actual in zip(predictions, y_test))
        print(f"Correctly predicted {correct_count} out of {len(predictions)} ({correct_count / len(predictions) * 100:.2f}%)")
        return
    
    # Train a new model if none provided
    model = build_model()
    
    # Image augmentation
    augmentor = ImageDataGenerator(
        rotation_range=10, zoom_range=0.15,
        width_shift_range=0.1, height_shift_range=0.1,
        horizontal_flip=False, fill_mode='nearest'
    )
    
    # Early stopping callback
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )
    
    # Train the model
    model.fit(
        augmentor.flow(x_train, y_train, batch_size=32),
        epochs=EPOCHS, validation_data=(x_test, y_test),
        callbacks=[early_stop]
    )
    
    # Evaluate model
    model.evaluate(x_test, y_test, verbose=2)
    
    # Save model
    save_path = sys.argv[2] if len(sys.argv) == 3 else "best_model.h5"
    model.save(save_path)
    print(f"Model stored in {save_path}")

def load_dataset(directory):
    """
    Load images and labels from the dataset directory.
    """
    images, labels = [], []
    for category in range(NUM_CLASSES):
        category_folder = os.path.join(directory, str(category))
        if not os.path.isdir(category_folder):
            continue
        
        for filename in os.listdir(category_folder):
            file_path = os.path.join(category_folder, filename)
            try:
                image = cv2.imread(file_path)
                if image is not None:
                    images.append(cv2.resize(image, IMG_SIZE))
                    labels.append(category)
            except Exception as error:
                print(f"Skipped {file_path}: {error}")
                continue
    return images, labels

def build_model():
    """Creates and compiles a CNN model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(*IMG_SIZE, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    main()
