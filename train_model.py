import numpy as np
import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Config
DATA_FILE = 'gesture_data.csv'
MODEL_FILE = 'gesture_model.h5'

def load_data(file_path):
    X = []
    y = []
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader) # Skip header
        
        for row in reader:
            # Parse landmarks (first 63 values)
            landmarks = [float(x) for x in row[:63]]
            label = int(row[63])
            
            # Direct mapping from collect_data.py
            # 0: Volume
            # 1: Bright Up
            # 2: Bright Down
            # 3: Show Desktop
            
            if 0 <= label <= 3:
                X.append(landmarks)
                y.append(label)
            
    return np.array(X), np.array(y)

def main():
    print("Loading data...")
    try:
        X, y = load_data(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Run collect_data.py first.")
        return

    # Check classes
    unique_classes = np.unique(y)
    print(f"Found {len(X)} samples.")
    print(f"Classes found: {unique_classes}")
    
    # We ideally need all 4 classes, but checks are loose to allow partial testing
    if len(unique_classes) < 2:
        print("Error: Need multiple classes to train.")
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Compute class weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weights_dict}")

    # Build Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(63,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax') # 0=Vol, 1=B_Up, 2=B_Down, 3=Desktop
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    # Train
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weights_dict,
        verbose=1
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Save
    model.save(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()
