"""
TensorFlow implementation for semantic segmentation using U-Net
Dataset: Medical image segmentation
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import copy
from sklearn.metrics import accuracy_score


# ==================== Configuration ====================
# Default dataset path (use Docker mount point)
DEFAULT_DATASET = "/app/datasets/Segmentation"

# Check for command-line argument
if len(sys.argv) > 1:
    dataset = sys.argv[1]
    print(f"Using custom dataset path: {dataset}")
else:
    dataset = DEFAULT_DATASET
    print(f"Using default dataset path: {dataset}")

# Verify dataset exists
if not os.path.exists(dataset):
    print(f"Dataset path does not exist: {dataset}")
    print("Please update the 'dataset' variable with the correct path to your local dataset folder.")
else:
    print(f"Dataset path found: {dataset}")


# ==================== Hyperparameters ====================
params = {
    'x': 192, 
    'y': 272,
    'batch_size': 8,
    'n_channels_mask': 1,
    'n_channels': 1,
    'shuffle': True, 
    'learningRate': 0.0001,
    'momentum': 0.98,
    'nEpoches': 20,
}


# ==================== Data Loading ====================
def load_training_data(dataset_path, params):
    """Load training images and masks from images/files/ directory where masks end with _mask.png"""
    
    # Define directories - check for both images/ and images/files/
    training_dir = os.path.join(dataset_path, "training")
    images_files_dir = os.path.join(training_dir, "images", "files")
    images_dir = os.path.join(training_dir, "images")
    
    # Use images/files/ if it exists, otherwise fall back to images/
    if os.path.exists(images_files_dir):
        data_dir = images_files_dir
        print(f"Using images/files/ directory: {data_dir}")
    elif os.path.exists(images_dir):
        data_dir = images_dir
        print(f"Using images/ directory: {data_dir}")
    else:
        print(f"ERROR: No images directory found at {images_dir}")
        return {}, {}
    
    # Get list of unique image files (exclude _mask files)
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])
    image_files = sorted(list(set([f for f in all_files if not f.endswith('_mask.png')])))
    
    length_training = len(image_files)
    length_validation = 0
    
    params['length_training'] = length_training
    params['length_validation'] = length_validation
    
    print(f"Training images found: {length_training}")
    print(f"Validation images found: {length_validation}")
    print("No validation data available - using training data only")
    
    dim = (params['y'], params['x'])
    Xlist = {}
    Ylist = {}
    ipp = 0
    
    # Load training data
    X_train = np.empty((params['length_training'], params['x'], params['y'], params['n_channels']))
    y_train = np.empty((params['length_training'], params['x'], params['y'], params['n_channels_mask']))
    
    print("\nLoading training data...")
    for im in image_files:
        image_path = os.path.join(data_dir, im)
        
        # Construct mask path by inserting _mask before .png
        base_name = im.rsplit('.', 1)[0]  # Remove extension
        mask_filename = f"{base_name}_mask.png"
        mask_path = os.path.join(data_dir, mask_filename)
        
        # Load and process image
        image = cv2.imread(image_path, 0)
        if image is not None:
            image = cv2.resize(image, dim)
            mea = np.mean(image)
            ss = np.std(image)
            image = (image - mea) / ss
            X_train[ipp, :, :, 0] = image
            print(f"  Loaded image: {im}")
        else:
            print(f"  ERROR: Failed to load image: {im}")
        
        # Load and process mask
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, 0)
            if mask is not None:
                mask = cv2.resize(mask, dim)
                mask = mask / 255
                y_train[ipp, :, :, 0] = mask
                print(f"  Loaded mask: {mask_filename}")
            else:
                print(f"  ERROR: Failed to load mask: {mask_filename}")
        else:
            print(f"  WARNING: Mask not found: {mask_path}")
        
        ipp += 1
    
    Xlist['training'] = X_train
    Ylist['training'] = y_train
    print(f"\nTraining data loaded: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    
    # Create empty validation arrays
    X_val = np.empty((0, params['x'], params['y'], params['n_channels']))
    y_val = np.empty((0, params['x'], params['y'], params['n_channels_mask']))
    Xlist['validation'] = X_val
    Ylist['validation'] = y_val
    print("Validation data: empty (no validation files found)")
    
    return Xlist, Ylist


# ==================== Data Generator ====================
def create_data_generator(X_train, Y_train, params):
    """Create TensorFlow data generators with augmentation"""
    
    # Create tf.data dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    
    def augment(image, mask):
        """Apply data augmentation"""
        # Random rotation (simulated with random flips and shifts)
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)
        
        return image, mask
    
    # Apply augmentation and batching
    train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train))
    train_dataset = train_dataset.batch(params['batch_size'])
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    steps_per_epoch = len(X_train) // params['batch_size']
    
    print("TensorFlow Data augmentation pipeline created")
    print("Training data generator ready:")
    print(f"  - Batch size: {params['batch_size']}")
    print("  - Augmentation: Enabled (flip left/right, flip up/down)")
    print(f"  - Steps per epoch: {steps_per_epoch}")
    
    return train_dataset, steps_per_epoch


# ==================== U-Net Model ====================
def create_unet_model(input_shape=(192, 272, 1), num_classes=1):
    """Create U-Net model using TensorFlow/Keras"""
    
    inputs = keras.Input(shape=input_shape)
    
    # Encoder
    enc1 = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    enc1 = layers.Conv2D(64, 3, padding='same', activation='relu')(enc1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(enc1)
    
    enc2 = layers.Conv2D(128, 3, padding='same', activation='relu')(pool1)
    enc2 = layers.Conv2D(128, 3, padding='same', activation='relu')(enc2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(enc2)
    
    enc3 = layers.Conv2D(256, 3, padding='same', activation='relu')(pool2)
    enc3 = layers.Conv2D(256, 3, padding='same', activation='relu')(enc3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(enc3)
    
    enc4 = layers.Conv2D(512, 3, padding='same', activation='relu')(pool3)
    enc4 = layers.Conv2D(512, 3, padding='same', activation='relu')(enc4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(enc4)
    
    # Bottleneck
    bottleneck = layers.Conv2D(1024, 3, padding='same', activation='relu')(pool4)
    bottleneck = layers.Conv2D(1024, 3, padding='same', activation='relu')(bottleneck)
    bottleneck = layers.Dropout(0.5)(bottleneck)
    
    # Decoder
    up4 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(bottleneck)
    concat4 = layers.Concatenate()([up4, enc4])
    dec4 = layers.Conv2D(512, 3, padding='same', activation='relu')(concat4)
    dec4 = layers.Conv2D(512, 3, padding='same', activation='relu')(dec4)
    
    up3 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(dec4)
    concat3 = layers.Concatenate()([up3, enc3])
    dec3 = layers.Conv2D(256, 3, padding='same', activation='relu')(concat3)
    dec3 = layers.Conv2D(256, 3, padding='same', activation='relu')(dec3)
    
    up2 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(dec3)
    concat2 = layers.Concatenate()([up2, enc2])
    dec2 = layers.Conv2D(128, 3, padding='same', activation='relu')(concat2)
    dec2 = layers.Conv2D(128, 3, padding='same', activation='relu')(dec2)
    
    up1 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(dec2)
    concat1 = layers.Concatenate()([up1, enc1])
    dec1 = layers.Conv2D(64, 3, padding='same', activation='relu')(concat1)
    dec1 = layers.Conv2D(64, 3, padding='same', activation='relu')(dec1)
    
    # Output
    outputs = layers.Conv2D(num_classes, 1, padding='same', activation='sigmoid')(dec1)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# ==================== Training Function ====================
def train_model(model, train_generator, steps_per_epoch, params, output_dir):
    """Train the U-Net model"""
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["learningRate"]),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Training setup complete:")
    print("  - Loss function: binary_crossentropy")
    print(f"  - Optimizer: Adam (lr={params['learningRate']})")
    print(f"  - Steps per epoch: {steps_per_epoch}")
    print(f"  - Total epochs: {params['nEpoches']}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Callbacks
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(output_dir, 'best_model.h5'),
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    print(f"\nStarting training for {params['nEpoches']} epochs...")
    print(f"Checkpoint directory: {output_dir}\n")
    
    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=params['nEpoches'],
        callbacks=[checkpoint, reduce_lr],
        verbose=1
    )
    
    print("\nTraining completed!")
    print(f"Best checkpoint: {output_dir}/best_model.h5")
    
    return model, history


# ==================== Visualization ====================
def plot_training_history(history):
    """Plot training history"""
    
    if history is not None:
        epochs_range = range(1, len(history.history['loss']) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('TensorFlow Training Results')
        
        # Loss plot
        axes[0].plot(epochs_range, history.history['loss'], 'b-', linewidth=2, label='Training Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(epochs_range, history.history['accuracy'], 'g-', linewidth=2, label='Training Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("Training metrics plotted")
        print(f"  Final Loss: {history.history['loss'][-1]:.4f}")
        print(f"  Final Accuracy: {history.history['accuracy'][-1]:.4f}")
    else:
        print("No training history available")


# ==================== Metrics ====================
def dice(im1, im2, empty_score=1.0):
    """Compute Dice coefficient"""
    
    im1 = np.asarray(im1).astype(np.bool_)
    im2 = np.asarray(im2).astype(np.bool_)
    
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    
    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score
    
    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / im_sum


def compute_accuracy(test_mask, pred):
    """Compute accuracy score"""
    
    if test_mask is not None and pred is not None:
        try:
            test_mask_norm = test_mask / 255
            a = test_mask_norm.flatten()
            b = pred[0, :, :, 0].flatten()
            a = a.astype(int)
            b = b.astype(int)
            
            accuracy = accuracy_score(a, b)
            print(f"Testing Accuracy: {accuracy:.4f}")
            return accuracy
        except Exception as e:
            print(f"Could not calculate accuracy: {e}")
            return None
    else:
        print("Cannot calculate accuracy - missing data")
        return None


def compute_dice(test_mask, pred):
    """Compute Dice coefficient"""
    
    if test_mask is not None and pred is not None:
        try:
            dice_score = dice(test_mask, pred[0, :, :, 0])
            print(f"Dice Coefficient: {dice_score:.4f}")
            return dice_score
        except Exception as e:
            print(f"Could not calculate Dice: {e}")
            return None
    else:
        print("Cannot calculate Dice - missing data")
        return None


# ==================== Inference ====================
def make_prediction(model, dataset, params):
    """Make predictions on test data"""
    
    # Try to find test image/mask in common locations
    training_dir = os.path.join(dataset, "training", "images", "files")
    if not os.path.exists(training_dir):
        training_dir = os.path.join(dataset, "training", "images")
    
    # Look for any image file to test on (first non-mask image)
    test_image_path = None
    test_mask_path = None
    
    if os.path.exists(training_dir):
        all_files = sorted([f for f in os.listdir(training_dir) if f.endswith('.png')])
        for f in all_files:
            if not f.endswith('_mask.png'):
                test_image_path = os.path.join(training_dir, f)
                base_name = f.rsplit('.', 1)[0]
                test_mask_path = os.path.join(training_dir, f"{base_name}_mask.png")
                break
    
    test_image = None
    test_mask = None
    
    if test_image_path and os.path.exists(test_image_path):
        test_image = cv2.imread(test_image_path, 0)
        test_image = cv2.resize(test_image, (params["y"], params["x"]))
        mean_ = np.mean(test_image)
        test_image = test_image - mean_
        std = np.std(test_image)
        test_image = test_image / std
        test_image = np.expand_dims(np.expand_dims(test_image, axis=0), axis=-1)
        print(f"Test image loaded: {test_image.shape}")
    else:
        print(f"Test image not found")
    
    if test_mask_path and os.path.exists(test_mask_path):
        test_mask = cv2.imread(test_mask_path, 0)
        test_mask = cv2.resize(test_mask, (params["y"], params["x"]))
        print(f"Test mask loaded: {test_mask.shape}")
    else:
        print(f"Test mask not found")
    
    # Make prediction
    prediction = None
    pred = None
    if test_image is not None:
        prediction = model.predict(test_image)
        
        pred = copy.copy(prediction)
        pred[pred > 0.5] = 1
        pred[pred < 0.5] = 0
        print(f"Prediction generated: {pred.shape}")
    else:
        print("Cannot make prediction - no test image")
    
    return test_image, test_mask, prediction, pred


def plot_predictions(test_image, test_mask, prediction, pred):
    """Plot prediction results"""
    
    if test_image is not None and pred is not None and test_mask is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Prediction Results')
        
        if prediction is not None:
            im0 = axes[0, 0].imshow(prediction[0, :, :, 0], cmap='gray')
            axes[0, 0].set_title('Raw Prediction')
            plt.colorbar(im0, ax=axes[0, 0])
        
        axes[0, 1].imshow(pred[0, :, :, 0], cmap='gray')
        axes[0, 1].set_title('Thresholded Prediction')
        
        axes[1, 0].imshow(test_mask, cmap='gray')
        axes[1, 0].set_title('Ground Truth Mask')
        
        axes[1, 1].imshow(test_image[0, :, :, 0], cmap='gray')
        axes[1, 1].set_title('Test Image')
        
        plt.tight_layout()
        plt.show()
    else:
        print("Cannot display predictions - missing data")


# ==================== Main ====================
if __name__ == "__main__":
    print("="*60)
    print("TensorFlow U-Net Semantic Segmentation")
    print("="*60)
    
    # Load data
    Xlist, Ylist = load_training_data(dataset, params)
    X_train = Xlist['training']
    Y_train = Ylist['training']
    X_val = Xlist['validation']
    Y_val = Ylist['validation']
    
    # Create data generator
    train_generator, steps_per_epoch = create_data_generator(X_train, Y_train, params)
    
    # Create model
    model = create_unet_model(input_shape=(params['x'], params['y'], params['n_channels']), num_classes=1)
    print("\nU-Net model created")
    print(f"Total parameters: {model.count_params():,}\n")
    
    # Train model with output directory in results folder
    if dataset.startswith("/app/"):
        # Docker path
        output_dir = "/app/results/model_checkpoints_tensorflow"
    else:
        # Local path
        output_dir = os.path.join(os.path.dirname(os.path.dirname(dataset)), "results", "model_checkpoints_tensorflow")
    print(f"Model checkpoints will be saved to: {output_dir}")
    model, history = train_model(model, train_generator, steps_per_epoch, params, output_dir)
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model
    checkpoint_path = os.path.join(output_dir, "best_model.h5")
    if os.path.exists(checkpoint_path):
        model = keras.models.load_model(checkpoint_path)
        print(f"Best model weights loaded from: {checkpoint_path}")
    else:
        print(f"Checkpoint not found at: {checkpoint_path}")
    
    print("Model set to inference mode\n")
    
    # Make predictions
    test_image, test_mask, prediction, pred = make_prediction(model, dataset, params)
    
    # Plot predictions
    plot_predictions(test_image, test_mask, prediction, pred)
    
    # Compute metrics
    compute_accuracy(test_mask, pred)
    compute_dice(test_mask, pred)
    
    print("\n" + "="*60)
    print("TensorFlow NOTEBOOK EXECUTION COMPLETED")
    print("="*60)
