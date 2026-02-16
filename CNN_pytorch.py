"""
PyTorch implementation for semantic segmentation using U-Net
Dataset: Medical image segmentation
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import copy
from sklearn.metrics import accuracy_score


# ==================== Configuration ====================
# Default dataset path
DEFAULT_DATASET = "/home/noya/Documenti/Datasets/Segmentation"

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
    """Load training images and masks"""
    
    modality = ['training']
    
    # Calculate dataset lengths
    training_dir = os.path.join(dataset_path, "training/image/files")
    if os.path.exists(training_dir):
        all_files = os.listdir(training_dir)
        image_files = [f for f in all_files if not f.endswith('_mask.png') and f.endswith('.png')]
        length_training = len(image_files)
    else:
        length_training = 0
        image_files = []
    
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
    for im in sorted(image_files):
        if not im.endswith('_mask.png'):
            image_path = os.path.join(training_dir, im)
            mask_path = os.path.join(training_dir, im.replace('.png', '_mask.png'))
            
            # Load and process image
            image = cv2.imread(image_path, 0)
            if image is not None:
                image = cv2.resize(image, dim)
                mea = np.mean(image)
                ss = np.std(image)
                image = (image - mea) / ss
                X_train[ipp, :, :, 0] = image
            
            # Load and process mask
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, 0)
                if mask is not None:
                    mask = cv2.resize(mask, dim)
                    mask = mask / 255
                    y_train[ipp, :, :, 0] = mask
            
            ipp += 1
    
    Xlist['training'] = X_train
    Ylist['training'] = y_train
    print(f"Training data loaded: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    
    # Create empty validation arrays
    X_val = np.empty((0, params['x'], params['y'], params['n_channels']))
    y_val = np.empty((0, params['x'], params['y'], params['n_channels_mask']))
    Xlist['validation'] = X_val
    Ylist['validation'] = y_val
    print("Validation data: empty (no validation files found)")
    
    return Xlist, Ylist


# ==================== PyTorch Dataset ====================
class SegmentationDataset(Dataset):
    """Custom PyTorch Dataset for semantic segmentation"""
    
    def __init__(self, images, masks, augment=False):
        self.images = images
        self.masks = masks
        self.augment = augment
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32).copy()
        mask = self.masks[idx].astype(np.float32).copy()
        
        # Augmentation: random flips
        if self.augment:
            if np.random.rand() > 0.5:
                image = np.ascontiguousarray(np.fliplr(image))
                mask = np.ascontiguousarray(np.fliplr(mask))
            
            if np.random.rand() > 0.5:
                image = np.ascontiguousarray(np.flipud(image))
                mask = np.ascontiguousarray(np.flipud(mask))
        
        # Convert to PyTorch tensors (H, W, C -> C, H, W)
        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        mask = torch.from_numpy(np.ascontiguousarray(mask)).permute(2, 0, 1)
        
        return image, mask


# ==================== U-Net Model ====================
class UNet(nn.Module):
    """U-Net architecture for semantic segmentation"""
    
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.enc4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        self.dropout = nn.Dropout(0.5)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Output
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        enc4 = self.enc4(x)
        x = self.pool4(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)
        x = self.dropout(x)
        
        # Decoder with skip connections
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # Output
        x = self.final(x)
        x = self.sigmoid(x)
        return x


# ==================== Training Function ====================
def train_model(model, train_loader, params, device, output_dir):
    """Train the U-Net model"""
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learningRate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    steps_per_epoch = int(np.ceil(params["length_training"] / params["batch_size"]))
    
    print("Training setup complete:")
    print("  - Loss function: BCELoss")
    print(f"  - Optimizer: Adam (lr={params['learningRate']})")
    print("  - LR Scheduler: ReduceLROnPlateau")
    print(f"  - Steps per epoch: {steps_per_epoch}")
    print(f"  - Total epochs: {params['nEpoches']}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Training history
    train_losses = []
    train_accuracies = []
    
    print(f"\nStarting training for {params['nEpoches']} epochs...")
    print(f"Checkpoint directory: {output_dir}\n")
    
    best_loss = float('inf')
    
    for epoch in range(params['nEpoches']):
        model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Metrics
            epoch_loss += loss.item()
            batch_pred = (outputs > 0.5).float()
            batch_acc = (batch_pred == masks).float().mean()
            epoch_accuracy += batch_acc.item()
            num_batches += 1
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{params['nEpoches']}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # Average metrics
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        train_losses.append(avg_loss)
        train_accuracies.append(avg_accuracy)
        
        # Learning rate scheduler
        scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved (loss: {avg_loss:.4f}) at {checkpoint_path}")
        
        print(f"\nEpoch {epoch+1}/{params['nEpoches']} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}\n")
    
    print("\nTraining completed!")
    print(f"Final loss: {train_losses[-1]:.4f}")
    print(f"Best checkpoint: {output_dir}/best_model.pth")
    
    history = type('obj', (object,), {
        'history': {
            'loss': train_losses,
            'accuracy': train_accuracies,
            'val_loss': train_losses,
            'val_accuracy': train_accuracies,
        }
    })()
    
    return model, history


# ==================== Visualization ====================
def plot_training_history(history):
    """Plot training history"""
    
    if history is not None and hasattr(history, 'history'):
        epochs_range = range(1, len(history.history['loss']) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('PyTorch Training Results')
        
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
def make_prediction(model, device, dataset, params):
    """Make predictions on test data"""
    
    test_image_path = os.path.join(dataset, "MCUCXR_0254_imm.png")
    test_mask_path = os.path.join(dataset, "MCUCXR_0254.png")
    
    test_image = None
    test_mask = None
    
    if os.path.exists(test_image_path):
        test_image = cv2.imread(test_image_path, 0)
        test_image = cv2.resize(test_image, (params["y"], params["x"]))
        mean_ = np.mean(test_image)
        test_image = test_image - mean_
        std = np.std(test_image)
        test_image = test_image / std
        test_image = torch.from_numpy(test_image).float().unsqueeze(0).unsqueeze(0).to(device)
        print(f"Test image loaded: {test_image.shape}")
    else:
        print(f"Test image not found: {test_image_path}")
    
    if os.path.exists(test_mask_path):
        test_mask = cv2.imread(test_mask_path, 0)
        test_mask = cv2.resize(test_mask, (params["y"], params["x"]))
        print(f"Test mask loaded: {test_mask.shape}")
    else:
        print(f"Test mask not found: {test_mask_path}")
    
    # Make prediction
    prediction = None
    pred = None
    if test_image is not None:
        model.eval()
        with torch.no_grad():
            prediction = model(test_image).cpu().numpy()
        
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
    print("PyTorch U-Net Semantic Segmentation")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    Xlist, Ylist = load_training_data(dataset, params)
    X_train = Xlist['training']
    Y_train = Ylist['training']
    X_val = Xlist['validation']
    Y_val = Ylist['validation']
    
    # Create data loaders
    train_dataset = SegmentationDataset(X_train, Y_train, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=0)
    
    print("PyTorch Data augmentation pipeline created")
    print("Training dataloader ready:")
    print(f"  - Batch size: {params['batch_size']}")
    print(f"  - Total batches: {len(train_loader)}")
    print("  - Augmentation: Enabled (random flip)\n")
    
    # Create model
    model = UNet(in_channels=1, out_channels=1).to(device)
    print(f"U-Net model created (device: {device})")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Train model
    output_dir = "/home/rocco/model_checkpoints_pytorch"
    model, history = train_model(model, train_loader, params, device, output_dir)
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model and make predictions
    checkpoint_path = f"{output_dir}/best_model.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Best model weights loaded from: {checkpoint_path}")
    else:
        print(f"Checkpoint not found at: {checkpoint_path}")
    
    model.eval()
    print("Model set to evaluation mode\n")
    
    # Make predictions
    test_image, test_mask, prediction, pred = make_prediction(model, device, dataset, params)
    
    # Plot predictions
    plot_predictions(test_image, test_mask, prediction, pred)
    
    # Compute metrics
    compute_accuracy(test_mask, pred)
    compute_dice(test_mask, pred)
    
    print("\n" + "="*60)
    print("PyTorch NOTEBOOK EXECUTION COMPLETED")
    print("="*60)
