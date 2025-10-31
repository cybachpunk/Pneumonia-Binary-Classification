from pathlib import Path
import pydicom
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os

# --- 1. Data Loading and Preprocessing ---

labels = pd.read_csv("stage_2_train_labels.csv")
labels.head(5)

labels = labels.drop_duplicates("patientId")

root_path = Path("stage_2_train_images/")
save_path = Path("Processed")

# Plotting code
fig, axis = plt.subplots(3, 3, figsize=(9, 9))
c = 0
for i in range(3):
    for j in range(3):
        patient_id = labels.patientId.iloc[c]
        dcm_path = root_path / patient_id
        dcm_path = dcm_path.with_suffix(".dcm")
        dcm = pydicom.read_file(dcm_path).pixel_array
        label = labels["Target"].iloc[c]
        axis[i][j].imshow(dcm, cmap="bone")
        axis[i][j].set_title(label)
        c += 1

sums, sums_squared = 0, 0

# Setting split point as a clear variable
TRAIN_SPLIT_INDEX = 24000

# itertuples() for a clean and fast loop

for c, row in enumerate(tqdm(labels.itertuples(), total=len(labels))):
    patient_id = row.patientId
    dcm_path = root_path / patient_id
    dcm_path = dcm_path.with_suffix(".dcm")
    
    # Load the full DICOM file to read metadata
    dcm = pydicom.read_file(dcm_path)
    
    # --- Dynamic DICOM Normalization ---
    # DICOM images can have different bit depths (e.g., 10, 12, 16 bits)
    # Read the pixel data
    pixel_array = dcm.pixel_array
    
    # Find the bits stored (e.g., 10, 12, 16) from metadata
    # We use getattr() to provide a safe default (e.g., 10 or 12) 
    # if the tag is missing. For this dataset 10 is standard.
    bits_stored = getattr(dcm, 'BitsStored', 10)
    
    # Calculate the true maximum value for unsigned data
    # (e.g., 2**10 - 1 = 1023)
    max_pixel_value = (2**bits_stored) - 1
    
    # Normalize the array by the true max value, not 255
    # We add 1e-6 to avoid any potential divide-by-zero, though
    # max_pixel_value should never be zero here.
    normalized_array = pixel_array / (max_pixel_value + 1e-6)
    # --- End Normalization ---
    
    # Resize the correctly normalized array.
    dcm_array = cv2.resize(normalized_array, (224, 224)).astype(np.float16)
    
    label = row.Target
    
    train_or_val = "train" if c < TRAIN_SPLIT_INDEX else "val"
    
    current_save_path = save_path / train_or_val / str(label)
    current_save_path.mkdir(parents=True, exist_ok=True)
    np.save(current_save_path / patient_id, dcm_array)
    
    normalizer = 224 * 224
    if train_or_val == "train":
        sums += np.sum(dcm_array) / normalizer
        sums_squared += (dcm_array ** 2).sum() / normalizer

# Calculate mean and std based on the correctly normalized training data
mean = sums / TRAIN_SPLIT_INDEX
std = np.sqrt((sums_squared / TRAIN_SPLIT_INDEX) - mean**2)

print(f"CALCULATED MEAN (from {bits_stored}-bit data): {mean:.4f}")
print(f"CALCULATED STD (from {bits_stored}-bit data): {std:.4f}")   

# --- 2. Training: Data Loader ---

import torch
import torchvision
from torchvision import transforms
import torchmetrics
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
# from tqdm.notebook import tqdm # Already imported
import numpy as np
import matplotlib.pyplot as plt

def load_file(path):
    return np.load(path).astype(np.float32)

# create a list of transforms
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomAffine(degrees=(-5, 5), translate=(0, 0.05), scale=(0.9, 1.1)),
    transforms.RandomResizedCrop((224, 224), scale=(0.35, 1.0))
])

# only normalize on this transform
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_dataset = torchvision.datasets.DatasetFolder(
    "Processed/train/", 
    loader=load_file, 
    extensions="npy", 
    transform=train_transforms
)

val_dataset = torchvision.datasets.DatasetFolder(
    "Processed/val/", 
    loader=load_file, 
    extensions="npy", 
    transform=val_transforms
)

fig, axis = plt.subplots(2, 2, figsize=(9, 9))
for i in range(2):
    for j in range(2):
        random_index = np.random.randint(0, TRAIN_SPLIT_INDEX)
        x_ray, label = train_dataset[random_index]
        axis[i][j].imshow(x_ray[0], cmap="bone")
        axis[i][j].set_title(label)

batch_size = 64

# Dynamically setting num_workers, utilizing half available cores assuming other tasks to prevent
# overloading. This is maxed at 8 vCores for this example, but can be adjusted based on system 
# capabilities as well as other dataset variables within the classification use case. This increases
# efficiency and portability across different production environments.
num_workers = min(os.cpu_count() // 2, 8)

# Dataloader definitions: pin_memory=True, to speed up CPU-to-GPU data transfer.
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    num_workers=num_workers, 
    shuffle=True,
    pin_memory=True 
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    num_workers=num_workers, 
    shuffle=False,
    pin_memory=True
)

# --- 3. Training: Model Creation ---

class PneumoniaModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, pos_weight_val=3.0):
        super().__init__()
        # Save hyperparameters for logging and checkpointing
        self.save_hyperparameters()

        # Use pretrained weights. Even though X-rays are different from ImageNet, the initial layers 
        # (edge detectors, etc.) are often useful transfer-learning starting points. Use the modern 
        # 'weights' API here.
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        
        # Adapt the first layer for 1-channel (grayscale) input
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Adapt the final layer for 1-output (binary classification)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)
        
        # Define loss function using the hyperparameter for positive class weighting
        # The pos_weight handles the 3:1 imbalance in the Pneumonia dataset used, and can be adapted
        # to other datasets in Healthcare that experience imbalance issues in sources such as rare disease detection.
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hparams.pos_weight_val]))
        
        # Use MetricCollection for F1, Accuracy, etc. F1 is crucial for imbalanced datasets.
        # 'task="binary"' is required for torchmetrics >= 0.7
        metrics = MetricCollection([
            Accuracy(task="binary"),
            Precision(task="binary"),
            Recall(task="binary"),
            F1Score(task="binary")
        ])
        # Create separate metric instances for train and val
        self.train_metrics = metrics.clone(prefix='Train/')
        self.val_metrics = metrics.clone(prefix='Val/')

    def forward(self, data):
        return self.model(data)
        
    def training_step(self, batch, batch_idx):
        x_ray, label = batch
        label = label.float() # BCEWithLogitsLoss expects float labels
        pred = self(x_ray)[:, 0] # Squeeze output
        loss = self.loss_fn(pred, label)
        
        # Log loss and metrics with on_step/on_epoch
        self.log("Train/Loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Calculate metrics, passing torch.sigmoid(pred) probabilities to the metrics
        metrics_output = self.train_metrics(torch.sigmoid(pred), label.int())
        self.log_dict(metrics_output, on_step=False, on_epoch=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        x_ray, label = batch
        label = label.float()
        pred = self(x_ray)[:, 0]
        loss = self.loss_fn(pred, label)
        
        # Log validation loss
        self.log("Val/Loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Calculate validation metrics
        metrics_output = self.val_metrics(torch.sigmoid(pred), label.int())
        self.log_dict(metrics_output, on_step=False, on_epoch=True)
        
    # Pytorch Lightning handles the aggregation and logging automatically for epoch definitions

    def configure_optimizers(self):
        # Define optimizer and scheduler 
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

        # Define learning rate scheduler.
        # ReduceLROnPlateau will reduce the learning rate if the Val F1 score stops improving.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',      # We want to maximize F1-Score
            factor=0.1,      # Reduce LR by 90%
            patience=3,      # Wait 3 epochs of no improvement
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "Val/F1Score", # Monitor the F1 score
                "interval": "epoch",
                "frequency": 1,
            },
        }

# --- 4. Training: Model Training ---

# Use ModelCheckpoint callback to save best models based on Val F1-Score
checkpoint_callback = ModelCheckpoint(
    monitor="Val/F1Score", 
    mode="max",
    save_top_k=3, # Saving 10 is fine, 3-5 is also common
    dirpath="./checkpoints/",
    filename="pneumonia-epoch{epoch:02d}-f1{Val/F1Score:.4f}"
)

# Add a LearningRateMonitor callback to log the LR in TensorBoard
lr_monitor = LearningRateMonitor(logging_interval='epoch')

# Instantiate the model
model = PneumoniaModel()

# Use the 'devices' and 'accelerator' flags
# Set log_every_n_steps to 50 to reduce logging overhead and improve training speed.

trainer = pl.Trainer(
    accelerator="auto", # Automatically uses "gpu" if available, else "cpu"
    devices=1,
    logger=TensorBoardLogger(save_dir="./logs"), 
    log_every_n_steps=50, 
    callbacks=[checkpoint_callback, lr_monitor], 
    max_epochs=35,
    # Add precision=16 for mixed-precision training
    # This trains ~2x faster and uses ~half the VRAM with minimal/no accuracy loss
    precision="16-mixed" 
)

# Start training
print("--- Starting Model Training ---")
trainer.fit(model, train_loader, val_loader)

# --- 5. Model Evaluation ---


best_model_path = checkpoint_callback.best_model_path
if not best_model_path:
    print("No best model found, using last checkpoint.")
    # If no best model (e.g., if training was interrupted), use last checkpoint
    best_model_path = trainer.checkpoint_callback.last_model_path   

print(f"Loading best model from: {best_model_path}")

# Again, this check assumes Nvidia GPU with CUDA is used. There may be cases where Apple Silicon or AMD GPUs are used.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model from the best checkpoint
model = PneumoniaModel.load_from_checkpoint(best_model_path)
model.eval()
model.to(device)

preds = []
labels = []

# Use val_loader for batched inference
with torch.no_grad():
    for batch in tqdm(val_loader):
        data, label = batch
        data = data.to(device)
        
        pred = model(data)
        pred = torch.sigmoid(pred[:, 0]).cpu() # Get probabilities
        
        preds.append(pred)
        labels.append(label)

# Concatenate all batch results
preds = torch.cat(preds)
labels = torch.cat(labels).int()

# Calculate all relevant metrics
acc = torchmetrics.Accuracy(task="binary")(preds, labels)
precision = torchmetrics.Precision(task="binary")(preds, labels)
recall = torchmetrics.Recall(task="binary")(preds, labels)
f1 = torchmetrics.F1Score(task="binary")(preds, labels)
cm = torchmetrics.ConfusionMatrix(num_classes=2, task="binary")(preds, labels)

print(f"Val Accuracy: {acc:.4f}")
print(f"Val Precision: {precision:.4f}")
print(f"Val Recall: {recall:.4f}")
print(f"Val F1-Score: {f1:.4f}") # <-- The most important metric
print(f"Confusion Matrix: \n {cm}")

# --- 6. Model Interpretability and Transparency (CAM) ---

# Below is an example of generating Class Activation Maps (CAM)
# for interpretability using the best model with Pytorch Lightning Hooks
# to capture Feature Maps from the last convolutional layer. Hooks are helpful 
# in reducing code duplication, improving modularity, and reducing GPU memory consumption
# Especially for larger models that span beyond the dataset at hand.

# Define a helper class to store extracted features
class FeatureExtractor:
    def __init__(self, model, layer_name):
        self.features = None

    def __call__(self, module, input, output):
        # store the output features
        self.features = output.detach()

# Instantiate the feature extractor for the last conv layer
feature_extractor = FeatureExtractor()
handle = model.model.layer4.register_forward_hook(feature_extractor)

# Rewrite the get_cam function to work with the Lightning model
def get_cam(model, img):
    with torch.no_grad():
        # Run forward pass. The hook will automatically capture
        # Output of 'layer4' in feature_extractor.features
        pred = model(img.unsqueeze(0).to(device))
                     
    # Get captured feature maps. Shape is [1,512,7,7]
    features = feature_extractor.features.cpu()

    # reshape for matrix multiplication
    features = features.reshape(features.shape[512], 49)

    # Get weights from the final fully connected layer
    weights = model.model.fc.weight.data.cpu()[0]

    # Calculate CAM
    cam = torch.matmul(weight, features_reshaped)
    cam_img = cam.reshape(7, 7)

    return cam_img, torch.sigmoid(pred.cpu())

def visualize(img, cam, pred, actual_label):
    img = img[0] # Remove channel dim
    # Resize 7x7 CAM to 224x224
    cam = transforms.functional.resize(cam.unsqueeze(0), (224,224))[0]

    fig, axis = plt.subplots(1, 2, figsize=(10, 5))

    actual_str = "Pneumonia" if actual_label == 1 else "Normal"
    axis[0].imshow(img, cmap="bone")
    axis[0].set_title(f"Original (Actual: {actual_str})")
    axis[0].axis("off")

    axis[1].imshow(img, cmap="bone")
    axis[1].imshow(cam, alpha=0.5, cmap="jet")

    pred_val = pred.item()
    pred_str = "Pneumonia" if pred_val >= 0.5 else "Normal"
    axis[1].set_title(f"CAM Overlay (Pred: {pred_str} [{pred_val:.3f}])")
    axis[1].axis("off")

    plt.show()

# Create a cleaned validation dataset for visualization
val_dataset_viz = torchvision.datasets.DatasetFolder(
    "Processed/val/", 
    loader=load_file, 
    extensions="npy", 
    transform=val_transforms
)

# Run visualizations on random samples from the validation set
for i in [-1, -5, -10, -20]:
    img, label = val_dataset_viz[i]
    activation_map, pred = get_cam(model, img)
    visualize(img, activation_map, pred, label)

# Remove the hook to prevent any memory leaks
handle.remove()
print("CAM visualizations complete and hook removed.")