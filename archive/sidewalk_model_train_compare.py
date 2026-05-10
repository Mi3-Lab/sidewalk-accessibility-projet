import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ultralytics import YOLO
from scipy import ndimage
import random

# ============================================================
# 1. Mobility Classes
# ============================================================
mobility_classes = ["Cane", "Walker", "MobilityScooter", "ManualWheelchair", "MotorizedWheelchair"]

# ============================================================
# 2. FIXED Aggregate Ratings
# ============================================================
def normalize_mobility_name(name: str) -> str:
    """Normalize variations like 'Manual wheelchair' vs 'ManualWheelchair'."""
    key = name.strip().lower().replace(" ", "")
    mapping = {
        "manualwheelchair": "ManualWheelchair",
        "mobilityscooter": "MobilityScooter",
        "motorizedwheelchair": "MotorizedWheelchair",
        "walker": "Walker",
        "walkingcane": "Cane",  
        "cane": "Cane",
    }
    return mapping.get(key, name)


def aggregate_ratings_from_q(image_q_csv, mobility_classes):
    """
    Aggregate Q scores (out of 10) per ImageID and MobilityAid.
    Q represents accessibility score (higher = more accessible).
    """
    df = pd.read_csv(image_q_csv)

    # Normalize names 
    df["MobilityAid"] = df["MobilityAid"].apply(normalize_mobility_name)

    # Handle missing or invalid values
    df = df[df["Q"].notnull()].copy()

    # Clip Q values to [0,10] and normalize to [0,1] for training
    df["Q_norm"] = df["Q"].clip(0, 10) / 10.0

    # Group by ImageID and MobilityAid and average if multiple entries exist
    pivot = df.groupby(["ImageID", "MobilityAid"])["Q_norm"].mean().unstack(fill_value=0.25)

    # Merge duplicate columns , sometimes happens after grouping
    pivot.columns = [normalize_mobility_name(c) for c in pivot.columns]
    pivot = pivot.T.groupby(level=0).mean().T

    pivot = pivot.reset_index()
    return pivot


# ============================================================
# 3. Dataset Class
# ============================================================
class SidewalkDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, mask_model=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.mask_model = mask_model
        self.mobility_classes = mobility_classes

        # Build mapping {imageID: filepath}
        self.image_map = {}
        for path in glob.glob(os.path.join(image_dir, "*.png")):
            filename = os.path.basename(path)
            parts = filename.split("-")
            if len(parts) >= 4:
                image_id = parts[2]
                self.image_map[image_id] = path

        # Filter out rows where we don't have images
        valid_ids = set(self.image_map.keys())
        self.df = self.df[self.df["ImageID"].astype(str).isin(valid_ids)].reset_index(drop=True)
        print(f"Dataset size after filtering: {len(self.df)} samples")

    def get_accessibility_vector(self, row):
        vector = row[self.mobility_classes].values.astype(np.float32)
        return torch.tensor(vector, dtype=torch.float32)

    def extract_sidewalk_crop(self, scene_image, crop_size=(224, 224)):
        """Extract sidewalk region using YOLO model"""
        if self.mask_model is None:
            return scene_image.resize(crop_size)
            
        results = self.mask_model.predict(scene_image, classes=[1], conf=0.5, verbose=False)
        mask_arr = np.zeros((scene_image.height, scene_image.width), dtype=bool)

        if results and results[0].masks is not None:
            masks_data = results[0].masks.data
            composite_mask = torch.any(masks_data, dim=0).cpu().numpy().astype("uint8")
            mask_pil = Image.fromarray(composite_mask * 255).convert("L")
            if mask_pil.size != scene_image.size:
                mask_pil = mask_pil.resize(scene_image.size, resample=Image.NEAREST)
            mask_arr = np.array(mask_pil) > 0

        if mask_arr.sum() == 0:
            # No sidewalk detected, use center crop
            w, h = scene_image.size
            crop_w, crop_h = min(w, h), min(w, h)
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
            sidewalk_crop = scene_image.crop((left, top, left + crop_w, top + crop_h))
        else:
            # Find largest connected component
            labeled, ncomponents = ndimage.label(mask_arr)
            if ncomponents > 0:
                sizes = ndimage.sum(mask_arr, labeled, range(1, ncomponents + 1))
                largest_component = (labeled == (np.argmax(sizes) + 1))
                ys, xs = np.where(largest_component)
                miny, maxy = ys.min(), ys.max()
                minx, maxx = xs.min(), xs.max()
                
                # Add padding
                pad = 16
                minx = max(0, minx - pad)
                miny = max(0, miny - pad)
                maxx = min(scene_image.width - 1, maxx + pad)
                maxy = min(scene_image.height - 1, maxy + pad)
                
                sidewalk_crop = scene_image.crop((minx, miny, maxx + 1, maxy + 1))
            else:
                sidewalk_crop = scene_image.copy()

        return sidewalk_crop.resize(crop_size, resample=Image.BILINEAR)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(int(row["ImageID"]))

        image_path = self.image_map.get(image_id)
        if image_path is None:
            raise FileNotFoundError(f"No image found for ImageID={image_id}")

        scene_image = Image.open(image_path).convert("RGB")
        sidewalk_image = self.extract_sidewalk_crop(scene_image)

        # Apply transforms
        if self.transform:
            scene_tensor = self.transform(scene_image)
            sidewalk_tensor = self.transform(sidewalk_image)
        else:
            to_tensor = transforms.ToTensor()
            scene_tensor = to_tensor(scene_image)
            sidewalk_tensor = to_tensor(sidewalk_image)

        target = self.get_accessibility_vector(row)

        return {
            "scene": scene_tensor,
            "sidewalk": sidewalk_tensor,
            "target": target,
            "image_id": image_id
        }

    def __len__(self):
        return len(self.df)

# ============================================================
# 4. Model Architecture
# ============================================================
class PatchEncoder(nn.Module):
    def __init__(self, input_channels=3, embed_dim=256, img_size=224, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.conv = nn.Conv2d(input_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embeddings
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B = x.shape[0]
        
        # Create patches
        x = self.conv(x)  # B x embed_dim x H' x W'
        x = x.flatten(2).transpose(1, 2)  # B x num_patches x embed_dim
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # Normalize and dropout
        x = self.norm(x)
        x = self.dropout(x)
        
        return x

class CrossAttentionBlock(nn.Module):
    # 8 num heads? Not sure how many to use, 0.1 dropout seems fine 
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed forward
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, sidewalk_tokens, scene_tokens):
        # Cross attention: sidewalk queries scene
        attn_out, _ = self.cross_attn(
            query=sidewalk_tokens, 
            key=scene_tokens, 
            value=scene_tokens
        )
        sidewalk_tokens = self.norm1(sidewalk_tokens + attn_out)
        
        # Feed forward
        ff_out = self.ff(sidewalk_tokens)
        sidewalk_tokens = self.norm2(sidewalk_tokens + ff_out)
        
        return sidewalk_tokens

class AccessibilityModel(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Encoders
        self.scene_encoder = PatchEncoder(embed_dim=embed_dim)
        self.sidewalk_encoder = PatchEncoder(embed_dim=embed_dim)
        
        # Cross attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # Output head
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, len(mobility_classes)),
            nn.Sigmoid()  # Ensure outputs are in [0, 1] range
            # TODO: Replace with tangential hyperbolic for -1 to 1 range
        )

    def forward(self, scene, sidewalk):
        # Encode both images
        scene_tokens = self.scene_encoder(scene)
        sidewalk_tokens = self.sidewalk_encoder(sidewalk)
        
        # Remove CLS token from scene for key/value
        scene_kv = scene_tokens[:, 1:, :]
        
        # Apply cross attention layers
        for layer in self.cross_attn_layers:
            sidewalk_tokens = layer(sidewalk_tokens, scene_kv)
        
        # Use CLS token for final prediction
        cls_token = sidewalk_tokens[:, 0, :]
        cls_token = self.output_norm(cls_token)
        
        output = self.output_head(cls_token)
        return output

# ============================================================
# 5. Training Setup
# ============================================================
def create_data_loaders(csv_path, image_dir, mask_model, batch_size=16, test_size=0.2):
    """Create train/validation data loaders"""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load full dataset
    full_dataset = SidewalkDataset(csv_path, image_dir, train_transform, mask_model)
    
    # Split indices manually
    indices = list(range(len(full_dataset)))
    random.seed(42)
    random.shuffle(indices)
    
    split_idx = int(len(indices) * (1 - test_size))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

def train_model():
    """Main training function"""
    
    # Regenerate the aggregated data
    print("Regenerating aggregated data with Q scores...")
    aggregated_df = aggregate_ratings_from_q("Project_Sidewalk_Data/image_comparison.csv", mobility_classes)
    aggregated_df.to_csv("Project_Sidewalk_Data/aggregated_q.csv", index=False)
    print(f"Aggregated data saved. Shape: {aggregated_df.shape}")

    
    # Print some statistics
    print("\nAccessibility score statistics:")
    for col in mobility_classes:
        scores = aggregated_df[col]
        print(f"{col}: mean={scores.mean():.3f}, std={scores.std():.3f}, min={scores.min():.3f}, max={scores.max():.3f}")
    
    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load mask model
    mask_model = YOLO("bestv12.pt")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        "Project_Sidewalk_Data/aggregated_q.csv",
        "Project_Sidewalk_Data/sidewalk-images",
        mask_model,
        batch_size=8
    )

    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize model
    model = AccessibilityModel(embed_dim=256, num_heads=8, num_layers=3).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    num_epochs = 10
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            scene = batch["scene"].to(device)
            sidewalk = batch["sidewalk"].to(device)
            target = batch["target"].to(device)
            
            optimizer.zero_grad()
            output = model(scene, sidewalk)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                scene = batch["scene"].to(device)
                sidewalk = batch["sidewalk"].to(device)
                target = batch["target"].to(device)
                
                output = model(scene, sidewalk)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "mobility_classes": mobility_classes
            }, "accessibility_checkpoint_compare.pth")
            print(f"New best model saved! Val Loss: {best_val_loss:.4f}")
    
    # Save training history
    import json
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    with open('training_history.json', 'w') as f:
        json.dump(history, f)
    
    print("Training completed!")
    return model

if __name__ == "__main__":
    model = train_model()