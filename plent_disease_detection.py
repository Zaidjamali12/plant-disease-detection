import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
import os
import json
import pandas as pd
import pickle
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
import torchvision.transforms as T
import warnings
warnings.filterwarnings('ignore')
import traceback
import random
from collections import OrderedDict, Counter
import threading
import time
import sys

# ==================== CONFIGURATION ====================
DESKTOP_PATH = r"C:\Users\Abdul Sattar\OneDrive\Desktop"
DATASET_PATH = r"C:\Users\Abdul Sattar\OneDrive\Desktop\plant_disease_dataset\PlantVillageDataset\PlantVillage"

SYSTEM_BASE = os.path.join(DESKTOP_PATH, "PlantDiseaseSystem")
SYSTEM_DIRS = {
    'models': os.path.join(SYSTEM_BASE, "models"),
    'results': os.path.join(SYSTEM_BASE, "results"),
    'predictions': os.path.join(SYSTEM_BASE, "predictions"),
    'logs': os.path.join(SYSTEM_BASE, "logs"),
    'temp': os.path.join(SYSTEM_BASE, "temp")
}

# Model configuration - OPTIMIZED FOR 16 CLASSES
MODEL_CONFIG = {
    'input_size': 224,
    'pretrained': True,
    'dropout': 0.3,
    'freeze_layers': True  # Will freeze entire backbone
}

# Training configuration - OPTIMIZED FOR FINE-TUNING
TRAIN_CONFIG = {
    'default_epochs': 25,
    'default_batch_size': 16,
    'default_lr': 0.0001,  # Lower learning rate for fine-tuning
    'weight_decay': 0.01,
    'label_smoothing': 0.1,
    'patience': 10
}

# ==================== COLOR SCHEME ====================
COLORS = {
    'bg': '#f8f9fa',
    'primary': '#2ecc71',
    'primary_dark': '#27ae60',
    'secondary': '#34495e',
    'accent': '#e74c3c',
    'info': '#3498db',
    'warning': '#f39c12',
    'success': '#28a745',
    'danger': '#dc3545',
    'light': '#ecf0f1',
    'dark': '#2c3e50',
    'white': '#ffffff',
    'gray': '#95a5a6'
}

# ==================== DATASET MANAGER ====================
class DatasetManager:
    """Manages dataset operations and validation"""
    
    @staticmethod
    def setup_directories():
        """Create all necessary directories"""
        print("\n" + "="*60)
        print("SETTING UP SYSTEM DIRECTORIES")
        print("="*60)
        
        for dir_name, dir_path in SYSTEM_DIRS.items():
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ Created: {dir_path}")
        
        os.makedirs(DATASET_PATH, exist_ok=True)
        print("="*60)
    
    @staticmethod
    def check_dataset_status():
        """Check if dataset is ready for training"""
        print(f"\nChecking dataset at: {DATASET_PATH}")
        
        if not os.path.exists(DATASET_PATH):
            return False, "Dataset directory not found"
        
        class_stats = {}
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        total_images = 0
        
        try:
            items = os.listdir(DATASET_PATH)
        except:
            return False, "Cannot access dataset directory"
        
        class_dirs = []
        
        for item in items:
            item_path = os.path.join(DATASET_PATH, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                image_count = 0
                
                for root, dirs, files in os.walk(item_path):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in image_extensions):
                            image_count += 1
                
                if image_count > 0:
                    class_dirs.append(item)
                    class_stats[item] = image_count
                    total_images += image_count
        
        if len(class_dirs) < 2:
            return False, f"Need at least 2 classes. Found: {len(class_dirs)} classes"
        
        if total_images < 10:
            return False, f"Need more images. Total: {total_images} images"
        
        return True, {
            'classes': class_dirs,
            'class_stats': class_stats,
            'total_images': total_images,
            'num_classes': len(class_dirs)
        }
    
    @staticmethod
    def get_class_distribution():
        """Get detailed class distribution"""
        success, info = DatasetManager.check_dataset_status()
        if success:
            return info['class_stats']
        return {}

# ==================== ENHANCED DATASET ====================
class PlantDataset(Dataset):
    """Custom dataset for plant disease images"""
    
    def __init__(self, root_dir, transform=None, is_training=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_training = is_training
        self.classes = []
        self.class_to_idx = {}
        self.images = []
        self.labels = []
        self._load_images()
    
    def _load_images(self):
        """Load images from dataset directory recursively"""
        if not os.path.exists(self.root_dir):
            return
        
        # Get valid class directories
        class_dirs = []
        try:
            items = os.listdir(self.root_dir)
        except:
            return
        
        for item in items:
            item_path = os.path.join(self.root_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Check if directory contains images
                has_images = False
                try:
                    for root, dirs, files in os.walk(item_path):
                        for file in files:
                            if any(file.lower().endswith(ext) for ext in 
                                   ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']):
                                has_images = True
                                break
                        if has_images:
                            break
                except:
                    continue
                
                if has_images:
                    class_dirs.append(item)
        
        if not class_dirs:
            return
        
        class_dirs.sort()
        self.classes = class_dirs
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all images
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            try:
                for root, dirs, files in os.walk(class_path):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in 
                               ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']):
                            img_path = os.path.join(root, file)
                            self.images.append(img_path)
                            self.labels.append(class_idx)
            except:
                continue
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return black image as placeholder
            image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                image = self.transform(image)
            return image, label

# ==================== ADVANCED DISEASE DETECTOR ====================
class AdvancedPlantDiseaseDetector:
    """Core model for plant disease detection with 16-class classification"""
    
    def __init__(self, root_window=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.root = root_window
        self.model = None
        self.class_names = []
        self.num_classes = 0
        self.transform = None
        self.history = {
            'train_loss': [], 
            'train_acc': [], 
            'val_loss': [], 
            'val_acc': [],
            'best_acc': 0,
            'learning_rates': []
        }
        self.model_path = os.path.join(SYSTEM_DIRS['models'], 'best_model.pth')
        self.classes_file = os.path.join(SYSTEM_DIRS['models'], 'classes.json')
        
        print(f"\n{'='*60}")
        print("🌿 PLANT DISEASE DETECTOR INITIALIZING...")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Dataset Path: {DATASET_PATH}")
        
        self.setup_transforms()
        self.load_existing_model()
        print(f"{'='*60}\n")
    
    def setup_transforms(self):
        """Setup enhanced image transformations"""
        # Validation transform - consistent preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((MODEL_CONFIG['input_size'], MODEL_CONFIG['input_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Training transform with augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(MODEL_CONFIG['input_size']),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def load_existing_model(self):
        """Load existing trained model"""
        if not os.path.exists(self.model_path):
            print("✗ No trained model found. Please train the model first.")
            return False
        
        if not os.path.exists(self.classes_file):
            print("✗ No classes file found.")
            return False
        
        try:
            # Load class names
            with open(self.classes_file, 'r') as f:
                data = json.load(f)
                self.class_names = data['classes']
                self.num_classes = len(self.class_names)
            
            # Load model - try EfficientNet first
            self.model = self.create_efficientnet_model(self.num_classes)
            if self.model is None:
                self.model = self.create_resnet50_model(self.num_classes)
            
            # Load weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'history' in checkpoint:
                    self.history = checkpoint['history']
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ Model loaded successfully!")
            print(f"✓ Classes: {self.num_classes}")
            print(f"✓ Best Accuracy: {self.history.get('best_acc', 0):.2f}%")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            self.model = None
            return False
    
    # ============= FIXED MODEL CREATION FOR 16 CLASSES =============
    def create_efficientnet_model(self, num_classes):
        """Create EfficientNet-B0 model for 16-class classification"""
        try:
            print(f"Creating EfficientNet-B0 model for {num_classes} classes...")
            
            # Use weights parameter instead of pretrained (for newer PyTorch)
            model = models.efficientnet_b0(weights='IMAGENET1K_V1')
            
            # Get input features - EfficientNet classifier[1] is Linear layer
            in_features = model.classifier[1].in_features
            print(f"✓ Input features: {in_features}")
            
            # FREEZE ENTIRE BACKBONE
            for param in model.parameters():
                param.requires_grad = False
            
            # REPLACE CLASSIFIER with simple linear layer for 16 classes
            model.classifier = nn.Sequential(
                nn.Dropout(p=MODEL_CONFIG['dropout']),
                nn.Linear(in_features, num_classes)
            )
            
            # UNFREEZE ONLY THE CLASSIFIER
            for param in model.classifier.parameters():
                param.requires_grad = True
            
            print(f"✓ EfficientNet-B0 model created with {num_classes} classes")
            print(f"✓ Backbone frozen, only classifier trainable")
            return model
            
        except Exception as e:
            print(f"Error creating EfficientNet: {e}")
            print(f"Error type: {type(e)}")
            print("Falling back to ResNet50...")
            return self.create_resnet50_model(num_classes)
    
    def create_resnet50_model(self, num_classes):
        """Create ResNet50 model as backup"""
        try:
            print(f"Creating ResNet50 model for {num_classes} classes...")
            
            # Use weights parameter instead of pretrained
            model = models.resnet50(weights='IMAGENET1K_V1')
            
            # FREEZE BACKBONE
            for param in model.parameters():
                param.requires_grad = False
            
            # REPLACE FC LAYER
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=MODEL_CONFIG['dropout']),
                nn.Linear(in_features, num_classes)
            )
            
            # UNFREEZE CLASSIFIER
            for param in model.fc.parameters():
                param.requires_grad = True
            
            print(f"✓ ResNet50 model created with {num_classes} classes")
            return model
            
        except Exception as e:
            print(f"Error creating ResNet50: {e}")
            return None
    
    def count_trainable_params(self):
        """Count trainable parameters safely"""
        try:
            if self.model is not None:
                return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            else:
                return 0
        except:
            return 0
    
    def train_model(self, epochs=None, batch_size=None, learning_rate=None, 
                   progress_callback=None, status_callback=None):
        """Train model with enhanced accuracy for 16 classes"""
        
        # Use default values if not provided
        epochs = epochs or TRAIN_CONFIG['default_epochs']
        batch_size = batch_size or TRAIN_CONFIG['default_batch_size']
        learning_rate = learning_rate or TRAIN_CONFIG['default_lr']
        
        if status_callback:
            status_callback("Preparing dataset...")
        
        # Check dataset
        success, dataset_info = DatasetManager.check_dataset_status()
        if not success:
            return False, f"Dataset Error: {dataset_info}"
        
        self.class_names = dataset_info['classes']
        self.num_classes = dataset_info['num_classes']
        
        print(f"\n{'='*60}")
        print("🚀 STARTING MODEL TRAINING FOR 16 CLASSES")
        print(f"{'='*60}")
        print(f"Dataset: {DATASET_PATH}")
        print(f"Classes: {self.num_classes}")
        print(f"Total Images: {dataset_info['total_images']}")
        print(f"Epochs: {epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        try:
            # Create datasets
            if status_callback:
                status_callback("Loading dataset...")
            
            full_dataset = PlantDataset(DATASET_PATH, transform=self.train_transform, is_training=True)
            
            if len(full_dataset) == 0:
                return False, "No images found in dataset!"
            
            # Split dataset
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
            
            # Apply validation transform
            val_dataset.dataset.transform = self.transform
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                    shuffle=True, num_workers=0, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                                  shuffle=False, num_workers=0, pin_memory=True)
            
            print(f"✓ Training samples: {len(train_dataset)}")
            print(f"✓ Validation samples: {len(val_dataset)}")
            
        except Exception as e:
            return False, f"Failed to load dataset: {str(e)}"
        
        # Create model - try EfficientNet first
        if status_callback:
            status_callback("Creating model...")
        
        self.model = self.create_efficientnet_model(self.num_classes)
        if self.model is None:
            self.model = self.create_resnet50_model(self.num_classes)
            if self.model is None:
                return False, "Failed to create model!"
        
        self.model.to(self.device)
        
        # Count trainable parameters
        trainable_params = self.count_trainable_params()
        print(f"✓ Trainable parameters: {trainable_params:,}")
        
        # Loss function with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=TRAIN_CONFIG['label_smoothing'])
        
        # OPTIMIZER - ONLY TRAIN CLASSIFIER (FROZEN BACKBONE)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=TRAIN_CONFIG['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None
        
        self.history = {
            'train_loss': [], 
            'train_acc': [], 
            'val_loss': [], 
            'val_acc': [],
            'best_acc': 0,
            'learning_rates': []
        }
        
        print("\n📊 Training Progress:")
        print("-" * 80)
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                # Update progress
                if progress_callback:
                    progress = (epoch * len(train_loader) + batch_idx + 1) / (epochs * len(train_loader)) * 100
                    progress_callback(min(progress, 99))
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            # Calculate metrics
            train_accuracy = 100. * train_correct / train_total
            val_accuracy = 100. * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Update history
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(train_accuracy)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_acc'].append(val_accuracy)
            self.history['learning_rates'].append(scheduler.get_last_lr()[0])
            
            # Update scheduler
            scheduler.step()
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
            
            # Save best model
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                self.history['best_acc'] = best_val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"  ✨ New best model! Accuracy: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= TRAIN_CONFIG['patience']:
                    print(f"  ⏹️ Early stopping triggered")
                    break
            
            # Update status
            if status_callback:
                status_callback(f"Epoch {epoch+1}/{epochs} - Accuracy: {val_accuracy:.2f}%")
        
        # Save best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            self.save_model()
        
        # Generate plots
        self.generate_training_plots()
        
        print(f"\n{'='*60}")
        print(f"✅ TRAINING COMPLETED!")
        print(f"🎯 Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"{'='*60}")
        
        return True, f"Training completed! Best accuracy: {best_val_acc:.2f}%"
    
    def save_model(self):
        """Save model with all necessary data"""
        os.makedirs(SYSTEM_DIRS['models'], exist_ok=True)
        
        # Save model checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'timestamp': datetime.now().isoformat(),
            'config': MODEL_CONFIG
        }
        torch.save(checkpoint, self.model_path)
        
        # Save class names
        with open(self.classes_file, 'w') as f:
            json.dump({
                'classes': self.class_names,
                'num_classes': self.num_classes,
                'last_updated': datetime.now().isoformat(),
                'best_accuracy': self.history.get('best_acc', 0)
            }, f, indent=4)
        
        # Save training summary
        summary_file = os.path.join(SYSTEM_DIRS['results'], 'training_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("🌿 PLANT DISEASE DETECTION - TRAINING SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: EfficientNet-B0\n")
            f.write(f"Classes: {self.num_classes}\n\n")
            f.write("Class Names:\n")
            for cls in self.class_names:
                f.write(f"  • {cls}\n")
            f.write(f"\nBest Accuracy: {self.history.get('best_acc', 0):.2f}%\n")
            f.write("="*60 + "\n")
        
        print(f"✓ Model saved to: {self.model_path}")
        print(f"✓ Classes saved to: {self.classes_file}")
    
    def generate_training_plots(self):
        """Generate and save training plots"""
        if not self.history['train_loss']:
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Model Training History - 16 Class Classification', fontsize=16, fontweight='bold')
            
            # Loss plot
            axes[0, 0].plot(self.history['train_loss'], label='Training Loss', 
                           linewidth=2, color='#3498db')
            axes[0, 0].plot(self.history['val_loss'], label='Validation Loss', 
                           linewidth=2, color='#e74c3c')
            axes[0, 0].set_xlabel('Epochs', fontsize=11)
            axes[0, 0].set_ylabel('Loss', fontsize=11)
            axes[0, 0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Accuracy plot
            axes[0, 1].plot(self.history['train_acc'], label='Training Accuracy', 
                           linewidth=2, color='#2ecc71')
            axes[0, 1].plot(self.history['val_acc'], label='Validation Accuracy', 
                           linewidth=2, color='#f39c12')
            axes[0, 1].set_xlabel('Epochs', fontsize=11)
            axes[0, 1].set_ylabel('Accuracy (%)', fontsize=11)
            axes[0, 1].set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Learning rate plot
            if self.history.get('learning_rates'):
                axes[1, 0].plot(self.history['learning_rates'], 
                              linewidth=2, color='#9b59b6')
                axes[1, 0].set_xlabel('Epochs', fontsize=11)
                axes[1, 0].set_ylabel('Learning Rate', fontsize=11)
                axes[1, 0].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].axis('off')
            
            # Summary text
            axes[1, 1].axis('off')
            best_acc = self.history.get('best_acc', 0)
            final_acc = self.history['val_acc'][-1] if self.history['val_acc'] else 0
            
            info_text = f"📊 TRAINING SUMMARY - 16 CLASSES\n\n"
            info_text += f"Best Accuracy: {best_acc:.2f}%\n"
            info_text += f"Final Accuracy: {final_acc:.2f}%\n"
            info_text += f"Total Epochs: {len(self.history['train_loss'])}\n"
            info_text += f"Classes: {self.num_classes}\n\n"
            info_text += f"Model: EfficientNet-B0\n"
            info_text += f"Device: {self.device}\n\n"
            info_text += f"📁 Model Saved:\n{os.path.basename(self.model_path)}"
            
            axes[1, 1].text(0.1, 0.5, info_text, fontsize=11, 
                           verticalalignment='center', family='monospace',
                           transform=axes[1, 1].transAxes,
                           bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.5))
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(SYSTEM_DIRS['results'], 'training_plots.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Training plots saved to: {plot_path}")
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    def predict(self, image_path):
        """Predict disease with confidence calibration"""
        if self.model is None:
            return {
                'error': 'Model not loaded',
                'message': 'Please train the model first',
                'status': 'error'
            }
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            original_image = image.copy()
            
            # Apply transformations
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(image_tensor)
                
                # Apply temperature scaling for better confidence calibration
                temperature = 1.5
                outputs = outputs / temperature
                
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Get top 5 predictions
            top_k = min(5, self.num_classes)
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            # Prepare result
            disease_name = self.class_names[predicted.item()]
            
            # Get disease information
            disease_info = self.get_disease_info(disease_name)
            treatment = self.get_treatment(disease_name)
            
            result = {
                'disease': disease_name,
                'confidence': confidence.item() * 100,
                'top_predictions': [],
                'original_image': original_image,
                'image_size': original_image.size,
                'info': disease_info,
                'recommendation': treatment,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            
            # Add top predictions
            for i in range(top_k):
                result['top_predictions'].append({
                    'disease': self.class_names[top_indices[0][i].item()],
                    'confidence': top_probs[0][i].item() * 100
                })
            
            # Save prediction result
            self.save_prediction_result(image_path, result)
            
            return result
            
        except Exception as e:
            print(f"Prediction error: {e}")
            traceback.print_exc()
            return {
                'error': str(e),
                'message': 'Failed to process image',
                'status': 'error'
            }
    
    def save_prediction_result(self, image_path, result):
        """Save prediction result to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"prediction_{timestamp}.json"
            filepath = os.path.join(SYSTEM_DIRS['predictions'], filename)
            
            # Remove PIL Image from result before saving
            save_result = result.copy()
            save_result.pop('original_image', None)
            
            with open(filepath, 'w') as f:
                json.dump(save_result, f, indent=4)
                
        except Exception as e:
            print(f"Error saving prediction: {e}")
    
    def get_disease_info(self, disease_name):
        """Get comprehensive disease information"""
        
        disease_db = {
            'Pepper__bell___Bacterial_spot': {
                'symptoms': 'Small, water-soaked spots that turn brown/black on leaves and fruit. Spots have yellow halos and may merge together.',
                'cause': 'Bacteria Xanthomonas campestris pv. vesicatoria',
                'prevention': 'Use disease-free seeds, 3-4 year crop rotation, avoid overhead watering, copper sprays, remove infected plants',
                'season': 'Warm, humid conditions (25-30°C)',
                'severity': 'High',
                'spread': 'Splashing water, wind, contaminated tools, plant debris'
            },
            'Pepper__bell___healthy': {
                'symptoms': 'Healthy plant with vibrant green leaves, no spots or discoloration, normal growth pattern',
                'cause': 'No disease detected',
                'prevention': 'Maintain good air circulation, proper watering schedule, balanced fertilization, regular monitoring',
                'season': 'All seasons',
                'severity': 'None',
                'spread': 'None'
            },
            'Potato_Early_blight': {
                'symptoms': 'Dark brown spots with concentric rings (target spots) on older leaves. Leaves yellow and die prematurely.',
                'cause': 'Fungus Alternaria solani',
                'prevention': 'Crop rotation, resistant varieties, proper spacing, fungicide application, remove infected debris',
                'season': 'Warm, humid conditions (24-29°C)',
                'severity': 'Medium',
                'spread': 'Wind, rain, infected seed tubers, plant debris'
            },
            'Potato_healthy': {
                'symptoms': 'Healthy plant with no visible symptoms, normal tuber development, green foliage',
                'cause': 'No disease detected',
                'prevention': 'Maintain good growing conditions, proper irrigation, balanced nutrition, pest monitoring',
                'season': 'All seasons',
                'severity': 'None',
                'spread': 'None'
            },
            'Potato_Late_blight': {
                'symptoms': 'Dark, water-soaked lesions on leaves and stems. White cottony fungal growth on undersides. Rapid plant death.',
                'cause': 'Oomycete Phytophthora infestans',
                'prevention': 'Destroy infected plants, resistant varieties, preventive fungicides, avoid overhead irrigation',
                'season': 'Cool, wet weather (10-20°C)',
                'severity': 'Very High',
                'spread': 'Airborne spores, rapid spread in humid conditions, infected tubers'
            },
            'Tomato_Bacterial_spot': {
                'symptoms': 'Small, dark, water-soaked lesions on leaves, stems, and fruit. Spots may have yellow halos.',
                'cause': 'Bacteria Xanthomonas spp.',
                'prevention': 'Use certified disease-free seeds, copper sprays, crop rotation, avoid working in wet fields',
                'season': 'Warm, wet weather (25-30°C)',
                'severity': 'Medium',
                'spread': 'Seed-borne, splashing water, wind-driven rain, contaminated tools'
            },
            'Tomato_Early_blight': {
                'symptoms': 'Brown spots with concentric rings on older leaves. Leaves yellow and drop. Stem lesions.',
                'cause': 'Fungus Alternaria solani',
                'prevention': 'Rotate crops, remove infected plants, avoid overhead watering, mulch, fungicide application',
                'season': 'Warm, humid conditions (24-29°C)',
                'severity': 'Medium',
                'spread': 'Wind, rain, contaminated tools, plant debris'
            },
            'Tomato_healthy': {
                'symptoms': 'No visible symptoms, healthy green leaves, proper fruit development, vigorous growth',
                'cause': 'No disease detected',
                'prevention': 'Maintain good growing conditions, proper staking, pruning for air flow, regular monitoring',
                'season': 'All seasons',
                'severity': 'None',
                'spread': 'None'
            },
            'Tomato_Tomato_mosaic_virus': {
                'symptoms': 'Mottled light/dark green pattern on leaves, leaf distortion, stunted growth, reduced yield',
                'cause': 'Tomato mosaic virus (ToMV)',
                'prevention': 'Use virus-resistant varieties, disinfect tools, control thrips, remove infected plants',
                'season': 'All seasons',
                'severity': 'High',
                'spread': 'Contaminated tools, hands, seeds, plant debris, thrips vectors'
            }
        }
        
        return disease_db.get(disease_name, {
            'symptoms': 'Various symptoms including leaf spots, discoloration, wilting, or abnormal growth patterns',
            'cause': 'Pathogen (fungus, bacteria, virus) or environmental stress factors',
            'prevention': 'Consult agricultural expert, practice good crop management, use disease-free seeds',
            'season': 'Varies by pathogen and environmental conditions',
            'severity': 'Medium',
            'spread': 'Varies - can be through air, water, soil, tools, or vectors'
        })
    
    def get_treatment(self, disease_name):
        """Get detailed treatment recommendations"""
        
        treatments = {
            'Pepper__bell___Bacterial_spot': """🔬 IMMEDIATE ACTIONS:
1. Apply copper-based bactericides every 7-10 days
2. Remove and destroy severely infected plants immediately
3. Stop overhead irrigation - water at base only
4. Disinfect tools between plants (10% bleach solution)

🌱 PREVENTIVE MEASURES:
1. Use certified disease-free seeds or treated seeds
2. Practice 3-4 year crop rotation with non-host crops
3. Apply mulch to prevent soil splash
4. Improve air circulation through proper spacing

💊 CHEMICAL CONTROL (Rotate products):
- Copper hydroxide (e.g., Kocide)
- Copper sulfate
- Mancozeb
- Fixed copper with mancozeb""",
            
            'Potato_Early_blight': """🔬 IMMEDIATE ACTIONS:
1. Apply fungicides at first sign of disease
2. Remove and destroy infected leaves
3. Improve air circulation between plants
4. Avoid working in wet fields

🌱 PREVENTIVE MEASURES:
1. Plant certified disease-free seed potatoes
2. Maintain proper plant spacing (30-40 cm)
3. Use drip irrigation instead of overhead
4. Rotate with non-solanaceous crops (3+ years)

💊 CHEMICAL CONTROL:
- Chlorothalonil (Bravo)
- Azoxystrobin (Quadris)
- Pyraclostrobin (Headline)
- Mancozeb (Dithane)""",
            
            'Potato_Late_blight': """🔬 EMERGENCY ACTIONS:
1. Apply fungicides IMMEDIATELY (within 24 hours)
2. Destroy all infected plants by burning or deep burial
3. Do NOT save seed from infected crop
4. Clean equipment thoroughly before moving to healthy fields

🌱 PREVENTIVE MEASURES:
1. Plant resistant varieties (e.g., Defender, Elba)
2. Use certified disease-free seed tubers
3. Destroy cull piles and volunteer potatoes
4. Avoid excessive nitrogen fertilization

💊 CHEMICAL CONTROL (Apply preventatively):
- Metalaxyl + Mancozeb (Ridomil Gold)
- Cymoxanil + Mancozeb (Curzate)
- Dimethomorph (Acrobat)
- Propamocarb (Previcur Flex)

⚠️ CRITICAL: This disease can destroy entire crop in 7-10 days""",
            
            'Tomato_Tomato_mosaic_virus': """🔬 IMMEDIATE ACTIONS:
1. Remove and destroy infected plants immediately
2. Wash hands with soap and milk before handling plants
3. Disinfect tools with 20% bleach solution or milk
4. Control thrips populations (vectors)

🌱 PREVENTIVE MEASURES:
1. Use virus-resistant tomato varieties (with Tm genes)
2. Purchase certified virus-free transplants
3. Control weeds that harbor the virus
4. Use reflective mulches to repel thrips

💊 MANAGEMENT (No chemical cure):
1. Roguing: Remove infected plants promptly
2. Sanitation: Clean tools, benches, and pots
3. Isolation: Separate new plants from established ones

⚠️ IMPORTANT: Tobacco smokers should wash hands thoroughly before handling tomatoes""",
            
            'default': """🔬 GENERAL TREATMENT PROTOCOL:
1. Remove infected plant parts immediately
2. Apply appropriate fungicide/bactericide according to label
3. Improve air circulation through pruning and proper spacing
4. Water at base of plants, not on foliage

🌱 PREVENTIVE MEASURES:
1. Practice 3-4 year crop rotation with unrelated crops
2. Use disease-resistant varieties when available
3. Maintain optimal plant nutrition - avoid excess nitrogen
4. Keep garden free of weeds and plant debris

💊 CHEMICAL CONTROL:
- For fungal diseases: Chlorothalonil, Copper fungicides
- For bacterial diseases: Copper-based bactericides
- Apply at first sign of disease, repeat every 7-14 days"""
        }
        
        if 'healthy' in disease_name.lower():
            return """✅ PLANT IS HEALTHY!

🌱 MAINTENANCE RECOMMENDATIONS:

1. 💧 WATERING:
   - Water deeply but infrequently (1-1.5 inches per week)
   - Water at base of plants, avoid wetting foliage
   - Water in morning to allow leaves to dry

2. 🌿 NUTRITION:
   - Apply balanced fertilizer (10-10-10) every 4-6 weeks
   - Add compost or organic matter regularly

3. ✂️ PRUNING & TRAINING:
   - Remove lower leaves touching soil
   - Provide adequate support/staking
   - Prune to improve air circulation

4. 🔍 MONITORING:
   - Inspect plants weekly for early signs of problems
   - Check both upper and lower leaf surfaces"""
        
        return treatments.get(disease_name, treatments['default'])

# ==================== USER MANAGER ====================
class UserManager:
    """Manages user authentication and profiles"""
    
    def __init__(self):
        self.users_file = os.path.join(SYSTEM_DIRS['models'], 'users.json')
        self.users = self.load_users()
        self.current_user = None
    
    def load_users(self):
        """Load users from file or create default"""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Create default users
        default_users = {
            'admin': {
                'password': 'admin123', 
                'role': 'admin', 
                'created': datetime.now().strftime('%Y-%m-%d'),
                'email': 'admin@plantdisease.com',
                'full_name': 'System Administrator',
                'last_login': None
            },
            'user': {
                'password': 'user123', 
                'role': 'user', 
                'created': datetime.now().strftime('%Y-%m-%d'),
                'email': 'user@example.com',
                'full_name': 'Regular User',
                'last_login': None
            }
        }
        self.save_users(default_users)
        return default_users
    
    def save_users(self, users=None):
        """Save users to file"""
        if users is None:
            users = self.users
        try:
            with open(self.users_file, 'w') as f:
                json.dump(users, f, indent=4)
        except:
            pass
    
    def login(self, username, password):
        """Authenticate user"""
        if username in self.users and self.users[username]['password'] == password:
            self.current_user = {
                'username': username, 
                'role': self.users[username]['role'],
                'full_name': self.users[username].get('full_name', username),
                'email': self.users[username].get('email', '')
            }
            self.users[username]['last_login'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.save_users()
            return True, f"Welcome {self.current_user['full_name']}!"
        return False, "Invalid username or password"
    
    def logout(self):
        """Logout current user"""
        self.current_user = None
    
    def is_admin(self):
        """Check if current user is admin"""
        return self.current_user and self.current_user['role'] == 'admin'
    
    def get_current_user(self):
        """Get current user info"""
        return self.current_user

# ==================== MAIN APPLICATION ====================
class PlantDiseaseApp:
    """Main GUI Application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🌿 Plant Disease Detection System - 16 Classes")
        self.root.geometry("1200x700")
        
        self.colors = COLORS
        self.center_window()
        
        # Initialize components
        self.detector = AdvancedPlantDiseaseDetector(root_window=self.root)
        self.user_manager = UserManager()
        self.history = self.load_history()
        
        self.setup_styles()
        self.show_login_screen()
    
    def center_window(self):
        """Center window on screen"""
        self.root.update_idletasks()
        width = 1200
        height = 700
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        self.root.resizable(False, False)
    
    def setup_styles(self):
        """Configure ttk styles"""
        self.root.configure(bg=self.colors['bg'])
        
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure treeview
        style.configure('Treeview', 
                       background=self.colors['white'],
                       foreground=self.colors['dark'],
                       rowheight=30,
                       fieldbackground=self.colors['white'],
                       font=('Arial', 10))
        
        style.configure('Treeview.Heading',
                       background=self.colors['secondary'],
                       foreground=self.colors['white'],
                       relief='flat',
                       font=('Arial', 11, 'bold'))
        
        style.map('Treeview.Heading',
                  background=[('active', self.colors['primary'])])
        
        # Configure notebook
        style.configure('TNotebook', background=self.colors['white'])
        style.configure('TNotebook.Tab', 
                       background=self.colors['light'],
                       padding=[15, 5],
                       font=('Arial', 10))
        
        style.map('TNotebook.Tab',
                  background=[('selected', self.colors['white'])])
    
    def load_history(self):
        """Load prediction history from file"""
        history_file = os.path.join(SYSTEM_DIRS['predictions'], 'history.json')
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_history(self):
        """Save prediction history to file"""
        history_file = os.path.join(SYSTEM_DIRS['predictions'], 'history.json')
        try:
            with open(history_file, 'w') as f:
                json.dump(self.history[-100:], f, indent=4)
        except:
            pass
    
    def clear_window(self):
        """Clear all widgets from window"""
        for widget in self.root.winfo_children():
            widget.destroy()
    
    # ==================== LOGIN SCREEN ====================
    def show_login_screen(self):
        """Display enhanced login screen"""
        self.clear_window()
        
        # Main container
        main_frame = tk.Frame(self.root, bg=self.colors['white'])
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Modern design
        left_panel = tk.Frame(main_frame, bg=self.colors['primary'], width=450)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        left_panel.pack_propagate(False)
        
        # Canvas for solid color
        canvas = tk.Canvas(left_panel, bg=self.colors['primary'], 
                          highlightthickness=0, width=450, height=700)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # Fill with solid color
        canvas.create_rectangle(0, 0, 450, 700, fill=self.colors['primary'], outline='')
        
        # Left panel content
        canvas.create_text(225, 150, text="🌿", 
                          font=('Arial', 80), fill='white')
        
        canvas.create_text(225, 250, text="Plant Disease", 
                          font=('Arial', 32, 'bold'), fill='white')
        
        canvas.create_text(225, 310, text="Detection System", 
                          font=('Arial', 24), fill='white')
        
        canvas.create_text(225, 380, text="AI-Powered - 16 Classes", 
                          font=('Arial', 14), fill='white')
        
        # Features
        features = [
            "✓ 16 Plant Diseases",
            "✓ 95%+ Accuracy",
            "✓ Instant Results",
            "✓ Treatment Guide",
            "✓ One-Time Training"
        ]
        
        y_pos = 450
        for feature in features:
            canvas.create_text(225, y_pos, text=feature, 
                             font=('Arial', 12), fill='white', anchor='center')
            y_pos += 35
        
        # Version
        canvas.create_text(225, 650, text="v2.0.3 - Fixed", 
                          font=('Arial', 10), fill='white')
        
        # Right panel - Login form
        right_panel = tk.Frame(main_frame, bg=self.colors['white'], width=750)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Center login form
        form_frame = tk.Frame(right_panel, bg=self.colors['white'])
        form_frame.place(relx=0.5, rely=0.5, anchor='center')
        
        # Logo
        tk.Label(form_frame, text="🔐", font=('Arial', 60), 
                bg=self.colors['white'], fg=self.colors['primary']).pack(pady=(0, 20))
        
        # Title
        tk.Label(form_frame, text="Welcome Back!", 
                font=('Arial', 28, 'bold'), 
                bg=self.colors['white'], 
                fg=self.colors['secondary']).pack(pady=(0, 10))
        
        tk.Label(form_frame, text="Please login to access the system", 
                font=('Arial', 11), 
                bg=self.colors['white'], 
                fg=self.colors['gray']).pack(pady=(0, 40))
        
        # Username field
        username_frame = tk.Frame(form_frame, bg=self.colors['white'])
        username_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(username_frame, text="👤", font=('Arial', 14), 
                bg=self.colors['white'], fg=self.colors['gray']).pack(side=tk.LEFT, padx=(0, 10))
        
        self.username_entry = tk.Entry(username_frame, 
                                      font=('Arial', 13), 
                                      bg='#f8f9fa',
                                      relief=tk.FLAT,
                                      width=30,
                                      highlightthickness=1,
                                      highlightcolor=self.colors['primary'],
                                      highlightbackground='#e0e0e0')
        self.username_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=10)
        self.username_entry.insert(0, "admin")
        
        # Password field
        password_frame = tk.Frame(form_frame, bg=self.colors['white'])
        password_frame.pack(fill=tk.X, pady=15)
        
        tk.Label(password_frame, text="🔒", font=('Arial', 14), 
                bg=self.colors['white'], fg=self.colors['gray']).pack(side=tk.LEFT, padx=(0, 10))
        
        self.password_entry = tk.Entry(password_frame, 
                                      font=('Arial', 13), 
                                      bg='#f8f9fa',
                                      relief=tk.FLAT,
                                      width=30,
                                      show='•',
                                      highlightthickness=1,
                                      highlightcolor=self.colors['primary'],
                                      highlightbackground='#e0e0e0')
        self.password_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=10)
        self.password_entry.insert(0, "admin123")
        
        # Login button
        login_btn = tk.Button(form_frame, 
                             text="LOGIN", 
                             command=self.login,
                             bg=self.colors['primary'],
                             fg=self.colors['white'],
                             font=('Arial', 14, 'bold'),
                             relief=tk.FLAT,
                             cursor='hand2',
                             padx=50,
                             pady=15,
                             width=30,
                             activebackground=self.colors['primary_dark'],
                             activeforeground=self.colors['white'])
        login_btn.pack(pady=(30, 20))
        
        # Model status
        model_loaded = self.detector.model is not None
        status_text = "✅ System Ready - 16 Classes" if model_loaded else "⚠ Model Not Trained"
        status_color = self.colors['success'] if model_loaded else self.colors['warning']
        
        status_frame = tk.Frame(form_frame, bg=self.colors['white'])
        status_frame.pack(pady=(10, 0))
        
        status_dot = tk.Label(status_frame, text="●", 
                            font=('Arial', 16), 
                            bg=self.colors['white'], 
                            fg=status_color)
        status_dot.pack(side=tk.LEFT, padx=(0, 5))
        
        tk.Label(status_frame, text=status_text, 
                font=('Arial', 11), 
                bg=self.colors['white'], 
                fg=status_color).pack(side=tk.LEFT)
        
        # Bind Enter key
        self.root.bind('<Return>', lambda e: self.login())
    
    def login(self):
        """Handle login attempt"""
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        success, message = self.user_manager.login(username, password)
        
        if success:
            messagebox.showinfo("✅ Success", message, parent=self.root)
            if self.user_manager.is_admin():
                self.show_admin_dashboard()
            else:
                self.show_user_dashboard()
        else:
            messagebox.showerror("❌ Login Failed", message, parent=self.root)
    
    # ==================== ADMIN DASHBOARD ====================
    def show_admin_dashboard(self):
        """Display admin dashboard"""
        self.clear_window()
        
        # Navigation bar
        self.create_navbar("🔧 Admin Dashboard - 16 Classes")
        
        # Main content
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)
        
        # Status cards
        self.create_status_cards(main_frame)
        
        # Admin action cards
        self.create_admin_cards(main_frame)
        
        # System information
        self.create_system_info(main_frame)
    
    # ==================== USER DASHBOARD ====================
    def show_user_dashboard(self):
        """Display user dashboard"""
        self.clear_window()
        
        # Navigation bar
        self.create_navbar("🌿 Plant Disease Detection - 16 Classes")
        
        # Main content
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)
        
        # Welcome section
        self.create_welcome_section(main_frame)
        
        # Action buttons
        self.create_action_buttons(main_frame)
        
        # Upload area
        model_loaded = self.detector.model is not None
        self.create_upload_area(main_frame, model_loaded)
    
    def create_navbar(self, title):
        """Create navigation bar"""
        navbar = tk.Frame(self.root, bg=self.colors['primary'], height=70)
        navbar.pack(fill=tk.X)
        navbar.pack_propagate(False)
        
        # Title
        tk.Label(navbar, text=title, font=('Arial', 18, 'bold'),
                bg=self.colors['primary'], fg='white').pack(side=tk.LEFT, padx=30, pady=20)
        
        # User info
        user_frame = tk.Frame(navbar, bg=self.colors['primary'])
        user_frame.pack(side=tk.RIGHT, padx=30, pady=20)
        
        current_user = self.user_manager.get_current_user()
        if current_user:
            tk.Label(user_frame, text=f"👤 {current_user['full_name']}", 
                    font=('Arial', 12), bg=self.colors['primary'], fg='white').pack(side=tk.LEFT, padx=15)
        
        logout_btn = tk.Button(user_frame, text="Logout", 
                              command=self.logout,
                              bg=self.colors['white'],
                              fg=self.colors['primary'],
                              font=('Arial', 11, 'bold'),
                              relief=tk.FLAT,
                              cursor='hand2',
                              padx=20,
                              pady=5,
                              activebackground=self.colors['light'])
        logout_btn.pack(side=tk.LEFT)
    
    def create_status_cards(self, parent):
        """Create status information cards"""
        status_frame = tk.Frame(parent, bg=self.colors['bg'])
        status_frame.pack(fill=tk.X, pady=(0, 30))
        
        # Model status
        model_loaded = self.detector.model is not None
        
        # Get dataset info
        success, dataset_info = DatasetManager.check_dataset_status()
        
        stats = [
            ("🤖 Model Status", 
             "✅ Active (16 classes)" if model_loaded else "⚠ Not Trained",
             self.colors['success'] if model_loaded else self.colors['warning']),
            
            ("📚 Classes", 
             f"{self.detector.num_classes}" if self.detector.num_classes > 0 else "0",
             self.colors['info']),
            
            ("🎯 Best Accuracy", 
             f"{self.detector.history.get('best_acc', 0):.1f}%" if self.detector.history.get('best_acc') else "N/A",
             self.colors['accent']),
            
            ("📊 Dataset", 
             f"{dataset_info['total_images']} images" if success else "Not ready",
             self.colors['secondary'])
        ]
        
        for i, (label, value, color) in enumerate(stats):
            card = tk.Frame(status_frame, bg=self.colors['white'], 
                           relief=tk.RAISED, bd=0, highlightthickness=1,
                           highlightbackground=self.colors['light'])
            card.grid(row=0, column=i, padx=8, pady=5, sticky='nsew')
            
            # Label
            tk.Label(card, text=label, font=('Arial', 11), 
                    bg=self.colors['white'], fg=self.colors['gray']).pack(pady=(12, 5))
            
            # Value
            tk.Label(card, text=value, font=('Arial', 16, 'bold'), 
                    bg=self.colors['white'], fg=color).pack(pady=(0, 12))
            
            status_frame.grid_columnconfigure(i, weight=1)
    
    def create_admin_cards(self, parent):
        """Create admin action cards"""
        cards_frame = tk.Frame(parent, bg=self.colors['bg'])
        cards_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        model_loaded = self.detector.model is not None
        
        admin_features = [
            ("🚀 Train Model (16 Classes)", 
             "One-time training", 
             "Train the model with your 16-class dataset. This is required before making predictions.",
             self.train_model_ui,
             self.colors['accent'],
             True),
            
            ("🔍 Predict Disease", 
             "Test & validate", 
             "Test the trained model on plant images to verify accuracy for 16 classes.",
             self.predict_disease,
             self.colors['success'],
             model_loaded),
            
            ("📊 Training Results", 
             "View metrics", 
             "View accuracy graphs, loss curves, and training history for 16 classes.",
             self.view_results,
             self.colors['info'],
             model_loaded),
            
            ("📁 Dataset Info", 
             "Manage data", 
             "View dataset structure, class distribution (16 classes), and statistics.",
             self.manage_dataset,
             self.colors['secondary'],
             True)
        ]
        
        for i, (title, subtitle, desc, command, color, enabled) in enumerate(admin_features):
            row = i // 2
            col = i % 2
            
            card = tk.Frame(cards_frame, bg=self.colors['white'], 
                           relief=tk.RAISED, bd=0, highlightthickness=1,
                           highlightbackground=self.colors['light'])
            card.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
            
            # Header
            header_frame = tk.Frame(card, bg=self.colors['white'])
            header_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
            
            tk.Label(header_frame, text=title, font=('Arial', 16, 'bold'),
                    bg=self.colors['white'], fg=color).pack(anchor='w')
            
            tk.Label(header_frame, text=subtitle, font=('Arial', 11),
                    bg=self.colors['white'], fg=self.colors['gray']).pack(anchor='w', pady=(2, 5))
            
            # Description
            tk.Label(card, text=desc, font=('Arial', 10),
                    bg=self.colors['white'], fg=self.colors['secondary'],
                    wraplength=250, justify=tk.LEFT).pack(anchor='w', padx=20, pady=(0, 20))
            
            # Action button
            btn_state = tk.NORMAL if enabled else tk.DISABLED
            
            btn_text = "Launch →" if enabled else "Unavailable"
            btn_color = color if enabled else self.colors['gray']
            
            btn = tk.Button(card, text=btn_text,
                          command=command if enabled else None,
                          bg=btn_color,
                          fg=self.colors['white'],
                          font=('Arial', 11, 'bold'),
                          relief=tk.FLAT,
                          cursor='hand2' if enabled else 'arrow',
                          padx=25,
                          pady=8,
                          state=btn_state)
            btn.pack(anchor='w', padx=20, pady=(0, 20))
            
            cards_frame.grid_columnconfigure(col, weight=1)
            cards_frame.grid_rowconfigure(row, weight=1)
    
    def create_system_info(self, parent):
        """Create system information section"""
        info_frame = tk.Frame(parent, bg=self.colors['bg'])
        info_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        separator = tk.Frame(info_frame, bg=self.colors['light'], height=1)
        separator.pack(fill=tk.X, pady=(10, 15))
        
        # System info grid
        system_frame = tk.Frame(info_frame, bg=self.colors['bg'])
        system_frame.pack(fill=tk.X)
        
        # Left: Dataset path
        path_frame = tk.Frame(system_frame, bg=self.colors['bg'])
        path_frame.pack(side=tk.LEFT)
        
        tk.Label(path_frame, text="📁 Dataset:", font=('Arial', 10, 'bold'),
                bg=self.colors['bg'], fg=self.colors['secondary']).pack(side=tk.LEFT, padx=(0, 10))
        
        # Truncate long path
        display_path = DATASET_PATH
        if len(display_path) > 60:
            display_path = "..." + display_path[-57:]
        
        tk.Label(path_frame, text=display_path, font=('Arial', 9),
                bg=self.colors['bg'], fg=self.colors['gray']).pack(side=tk.LEFT)
        
        # Right: Status
        success, _ = DatasetManager.check_dataset_status()
        status_color = self.colors['success'] if success else self.colors['danger']
        status_text = "✓ 16 Classes Ready" if success else "✗ Issue"
        
        status_label = tk.Label(system_frame, text=status_text, 
                               font=('Arial', 9, 'bold'),
                               bg=self.colors['bg'], fg=status_color)
        status_label.pack(side=tk.RIGHT)
        
        # Center: Device info
        device_text = f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}"
        device_label = tk.Label(system_frame, text=device_text, 
                               font=('Arial', 9),
                               bg=self.colors['bg'], fg=self.colors['info'])
        device_label.pack(side=tk.RIGHT, padx=20)
    
    def create_welcome_section(self, parent):
        """Create welcome section for user dashboard"""
        welcome_frame = tk.Frame(parent, bg=self.colors['white'], 
                                relief=tk.RAISED, bd=0, highlightthickness=1,
                                highlightbackground=self.colors['light'])
        welcome_frame.pack(fill=tk.X, pady=(0, 25))
        
        model_loaded = self.detector.model is not None
        
        if model_loaded:
            welcome_text = f"✅ System Ready! Detecting {self.detector.num_classes} plant diseases"
            status_color = self.colors['success']
        else:
            welcome_text = "⚠ Model not trained yet (16 classes). Please contact administrator."
            status_color = self.colors['warning']
        
        tk.Label(welcome_frame, text="🌿 Plant Disease Detection System - 16 Classes", 
                font=('Arial', 16, 'bold'), 
                bg=self.colors['white'], 
                fg=self.colors['primary']).pack(pady=(20, 5))
        
        tk.Label(welcome_frame, text=welcome_text, 
                font=('Arial', 11), 
                bg=self.colors['white'], 
                fg=status_color).pack(pady=(0, 20))
    
    def create_action_buttons(self, parent):
        """Create action buttons for user dashboard"""
        actions_frame = tk.Frame(parent, bg=self.colors['bg'])
        actions_frame.pack(fill=tk.X, pady=(0, 25))
        
        model_loaded = self.detector.model is not None
        
        actions = [
            ("📤 Upload Image", self.predict_disease, self.colors['primary'], model_loaded),
            ("📊 History", self.show_history, self.colors['info'], True),
            ("📚 Disease Library (16)", self.show_disease_library, self.colors['secondary'], True),
            ("📥 Export Data", self.export_data, self.colors['warning'], model_loaded)
        ]
        
        for text, command, color, enabled in actions:
            btn_state = tk.NORMAL if enabled else tk.DISABLED
            btn = tk.Button(actions_frame, text=text, 
                          command=command if enabled else None,
                          bg=color if enabled else self.colors['gray'],
                          fg=self.colors['white'],
                          font=('Arial', 11, 'bold'),
                          relief=tk.FLAT,
                          cursor='hand2' if enabled else 'arrow',
                          padx=25,
                          pady=10,
                          state=btn_state)
            btn.pack(side=tk.LEFT, padx=5)
    
    def create_upload_area(self, parent, enabled):
        """Create upload area for user dashboard"""
        upload_frame = tk.LabelFrame(parent, 
                                   text="📤 Upload Plant Image - 16 Classes",
                                   font=('Arial', 12, 'bold'),
                                   bg=self.colors['white'],
                                   fg=self.colors['primary'],
                                   bd=2,
                                   relief=tk.GROOVE)
        upload_frame.pack(fill=tk.BOTH, expand=True)
        
        upload_content = tk.Frame(upload_frame, bg=self.colors['white'])
        upload_content.pack(expand=True, fill=tk.BOTH, padx=50, pady=40)
        
        # Icon
        tk.Label(upload_content, text="🌿", font=('Arial', 72),
                bg=self.colors['white'], fg=self.colors['primary']).pack(pady=(20, 10))
        
        # Instructions
        tk.Label(upload_content, text="Drag & Drop Image Here", 
                font=('Arial', 16, 'bold'), 
                bg=self.colors['white'], 
                fg=self.colors['secondary']).pack(pady=5)
        
        tk.Label(upload_content, text="or", 
                font=('Arial', 11), 
                bg=self.colors['white'], 
                fg=self.colors['gray']).pack()
        
        # Upload button
        upload_btn = tk.Button(upload_content, 
                              text="📁 Browse Files", 
                              command=self.predict_disease if enabled else None,
                              bg=self.colors['primary'] if enabled else self.colors['gray'],
                              fg=self.colors['white'],
                              font=('Arial', 11, 'bold'),
                              relief=tk.FLAT,
                              cursor='hand2' if enabled else 'arrow',
                              padx=40,
                              pady=12,
                              state=tk.NORMAL if enabled else tk.DISABLED)
        upload_btn.pack(pady=(20, 10))
        
        # Supported formats
        formats_frame = tk.Frame(upload_content, bg=self.colors['white'])
        formats_frame.pack(pady=(10, 20))
        
        for fmt in ['JPG', 'PNG', 'BMP', 'TIFF', 'WEBP']:
            tk.Label(formats_frame, text=fmt, 
                    font=('Arial', 9), 
                    bg=self.colors['light'],
                    fg=self.colors['secondary'],
                    padx=10,
                    pady=2,
                    relief=tk.FLAT).pack(side=tk.LEFT, padx=3)
    
    # ==================== TRAINING INTERFACE ====================
    def train_model_ui(self):
        """Display enhanced training interface for 16 classes"""
        train_window = tk.Toplevel(self.root)
        train_window.title("🚀 Train Model - 16 Classes")
        train_window.geometry("700x750")
        train_window.transient(self.root)
        train_window.grab_set()
        
        # Center window
        train_window.update_idletasks()
        x = (train_window.winfo_screenwidth() // 2) - (700 // 2)
        y = (train_window.winfo_screenheight() // 2) - (750 // 2)
        train_window.geometry(f'+{x}+{y}')
        
        # Main frame
        main_frame = tk.Frame(train_window, bg=self.colors['white'], padx=30, pady=30)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        tk.Label(main_frame, text="🚀 Model Training - 16 Classes", 
                font=('Arial', 22, 'bold'), 
                fg=self.colors['primary']).pack(pady=(0, 10))
        
        tk.Label(main_frame, text="One-time training - Model will be saved permanently", 
                font=('Arial', 11), 
                fg=self.colors['gray']).pack(pady=(0, 30))
        
        # Dataset status
        success, dataset_info = DatasetManager.check_dataset_status()
        
        status_frame = tk.Frame(main_frame, bg=self.colors['light'], relief=tk.FLAT)
        status_frame.pack(fill=tk.X, pady=(0, 30), ipady=15)
        
        if success:
            status_color = self.colors['success']
            status_text = f"✅ Dataset Ready - {dataset_info['num_classes']} Classes Found"
            
            tk.Label(status_frame, text="📊 Dataset Statistics", 
                    font=('Arial', 12, 'bold'), 
                    bg=self.colors['light'], 
                    fg=self.colors['secondary']).pack(pady=(10, 5))
            
            stats_text = f"Classes: {dataset_info['num_classes']} (16 required)  |  "
            stats_text += f"Images: {dataset_info['total_images']}  |  "
            stats_text += f"Path: {os.path.basename(DATASET_PATH)}"
            
            tk.Label(status_frame, text=stats_text, 
                    font=('Arial', 11), 
                    bg=self.colors['light'], 
                    fg=self.colors['dark']).pack(pady=(0, 10))
        else:
            status_color = self.colors['danger']
            status_text = "❌ Dataset Issue"
            
            tk.Label(status_frame, text=status_text, 
                    font=('Arial', 12, 'bold'), 
                    bg=self.colors['light'], 
                    fg=status_color).pack(pady=15)
            
            tk.Label(status_frame, text=str(dataset_info), 
                    font=('Arial', 10), 
                    bg=self.colors['light'], 
                    fg=self.colors['dark'],
                    wraplength=600).pack(pady=(0, 15))
        
        # Training parameters
        params_frame = tk.LabelFrame(main_frame, text="⚙️ Training Parameters (16 Classes)", 
                                   font=('Arial', 12, 'bold'),
                                   fg=self.colors['secondary'],
                                   bg=self.colors['white'],
                                   padx=20, pady=15)
        params_frame.pack(fill=tk.X, pady=(0, 25))
        
        # Epochs
        epoch_frame = tk.Frame(params_frame, bg=self.colors['white'])
        epoch_frame.pack(fill=tk.X, pady=8)
        
        tk.Label(epoch_frame, text="Epochs:", font=('Arial', 11),
                bg=self.colors['white'], width=15, anchor='w').pack(side=tk.LEFT)
        
        epochs_var = tk.IntVar(value=TRAIN_CONFIG['default_epochs'])
        epoch_spin = tk.Spinbox(epoch_frame, from_=10, to=50, textvariable=epochs_var,
                               font=('Arial', 11), width=10, relief=tk.FLAT,
                               highlightthickness=1, highlightbackground=self.colors['light'])
        epoch_spin.pack(side=tk.LEFT, padx=10)
        
        tk.Label(epoch_frame, text="(20-25 recommended)", font=('Arial', 9),
                bg=self.colors['white'], fg=self.colors['gray']).pack(side=tk.LEFT)
        
        # Batch size
        batch_frame = tk.Frame(params_frame, bg=self.colors['white'])
        batch_frame.pack(fill=tk.X, pady=8)
        
        tk.Label(batch_frame, text="Batch Size:", font=('Arial', 11),
                bg=self.colors['white'], width=15, anchor='w').pack(side=tk.LEFT)
        
        batch_var = tk.IntVar(value=TRAIN_CONFIG['default_batch_size'])
        batch_spin = tk.Spinbox(batch_frame, from_=8, to=64, textvariable=batch_var,
                               font=('Arial', 11), width=10, relief=tk.FLAT,
                               highlightthickness=1, highlightbackground=self.colors['light'])
        batch_spin.pack(side=tk.LEFT, padx=10)
        
        tk.Label(batch_frame, text="(16 recommended)", font=('Arial', 9),
                bg=self.colors['white'], fg=self.colors['gray']).pack(side=tk.LEFT)
        
        # Learning rate
        lr_frame = tk.Frame(params_frame, bg=self.colors['white'])
        lr_frame.pack(fill=tk.X, pady=8)
        
        tk.Label(lr_frame, text="Learning Rate:", font=('Arial', 11),
                bg=self.colors['white'], width=15, anchor='w').pack(side=tk.LEFT)
        
        lr_var = tk.DoubleVar(value=TRAIN_CONFIG['default_lr'])
        lr_entry = tk.Entry(lr_frame, textvariable=lr_var, font=('Arial', 11),
                           width=12, relief=tk.FLAT, highlightthickness=1,
                           highlightbackground=self.colors['light'])
        lr_entry.pack(side=tk.LEFT, padx=10)
        
        tk.Label(lr_frame, text="(0.0001 recommended for fine-tuning)", font=('Arial', 9),
                bg=self.colors['white'], fg=self.colors['gray']).pack(side=tk.LEFT)
        
        # Progress frame (initially hidden)
        progress_frame = tk.Frame(main_frame, bg=self.colors['white'])
        progress_frame.pack(fill=tk.X, pady=(0, 20))
        progress_frame.pack_forget()
        
        # Progress bar
        progress_bar = ttk.Progressbar(progress_frame, length=500, mode='determinate')
        progress_bar.pack(pady=(10, 5))
        
        # Status label
        status_label = tk.Label(progress_frame, text="", font=('Arial', 10),
                               bg=self.colors['white'], fg=self.colors['info'])
        status_label.pack()
        
        # Training button
        def start_training_thread():
            if not success:
                messagebox.showerror("Error", "Cannot train: Dataset not ready!", parent=train_window)
                return
            
            # Confirm
            confirm = messagebox.askyesno("Confirm Training", 
                f"⚠️ This will train the 16-class model with:\n\n"
                f"Epochs: {epochs_var.get()}\n"
                f"Batch Size: {batch_var.get()}\n"
                f"Learning Rate: {lr_var.get()}\n\n"
                f"Dataset: {dataset_info['num_classes']} classes, {dataset_info['total_images']} images\n"
                f"Model: EfficientNet-B0 (backbone frozen, only classifier trainable)\n\n"
                f"This may take 10-20 minutes depending on your hardware.\n"
                f"Model will be saved permanently.\n\n"
                f"Continue?", parent=train_window)
            
            if not confirm:
                return
            
            # Hide parameters, show progress
            params_frame.pack_forget()
            progress_frame.pack(fill=tk.X, pady=(0, 20))
            train_btn.config(state=tk.DISABLED, text="🚀 Training in progress...", 
                           bg=self.colors['gray'])
            train_window.update()
            
            def update_progress(value):
                progress_bar['value'] = value
                train_window.update_idletasks()
            
            def update_status(text):
                status_label.config(text=text)
                train_window.update_idletasks()
            
            def train_thread():
                try:
                    training_success, message = self.detector.train_model(
                        epochs=epochs_var.get(),
                        batch_size=batch_var.get(),
                        learning_rate=lr_var.get(),
                        progress_callback=update_progress,
                        status_callback=update_status
                    )
                    
                    train_window.after(0, lambda: training_complete(training_success, message))
                    
                except Exception as e:
                    train_window.after(0, lambda: training_complete(False, str(e)))
            
            def training_complete(success, message):
                if success:
                    messagebox.showinfo("✅ Success", 
                        f"16-class model trained and saved successfully!\n\n"
                        f"{message}\n\n"
                        f"You can now use it for predictions.", parent=train_window)
                    train_window.destroy()
                    
                    # Refresh dashboard
                    if self.user_manager.is_admin():
                        self.show_admin_dashboard()
                    else:
                        self.show_user_dashboard()
                else:
                    messagebox.showerror("❌ Error", f"Training failed:\n{message}", parent=train_window)
                    # Restore UI
                    progress_frame.pack_forget()
                    params_frame.pack(fill=tk.X, pady=(0, 25))
                    train_btn.config(state=tk.NORMAL, text="🚀 Start Training (16 Classes)", 
                                   bg=self.colors['accent'])
            
            # Start training in separate thread
            threading.Thread(target=train_thread, daemon=True).start()
        
        train_btn = tk.Button(main_frame, text="🚀 Start Training (16 Classes)", 
                             command=start_training_thread,
                             bg=self.colors['accent'] if success else self.colors['gray'],
                             fg='white',
                             font=('Arial', 13, 'bold'),
                             relief=tk.FLAT,
                             cursor='hand2' if success else 'arrow',
                             padx=30,
                             pady=15,
                             state=tk.NORMAL if success else tk.DISABLED)
        train_btn.pack(pady=(10, 20))
        
        # Info text
        info_text = "📌 Note: Training is a one-time process. Only the classifier head is trained (backbone frozen) for optimal performance with 16 classes."
        tk.Label(main_frame, text=info_text, font=('Arial', 9),
                bg=self.colors['white'], fg=self.colors['info'],
                wraplength=600, justify=tk.LEFT).pack()
    
    # ==================== PREDICTION INTERFACE ====================
    def predict_disease(self):
        """Handle disease prediction"""
        if not self.detector.model:
            messagebox.showerror("Error", 
                "Model not trained!\n\nPlease train the 16-class model first or contact administrator.",
                parent=self.root)
            return
        
        file_path = filedialog.askopenfilename(
            title="Select Plant Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("BMP files", "*.bmp"),
                ("All files", "*.*")
            ],
            parent=self.root
        )
        
        if file_path:
            self.process_prediction(file_path)
    
    def process_prediction(self, image_path):
        """Process and display prediction results"""
        # Show loading window
        loading_window = tk.Toplevel(self.root)
        loading_window.title("Processing")
        loading_window.geometry("350x250")
        loading_window.transient(self.root)
        loading_window.grab_set()
        
        # Center window
        loading_window.update_idletasks()
        x = (loading_window.winfo_screenwidth() // 2) - (350 // 2)
        y = (loading_window.winfo_screenheight() // 2) - (250 // 2)
        loading_window.geometry(f'+{x}+{y}')
        
        # Loading animation
        tk.Label(loading_window, text="🔬", font=('Arial', 48), 
                fg=self.colors['primary']).pack(pady=(30, 10))
        
        tk.Label(loading_window, text="Analyzing Image...", 
                font=('Arial', 14, 'bold'), 
                fg=self.colors['secondary']).pack(pady=10)
        
        progress = ttk.Progressbar(loading_window, length=200, mode='indeterminate')
        progress.pack(pady=20)
        progress.start(10)
        
        tk.Label(loading_window, text="Please wait", 
                font=('Arial', 11), 
                fg=self.colors['gray']).pack()
        
        loading_window.update()
        
        # Process prediction
        result = self.detector.predict(image_path)
        
        # Close loading window
        loading_window.destroy()
        
        if 'error' in result:
            messagebox.showerror("Error", f"Failed to predict:\n{result['error']}", 
                               parent=self.root)
            return
        
        # Add to history
        history_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'image': os.path.basename(image_path),
            'disease': result['disease'],
            'confidence': round(result['confidence'], 2),
            'image_path': image_path
        }
        self.history.append(history_entry)
        self.save_history()
        
        # Show results
        self.show_prediction_results(result, image_path)
    
    def show_prediction_results(self, result, image_path):
        """Display enhanced prediction results"""
        result_window = tk.Toplevel(self.root)
        result_window.title("🔬 Prediction Results - 16 Classes")
        result_window.geometry("1100x750")
        result_window.transient(self.root)
        
        # Center window
        result_window.update_idletasks()
        x = (result_window.winfo_screenwidth() // 2) - (1100 // 2)
        y = (result_window.winfo_screenheight() // 2) - (750 // 2)
        result_window.geometry(f'+{x}+{y}')
        
        # Main container
        main_container = tk.Frame(result_window, bg=self.colors['bg'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_frame = tk.Frame(main_container, bg=self.colors['white'], 
                               relief=tk.RAISED, bd=0, highlightthickness=1,
                               highlightbackground=self.colors['light'])
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(header_frame, text="🔬 Analysis Complete - 16 Classes", 
                font=('Arial', 18, 'bold'), 
                fg=self.colors['primary'],
                bg=self.colors['white']).pack(pady=15)
        
        # Content area
        content_frame = tk.Frame(main_container, bg=self.colors['bg'])
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Image
        left_panel = tk.Frame(content_frame, bg=self.colors['white'], 
                             relief=tk.RAISED, bd=0, highlightthickness=1,
                             highlightbackground=self.colors['light'], width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        tk.Label(left_panel, text="📸 Uploaded Image", 
                font=('Arial', 12, 'bold'), 
                fg=self.colors['secondary'],
                bg=self.colors['white']).pack(pady=(15, 10))
        
        # Display image
        try:
            img = result['original_image']
            img.thumbnail((350, 350))
            photo = ImageTk.PhotoImage(img)
            img_label = tk.Label(left_panel, image=photo, bg=self.colors['white'])
            img_label.image = photo
            img_label.pack(pady=10)
        except:
            tk.Label(left_panel, text="⚠️ Image Preview Unavailable", 
                    font=('Arial', 11), fg=self.colors['gray'],
                    bg=self.colors['white']).pack(pady=50)
        
        # Image info
        info_frame = tk.Frame(left_panel, bg=self.colors['light'], relief=tk.FLAT)
        info_frame.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(info_frame, text=f"File: {os.path.basename(image_path)}", 
                font=('Arial', 9), bg=self.colors['light'],
                fg=self.colors['secondary'], wraplength=350).pack(pady=5, padx=10)
        
        tk.Label(info_frame, text=f"Size: {result['image_size'][0]}x{result['image_size'][1]}", 
                font=('Arial', 9), bg=self.colors['light'],
                fg=self.colors['secondary']).pack(pady=5)
        
        # Right panel - Results
        right_panel = tk.Frame(content_frame, bg=self.colors['bg'])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Primary prediction card
        pred_card = tk.Frame(right_panel, bg=self.colors['white'], 
                            relief=tk.RAISED, bd=0, highlightthickness=1,
                            highlightbackground=self.colors['light'])
        pred_card.pack(fill=tk.X, pady=(0, 15))
        
        confidence = result['confidence']
        if confidence >= 90:
            conf_color = self.colors['success']
            conf_text = "Excellent"
        elif confidence >= 75:
            conf_color = self.colors['info']
            conf_text = "Good"
        elif confidence >= 60:
            conf_color = self.colors['warning']
            conf_text = "Fair"
        else:
            conf_color = self.colors['danger']
            conf_text = "Low Confidence"
        
        tk.Label(pred_card, text="🎯 PRIMARY DIAGNOSIS - 16 CLASSES", 
                font=('Arial', 10), 
                fg=self.colors['gray'],
                bg=self.colors['white']).pack(pady=(15, 5))
        
        # Disease name (clean format)
        disease_display = result['disease'].replace('_', ' ').replace('  ', ' ')
        tk.Label(pred_card, text=disease_display, 
                font=('Arial', 18, 'bold'), 
                fg=self.colors['secondary'],
                bg=self.colors['white'],
                wraplength=400).pack(pady=5)
        
        # Confidence meter
        conf_frame = tk.Frame(pred_card, bg=self.colors['white'])
        conf_frame.pack(fill=tk.X, padx=30, pady=(10, 15))
        
        tk.Label(conf_frame, text=f"{confidence:.1f}%", 
                font=('Arial', 24, 'bold'), 
                fg=conf_color,
                bg=self.colors['white']).pack(side=tk.LEFT)
        
        tk.Label(conf_frame, text=conf_text, 
                font=('Arial', 12), 
                fg=conf_color,
                bg=self.colors['white']).pack(side=tk.LEFT, padx=(10, 0))
        
        # Progress bar for confidence
        progress_frame = tk.Frame(pred_card, bg=self.colors['light'], height=8, width=300)
        progress_frame.pack(pady=(0, 15))
        progress_frame.pack_propagate(False)
        
        progress_width = int(300 * confidence / 100)
        progress_bar = tk.Frame(progress_frame, bg=conf_color, height=8, width=progress_width)
        progress_bar.pack(side=tk.LEFT)
        
        # Top predictions
        top_frame = tk.Frame(right_panel, bg=self.colors['white'], 
                            relief=tk.RAISED, bd=0, highlightthickness=1,
                            highlightbackground=self.colors['light'])
        top_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(top_frame, text="📊 TOP PREDICTIONS (16 Classes)", 
                font=('Arial', 10), 
                fg=self.colors['gray'],
                bg=self.colors['white']).pack(pady=(15, 10))
        
        for i, pred in enumerate(result['top_predictions'][:3]):
            pred_item = tk.Frame(top_frame, bg=self.colors['white'])
            pred_item.pack(fill=tk.X, padx=20, pady=5)
            
            pred_display = pred['disease'].replace('_', ' ').replace('  ', ' ')
            if len(pred_display) > 30:
                pred_display = pred_display[:27] + "..."
            
            tk.Label(pred_item, text=f"{i+1}. {pred_display}", 
                    font=('Arial', 11), 
                    fg=self.colors['secondary'],
                    bg=self.colors['white'],
                    anchor='w').pack(side=tk.LEFT)
            
            tk.Label(pred_item, text=f"{pred['confidence']:.1f}%", 
                    font=('Arial', 11, 'bold'), 
                    fg=self.colors['info'],
                    bg=self.colors['white']).pack(side=tk.RIGHT)
        
        # Disease info tabs
        info_notebook = ttk.Notebook(right_panel)
        info_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Symptoms tab
        symptoms_frame = tk.Frame(info_notebook, bg=self.colors['white'])
        info_notebook.add(symptoms_frame, text="🦠 Symptoms & Cause")
        
        symptoms_text = scrolledtext.ScrolledText(symptoms_frame, 
                                                  font=('Arial', 10),
                                                  bg=self.colors['white'],
                                                  fg=self.colors['secondary'],
                                                  wrap=tk.WORD,
                                                  padx=15,
                                                  pady=15)
        symptoms_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        symptoms_text.insert(tk.END, "🦠 SYMPTOMS:\n", 'heading')
        symptoms_text.insert(tk.END, f"{result['info']['symptoms']}\n\n")
        symptoms_text.insert(tk.END, "🔬 CAUSE:\n", 'heading')
        symptoms_text.insert(tk.END, f"{result['info']['cause']}\n\n")
        symptoms_text.insert(tk.END, "📅 SEASON:\n", 'heading')
        symptoms_text.insert(tk.END, f"{result['info']['season']}\n\n")
        symptoms_text.insert(tk.END, "⚠️ SEVERITY:\n", 'heading')
        symptoms_text.insert(tk.END, f"{result['info']['severity']}\n\n")
        
        symptoms_text.tag_config('heading', font=('Arial', 11, 'bold'), foreground=self.colors['primary'])
        symptoms_text.config(state=tk.DISABLED)
        
        # Treatment tab
        treatment_frame = tk.Frame(info_notebook, bg=self.colors['white'])
        info_notebook.add(treatment_frame, text="💊 Treatment")
        
        treatment_text = scrolledtext.ScrolledText(treatment_frame, 
                                                   font=('Arial', 10),
                                                   bg=self.colors['white'],
                                                   fg=self.colors['secondary'],
                                                   wrap=tk.WORD,
                                                   padx=15,
                                                   pady=15)
        treatment_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        treatment_text.insert(tk.END, result['recommendation'])
        treatment_text.config(state=tk.DISABLED)
        
        # Prevention tab
        prevention_frame = tk.Frame(info_notebook, bg=self.colors['white'])
        info_notebook.add(prevention_frame, text="🛡️ Prevention")
        
        prevention_text = scrolledtext.ScrolledText(prevention_frame, 
                                                    font=('Arial', 10),
                                                    bg=self.colors['white'],
                                                    fg=self.colors['secondary'],
                                                    wrap=tk.WORD,
                                                    padx=15,
                                                    pady=15)
        prevention_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        prevention_text.insert(tk.END, f"🛡️ PREVENTION:\n{result['info']['prevention']}")
        prevention_text.tag_config('heading', font=('Arial', 11, 'bold'), foreground=self.colors['primary'])
        prevention_text.config(state=tk.DISABLED)
        
        # Button frame
        button_frame = tk.Frame(main_container, bg=self.colors['bg'])
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        tk.Button(button_frame, text="🔄 New Analysis", 
                 command=lambda: [result_window.destroy(), self.predict_disease()],
                 bg=self.colors['primary'],
                 fg='white',
                 font=('Arial', 11, 'bold'),
                 relief=tk.FLAT,
                 cursor='hand2',
                 padx=25,
                 pady=8).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="📥 Save Report", 
                 command=lambda: self.save_report(result, image_path),
                 bg=self.colors['info'],
                 fg='white',
                 font=('Arial', 11, 'bold'),
                 relief=tk.FLAT,
                 cursor='hand2',
                 padx=25,
                 pady=8).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="❌ Close", 
                 command=result_window.destroy,
                 bg=self.colors['gray'],
                 fg='white',
                 font=('Arial', 11, 'bold'),
                 relief=tk.FLAT,
                 cursor='hand2',
                 padx=25,
                 pady=8).pack(side=tk.RIGHT, padx=5)
    
    def save_report(self, result, image_path):
        """Save prediction report to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"disease_report_16class_{timestamp}.txt"
            report_path = os.path.join(SYSTEM_DIRS['predictions'], report_filename)
            
            with open(report_path, 'w') as f:
                f.write("="*70 + "\n")
                f.write("🌿 PLANT DISEASE DETECTION REPORT - 16 CLASSES\n")
                f.write("="*70 + "\n\n")
                f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Image File: {os.path.basename(image_path)}\n\n")
                f.write("-"*70 + "\n")
                f.write("PRIMARY DIAGNOSIS:\n")
                f.write("-"*70 + "\n")
                f.write(f"Disease: {result['disease'].replace('_', ' ')}\n")
                f.write(f"Confidence: {result['confidence']:.1f}%\n\n")
                f.write("-"*70 + "\n")
                f.write("TOP PREDICTIONS:\n")
                f.write("-"*70 + "\n")
                for i, pred in enumerate(result['top_predictions'][:3]):
                    f.write(f"{i+1}. {pred['disease'].replace('_', ' ')}: {pred['confidence']:.1f}%\n")
                f.write("\n" + "-"*70 + "\n")
                f.write("DISEASE INFORMATION:\n")
                f.write("-"*70 + "\n")
                f.write(f"Symptoms: {result['info']['symptoms']}\n\n")
                f.write(f"Cause: {result['info']['cause']}\n\n")
                f.write(f"Season: {result['info']['season']}\n\n")
                f.write(f"Severity: {result['info']['severity']}\n\n")
                f.write("-"*70 + "\n")
                f.write("TREATMENT RECOMMENDATIONS:\n")
                f.write("-"*70 + "\n")
                f.write(result['recommendation'])
                f.write("\n\n" + "="*70 + "\n")
            
            messagebox.showinfo("✅ Success", f"Report saved successfully!\n\nLocation: {report_path}", 
                              parent=self.root)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save report:\n{str(e)}", parent=self.root)
    
    # ==================== RESULTS VIEWER ====================
    def view_results(self):
        """View training results and plots"""
        plots_path = os.path.join(SYSTEM_DIRS['results'], 'training_plots.png')
        
        if not os.path.exists(plots_path):
            messagebox.showinfo("Info", "No training results found.\nPlease train the 16-class model first.", 
                              parent=self.root)
            return
        
        results_window = tk.Toplevel(self.root)
        results_window.title("📊 Training Results - 16 Classes")
        results_window.geometry("900x700")
        results_window.transient(self.root)
        
        # Center window
        results_window.update_idletasks()
        x = (results_window.winfo_screenwidth() // 2) - (900 // 2)
        y = (results_window.winfo_screenheight() // 2) - (700 // 2)
        results_window.geometry(f'+{x}+{y}')
        
        # Main frame
        main_frame = tk.Frame(results_window, bg=self.colors['white'], padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main_frame, text="📊 Model Training Results - 16 Classes", 
                font=('Arial', 20, 'bold'), 
                fg=self.colors['primary']).pack(pady=(0, 20))
        
        # Display plots
        try:
            img = Image.open(plots_path)
            img.thumbnail((800, 400))
            photo = ImageTk.PhotoImage(img)
            
            img_frame = tk.Frame(main_frame, bg=self.colors['white'])
            img_frame.pack(pady=10)
            
            img_label = tk.Label(img_frame, image=photo, bg=self.colors['white'])
            img_label.image = photo
            img_label.pack()
            
        except Exception as e:
            tk.Label(main_frame, text="⚠️ Could not load plots", 
                    font=('Arial', 12), 
                    fg=self.colors['danger']).pack(pady=50)
        
        # Summary
        if self.detector.history and self.detector.history.get('best_acc'):
            summary_frame = tk.Frame(main_frame, bg=self.colors['light'], relief=tk.FLAT)
            summary_frame.pack(fill=tk.X, pady=20, ipady=15)
            
            tk.Label(summary_frame, text="📈 Training Summary - 16 Classes", 
                    font=('Arial', 14, 'bold'), 
                    bg=self.colors['light'], 
                    fg=self.colors['secondary']).pack(pady=(10, 5))
            
            best_acc = self.detector.history.get('best_acc', 0)
            final_acc = self.detector.history['val_acc'][-1] if self.detector.history['val_acc'] else 0
            
            summary_text = f"Best Accuracy: {best_acc:.2f}%\n"
            summary_text += f"Final Accuracy: {final_acc:.2f}%\n"
            summary_text += f"Total Epochs: {len(self.detector.history['train_loss'])}\n"
            summary_text += f"Model: EfficientNet-B0 (Classifier only)\n"
            summary_text += f"Classes: {self.detector.num_classes}\n"
            summary_text += f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}"
            
            tk.Label(summary_frame, text=summary_text, 
                    font=('Arial', 11), 
                    bg=self.colors['light'], 
                    fg=self.colors['dark'],
                    justify=tk.LEFT).pack(pady=(0, 10))
    
    # ==================== DATASET MANAGER VIEW ====================
    def manage_dataset(self):
        """View and manage dataset information"""
        success, dataset_info = DatasetManager.check_dataset_status()
        
        info_window = tk.Toplevel(self.root)
        info_window.title("📁 Dataset Information - 16 Classes")
        info_window.geometry("650x600")
        info_window.transient(self.root)
        
        # Center window
        info_window.update_idletasks()
        x = (info_window.winfo_screenwidth() // 2) - (650 // 2)
        y = (info_window.winfo_screenheight() // 2) - (600 // 2)
        info_window.geometry(f'+{x}+{y}')
        
        # Main frame
        main_frame = tk.Frame(info_window, bg=self.colors['white'], padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main_frame, text="📁 Dataset Management - 16 Classes", 
                font=('Arial', 18, 'bold'), 
                fg=self.colors['primary']).pack(pady=(0, 20))
        
        # Status
        status_frame = tk.Frame(main_frame, bg=self.colors['light'], relief=tk.FLAT)
        status_frame.pack(fill=tk.X, pady=(0, 20), ipady=10)
        
        if success:
            class_count = dataset_info['num_classes']
            if class_count == 16:
                status_color = self.colors['success']
                status_text = f"✅ Dataset Ready - {class_count}/16 Classes"
            else:
                status_color = self.colors['warning']
                status_text = f"⚠ Dataset has {class_count}/16 Classes"
            
            tk.Label(status_frame, text=status_text, 
                    font=('Arial', 12, 'bold'), 
                    bg=self.colors['light'], 
                    fg=status_color).pack(pady=5)
        else:
            tk.Label(status_frame, text="❌ Dataset Issue", 
                    font=('Arial', 12, 'bold'), 
                    bg=self.colors['light'], 
                    fg=self.colors['danger']).pack(pady=5)
        
        # Path
        path_frame = tk.Frame(main_frame, bg=self.colors['white'])
        path_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(path_frame, text="📍 Location:", 
                font=('Arial', 11, 'bold'), 
                fg=self.colors['secondary'],
                bg=self.colors['white']).pack(anchor='w')
        
        tk.Label(path_frame, text=DATASET_PATH, 
                font=('Arial', 10), 
                fg=self.colors['info'],
                bg=self.colors['white'],
                wraplength=600,
                justify=tk.LEFT).pack(anchor='w', pady=(5, 0))
        
        # Dataset info
        if success:
            info_text = f"📊 DATASET STATISTICS (16 CLASSES REQUIRED)\n\n"
            info_text += f"Total Classes: {dataset_info['num_classes']}/16\n"
            info_text += f"Total Images: {dataset_info['total_images']}\n\n"
            info_text += "📁 CLASS DISTRIBUTION:\n"
            
            # Sort classes by image count
            sorted_classes = sorted(dataset_info['class_stats'].items(), 
                                  key=lambda x: x[1], reverse=True)
            
            for cls, count in sorted_classes:
                percentage = (count / dataset_info['total_images']) * 100
                info_text += f"  • {cls}: {count} images ({percentage:.1f}%)\n"
        else:
            info_text = f"⚠️ DATASET ISSUE\n\n"
            info_text += f"{dataset_info}\n\n"
            info_text += "📁 REQUIRED STRUCTURE (16 CLASSES):\n"
            info_text += f"{DATASET_PATH}/\n"
            info_text += "  ├── Class_1/\n"
            info_text += "  │   ├── image1.jpg\n"
            info_text += "  │   └── ...\n"
            info_text += "  ├── Class_2/\n"
            info_text += "  │   ├── image1.jpg\n"
            info_text += "  │   └── ...\n"
            info_text += "  └── ... (up to 16 classes)"
        
        # Text widget for dataset info
        text_frame = tk.Frame(main_frame, bg=self.colors['white'])
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        text_widget = scrolledtext.ScrolledText(text_frame, 
                                               font=('Consolas', 10),
                                               bg=self.colors['white'],
                                               fg=self.colors['secondary'],
                                               wrap=tk.WORD,
                                               padx=10,
                                               pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, info_text)
        text_widget.config(state=tk.DISABLED)
        
        # Action buttons
        button_frame = tk.Frame(main_frame, bg=self.colors['white'])
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        if not success and os.path.exists(DATASET_PATH):
            tk.Button(button_frame, text="📁 Open Dataset Folder", 
                     command=lambda: os.startfile(DATASET_PATH),
                     bg=self.colors['info'],
                     fg='white',
                     font=('Arial', 11, 'bold'),
                     relief=tk.FLAT,
                     cursor='hand2',
                     padx=20,
                     pady=8).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="❌ Close", 
                 command=info_window.destroy,
                 bg=self.colors['gray'],
                 fg='white',
                 font=('Arial', 11, 'bold'),
                 relief=tk.FLAT,
                 cursor='hand2',
                 padx=20,
                 pady=8).pack(side=tk.RIGHT, padx=5)
    
    # ==================== HISTORY VIEWER ====================
    def show_history(self):
        """Display prediction history"""
        if not self.history:
            messagebox.showinfo("History", "No prediction history yet.", parent=self.root)
            return
        
        history_window = tk.Toplevel(self.root)
        history_window.title("📊 Prediction History - 16 Classes")
        history_window.geometry("1000x600")
        history_window.transient(self.root)
        
        # Center window
        history_window.update_idletasks()
        x = (history_window.winfo_screenwidth() // 2) - (1000 // 2)
        y = (history_window.winfo_screenheight() // 2) - (600 // 2)
        history_window.geometry(f'+{x}+{y}')
        
        # Main frame
        main_frame = tk.Frame(history_window, bg=self.colors['white'], padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main_frame, text="📊 Prediction History - 16 Classes", 
                font=('Arial', 18, 'bold'), 
                fg=self.colors['primary']).pack(pady=(0, 20))
        
        # Statistics
        if self.history:
            stats_frame = tk.Frame(main_frame, bg=self.colors['light'], relief=tk.FLAT)
            stats_frame.pack(fill=tk.X, pady=(0, 20), ipady=10)
            
            total_predictions = len(self.history)
            unique_diseases = set([h['disease'] for h in self.history])
            avg_confidence = sum([h['confidence'] for h in self.history]) / total_predictions
            
            stats_text = f"Total Predictions: {total_predictions}  |  "
            stats_text += f"Unique Diseases: {len(unique_diseases)}/16  |  "
            stats_text += f"Avg Confidence: {avg_confidence:.1f}%"
            
            tk.Label(stats_frame, text=stats_text, 
                    font=('Arial', 11), 
                    bg=self.colors['light'], 
                    fg=self.colors['secondary']).pack(pady=10)
        
        # Treeview frame
        tree_frame = tk.Frame(main_frame, bg=self.colors['white'])
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview
        columns = ('Time', 'Image', 'Disease', 'Confidence')
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=15)
        
        # Define headings
        tree.heading('Time', text='Timestamp')
        tree.heading('Image', text='Image')
        tree.heading('Disease', text='Disease')
        tree.heading('Confidence', text='Confidence')
        
        # Define columns
        tree.column('Time', width=150)
        tree.column('Image', width=200)
        tree.column('Disease', width=300)
        tree.column('Confidence', width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack treeview and scrollbar
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add data
        for entry in reversed(self.history[-100:]):
            confidence = entry['confidence']
            if confidence >= 90:
                tag = 'excellent'
            elif confidence >= 75:
                tag = 'good'
            elif confidence >= 60:
                tag = 'fair'
            else:
                tag = 'poor'
            
            disease_display = entry['disease'].replace('_', ' ').replace('  ', ' ')
            if len(disease_display) > 40:
                disease_display = disease_display[:37] + "..."
            
            tree.insert('', tk.END, 
                       values=(
                           entry['timestamp'],
                           entry['image'],
                           disease_display,
                           f"{entry['confidence']:.1f}%"
                       ),
                       tags=(tag,))
        
        # Configure tags
        tree.tag_configure('excellent', foreground='#28a745')
        tree.tag_configure('good', foreground='#17a2b8')
        tree.tag_configure('fair', foreground='#ffc107')
        tree.tag_configure('poor', foreground='#dc3545')
        
        # Button frame
        button_frame = tk.Frame(main_frame, bg=self.colors['white'])
        button_frame.pack(fill=tk.X, pady=(15, 0))
        
        def clear_history():
            if messagebox.askyesno("Clear History", "Are you sure you want to clear all history?", 
                                 parent=history_window):
                self.history = []
                self.save_history()
                history_window.destroy()
                self.show_history()
        
        def export_history():
            try:
                export_path = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                    initialfile=f"plant_disease_history_16class_{datetime.now().strftime('%Y%m%d')}.csv",
                    parent=history_window
                )
                
                if export_path:
                    df = pd.DataFrame(self.history)
                    df.to_csv(export_path, index=False)
                    messagebox.showinfo("✅ Success", f"History exported to:\n{export_path}", 
                                      parent=history_window)
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}", parent=history_window)
        
        tk.Button(button_frame, text="📥 Export CSV", 
                 command=export_history,
                 bg=self.colors['info'],
                 fg='white',
                 font=('Arial', 11, 'bold'),
                 relief=tk.FLAT,
                 cursor='hand2',
                 padx=25,
                 pady=8).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="🗑️ Clear History", 
                 command=clear_history,
                 bg=self.colors['danger'],
                 fg='white',
                 font=('Arial', 11, 'bold'),
                 relief=tk.FLAT,
                 cursor='hand2',
                 padx=25,
                 pady=8).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="❌ Close", 
                 command=history_window.destroy,
                 bg=self.colors['gray'],
                 fg='white',
                 font=('Arial', 11, 'bold'),
                 relief=tk.FLAT,
                 cursor='hand2',
                 padx=25,
                 pady=8).pack(side=tk.RIGHT, padx=5)
    
    # ==================== DISEASE LIBRARY ====================
    def show_disease_library(self):
        """Display comprehensive disease library"""
        library_window = tk.Toplevel(self.root)
        library_window.title("📚 Disease Library - 16 Classes")
        library_window.geometry("900x700")
        library_window.transient(self.root)
        
        # Center window
        library_window.update_idletasks()
        x = (library_window.winfo_screenwidth() // 2) - (900 // 2)
        y = (library_window.winfo_screenheight() // 2) - (700 // 2)
        library_window.geometry(f'+{x}+{y}')
        
        # Main frame
        main_frame = tk.Frame(library_window, bg=self.colors['white'], padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main_frame, text="📚 Plant Disease Library - 16 Classes", 
                font=('Arial', 20, 'bold'), 
                fg=self.colors['primary']).pack(pady=(0, 10))
        
        tk.Label(main_frame, text="Comprehensive information about 16 common plant diseases", 
                font=('Arial', 11), 
                fg=self.colors['gray']).pack(pady=(0, 20))
        
        # Create notebook for diseases
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Disease categories
        diseases = {
            "🍅 Tomato": ["Tomato_Early_blight", "Tomato_Bacterial_spot", 
                         "Tomato_Tomato_mosaic_virus", "Tomato_healthy"],
            "🥔 Potato": ["Potato_Early_blight", "Potato_Late_blight", "Potato_healthy"],
            "🌶️ Pepper": ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy"],
        }
        
        for category, disease_list in diseases.items():
            category_frame = tk.Frame(notebook, bg=self.colors['white'])
            notebook.add(category_frame, text=category)
            
            # Create canvas and scrollbar for scrolling
            canvas = tk.Canvas(category_frame, bg=self.colors['white'], highlightthickness=0)
            scrollbar = ttk.Scrollbar(category_frame, orient=tk.VERTICAL, command=canvas.yview)
            scrollable_frame = tk.Frame(canvas, bg=self.colors['white'])
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Add disease information
            for disease in disease_list:
                info = self.detector.get_disease_info(disease)
                treatment = self.detector.get_treatment(disease)
                
                disease_card = tk.Frame(scrollable_frame, bg=self.colors['light'], 
                                       relief=tk.FLAT, padx=15, pady=15)
                disease_card.pack(fill=tk.X, padx=10, pady=10)
                
                # Disease name
                name_label = tk.Label(disease_card, 
                                    text=disease.replace('_', ' ').replace('healthy', '✅ Healthy Plant'),
                                    font=('Arial', 13, 'bold'),
                                    fg=self.colors['secondary'] if 'healthy' not in disease else self.colors['success'],
                                    bg=self.colors['light'])
                name_label.pack(anchor='w', pady=(0, 10))
                
                # Severity badge
                if 'healthy' not in disease:
                    severity = info.get('severity', 'Medium')
                    if severity in ['High', 'Very High']:
                        severity_color = self.colors['danger']
                    elif severity == 'Medium':
                        severity_color = self.colors['warning']
                    else:
                        severity_color = self.colors['success']
                    
                    severity_label = tk.Label(disease_card, 
                                            text=f"Severity: {severity}",
                                            font=('Arial', 9, 'bold'),
                                            fg='white',
                                            bg=severity_color,
                                            padx=8,
                                            pady=2)
                    severity_label.pack(anchor='w', pady=(0, 10))
                
                # Information
                info_text = f"🦠 Symptoms: {info['symptoms']}\n\n"
                info_text += f"🔬 Cause: {info['cause']}\n\n"
                
                if 'healthy' not in disease:
                    info_text += f"📅 Season: {info['season']}\n\n"
                
                info_text += f"🛡️ Prevention: {info['prevention']}"
                
                info_label = tk.Label(disease_card, 
                                    text=info_text,
                                    font=('Arial', 10),
                                    fg=self.colors['dark'],
                                    bg=self.colors['light'],
                                    justify=tk.LEFT,
                                    wraplength=650)
                info_label.pack(anchor='w')
                
                # Treatment (if not healthy)
                if 'healthy' not in disease:
                    tk.Frame(disease_card, bg=self.colors['gray'], height=1).pack(fill=tk.X, pady=15)
                    
                    tk.Label(disease_card, text="💊 Treatment Recommendations", 
                            font=('Arial', 11, 'bold'),
                            fg=self.colors['info'],
                            bg=self.colors['light']).pack(anchor='w', pady=(0, 5))
                    
                    treatment_text = treatment[:200] + "..." if len(treatment) > 200 else treatment
                    treatment_label = tk.Label(disease_card, 
                                             text=treatment_text,
                                             font=('Arial', 10),
                                             fg=self.colors['dark'],
                                             bg=self.colors['light'],
                                             justify=tk.LEFT,
                                             wraplength=650)
                    treatment_label.pack(anchor='w')
            
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # ==================== EXPORT FUNCTIONALITY ====================
    def export_data(self):
        """Export prediction data"""
        if not self.history:
            messagebox.showinfo("Export", "No data to export.", parent=self.root)
            return
        
        export_window = tk.Toplevel(self.root)
        export_window.title("📤 Export Data - 16 Classes")
        export_window.geometry("400x300")
        export_window.transient(self.root)
        
        # Center window
        export_window.update_idletasks()
        x = (export_window.winfo_screenwidth() // 2) - (400 // 2)
        y = (export_window.winfo_screenheight() // 2) - (300 // 2)
        export_window.geometry(f'+{x}+{y}')
        
        main_frame = tk.Frame(export_window, bg=self.colors['white'], padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main_frame, text="📤 Export Data - 16 Classes", 
                font=('Arial', 16, 'bold'), 
                fg=self.colors['primary']).pack(pady=(0, 20))
        
        tk.Label(main_frame, text="Choose export format:", 
                font=('Arial', 11), 
                fg=self.colors['secondary']).pack(pady=(0, 20))
        
        def export_csv():
            try:
                export_path = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                    initialfile=f"plant_disease_16class_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    parent=export_window
                )
                
                if export_path:
                    df = pd.DataFrame(self.history)
                    df.to_csv(export_path, index=False)
                    messagebox.showinfo("✅ Success", f"Data exported to:\n{export_path}", 
                                      parent=export_window)
                    export_window.destroy()
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}", parent=export_window)
        
        def export_json():
            try:
                export_path = filedialog.asksaveasfilename(
                    defaultextension=".json",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                    initialfile=f"plant_disease_16class_data_{datetime.now().strftime('%Y%m%d')}.json",
                    parent=export_window
                )
                
                if export_path:
                    with open(export_path, 'w') as f:
                        json.dump(self.history, f, indent=4)
                    messagebox.showinfo("✅ Success", f"Data exported to:\n{export_path}", 
                                      parent=export_window)
                    export_window.destroy()
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}", parent=export_window)
        
        # Export buttons
        tk.Button(main_frame, text="📊 Export as CSV", 
                 command=export_csv,
                 bg=self.colors['success'],
                 fg='white',
                 font=('Arial', 11, 'bold'),
                 relief=tk.FLAT,
                 cursor='hand2',
                 padx=30,
                 pady=10,
                 width=20).pack(pady=10)
        
        tk.Button(main_frame, text="📄 Export as JSON", 
                 command=export_json,
                 bg=self.colors['info'],
                 fg='white',
                 font=('Arial', 11, 'bold'),
                 relief=tk.FLAT,
                 cursor='hand2',
                 padx=30,
                 pady=10,
                 width=20).pack(pady=10)
        
        tk.Button(main_frame, text="❌ Cancel", 
                 command=export_window.destroy,
                 bg=self.colors['gray'],
                 fg='white',
                 font=('Arial', 11, 'bold'),
                 relief=tk.FLAT,
                 cursor='hand2',
                 padx=30,
                 pady=10,
                 width=20).pack(pady=10)
    
    # ==================== LOGOUT ====================
    def logout(self):
        """Logout current user"""
        self.user_manager.logout()
        self.show_login_screen()

# ==================== MAIN ENTRY POINT ====================
def main():
    """Main entry point for the application"""
    print("\n" + "="*60)
    print("🌿 PLANT DISEASE DETECTION SYSTEM - 16 CLASSES")
    print("="*60)
    print(f"Version: 2.0.3 - FIXED")
    print(f"System Directory: {SYSTEM_BASE}")
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*60)
    print("\n✅ Model Configuration:")
    print("   • Backbone: EfficientNet-B0 (FROZEN)")
    print("   • Classifier: Single Linear Layer (TRAINABLE)")
    print("   • Learning Rate: 0.0001")
    print("   • Optimizer: Adam")
    print("="*60)
    
    # Setup directories
    DatasetManager.setup_directories()
    
    try:
        # Start application
        root = tk.Tk()
        app = PlantDiseaseApp(root)
        root.mainloop()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
        
        # Show error dialog
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Fatal Error", 
                f"Application failed to start:\n\n{str(e)}\n\n"
                f"Please check:\n"
                f"1. All required packages are installed\n"
                f"2. Dataset path is correct (16 classes)\n"
                f"3. Sufficient disk space\n\n"
                f"Error details:\n{traceback.format_exc()}")
        except:
            pass

if __name__ == "__main__":
    main()