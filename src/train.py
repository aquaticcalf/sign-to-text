import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import argparse

from data.isl_dataloader import get_isl_dataloaders
from models.sign_language_model import SignLanguageFrameCNN, SignLanguageSequenceModel

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device='cuda'):
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Record history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # Deep copy the model if it's the best accuracy
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), 'best_model.pth')
    
    return model, history

def main(args):
    # Get class mapping to determine number of classes
    mapping_path = Path(args.data_dir) / args.level / 'class_mapping.json'
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
        classes = mapping['classes']
        num_classes = len(classes)
    
    print(f"Dataset has {num_classes} classes")
    
    # Create dataloaders
    dataloaders = get_isl_dataloaders(
        data_dir=args.data_dir,
        level=args.level,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_sequences=args.use_sequences,
        max_seq_len=args.seq_len
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model based on sequence or frame mode
    if args.use_sequences:
        model = SignLanguageSequenceModel(num_classes=num_classes, pretrained=True)
    else:
        model = SignLanguageFrameCNN(num_classes=num_classes, pretrained=True)
    
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train the model
    model, history = train_model(
        model, 
        dataloaders, 
        criterion, 
        optimizer, 
        num_epochs=args.epochs,
        device=device
    )
    
    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'classes': classes
    }, 'final_model.pth')
    
    print("Training complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sign Language Recognition Model")
    parser.add_argument('--data_dir', type=str, default='data/isl_dataset',
                        help='Directory with processed dataset')
    parser.add_argument('--level', type=str, default='word_level',
                        choices=['word_level', 'sentence_level'],
                        help='Level of sign language to use (word or sentence)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--use_sequences', action='store_true',
                        help='Use sequence model instead of frame-by-frame')
    parser.add_argument('--seq_len', type=int, default=20,
                        help='Maximum sequence length for sequence model')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    
    args = parser.parse_args()
    main(args) 