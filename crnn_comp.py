import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from jiwer import cer, wer
import shutil


# ============================================
# PART 1: Dataset Class for CRNN
# ============================================
class CRNNDataset(Dataset):
    def __init__(self, dataframe, root_dir, converter, img_height=32, img_width=256, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.converter = converter
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_height, img_width)),
                transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        label = self.dataframe.iloc[idx, 1]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Encode label
        encoded_label = [self.converter.char_to_idx.get(char, 0) for char in label]
        
        return image, torch.LongTensor(encoded_label), len(label)


def collate_fn(batch):
    """Custom collate function to handle variable length labels"""
    images, labels, label_lengths = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Pad labels to same length
    max_len = max(label_lengths)
    padded_labels = []
    for label in labels:
        padded = torch.cat([label, torch.zeros(max_len - len(label), dtype=torch.long)])
        padded_labels.append(padded)
    
    padded_labels = torch.stack(padded_labels, 0)
    label_lengths = torch.LongTensor(label_lengths)
    
    return images, padded_labels, label_lengths


# ============================================
# PART 2: CRNN Model (from previous artifact)
# ============================================
class CRNN(nn.Module):
    def __init__(self, img_height, num_chars, rnn_hidden=256, num_rnn_layers=2):
        super(CRNN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),
            
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        self.cnn_output_height = img_height // 16
        self.rnn_input_size = 512 * self.cnn_output_height
        
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=rnn_hidden,
            num_layers=num_rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.3 if num_rnn_layers > 1 else 0
        )
        
        self.fc = nn.Linear(rnn_hidden * 2, num_chars)
        
    def forward(self, x):
        conv = self.cnn(x)
        batch, channels, height, width = conv.size()
        conv = conv.permute(0, 3, 1, 2)
        conv = conv.contiguous().view(batch, width, channels * height)
        
        rnn_out, _ = self.rnn(conv)
        output = self.fc(rnn_out)
        log_probs = torch.log_softmax(output, dim=2)
        log_probs = log_probs.permute(1, 0, 2)
        
        return log_probs


class CTCLabelConverter:
    def __init__(self, characters):
        self.characters = characters
        self.num_chars = len(characters) + 1
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(characters)}
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(characters)}
        self.idx_to_char[0] = '[blank]'
        
    def decode(self, indices_batch):
        if isinstance(indices_batch, torch.Tensor):
            indices_batch = indices_batch.cpu().numpy()
        
        texts = []
        for indices in indices_batch:
            decoded = []
            prev_idx = 0
            for curr_idx in indices:
                if curr_idx != 0 and curr_idx != prev_idx:
                    decoded.append(self.idx_to_char.get(curr_idx, ''))
                prev_idx = curr_idx
            texts.append(''.join(decoded))
        
        return texts
    
    def decode_greedy(self, log_probs_batch):
        _, indices = torch.max(log_probs_batch, dim=2)
        indices = indices.permute(1, 0)
        return self.decode(indices)


# ============================================
# PART 3: Helper Functions
# ============================================
def load_dataset(file_path):
    data = pd.read_csv(file_path, sep="\t", header=None, names=["image", "label"])
    return data


def build_character_set(data_file):
    data = pd.read_csv(data_file, sep='\t', header=None, names=['image', 'label'])
    all_chars = set()
    for label in data['label']:
        all_chars.update(label)
    characters = ''.join(sorted(all_chars))
    print(f"Found {len(characters)} unique characters")
    return characters


def save_checkpoint(model, optimizer, epoch, step, converter, checkpoint_dir="checkpoint_latest"):
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'characters': converter.characters,
    }, os.path.join(checkpoint_dir, 'checkpoint.pt'))
    
    print(f"\n✓ Checkpoint saved: Epoch {epoch}, Step {step}")


# ============================================
# PART 4: Training and Validation Functions
# ============================================
def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} - Training")
    for images, labels, label_lengths in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        
        # Forward pass
        log_probs = model(images)
        
        # Calculate input lengths
        batch_size = log_probs.size(1)
        input_lengths = torch.full(
            size=(batch_size,),
            fill_value=log_probs.size(0),
            dtype=torch.long,
            device=device
        )
        
        # Flatten labels for CTC
        labels_flat = labels.view(-1)
        
        # Calculate loss
        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, valid_loader, criterion, converter, device, epoch, total_epochs):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        progress_bar = tqdm(valid_loader, desc=f"Epoch {epoch}/{total_epochs} - Validation")
        for images, labels, label_lengths in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)
            
            # Forward pass
            log_probs = model(images)
            
            # Calculate input lengths
            batch_size = log_probs.size(1)
            input_lengths = torch.full(
                size=(batch_size,),
                fill_value=log_probs.size(0),
                dtype=torch.long,
                device=device
            )
            
            # Calculate loss
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            total_loss += loss.item()
            
            # Decode predictions
            predictions = converter.decode_greedy(log_probs)
            
            # Decode references
            references = []
            for i in range(len(labels)):
                ref_indices = labels[i][:label_lengths[i]].cpu().numpy()
                ref_text = ''.join([converter.idx_to_char.get(int(idx), '') for idx in ref_indices])
                references.append(ref_text)
            
            all_predictions.extend(predictions)
            all_references.extend(references)
            
            progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(valid_loader)
    
    # Calculate metrics
    cer_score = cer(all_references, all_predictions)
    wer_score = wer(all_references, all_predictions)
    
    return avg_loss, cer_score, wer_score, all_predictions, all_references


# ============================================
# PART 5: Main Training Script
# ============================================
def main():
    # Configuration
    img_height = 32
    img_width = 256
    batch_size = 32
    epochs = 20
    learning_rate = 0.001
    data_path = "data_v1"
    checkpoint_dir = "checkpoint_latest"
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    print("\n=== Loading Datasets ===")
    train_data = load_dataset(f"{data_path}/train.txt")
    valid_data = load_dataset(f"{data_path}/valid.txt")
    test_data = load_dataset(f"{data_path}/test.txt")
    
    print(f"Train: {len(train_data)} samples")
    print(f"Valid: {len(valid_data)} samples")
    print(f"Test: {len(test_data)} samples")
    
    # Build character set
    print("\n=== Building Character Set ===")
    characters = build_character_set(f"{data_path}/train.txt")
    converter = CTCLabelConverter(characters)
    print(f"Vocabulary size: {converter.num_chars} (including blank)")
    
    # Create datasets and dataloaders
    print("\n=== Creating DataLoaders ===")
    train_dataset = CRNNDataset(train_data, f"{data_path}/train/", converter, img_height, img_width)
    valid_dataset = CRNNDataset(valid_data, f"{data_path}/valid/", converter, img_height, img_width)
    test_dataset = CRNNDataset(test_data, f"{data_path}/test/", converter, img_height, img_width)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, collate_fn=collate_fn, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=4, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_fn, pin_memory=True)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Valid batches: {len(valid_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Initialize model
    print("\n=== Initializing Model ===")
    model = CRNN(img_height=img_height, num_chars=converter.num_chars, 
                 rnn_hidden=256, num_rnn_layers=2)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                            factor=0.5, patience=3, verbose=True)
    
    # Training loop
    print("\n=== Starting Training ===")
    training_losses = []
    validation_losses = []
    cer_scores = []
    wer_scores = []
    global_step = 0
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs)
        training_losses.append(train_loss)
        global_step += len(train_loader)
        
        print(f"\nEpoch {epoch}/{epochs} - Training Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, cer_score, wer_score, predictions, references = validate(
            model, valid_loader, criterion, converter, device, epoch, epochs
        )
        validation_losses.append(val_loss)
        cer_scores.append(cer_score)
        wer_scores.append(wer_score)
        
        print(f"Epoch {epoch}/{epochs} - Validation Loss: {val_loss:.4f}")
        print(f"Epoch {epoch}/{epochs} - CER: {cer_score:.4f}, WER: {wer_score:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Show sample predictions
        print("\n=== Sample Predictions ===")
        for i in range(min(3, len(predictions))):
            print(f"Pred: {predictions[i]}")
            print(f"Ref:  {references[i]}")
            print("-" * 60)
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, global_step, converter, checkpoint_dir)
    
    # Plot results
    print("\n=== Plotting Results ===")
    epochs_range = range(1, epochs + 1)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, training_losses, label='Training Loss', marker='o')
    plt.plot(epochs_range, validation_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, cer_scores, label='CER', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('CER')
    plt.title('Character Error Rate')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, wer_scores, label='WER', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('WER')
    plt.title('Word Error Rate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()
    
    # Test the model
    print("\n=== Testing Model ===")
    test_loss, test_cer, test_wer, test_predictions, test_references = validate(
        model, test_loader, criterion, converter, device, epochs, epochs
    )
    
    print(f"\n=== Test Results ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test CER: {test_cer:.4f}")
    print(f"Test WER: {test_wer:.4f}")
    
    print("\n=== Sample Test Predictions ===")
    for i in range(min(10, len(test_predictions))):
        print(f"\n{i+1}.")
        print(f"Prediction: {test_predictions[i]}")
        print(f"Reference:  {test_references[i]}")
        print("-" * 60)
    
    # Save final model
    print("\n=== Saving Final Model ===")
    final_save_path = "khmer_crnn_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'characters': converter.characters,
        'img_height': img_height,
        'img_width': img_width,
    }, final_save_path)
    print(f"Model saved to {final_save_path}")
    
    print("\n=== Training Complete ===")


if __name__ == "__main__":
    main()