import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    """
    CRNN (Convolutional Recurrent Neural Network) with CTC Loss
    Architecture: CNN (feature extraction) -> RNN (sequence modeling) -> CTC (decoding)
    """
    
    def __init__(self, img_height, num_chars, rnn_hidden=256, num_rnn_layers=2):
        """
        Args:
            img_height: Height of input images
            num_chars: Number of unique characters in vocabulary (including blank)
            rnn_hidden: Hidden size for RNN layers
            num_rnn_layers: Number of bidirectional LSTM layers
        """
        super(CRNN, self).__init__()
        
        self.img_height = img_height
        self.num_chars = num_chars
        
        # CNN layers for feature extraction
        # Input: (batch, 1, height, width) for grayscale or (batch, 3, height, width) for RGB
        self.cnn = nn.Sequential(
            # Conv Block 1: (3, H, W) -> (64, H/2, W/2)
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by 2
            
            # Conv Block 2: (64, H/2, W/2) -> (128, H/4, W/4)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by 2
            
            # Conv Block 3: (128, H/4, W/4) -> (256, H/4, W/8)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),  # Downsample height only
            
            # Conv Block 4: (256, H/4, W/8) -> (512, H/8, W/8)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),  # Downsample height only
            
            # Conv Block 5: (512, H/8, W/8) -> (512, H/8, W/8-2)
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Calculate CNN output height after all pooling operations
        # For input height 32: 32 -> 16 -> 8 -> 4 -> 2 -> 2
        self.cnn_output_height = img_height // 16  # Adjust based on your pooling
        
        # RNN input size is (batch, seq_len, features)
        # seq_len comes from width dimension, features = channels * height
        self.rnn_input_size = 512 * self.cnn_output_height
        
        # Bidirectional LSTM layers
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=rnn_hidden,
            num_layers=num_rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.3 if num_rnn_layers > 1 else 0
        )
        
        # Fully connected layer for classification
        # *2 because bidirectional LSTM
        self.fc = nn.Linear(rnn_hidden * 2, num_chars)
        
    def forward(self, x):
        """
        Args:
            x: Input images (batch, channels, height, width)
        Returns:
            log_probs: Log probabilities (batch, seq_len, num_chars)
        """
        # CNN feature extraction
        # x: (batch, 3, height, width)
        conv = self.cnn(x)  # (batch, 512, H', W')
        
        # Reshape for RNN: (batch, channels, height, width) -> (batch, width, channels*height)
        batch, channels, height, width = conv.size()
        conv = conv.permute(0, 3, 1, 2)  # (batch, width, channels, height)
        conv = conv.contiguous().view(batch, width, channels * height)  # (batch, seq_len, features)
        
        # RNN sequence modeling
        rnn_out, _ = self.rnn(conv)  # (batch, seq_len, hidden*2)
        
        # Fully connected layer
        output = self.fc(rnn_out)  # (batch, seq_len, num_chars)
        
        # Apply log_softmax for CTC loss
        log_probs = F.log_softmax(output, dim=2)  # (batch, seq_len, num_chars)
        
        # CTC expects (seq_len, batch, num_chars)
        log_probs = log_probs.permute(1, 0, 2)  # (seq_len, batch, num_chars)
        
        return log_probs


class CTCLabelConverter:
    """
    Convert between text labels and CTC-compatible encodings
    """
    
    def __init__(self, characters):
        """
        Args:
            characters: String of all unique characters (e.g., "abcdefgh...")
        """
        self.characters = characters
        self.num_chars = len(characters) + 1  # +1 for CTC blank token
        
        # Create char-to-index and index-to-char mappings
        # Index 0 is reserved for CTC blank token
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(characters)}
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(characters)}
        self.idx_to_char[0] = '[blank]'  # CTC blank token
        
    def encode(self, text_batch):
        """
        Encode text labels to indices
        Args:
            text_batch: List of strings
        Returns:
            encoded: List of encoded sequences
            lengths: Tensor of sequence lengths
        """
        encoded = []
        lengths = []
        
        for text in text_batch:
            encoded_text = [self.char_to_idx.get(char, 0) for char in text]
            encoded.append(encoded_text)
            lengths.append(len(encoded_text))
        
        # Pad sequences to same length
        max_len = max(lengths)
        padded_encoded = []
        for seq in encoded:
            padded_encoded.append(seq + [0] * (max_len - len(seq)))
        
        return torch.LongTensor(padded_encoded), torch.LongTensor(lengths)
    
    def decode(self, indices_batch, lengths=None):
        """
        Decode CTC output to text
        Args:
            indices_batch: Tensor of shape (batch, seq_len) or list of sequences
            lengths: Optional tensor of actual sequence lengths
        Returns:
            texts: List of decoded strings
        """
        if isinstance(indices_batch, torch.Tensor):
            indices_batch = indices_batch.cpu().numpy()
        
        texts = []
        for idx, indices in enumerate(indices_batch):
            if lengths is not None:
                indices = indices[:lengths[idx]]
            
            # CTC decoding: remove blanks and repeated characters
            decoded = []
            prev_idx = 0
            for curr_idx in indices:
                if curr_idx != 0 and curr_idx != prev_idx:  # Skip blank and repeats
                    decoded.append(self.idx_to_char.get(curr_idx, ''))
                prev_idx = curr_idx
            
            texts.append(''.join(decoded))
        
        return texts
    
    def decode_greedy(self, log_probs_batch):
        """
        Greedy decoding from log probabilities
        Args:
            log_probs_batch: Tensor of shape (seq_len, batch, num_chars)
        Returns:
            texts: List of decoded strings
        """
        # Get most likely indices at each time step
        _, indices = torch.max(log_probs_batch, dim=2)  # (seq_len, batch)
        indices = indices.permute(1, 0)  # (batch, seq_len)
        
        return self.decode(indices)


# Example usage and helper functions
def build_character_set(data_file):
    """
    Build character set from training data
    Args:
        data_file: Path to training labels file (tab-separated: image\tlabel)
    Returns:
        characters: String of all unique characters
    """
    import pandas as pd
    
    data = pd.read_csv(data_file, sep='\t', header=None, names=['image', 'label'])
    all_chars = set()
    
    for label in data['label']:
        all_chars.update(label)
    
    # Sort for consistency
    characters = ''.join(sorted(all_chars))
    print(f"Found {len(characters)} unique characters")
    print(f"Character set: {characters[:50]}..." if len(characters) > 50 else f"Character set: {characters}")
    
    return characters


def calculate_ctc_input_lengths(img_width, cnn_output_width_ratio=8):
    """
    Calculate CTC input lengths based on CNN downsampling
    Args:
        img_width: Input image width
        cnn_output_width_ratio: How much the width is downsampled (default: 8 for typical CRNN)
    Returns:
        input_length: Sequence length after CNN
    """
    return img_width // cnn_output_width_ratio


# Training example
def train_step(model, optimizer, criterion, images, labels, label_lengths, device):
    """
    Single training step
    Args:
        model: CRNN model
        optimizer: Optimizer
        criterion: CTC loss criterion
        images: Batch of images (batch, channels, height, width)
        labels: Encoded labels (batch, max_label_length)
        label_lengths: Actual label lengths (batch,)
        device: torch device
    Returns:
        loss: Loss value
    """
    model.train()
    images = images.to(device)
    labels = labels.to(device)
    
    # Forward pass
    log_probs = model(images)  # (seq_len, batch, num_chars)
    
    # Calculate input lengths (seq_len for all samples in batch)
    input_lengths = torch.full(
        size=(log_probs.size(1),), 
        fill_value=log_probs.size(0), 
        dtype=torch.long
    )
    
    # CTC Loss
    # log_probs: (T, N, C) - time, batch, classes
    # labels: (N, S) - batch, max label length
    # input_lengths: (N,) - length of each sequence
    # label_lengths: (N,) - length of each label
    
    # Flatten labels for CTC loss
    labels_flat = labels.view(-1)
    
    loss = criterion(log_probs, labels, input_lengths, label_lengths)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    
    optimizer.step()
    
    return loss.item()


# Example instantiation
if __name__ == "__main__":
    # Parameters
    img_height = 32
    img_width = 256
    batch_size = 16
    
    # Build character set from your data
    characters = "abcdefghijklmnopqrstuvwxyzកខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហឡអ" 
    # Initialize converter and model
    converter = CTCLabelConverter(characters)
    model = CRNN(
        img_height=img_height,
        num_chars=converter.num_chars,
        rnn_hidden=256,
        num_rnn_layers=2
    )
    
    # CTC Loss
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Model created successfully!")
    print(f"Number of characters: {converter.num_chars}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")