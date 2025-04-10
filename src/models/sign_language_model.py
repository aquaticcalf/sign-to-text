import torch
import torch.nn as nn
import torchvision.models as models

class SignLanguageFrameCNN(nn.Module):
    """CNN model for single frame sign language recognition"""
    def __init__(self, num_classes, pretrained=True):
        super(SignLanguageFrameCNN, self).__init__()
        # Use a pretrained ResNet as the backbone
        resnet = models.resnet50(weights='DEFAULT' if pretrained else None)
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Add a new fully connected layer for classification
        self.classifier = nn.Linear(resnet.fc.in_features, num_classes)
        
    def forward(self, x):
        # Extract features
        x = self.features(x)
        # Flatten
        x = torch.flatten(x, 1)
        # Classify
        x = self.classifier(x)
        return x

class SignLanguageSequenceModel(nn.Module):
    """Sequence model for sign language recognition"""
    def __init__(self, num_classes, pretrained=True, bidirectional=True):
        super(SignLanguageSequenceModel, self).__init__()
        # Use a pretrained ResNet as frame feature extractor
        resnet = models.resnet50(weights='DEFAULT' if pretrained else None)
        # Remove the final fully connected layer
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        
        # LSTM to process the sequence of frame features
        self.hidden_size = 512
        self.lstm = nn.LSTM(
            resnet.fc.in_features, 
            self.hidden_size, 
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Final classifier
        lstm_output_size = self.hidden_size * 2 if bidirectional else self.hidden_size
        self.classifier = nn.Linear(lstm_output_size, num_classes)
        
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        
        # Process each frame with CNN
        cnn_output = []
        for t in range(seq_len):
            # Extract features for each frame
            frame_features = self.cnn(x[:, t])
            frame_features = torch.flatten(frame_features, 1)
            cnn_output.append(frame_features)
            
        # Stack the frame features
        cnn_output = torch.stack(cnn_output, dim=1)  # [batch_size, seq_len, features]
        
        # Process the sequence with LSTM
        lstm_out, _ = self.lstm(cnn_output)
        
        # Use the final timestep output for classification
        final_out = lstm_out[:, -1]
        
        # Classify
        output = self.classifier(final_out)
        return output 