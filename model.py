import torch
import torch.nn as nn
import torchvision.models as models
import timm 


class CNN3D(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN3D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=3,         # RGB channels
                out_channels=32,
                kernel_size=(3, 3, 3), 
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) # Pool spatially, not temporally
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Linear(64, num_classes) # num_classes=2 for binary

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) 
        logits = self.fc(x)
        return logits
    
class ResnetVideoToSequenceEncoder(nn.Module):
    def __init__(self, model='resnet50_ft_caffe_vggface2'):
        super(ResnetVideoToSequenceEncoder, self).__init__()
        self.model = timm.create_model(
                model, 
                pretrained=True, 
                num_classes=0 # No classification layer
        )
        
    def forward(self, x):
        batch_size, num_frames, C, H, W = x.size()
        x = x.view(batch_size * num_frames, C, H, W)  # Merge batch and frames
        features = self.model(x)  # Extract features
        feature_dim = features.size(1)
        features = features.view(batch_size, num_frames, feature_dim)  # Reshape back
        return features
    
class GRUSequenceClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes=2):
        super(GRUSequenceClassifier, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        output, hidden = self.gru(x) 
        
        # NOTE:
        # The 'hidden' tensor contains the final state of the Forward pass 
        # AND the final state of the Backward pass (which implies it saw the whole video).
        # We take the last layer's forward and backward states
        # hidden[-2] is the last Forward layer
        # hidden[-1] is the last Backward layer
        
        final_state = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        logits = self.fc(final_state)
        return logits
    
    # NOTE: CROSSENTROPY LOSS  DOES SOFTMAX AUTOMATICALLY
    # NOTE: GRU APPLIES TANH ACTIVATION INTERNALLY
    
class VideoClassificationModel(nn.Module):
    def __init__(self, 
                 encoder_model='resnet50_ft_caffe_vggface2',
                 gru_hidden_dim=256,
                 gru_num_layers=2,
                 num_classes=2):
        super(VideoClassificationModel, self).__init__()
        self.encoder = ResnetVideoToSequenceEncoder(model=encoder_model)
        
        embed_dim = self.encoder.model.num_features
        
        self.classifier = GRUSequenceClassifier(
            input_dim=embed_dim, 
            hidden_dim=gru_hidden_dim, 
            num_layers=gru_num_layers, 
            num_classes=num_classes
        )
        
    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits