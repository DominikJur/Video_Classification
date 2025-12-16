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
    def __init__(self, model='resnet18'):
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
    

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        self.register_hooks()

    def save_feature_maps(self, module, input, output):
        # output shape is (Batch*Frames, Channels, Height, Width)
        self.feature_maps = output

    def save_gradients(self, grad):
        # grad shape is (Batch*Frames, Channels, Height, Width)
        self.gradients = grad

    def register_hooks(self):
        # 1. Register a hook to capture the output of the target layer (features)
        self.target_layer.register_forward_hook(self.save_feature_maps)
        
        # 2. Register a hook to capture the gradients flowing into the target layer
        self.target_layer.register_backward_hook(self.save_gradients)

    def __call__(self, input_tensor, target_category=None):
        # Ensure the model is in evaluation mode
        self.model.eval()

        # --- 1. Forward Pass ---
        # The model's output is the logits (Batch_size, Num_classes)
        logits = self.model(input_tensor)
        
        # --- 2. Target Identification ---
        if target_category is None:
            # Get the predicted class (the one with the highest logit)
            target_category = torch.argmax(logits, dim=1).item()
        
        # Zero gradients before backward pass
        self.model.zero_grad()
        
        # --- 3. Backward Pass ---
        # Create a tensor of 1s for the target class output for the backward pass
        # This is (Batch_size, Num_classes) but we only backpropagate the target class logit
        one_hot = torch.zeros_like(logits)
        one_hot[:, target_category] = 1
        
        # Backpropagate the gradient of the target class score
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # The gradients and feature_maps are now populated by the hooks
        gradients = self.gradients
        feature_maps = self.feature_maps # (B*T, C, H, W)

        # --- 4. Grad-CAM Computation ---
        batch_size, num_frames, _, _, _ = input_tensor.size()
        C, H, W = feature_maps.size(1), feature_maps.size(2), feature_maps.size(3)

        # Reshape the feature maps and gradients back to (B, T, C, H, W) for frame-wise processing
        feature_maps = feature_maps.view(batch_size, num_frames, C, H, W)
        gradients = gradients.view(batch_size, num_frames, C, H, W)
        
        # Initialize an empty list to store frame-level heatmaps
        cam_heatmaps = []

        for b in range(batch_size):
            # We process one video (sequence) at a time
            
            # Global Average Pooling on Gradients to get alpha_k^c (importance weights)
            # Take the mean over spatial dimensions (H, W) for each frame and channel
            # Shape is (Num_frames, Channels)
            alpha_k_c = torch.mean(gradients[b], dim=(2, 3), keepdim=False) 

            video_heatmaps = []
            for t in range(num_frames):
                # Calculate the weighted feature map for the current frame 't'
                # (Channels) @ (Channels, H, W) -> (H, W)
                weighted_features = (alpha_k_c[t].unsqueeze(-1).unsqueeze(-1) * feature_maps[b, t]).sum(dim=0)
                
                # Apply ReLU
                cam = F.relu(weighted_features)
                
                # Normalize and Upsample (assuming input image size is 224x224)
                cam = cam.cpu().detach().numpy()
                cam = cam - np.min(cam)
                cam = cam / (np.max(cam) + 1e-8)
                
                # Resize to match the original frame's spatial dimensions (e.g., 224x224)
                # You'll need to handle the actual resizing in your visualization code
                # For simplicity here, we just store the normalized CAM map
                video_heatmaps.append(cam)
                
            cam_heatmaps.append(video_heatmaps)

        return cam_heatmaps, target_category