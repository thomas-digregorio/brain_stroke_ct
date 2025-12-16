import torch
import torch.nn as nn
import timm

class StrokeClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b4', pretrained=True, dropout_rate=0.2):
        super(StrokeClassifier, self).__init__()
        
        # Load Pre-trained Backbone
        # num_classes=0: This is the key trick.
        # Normally, EfficientNet ends with a layer for 1000 ImageNet classes.
        # Setting this to 0 removes that final layer entirely.
        # The model now acts as a Feature Extractor only. It inputs an image and outputs a raw vector of numbers (features) instead of predictions.
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Determine Input Features (EfficientNet-B4 usually 1792)
        in_features = self.backbone.num_features
        
        # Custom Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate), # Tunable Parameter
            nn.Linear(in_features, 2) # Output 2 raw logits (Normal, Stroke)
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        # Classify
        logits = self.classifier(features)
        return logits

class StrokeSegmentor(nn.Module):
    """
    Stub for future Segmentation Task.
    """
    def __init__(self):
        super(StrokeSegmentor, self).__init__()
        # TODO: Implement U-Net style decoder from EfficientNet features
        pass
    
    def forward(self, x):
        pass
