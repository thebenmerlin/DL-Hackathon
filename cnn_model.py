import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class CNNFeatureExtractor(nn.Module):
    """
    CNN-based feature extractor using pre-trained ResNet-50.
    Removes the final classification layer to get feature vectors.
    """
    def __init__(self, embed_size=256):
        super(CNNFeatureExtractor, self).__init__()
        
        # Load pre-trained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Remove the final classification layer
        # Take all layers except the final FC layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Embedding layer to project features to desired dimension
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        
    def forward(self, images):
        """
        Extract features from images.
        Args:
            images: Tensor of shape (batch_size, 3, 224, 224)
        Returns:
            features: Tensor of shape (batch_size, embed_size)
        """
        with torch.no_grad():
            features = self.resnet(images)
        
        # Flatten
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
    def get_transform(self):
        """Returns the image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
