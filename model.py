import torch
import torch.nn as nn
import torchvision.models as models

class MRNet_ResNet18(nn.Module): # Renamed class
    def __init__(self, n_classes=1, pretrained=True):
        """
        Args:
            n_classes (int): Number of output classes (1 for binary classification).
            pretrained (bool): Whether to use ImageNet pretrained weights for ResNet18.
        """
        super().__init__()

        
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # ResNet18's last fully connected layer is resnet.fc
        # It has in_features (512 for ResNet18) and (default) 1000 outputs
        num_ftrs = resnet.fc.in_features # This will be 512 for ResNet18

        # Remove the original fully connected layer to get the backbone
        self.base_model = nn.Sequential(*list(resnet.children())[:-1])

        # Add our custom fully connected layer for classification
        self.fc = nn.Linear(num_ftrs, n_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_slices, channels, height, width).
                              For ResNet18, channels should ideally be 3.
        """
        batch_size, num_slices, channels, height, width = x.shape

        # ResNet18 expects 3-channel input. MRI slices are single-channel.
        # Dataloader should prepare 3-channel input, but this is a safeguard.
        if channels == 1:
            x = x.repeat(1, 1, 3, 1, 1) # (batch_size, num_slices, 3, height, width)
        elif channels != 3:
            raise ValueError(f"Input tensor expected to have 1 or 3 channels, got {channels}")


        # Merge batch_size and num_slices dimensions to process all slices through ResNet18
        x = x.view(batch_size * num_slices, 3, height, width)

        # Pass through ResNet18 backbone
        # Output shape of self.base_model is (batch_size * num_slices, num_ftrs, 1, 1)
        features = self.base_model(x)

        # Flatten the features
        features = features.view(batch_size * num_slices, -1)

        # Reshape features back to separate batch and slice dimensions
        features = features.view(batch_size, num_slices, -1) # (batch_size, num_slices, num_ftrs)

        # Perform max pooling across the slice dimension
        pooled_features, _ = torch.max(features, dim=1) # (batch_size, num_ftrs)

        # Pass the aggregated features through our custom fully connected layer
        out = self.fc(pooled_features) # (batch_size, n_classes)

        return out

if __name__ == '__main__':
    # Example usage:
    dummy_input = torch.randn(2, 10, 3, 224, 224) # Assuming 3-channel input from dataloader now

    model = MRNet_ResNet18(n_classes=1, pretrained=True)
    output = model(dummy_input)

    print("Dummy input shape:", dummy_input.shape)
    print("Model output shape:", output.shape) # Expected: torch.Size([2, 1])
    print(f"Number of features for FC layer (ResNet18): {model.fc.in_features}") # Should be 512

    # Test with single channel input to check repeat logic (though dataloader should handle it)
    dummy_input_single_channel = torch.randn(2, 10, 1, 224, 224)
    output_single = model(dummy_input_single_channel)
    print("Dummy single channel input shape:", dummy_input_single_channel.shape)
    print("Model output shape (from single channel):", output_single.shape)
