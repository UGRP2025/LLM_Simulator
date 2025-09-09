import torch
import torch.nn as nn
import torchvision.models as models

def create_model(num_outputs):
    """
    Creates a model with a pre-trained ResNet18 backbone.

    Args:
        num_outputs (int): The number of output classes (control bins).

    Returns:
        A PyTorch model.
    """
    # Load a pre-trained ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all the parameters in the pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    # Get the number of input features for the classifier
    num_ftrs = model.fc.in_features

    # Replace the final fully connected layer
    # A new layer with `num_outputs` is created. The parameters of this new layer are not frozen by default.
    model.fc = nn.Linear(num_ftrs, num_outputs)

    return model

if __name__ == '__main__':
    # Example of how to use the create_model function
    NUM_CONTROL_BINS = 6 # Example: 3 steering bins * 2 throttle bins
    model = create_model(NUM_CONTROL_BINS)
    print(model)

    # Create a dummy input tensor and pass it through the model
    dummy_input = torch.randn(1, 3, 224, 224) # (batch_size, channels, height, width)
    output = model(dummy_input)
    print(f"\nOutput shape: {output.shape}") # Should be (1, NUM_CONTROL_BINS)
