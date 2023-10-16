import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        
        self.model = nn.Sequential(
            # 1st layer
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16*224* 224
            
            # 2nd layer
            nn.Conv2d(16, 32, 3, padding=1),  # 16x112x112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 32x56x56
            
            #3rd layer
            nn.Conv2d(32, 64, 3, padding=1),  # -> 32x56x56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 64x28x28
            
            # Since we are using BatchNorm and data augmentation,
            # 4th layer
            nn.Conv2d(64, 128, 3, padding=1),  # -> 64x28x28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 128x14x14
            nn.Dropout(p = dropout),
            
            # 5th layer
            nn.Conv2d(128, 256, 3, padding=1),  # -> 128x14x14
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 256x7x7
            nn.Dropout(p = dropout),
            
            nn.Flatten(),  # -> 12544
            nn.Linear(12544, 6500),  # -> 6500
            nn.Dropout(p = dropout),
            # Add batch normalization (BatchNorm1d, NOT BatchNorm2d) here
            nn.BatchNorm1d(6500),
            nn.ReLU(),
            
            nn.Linear(6500, 3000),  # 3000
            nn.Dropout(p = dropout),
            nn.BatchNorm1d(3000),
            nn.ReLU(),
            
            nn.Linear(3000, num_classes), => 50
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
