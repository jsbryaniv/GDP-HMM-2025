import torch

import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage
if __name__ == "__main__":
    input_size = 10
    hidden_size = 20
    output_size = 1

    model = SimpleModel(input_size, hidden_size, output_size)
    print(model)

    # Example input
    x = torch.randn(1, input_size)
    output = model(x)
    print(output)