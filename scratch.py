
# Import libraries
import torch
import torch.nn as nn
import tracemalloc


# Define function to measure CPU memory
def measure_cpu_memory(model, x):

    # Move to CPU
    device = torch.device("cpu")  # Ensure execution on CPU
    model = model.to(device)

    # Start memory tracking
    tracemalloc.start()

    # Forward pass
    output = model(x)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Measure peak memory usage
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Return peak memory usage
    return peak_mem


# Main
if __name__ == '__main__':

    # Example Model
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(128 * 32 * 32, 10)
    )

    # Example input
    x = torch.randn(1, 3, 32, 32)

    # Measure CPU memory
    measure_cpu_memory(model, x, dtype=torch.float32)

    # Done
    print('Done')


# If you can do this...

import numpy as np

a = np.array([1, 2, 3, 4])
b = np.zeros((1, 4))
c = np.randn(1, 4)
d = a + b * c

# ...then you can do this

import torch

a = torch.tensor([1, 2, 3, 4])
b = torch.zeros(1, 4)
c = torch.randn(1, 4)
d = a + b * c




# If you can do this...
    
class MyClass:
    def __init__(self, z):
        self.y = 10
        self.z = z

    def myfunc(self, x):
        x = x + self.y + self.z
        return x

myclass = MyClass(10)
x = myclass.myfunc(20)

# ...then you can do this

class ParentClass:
    def __init__(self):
        self.y = 10

    def parentfunc(self, x):
        x = x + self.y
        return x
    
class ChildClass(ParentClass):
    def __init__(self, z):
        super().__init__()
        self.z = z

    def myfunc(self, x):
        x = self.parentfunc(x)
        x = x + self.z
        return x

myclass = ChildClass(10)
z = myclass.myfunc(20)




# If you can do this...
    
class ChildClass(ParentClass):
    def __init__(self, z):
        super().__init__()
        self.z = z

    def myfunc(self, x):
        x = self.parentfunc(x)
        x = x + self.z
        return x

myclass = ChildClass(10)
z = myclass.myfunc(20)

# ...then you can do this...

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_hidden):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(1, num_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_hidden, 1)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MyModel(10)
x = torch.randn(1, 1)
y = model(x)

# ...and you can do this

from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

mydataset = MyDataset([1, 2, 3, 4, 5])
x = mydataset[0]

# If you can do this...

data = np.array([1, 2, 3, 4, 5])

for x in data:
    y = myfunc(x)
    loss = sum((x-y)**2)

# ...then you can do this

for x in mydataset:
    torch.zero_grad()
    y = model(x)
    loss = sum((x-y)**2)
    loss.backward()
    optimizer.step()