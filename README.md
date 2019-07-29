# pytorch-cifar

Drop-in replacements for the `torchvision` CIFAR-N datasets.

Fixes CIFAR-100 with coarse labels (ie, "CIFAR-20").

```py
from torchcifar import CIFAR20
from torch.utils.data import DataLoader
import torchvision.transforms as tf


transform_train = tf.Compose([
    tf.RandomCrop(32, padding=4),
    tf.RandomHorizontalFlip(),
    tf.ToTensor(),
    tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = CIFAR20(root='./data', train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
```
