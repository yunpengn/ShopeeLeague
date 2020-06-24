import os
import pandas as pd
import time
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.autograd import Variable

# Defines where the model is located.
model_path = 'classify_resnet_152_16.pth'
# Defines where the test data is located.
test_folder = './data/test'
# Defines how often to print progress.
print_batch_size = 50

# Defines how to pre-process the image data.
transform = transforms.Compose([transforms.RandomResizedCrop(200),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

# Loads the model and sets to eval mode.
model = torch.load(model_path)
model.eval()

count = 0
output = []

# Iterates over all images in test folder.
for file_name in os.listdir(test_folder):
    if count % print_batch_size == 0:
        print('Progress: #{}'.format(count))

    # Checks the file type.
    if not file_name.endswith('.jpg'):
        print('Skip for file {} due to abnormal extension name.'.format(file_name))
        continue

    # Loads the image.
    file_path = os.path.join(test_folder, file_name)
    image_raw = Image.open(file_path).convert('RGB')
    image = transform(image_raw).float()
    x = Variable(image.cuda()).unsqueeze(0)

    # Predicts the label.
    y = model(x)
    label = torch.argmax(y, 1).item()

    # Appends to output.
    output.append([file_name, label])

    # Increments the counter.
    count += 1

# Saves the result.
result = pd.DataFrame(lst, columns =['filename', 'category'])
result.to_csv('output.csv', index=False)
