import os
import time
import torch
import torchvision.transforms as transforms

from PIL import Image

# Defines where the model is located.
model_path = 'classify_resnet_152_10.pth'
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

# Iterates over all images in test folder.
count = 0
for file_name in os.listdir(test_folder):
	if count % print_batch_size == 0:
		print('Progress: #{}'.format(count))

	# Checks the file type.
	if not file_name.endswith('.jpg'):
		print('Skip for file {} due to abnormal extension name.'.format(file_name))
		continue

	# Loads the image.
	file_path = os.path.join(test_folder, file_name)
	image = Image.open(file_path).convert('RGB')
	x = transform(image)

	# Predicts the label.
	y = model(x)
	label = torch.argmax(y, 1)

	# Increments the counter.
	count += 1
