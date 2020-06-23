import time
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models

from torch.autograd import Variable

# Defines the batch size to read from datasets.
batch_size = 32
# Defines the learning rate.
learning_rate = 0.01
# Defines epoch - # of times to train the model.
epoch = 100

# Defines how to pre-process the image data.
transform = transforms.Compose([transforms.RandomResizedCrop(200),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

# Defines where the image datasets are located.
train_data = dsets.ImageFolder('./data/train', transform=transform)
test_data = dsets.ImageFolder('./data/test', transform=transform)

# Defines the input data.
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# Creates the model.
model = models.resnet152(pretrained=True)							# Initializes ResNet with 512 layers.
model.fc = torch.nn.Linear(2048, 2)									# Changes the output FC (fully conected) layer.
model = model.cuda()												# Uses GPU to accelerate the training.
criterion = torch.nn.CrossEntropyLoss()								# Defines the loss function.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)	# Defines the optimizer function.

def train():
	# Enables the training mode.
    model.train()

    total_loss = 0

    # Iterates through each image in train set.
    for image, label in train_loader:
        image = Variable(image.cuda())
        label = Variable(label.cuda())
        optimizer.zero_grad()

        target = model(image)
        loss = criterion(target, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Returns loss rate.
    return total_loss / float(len(train_loader))

def evaluate():
	# Enables the evaluation mode.
    model.eval()

    corrects = 0
    eval_loss = 0

    # Iterates through each image in test set.
    for image, label in test_loader:
        image = Variable(image.cuda())
        label = Variable(label.cuda())
        pred = model(image)
        loss = criterion(pred, label)

        eval_loss += loss.item()
        corrects += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()

    # Returns loss rate.
    return eval_loss / float(len(test_loader)), corrects, corrects * 100.0 / len(test_loader), len(test_loader)

train_loss = []
valid_loss = []
accuracy = []

# Repeats for # of times.
for epoch in range(1, epoch + 1):
	# Marks the start of current epoch.
	print('-' * 10)
	print('| # epoch {:3d} | current time {}'.format(epoch, time.ctime()))
	print('-' * 10)

	# Performs training.
    start_time = time.time()
    loss = train()
    train_loss.append(loss * 1000.)

    # Prints result.
    time_elapse = time.time() - start_time
    print('| train | time: {:2.2f}s | loss {:5.6f}'.format(time_elapse, loss))
    print('-' * 10)

    # Performs evaluation.
    start_time = time.time()
    loss, corrects, acc, size = evaluate()
    valid_loss.append(loss * 1000.)
    accuracy.append(acc)

    # Prints result.
    time_elapse = time.time() - start_time
    print('| eval | time: {:2.2f}s | loss {:.4f} | accuracy {}% ({}/{})'.format(time_elapse, loss, acc, corrects, size))
    print('-' * 10)
