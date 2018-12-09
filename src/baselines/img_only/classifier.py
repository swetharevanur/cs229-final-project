import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import sys
import copy
import sklearn
import sklearn.metrics
torch.manual_seed(7)

model_name = 'inception_v3'
sys.stdout = open(os.path.join('output', 'logs', '%s.log' % model_name), 'a+')

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale = (0.9, 1.0)), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(size = (224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '../../../data/img_data'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title = None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.savefig(os.path.join('.', 'output', model_name, '%_predictions.png' % model_name))
    plt.pause(0.001) # pause a bit so that plots are updated

def visualize_model(model, num_images = 6):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 32, shuffle = True, num_workers = 16)
                   for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    was_training = model.training
    model.eval()
    images_so_far = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode = was_training)
                    return
        model.train(mode = was_training)        

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs = 10):
    # https://github.com/keras-team/keras/issues/5475; to overcome loading errors
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    finetuning = True
    
    train_loss_history = []
    train_acc_history, val_acc_history = [], []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
                # train entire network after 5 epochs
                if finetuning == True and epoch >= 5:
                    for param in model.parameters(): 
                        param.requires_grad = True
                    print('Finetuning complete. Training entire network now.')
                    finetuning = False
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        train_loss_history.append(loss) # per batch loss
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'train':
                train_acc_history.append(float(epoch_acc.numpy())) # per epoch
            else:
                val_acc_history.append(float(epoch_acc.numpy())) # per epoch
            
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc, train_loss_history, train_acc_history, val_acc_history
        
def train_handler(model_name):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 32, shuffle = True, num_workers = 16)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(dataset_sizes)
    class_names = image_datasets['train'].classes

    if model_name == 'resnet18':
        model_conv = torchvision.models.resnet18(pretrained = True)
    elif model_name == 'resnet50':
        model_conv = torchvision.models.resnet50(pretrained = True)
    
    # freeze all layers
    for param in model_conv.parameters():
        param.requires_grad = False

    # parameters of newly constructed modules have requires_grad = True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 6)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # observe that only parameters of final layer are being optimized as opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr = 0.001, momentum = 0.9)

    # decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size = 7, gamma = 0.1)

    model, best_acc, train_loss_history, train_acc_history, val_acc_history = train_model(
        model_conv, criterion, optimizer_conv, exp_lr_scheduler, dataloaders, dataset_sizes)
    
    visualize_model(model)
    return model, best_acc, train_loss_history, train_acc_history, val_acc_history

def test_handler(model):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms['val']) 
                      for x in ['test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 32, shuffle = True, num_workers = 16)
                   for x in ['test']}
    
    preds_list, gt_list = [], []
    
    model.eval()
    with torch.no_grad():
        for images, labels in dataloaders['test']:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds_list.extend(list(np.asarray(torch.max(outputs, 1)[1])))
            gt_list.extend(list(np.asarray(labels)))
    
    return preds_list, gt_list

