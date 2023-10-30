import os 
import argparse
import torch 
from torch import nn, optim
import torchvision 
from torchvision import datasets, transforms, models 

def argparser(): 
    parser = argeparse.ArgumentParser(description='parsing train.py')
    parser.add_argument('data_dir', help='data directory')
    parser.add_argument('--arch', dest='arch', action='store', default='vgg16', type=str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./fl_classifier.pth")
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    parser.add_argument('--learning_rate', type=float, dest='lr', action='store', default=0.001)
    parser.add_argument('--epochs', type=int, dest='epochs', action='store', default=5)
    args = parser.parse_args()
    return args
def main(): 
    args = argparser()
    
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=50)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=50)
    
    if arch == 'vgg16': 
        model = models.vgg16(pretrained=True)
        for param in model.parameters(): 
            param.requires_grad = False
            classifier = nn.Sequential(
        nn.Linear(25088, 120), 
        nn.ReLU(),
        nn.Dropout(0.5), 
        nn.Linear(120, 90), 
        nn.ReLU(),
        nn.Linear(90, 70), 
        nn.ReLU(),
        nn.Linear(70, 102),  
        nn.LogSoftmax(dim=1)
    )
    elif arch == 'resnet50': 
        model = models.resnet50(pretrained=True)
        
        for param in model.parameters(): 
            param.requires_grad=False
            classifier = nn.Sequential(
        nn.Linear(25088, 120), 
        nn.ReLU(),
        nn.Dropout(0.5), 
        nn.Linear(120, 90), 
        nn.ReLU(),
        nn.Linear(90, 70), 
        nn.ReLU(),
        nn.Linear(70, 102),  
        nn.LogSoftmax(dim=1)
    )
    elif arch=='Densenet': 
        model = models.densenet121(pretrained=True)
        for param in model.parameters(): 
            param.requires_grad=False
            classifier = nn.Sequential(
        nn.Linear(25088, 120), 
        nn.ReLU(),
        nn.Dropout(0.5), 
        nn.Linear(120, 90), 
        nn.ReLU(),
        nn.Linear(90, 70), 
        nn.ReLU(),
        nn.Linear(70, 102),  
        nn.LogSoftmax(dim=1)
    )
    elif arch=='alexnet': 
        model = models.alexnet(pretrained=True)
        for param in model.parameters(): 
            param.requires_grad=False
            classifier = nn.Sequential(
        nn.Linear(25088, 120), 
        nn.ReLU(),
        nn.Dropout(0.5), 
        nn.Linear(120, 90), 
        nn.ReLU(),
        nn.Linear(90, 70), 
        nn.ReLU(),
        nn.Linear(70, 102),  
        nn.LogSoftmax(dim=1)
    )
    else: 
        
        raise ValueError("Invalid architecture")
            
            
    
    model.classifier = classifier
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu == 'gpu' else "cpu")
    model.to(device)
    
    lr = args.learning_rate
    epochs = args.epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    print("Training process initializing .....\n")


    for epoch in range(epochs):
        running_loss = 0
        model.train()
        
        for ii, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
        model.eval()

        with torch.no_grad():
            test_loss, accuracy = validation(model, testloader, criterion, device)
        
        print(f"Epoch: {e+1}/{epochs} | Training Loss: {running_loss/len(trainloader):.4f} | Validation Loss: {test_loss/len(testloader):.4f} | Validation Accuracy: {accuracy/len(testloader):.4f}")
        
    model.class_to_idx = class_to_idx
    checkpoint = {'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'epochs': args.epochs,
                  'optim_stat_dict': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'architecture': args.arch
                 }

    torch.save(checkpoint, os.path.join(args.save_directory, "checkpoint.pth"))
    print("Model has been saved to {}".format(os.path.join(args.save_directory, "checkpoint.pth")))
    
                

if __name__ == '__main':
    main()
    
    