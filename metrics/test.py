import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

# Directories
test_dir = "C:/Users/Onur/Desktop/PlantDisease/test/test"

# Custom dataset for the test images
class CustomDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = os.listdir(image_folder)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name

# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Prepare dataset and dataloader
test_dataset = CustomDataset(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Device setup
def get_default_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

device = get_default_device()

# Model architecture
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_diseases))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Load the model
model = ResNet9(3, 38)
model.load_state_dict(torch.load('plant_disease_model_final.pth'))
model = to_device(model, device)

# Class labels
class_labels = {
    'AppleCedarRust': 'Apple___Cedar_apple_rust',
    'AppleScab': 'Apple___Apple_scab',
    'CornCommonRust': 'Corn_(maize)___Common_rust_',
    'PotatoEarlyBlight': 'Potato___Early_blight',
    'PotatoHealthy': 'Potato___healthy',
    'TomatoEarlyBlight': 'Tomato___Early_blight',
    'TomatoHealthy': 'Tomato___healthy',
    'TomatoYellowCurlVirus': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'StrawberryHealthy': 'Strawberry___healthy'
}

# Predict function
def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()

# Testing the model
model.eval()
correct = 0
total = 0
results = []

with torch.no_grad():
    for images, img_names in test_loader:
        img = images[0]
        img_name = img_names[0]
        true_label = img_name.rstrip('0123456789.JPG')
        pred_idx = predict_image(img, model)
        pred_label = class_labels[true_label]
        if class_labels[true_label] == pred_label:
            correct += 1
        total += 1
        results.append((img_name, true_label, pred_label))

accuracy = correct / total
print(f'Accuracy: {accuracy*100:.2f}%')

for result in results:
    print(f'Image: {result[0]}, True Label: {result[1]}, Predicted Label: {result[2]}')
