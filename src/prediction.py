import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

class BaseModel:
    def __init__(self, model, model_path, classnames_path, transform, is_binary=False):
        self.device = torch.device("cpu")
        self.is_binary = is_binary
        self.class_names = self.load_class_names(classnames_path)
        self.model = model
        self.load_model(model_path)
        self.transform = transform

    def load_class_names(self, classnames_path):
        with open(classnames_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            if self.is_binary:
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).int().item()
            else:
                _, preds = torch.max(outputs, 1)
        return self.class_names[preds]

# Model A - Brain-Chest Binary Classifier
class XrayClassifierModel(BaseModel):
    def __init__(self, model_path, classnames_path):
        model = models.resnet152(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        super().__init__(model, model_path, classnames_path, transform, is_binary=True)

# Model B - Brain Classifier
class BrainDiseaseModel(BaseModel):
    def __init__(self, model_path, classnames_path):
        model = models.mobilenet_v2(pretrained=False)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 3)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        super().__init__(model, model_path, classnames_path, transform)

# Model C - Chest Classifier
class ChestDiseaseModel(BaseModel):
    def __init__(self, model_path, classnames_path):
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 7)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])
        super().__init__(model, model_path, classnames_path, transform)
