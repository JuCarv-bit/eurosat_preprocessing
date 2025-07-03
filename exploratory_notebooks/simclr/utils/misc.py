import torch

def evaluate(classifier, backbone, loader, device):
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            features = backbone(images)
            outputs = classifier(features)
            total += labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    return correct / total * 100