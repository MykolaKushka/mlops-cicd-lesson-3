import torch
from torchvision import transforms
from PIL import Image
import urllib.request

# Завантажуємо модель
model = torch.jit.load("model.pt")
model.eval()

# Завантажуємо список класів ImageNet
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels_path = "imagenet_classes.txt"
urllib.request.urlretrieve(LABELS_URL, labels_path)
with open(labels_path, "r") as f:
    imagenet_classes = [line.strip() for line in f.readlines()]

# Преобразування зображення
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Функція передбачення
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top3 = torch.topk(probs, 3)
    for i, idx in enumerate(top3.indices):
        print(f"{i+1}. {imagenet_classes[idx]}: {probs[idx].item()*100:.2f}%")

# Тестуємо на зображенні
if __name__ == "__main__":
    img_path = "test.jpg"
    predict(img_path)

