import torch
import torchvision
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_path = "./test_img/airplane.png"
image = Image.open(image_path)
print(image)
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

model = torch.load("./trained_models/tudui_9.pth", map_location=torch.device('cuda'))
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
image = image.to(device)
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))