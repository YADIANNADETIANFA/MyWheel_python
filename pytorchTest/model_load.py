from model_save import *
import torchvision

# 方式1-》保存方式1，加载模型
model = torch.load("./trained_models/vgg16_method1.pth")
print(model)

# 方式2，加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("./trained_models/vgg16_method2.pth"))

# 加载自己的模型，前提使用  from model_save import *
model2 = torch.load('./trained_models/tudui_method1.pth')
print(model2)


