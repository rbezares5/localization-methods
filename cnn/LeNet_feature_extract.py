import torch
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models import vgg16, VGG16_Weights
from torchsummary import summary
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms

from lenet import LeNet
from PIL import Image

train_nodes, eval_nodes = get_graph_node_names(LeNet(1, 50))
print(train_nodes) #get layer names for feature extractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('dummy.pth').to(device)
model.eval()

summary(model, (1, 128, 512))
print(model)

# extract intermediate layer
model = create_feature_extractor(model, {'relu3': 'lenet_descr'})
    #this basically removes all downstream nodes after the last one we specified!
out = model(torch.rand(1, 1, 128, 512))
print([(k, v.shape) for k, v in out.items()])

#Get a descriptor from an image
img_path = 'Friburgo/Friburgo_Train/t1152902729.509119_x0.343810_y-0.001590_a-0.006566.jpeg'
image = Image.open(img_path).convert('L') #convert to grayscale
transform = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

input_tensor = transform(image).unsqueeze(0)
feat=model(input_tensor) #vale -> de aqui saco los vectores descriptores, ahora a ver cómo los separo y los uso adecuadamente!
    #nos devuelve un diccionario con los claves que pusimos como nombres antes y que contiene los tensores de info
    #ahora es cuestion de hacer algo con estos tensores
print(feat)

out=feat['lenet_descr']
print(out.size())
out=out.flatten()
print(out.size())
print(out)