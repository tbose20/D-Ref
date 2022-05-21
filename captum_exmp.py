import torch
import torchvision
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
from captum.attr import DeepLift
from torch.autograd import Variable
from captum.attr import Occlusion


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = torchvision.models.alexnet(pretrained=True).eval()
model.to(device)

response = requests.get("https://image.freepik.com/free-photo/two-beautiful-puppies-cat-dog_58409-6024.jpg")
img = Image.open(BytesIO(response.content))

center_crop = transforms.Compose([
     transforms.Resize(256),
     transforms.CenterCrop(224),
      ])

normalize = transforms.Compose([
        transforms.ToTensor(),               # converts the image to a tensor with values between 0 and 1
        transforms.Normalize(                # normalize to follow 0-centered imagenet pixel rgb distribution
                     mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]
                              )
            ])
strides = (3, 9, 9)               # smaller = more fine-grained attribution but slower
target=208,                       # Labrador index in ImageNet
sliding_window_shapes=(3,45, 45)  # choose size enough to change object appearance
baselines = 0 
input_img = normalize(center_crop(img)).unsqueeze(0)
input_img = Variable(input_img).to(device)
dl = DeepLift(model)

#occlusion = Occlusion(model)

#dl_attr = occlusion.attribute(input_img, strides = strides, target=1, sliding_window_shapes=sliding_window_shapes, baselines=baselines)
dl_attr = dl.attribute(input_img,target=1)
print("dl_attr: ",dl_attr)
