import torch
from torchvision import transforms,models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#loading the model vgg19
model = models.vgg19(pretrained=True).features
device = ("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.Resize(400),
                                transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

#loading the images
inpimg = Image.open('i.jpg').convert('RGB')
inpimg = transform(inpimg).to(device)
styleimg = Image.open('in.jpg').convert('RGB')
styleimg = transform(styleimg).to(device)

#freezing the layers of the neural network
for p in model.parameters():
    p.requires_grad = False
model.to(device)

#imgfeature extract the features of the img by learning through the network but we will just extract only features given by some layers
def imgfeature(model,img):
  layers = {
  '0' : 'conv1_1',
  '5' : 'conv2_1',
  '10': 'conv3_1',
  '19': 'conv4_1',
  '21': 'conv4_2',
  '28': 'conv5_1'
  }
  x = img
  x = x.unsqueeze(0)                                                            
  feature = {}
  for name,layer in model._modules.items():
    x = layer(x)
    if name in layers:
      feature[layers[name]] = x
  return feature


#converting the image back to normal matrix as we transorm it and also detach it from cuda in the process
def convertback(tensor):
  c = tensor.to("cpu").clone().detach().numpy().squeeze()
  c = c.transpose()
  c = c*np.array((0.5,0.5,0.5)) + np.array((0.5,0.5,0.5))
  return c

#gram matrix extract the correlation between the different channels of the matrix features extracterd through the neural network
def gram_matrix(imgfea):
  _,d,h,w = imgfea.shape
  imgfea = imgfea.view(d,h*w)
  res = torch.mm(imgfea,imgfea.t())
  return res

plt.imshow(convertback(inpimg))
plt.show()

plt.imshow(convertback(styleimg))
plt.show()

style_features = imgfeature(model,styleimg)
content_features = imgfeature(model,inpimg)

#every layer cannot be treated with same weitght as increasing the layer increase the content of the image and reduce style
style_wt_meas = {"conv1_1" : 1.0, 
                 "conv2_1" : 0.8,
                 "conv3_1" : 0.4,
                 "conv4_1" : 0.2,
                 "conv5_1" : 0.1}
#7.5e0
content_wt = 1e-2
style_wt = 1e4

print_after = 500
epochs = 5000

#target is the resulting image we are trying to form 
target = inpimg.clone().requires_grad_(True).to(device)


optimizer = torch.optim.Adam([target],lr = 0.07)

for i in range(1,epochs+1):
  target_feature = imgfeature(model,target)
  content_loss = torch.mean((content_features['conv4_2']-target_feature['conv4_2'])**2)
  style_loss = 0
  for lay,wt in (style_wt_meas).items():
#    _,d,h,w = (gram_matrix(style_features[lay])).shape
    
    target_gram = gram_matrix(target_feature[lay])
    _,d,h,w = target_feature[lay].shape
    style_gram = gram_matrix(style_features[lay])
    style_loss += (wt*torch.mean((target_gram-style_gram)**2))/d*h*w

  total_loss = style_loss + content_loss  

  optimizer.zero_grad()
  total_loss.backward()
  optimizer.step()

  if i%10 == 0:
    print(total_loss)

  if i%500 == 0:
    plt.imshow(convertback(target))
    plt.show()