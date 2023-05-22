from torchvision.models import vgg19


vgg = vgg19()

state_dict = vgg.state_dict()
print(type(state_dict))
print(state_dict.keys())

x = 1