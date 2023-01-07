from torchvision import models
import torch.nn as nn
import torch

# vgg = models.vgg19(pretrained=True)
# print(vgg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = models.vgg19(pretrained=True)
        ###################################################
        self.features = net.features

        ###################################################
    def forward(self, x):
        return self.features(x)

# load pretrained vgg 19 and froze it
def vgg_19():
    # vgg = VGG()
    # for param in vgg.parameters():
    #     param.requires_grad = False
    # vgg_19 = models.vgg19(pretrained=True)
    vgg_feature = models.vgg19(pretrained=True)
    vgg_feature_list = list(vgg_feature.children())
    vgg = nn.Sequential(*list(vgg_feature_list[0].children())[:-1])
    vgg = vgg.to(device)
    vgg = torch.nn.DataParallel(vgg)

    for param in vgg.parameters():
        param.requires_grad = False

    # for param in vgg19.parameters():
    #     param.requires_grad = False
    return vgg.to(device)

# vgg = vgg_19()
# print(vgg)