import os
import PIL.Image as Image
import numpy as np
import torch
import cv2 as cv
from torch.utils.data import Dataset
from torchvision import transforms
# from main import parse_args
#
# args = parse_args()

data_path = 'ebb_dataset'
train_original = os.path.join(data_path, 'train/original')
train_bokeh = os.path.join(data_path, 'train/bokeh')
train_depth = os.path.join(data_path, 'train/original_depth')

test_original = os.path.join(data_path, 'test/original')
test_bokeh = os.path.join(data_path, 'test/bokeh')
test_depth = os.path.join(data_path, 'test/original_depth')

transforms_color = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # transforms.ToPILImage(),
])

transforms_gray = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(0.456, 0.224),
])


input_size = 512
class EBBdataset(Dataset):
    def __init__(self, split="train"):
        assert split in ['train', 'test']
        self.dict = {}
        self.split = split

        if self.split == "train":
            self.path_original = train_original
            self.path_bokeh = train_bokeh
            self.path_depth = train_depth
        else:
            self.path_original = test_original
            self.path_bokeh = test_bokeh
            self.path_depth = test_depth

        self.img_original = os.listdir(self.path_original)
        self.img_bokeh = os.listdir(self.path_bokeh)
        self.img_depth = os.listdir(self.path_depth)

        self.transform_c = transforms_color
        self.transform_g = transforms_gray
        self.input = np.zeros((input_size, input_size, 4))

    def __getitem__(self, idx):

        original_path = os.path.join(self.path_original, self.img_original[idx])
        depth_path = os.path.join(self.path_depth, self.img_depth[idx])
        bokeh_path = os.path.join(self.path_bokeh, self.img_bokeh[idx])

        img = Image.open(original_path)
        img = img.resize((img.width//2, img.height//2))
        img_tensor = self.transform_c(img)
        # origianl image height is 1024 => 512
        new_width = input_size  # default 512
        img_tensor = img_tensor[:, :, :new_width]
        depth = Image.open(depth_path)
        depth_tensor = self.transform_g(depth)
        depth_tensor = depth_tensor[:, :, :new_width]
        # (4, 512, 512)
        input = torch.cat([img_tensor, depth_tensor], dim=0)

        # input = None
        target = Image.open(bokeh_path)
        target = target.resize((target.width//2, target.height//2))

        target_tensor = self.transform_c(target)
        # target_tensor.show()
        target_tensor = target_tensor[:, :, :new_width]

        return input, target_tensor

    def __len__(self):
        return len(self.img_original)


