from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
import torchvision.transforms.functional as TF
import torch
import imageio
import numpy as np
import cv2
import math
import sys
import torch.nn.functional as F

import msssim
from msssim import MSSSIM,SSIM
from loadvgg import vgg_19


np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data augmentation
def augmentation(input,target):
    input = input.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    for i in range(input.shape[0]):
        angle = np.random.randint(1, 100) % 4
        angle = angle * 90
        random_flip = np.random.randint(1, 100) % 2
        input[i] = np.rot90(input[i], angle)
        target[i] = np.rot90(target[i], angle)
        if random_flip == 1:
            input[i] = np.flipud(input[i])
            target[i] = np.flipud(target[i])

    input = torch.from_numpy(input).to(device)
    target =torch.from_numpy(target).to(device)
    return input,target

# resize image
def resize(target,shape):
    target = target.cpu().detach().numpy()
    # B,C,H,W -> B,W,H,C -> B,H,W,C
    target = np.swapaxes(target, 1, 3)
    target = np.swapaxes(target, 1, 2)

    target_resized = np.zeros((target.shape[0],shape[0],shape[1],target.shape[3]))

    for i in range(target.shape[0]):
        target_resized[i] = cv2.resize(target[i],(shape[1],shape[0]),interpolation=cv2.INTER_CUBIC)
    # B,H,W,C -> B,W,H,C -> B,C,H,W
    target_resized = np.swapaxes(target_resized,1,2)
    target_resized = np.swapaxes(target_resized, 1, 3)
    target = torch.from_numpy(target_resized).to(device)
    return target.float()

def toImage(image):
    image = image.cpu().detach().numpy()
    # B,C,H,W -> B,W,H,C -> B,H,W,C
    image = np.swapaxes(image, 1, 3)
    image = np.swapaxes(image, 1, 2)

    image = image*255

    return image[0]

def train(args, model, optimizer, dataloaders):
    torch.backends.cudnn.deterministic = True
    trainloader, testloader = dataloaders

    # Losses
    # use L1 loss for level 5,4,3,2,1 other losses only apply to level 0
    L1_loss = torch.nn.L1Loss()
    # L2_loss = torch.nn.L2Loss()
    SSIM_loss = SSIM()
    MSE_loss = torch.nn.MSELoss()
    vgg = vgg_19()

    # training
    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
        model.train()
        train_iter = iter(trainloader)
        for i in range(len(trainloader)):
            input, target = next(train_iter)
            # data augmentation
            input = input.to(device)
            target = target.to(device)
            input, target = augmentation(input,target)

            # input = input.to(device)
            # target = target.to(device)

            bokeh = model(input)

            target = resize(target,(bokeh.shape[2],bokeh.shape[3]))
            # loss_vgg = L2_loss(vgg(target),vgg(bokeh))
            if args.level==0:
                loss_l1 = L1_loss(target,bokeh)
                vgg_bokeh = vgg(bokeh)
                vgg_target = vgg(target)
                loss_content = MSE_loss(vgg_bokeh,vgg_target)
                loss_ssim = SSIM_loss(bokeh,target)
                loss= loss_l1 * 10 + loss_content * 0.1 + (1 - loss_ssim) * 10

            else:
                loss = L1_loss(target,bokeh)*100

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%100==0:
                print(i)
                print(loss.item())

    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "models/pynet_level_" + str(args.level) + "_epoch_" + str(args.epochs) + ".pth")

def evaluate(args,model,dataloaders):
    trainloader, testloader = dataloaders
    model.eval()
    # Larger images are needed for computing PSNR / SSIM scores (output of level 1 and level 0)
    loss_psnr_ = 0.0
    loss_ssim_ = 0.0
    loss_msssim_ = 0.0
    MSE_loss = torch.nn.MSELoss()
    SSIM_loss = SSIM().float()
    MSSSIM_loss = MSSSIM().float()
    test_size = len(testloader)
    with torch.no_grad():

        test_iter = iter(testloader)
        for j in range(len(testloader)):

            print("Processing image " + str(j))

            torch.cuda.empty_cache()
            raw,target = next(test_iter)
            raw = raw.to(device)
            target = target.to(device)
            # Run inference
            bokeh = model(raw.detach())
            target = resize(target, (bokeh.shape[2], bokeh.shape[3]))

            # compute loss
            loss_mse_temp = MSE_loss(bokeh,target).item()
            loss_psnr_temp = 20 * math.log10(1.0 / math.sqrt(loss_mse_temp))
            loss_ssim_temp = SSIM_loss(bokeh,target).item()
            loss_msssim_temp = MSSSIM_loss(bokeh,target).item()
            loss_psnr_ += loss_psnr_temp / test_size
            loss_ssim_ += loss_ssim_temp / test_size
            loss_msssim_ += loss_msssim_temp / test_size

            # save bokeh image (not implement)
            bokeh_image = toImage(bokeh)
            cv2.imwrite('./results/level'+str(args.level)+'_'+str(j)+'.png',bokeh_image)

    output_logs = "PSNR: %.4g, SSIM: %.4g, MS-SSIM: %.4g\n" % (loss_psnr_, loss_ssim_, loss_msssim_)
    print(output_logs)
