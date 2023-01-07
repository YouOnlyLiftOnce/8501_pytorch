import argparse
import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from model import PyNET
from dataset import EBBdataset
from train import train, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, default='exp_0')
    # when test，need set both resume and test to 1
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--resume_level', type=int, default=6, help='resume the trained model')
    parser.add_argument('--restore_epoch', type=int, default=1, help='restore epochs')
    parser.add_argument('--test', type=int, default=1, help='test with trained model')
    parser.add_argument('--level', type=int, default=5, help='output level')
    parser.add_argument('--input_size', type=int, default=512, help='input_size')
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--val_gap', type=float, default=1, help='gap for validation')
    parser.add_argument('--save_gap', type=float, default=10, help='gap for save model')


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # dataloaders
    trainloader = DataLoader(EBBdataset(split='train'),
        batch_size=args.batch_size, shuffle=True, num_workers=2)
    # test batch size must be 1. The can only compute SSIM with single input image (not batch of images)
    testloader = DataLoader(EBBdataset(split='test'),
        batch_size=1, shuffle=False, num_workers=2)
    dataloaders = (trainloader, testloader)

    # network
    model = PyNET(args.level).to(device)
    model = torch.nn.DataParallel(model)
    # model = model.double()
    # resume
    if args.resume_level <= 5:
        model.load_state_dict(torch.load("models/pynet_level_" + str(args.resume_level) +
                                             "_epoch_" + str(args.restore_epoch) + ".pth"), strict=False)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999))

    train(args, model, optimizer, dataloaders)

    # evaluate(args,model,dataloaders)


