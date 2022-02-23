from __future__ import absolute_import, division, print_function
from PIL import Image
import numpy as np
import argparse
from networks import MyNet
from networks import util
from networks import feature_extraction
import torch
import torch.nn.functional as F
from mydatasets import matterport3d
from mydatasets import matterport3d_test
import os
import tqdm
import time
import Utils
import cv2
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from losses import BerhuLoss
from networks import net360SD

parser = argparse.ArgumentParser(description='XMNet')
parser.add_argument('--model', default='net360SD', help='MYNet/net360SD')
parser.add_argument('--maxdisp', type=int, default=48, help='maxium disparity')
parser.add_argument('--use_amp', default=True, type=bool, help="use amp?")
parser.add_argument('--datapath', default='/media/xioamao/My Passport/i', help='datapath')
parser.add_argument('--checkpoint', default=None, help='load checkpoint path')
parser.add_argument('--train_file_list', default='./mydatasets/', help='datapath')
parser.add_argument('--epochs', type=int, default=0, help='number of epochs to train')
parser.add_argument('--step_pre_epochs', type=int, default=3800, help='number of epochs to train')
parser.add_argument('--save_steps', type=int, default=3799, help='number of epochs to train')
parser.add_argument('--loadmodel', default='360SD_finally.tar', help='load model')
parser.add_argument('--savemodel', default='./', help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
f1 = open('mydatasets/loss_360SD.txt', 'r+')
f2 = open('mydatasets/test_360SD.txt', 'r+')
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)



#--------------------Create Angle info-----------------------
angle_y = np.array([(i - 0.5) / 256 * 180 for i in range(128, -128, -1)], dtype=np.float32)
angle_ys = np.tile(angle_y[:, np.newaxis, np.newaxis], (1, 512, 1))
equi_info = angle_ys
# --------------------Load model ----------------------------------------------
if args.model == 'net360SD':
    print("args.model == 'net360SD'")
    model = net360SD(args.maxdisp)
elif args.model == 'MYNet':
    print("args.model == 'MYNet'")
    model = MyNet(args.maxdisp)
else:
    raise NotImplementedError('Model Not Implemented!!!')

if args.loadmodel is not None:
    print("Load pretrained model:")
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'])

# ---------------get datasets -----------------------------
train_dataset = matterport3d.Matterport3D(args.datapath, args.train_file_list, equi_info, 256,
                                          512, max_depth_meters=args.maxdisp, is_training=True)
test_dataset = matterport3d_test.Matterport3D(args.datapath, args.train_file_list, equi_info, 256,
                                              512, max_depth_meters=args.maxdisp, is_training=True)
dataset = DataLoader(
    train_dataset,
    batch_size=2,
    num_workers=8,
    drop_last=False,
    pin_memory=True,
    shuffle=False
)
test_dataset = DataLoader(
    test_dataset,
    batch_size=1,
    num_workers=1,
    drop_last=False,
    pin_memory=True,
    shuffle=False
)


# ----------------------------------------------------------
Loss = F.smooth_l1_loss
init_array = np.zeros((1, 1, 7, 1))  # 7 of filter
init_array[:, :, 3, :] = 28. / 540
init_array[:, :, 2, :] = 512. / 540
model.model_cost.forF.forfilter1.weight = torch.nn.Parameter(torch.tensor(init_array).float(), requires_grad=True)
# -----------------------------------------------------------------------------

# Optimizer -----------------------------------------------------
optimizer = optim.Adam(list(model.parameters()), lr=0.0001, betas=(0.9, 0.999))
print("optimizer:", optimizer)
# ---------------------------------------------------------------

# Load Checkpoint -------------------------------
start_epoch = 0
if args.checkpoint is not None:
    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict['state_dict'])
    start_epoch = state_dict['epoch']

print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))


def adjust_learning_rate(optimizer, epoch):
    if epoch == 5:
        lr = 0.0001
        print(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    if epoch == 5:
        lr = 0.0001
        print(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)


# -----------test---------------------
def test(imgU, imgD, disp_true, mask, batch_idx, a, name):
    model.eval()
    imgU, imgD, disp_true = imgU.cuda(), imgD.cuda(), disp_true.cuda()
    with torch.cuda.amp.autocast(enabled=not args.use_amp):
        output1, output2, output3 = model(imgU, imgD, training=False)

    mask1 = mask[0, 0, :, :, 0]
    img1 = torch.squeeze(output1)[mask1].cpu().detach().numpy()
    disp_true1 = disp_true[0, 0, :, :, 0][mask1].cpu().detach().numpy()
    mask = mask[:, 0, :, :, 0]
    disp_true = disp_true[:, 0, :, :, 0][mask]
    img = torch.squeeze(output1, 1)[mask]
    loss = Loss(img, disp_true, reduction='mean')
    print(loss.data)

    return loss.data.cpu()

# --------------------------------------------------
def train():
    """Validate the model on the validation set
    """
    model.train()

    for epoch in range(args.epochs):
        total_train_loss = 0
        adjust_learning_rate(optimizer, epoch)
        loss = train_one_epoch(epoch, total_train_loss)
        total_train_loss += loss


def train_one_epoch(epoch, total_train_loss):

    optimizer.zero_grad()

    pbar = tqdm.tqdm(iterable=dataset,
                     bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}')
    pbar.set_description("Training Epoch_{}".format(epoch))
    total_loss = 0
    for batch_idx, inputs in enumerate(pbar):
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            time1 = time.time()
            if batch_idx % 46 == 0:
                time.sleep(1)
                continue
            output1, output2, output3 = model(inputs["rgb_name_Up"].cuda(), inputs["rgb_name_Center"].cuda()
                                              , training=True)
            mask = inputs["val_mask"][:, 0, :, :, 0].cuda()
            target = inputs["depth_name_Up"][:, 0, :, :, 0].cuda().to(torch.float32)[mask]
            output1 = torch.squeeze(output1, 1)[mask]
            output2 = torch.squeeze(output2, 1)[mask]
            output3 = torch.squeeze(output3, 1)[mask]
            loss = 0.5 * Loss(output1, target, reduction='mean') + \
                   0.7 * Loss(output2, target, reduction='mean') + \
                   Loss(output3, target, reduction='mean')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            print("\n", loss.data, time.time() - time1)
            total_loss = total_loss + loss
            if (batch_idx + 1) % 10 == 0:
                f1.write(str(total_loss / 10))
                f1.write('\n')
                print(str(total_loss / 10))
                total_loss = 0
            if (batch_idx + 1) % args.save_steps == 0:
                savefilename = args.savemodel + '/checkpoint_' + str(epoch) + '_' + str(batch_idx) + '.tar'
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'train_loss': total_train_loss / len(pbar),
                }, savefilename)
            if (batch_idx + 1) % args.step_pre_epochs == 0:

                return loss.data
                # SAVE


def main():
    epoch = 0
    step = 0
    a = 1.
    start_full_time = time.time()
    print('Test Data Num:', len(train_dataset), "start_time:", start_full_time)

    train()

    # ------------- TEST ------------------------------------------------------------
    pbar = tqdm.tqdm(test_dataset)
    pbar.set_description("test")

    for batch_idx, inputs in enumerate(pbar):
        test_loss = test(inputs["rgb_name_Up"], inputs["rgb_name_Center"], inputs["depth_name_Up"], inputs["val_mask"], batch_idx, a, inputs["name"])
    # -----------------SAVE test information-----------------------------------------
    print('full training time = %.2f HR' %
          ((time.time() - start_full_time) / 3600))


if __name__ == "__main__":
    main()
