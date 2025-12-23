import argparse
import subprocess
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import BioIRTrainDataset
from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from net.model import BioIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
# import wandb
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import random

class FFTLoss(nn.Module):
    def __init__(self, loss_weight=0.1, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)
        
    def forward(self, pred, target):
        pred_fft = torch.fft.fft2(pred, dim=(-2,-1))
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        
        target_fft = torch.fft.fft2(target, dim=(-2,-1))
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        
        return self.loss_weight * self.criterion(pred_fft, target_fft)


class BioIRModel(pl.LightningModule):
    def __init__(self, opt, eval_interval):
        super().__init__()
        self.opt = opt
        self.net = BioIR()
        self.loss_fn  = nn.L1Loss()
        self.eval_datasets()
        self.eval_interval = eval_interval
        self.loss_fft = FFTLoss()

    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):

        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored, clean_patch)
        fft_loss = self.loss_fft(restored, clean_patch)
        loss += fft_loss
        self.log("train_loss", loss)
        return loss

    def lr_scheduler_step(self,scheduler, *args, **kwargs):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=180) # 150/180

        return [optimizer],[scheduler]
    
    # start evaluation
    def on_train_epoch_end(self, unused=None):
        if (self.current_epoch + 1) % self.eval_interval == 0:
            self.test_all()

    def test_all(self):
        self.test_mode_3()

    def test_mode_3(self):
        for testset in self.denoise_tests:
            if 'denoise_15' in self.opt.de_type:
                self.test_Denoise(testset, sigma=15)
            if 'denoise_25' in self.opt.de_type:
                self.test_Denoise(testset, sigma=25)
            if 'denoise_50' in self.opt.de_type:
                self.test_Denoise(testset, sigma=50)
        if 'derain' in self.opt.de_type:
            self.test_Derain_Dehaze(self.derain_set, task="derain")
        if 'enhance' in self.opt.de_type:
            self.test_Derain_Dehaze(self.enhance_set, task="enhance")      
        if 'dehaze' in self.opt.de_type:
            self.test_Derain_Dehaze(self.dehaze_set, task="dehaze")
        if 'deblur' in self.opt.de_type:    
            self.test_Derain_Dehaze(self.deblur_set, task="deblur")
        
    def eval_datasets(self):
        self.denoise_tests = []
        self.derain_tests = []
        self.dehaze_tests = []
        
        denoise_splits = ["bsd68/"]
        denoise_base_path = self.opt.denoise_path
        for i in denoise_splits:
            self.opt.denoise_path = os.path.join(denoise_base_path, i)
            denoise_testset = DenoiseTestDataset(self.opt)
            self.denoise_tests.append(denoise_testset)
            
        # derain
        derain_splits = ["Rain100L/"]
        derain_base_path = self.opt.derain_path
        for name in derain_splits:
            self.opt.derain_path = os.path.join(derain_base_path, name)
            self.derain_set = DerainDehazeDataset(self.opt,addnoise=False, sigma=15)
        

        self.opt.dehaze_path = self.opt.dehaze_path
        self.dehaze_set = DerainDehazeDataset(self.opt,addnoise=False,sigma=15)

        deblur_splits = ["gopro/"]
        deblur_base_path = self.opt.gopro_path
        for name in deblur_splits:

            self.opt.gopro_path = os.path.join(deblur_base_path,name)
            self.deblur_set = DerainDehazeDataset(self.opt,addnoise=False,sigma=15)

        enhance_splits = ["lol/"]
        enhance_base_path = self.opt.enhance_path
        for name in enhance_splits:

            self.opt.enhance_path = os.path.join(enhance_base_path,name)
            self.enhance_set = DerainDehazeDataset(self.opt,addnoise=False,sigma=15)



    def test_Denoise(self, dataset, sigma=15):
        dataset.set_sigma(sigma)
        testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

        psnr = AverageMeter()
        ssim = AverageMeter()

        with torch.no_grad():
            for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
                degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

                restored = self.forward(degrad_patch)
                temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)

                psnr.update(temp_psnr, N)
                ssim.update(temp_ssim, N)

            print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))
            self.log("psnr %d"% (sigma), psnr.avg)
            self.log("SSIM %d"% (sigma), ssim.avg)

    def test_Derain_Dehaze(self, dataset, task="derain"):
        dataset.set_dataset(task)
        testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

        psnr = AverageMeter()
        ssim = AverageMeter()

        with torch.no_grad():
            for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
                degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

                b, c, h, w = degrad_patch.shape
                h_n = (4 - h % 4) % 4
                w_n = (4 - w % 4) % 4
                degrad_patch = torch.nn.functional.pad(degrad_patch, (0, w_n, 0, h_n), mode='reflect')

                restored = self.forward(degrad_patch)[:, :, :h, :w]
                temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
                psnr.update(temp_psnr, N)
                ssim.update(temp_ssim, N)
            self.log("psnr %s" % (task), psnr.avg)
            self.log("SSIM %s" % (task), ssim.avg)

            print("PSNR_%s: %.2f, SSIM_%s: %.4f" % (task, psnr.avg, task, ssim.avg))

def main():

    parser = argparse.ArgumentParser()

    # parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--denoise_path', type=str, default="data/test/denoise/", help='save path of test noisy images')
    parser.add_argument('--derain_path', type=str, default="data/test/derain/", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="data/test/dehaze/", help='save path of test hazy images')
    parser.add_argument('--gopro_path', type=str, default="data/test/deblur/", help='save path of test blurry images')
    parser.add_argument('--enhance_path', type=str, default="data/test/enhance/", help='save path of test low light images')

    parser.add_argument('--epochs', type=int, default=150, help='maximum number of epochs to train the total model.')
    parser.add_argument('--batch_size', type=int,default=16,help="Batch size to use per GPU")
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')

    parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze', 'deblur', 'enhance'],
                        help='which type of degradations is training and testing for.')

    parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')

    # path
    parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='where clean images of denoising saves.')
    parser.add_argument('--denoise_dir', type=str, default='data/Train/Denoise/',
                        help='where clean images of denoising saves.')
    parser.add_argument('--gopro_dir', type=str, default='data/Train/Deblur/',
                        help='where clean images of denoising saves.')
    parser.add_argument('--enhance_dir', type=str, default='data/Train/Enhance/',
                        help='where clean images of denoising saves.')
    parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/',
                        help='where training images of deraining saves.')
    parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/',
                        help='where training images of dehazing saves.')
    parser.add_argument('--output_path', type=str, default="output/", help='output save path')
    parser.add_argument('--ckpt_path', type=str, default="ckpt/Denoise/", help='checkpoint save path')
    parser.add_argument("--wblogger",type=str,default="BioIR",help = "Determine to log to wandb or not and the project name")
    parser.add_argument("--ckpt_dir",type=str,default="BioIR",help = "Name of the Directory where the checkpoint is to be saved")
    parser.add_argument("--num_gpus",type=int,default= 2, help = "Number of GPUs to use for training")

    opt = parser.parse_args()

    path = opt.ckpt_dir+'_model'
    if not os.path.exists(path):
        os.makedirs(path)
    command = 'cp '+'net/model.py ' + path
    os.system(command)

    logger = TensorBoardLogger(save_dir = "bioir/")

    trainset = BioIRTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    
    model = BioIRModel(opt, eval_interval=10)
    trainer = pl.Trainer(max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,strategy="ddp_find_unused_parameters_true",logger=logger,callbacks=[checkpoint_callback])
    # trainer = pl.Trainer(max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,strategy="ddp_find_unused_parameters_true",logger=logger,callbacks=[checkpoint_callback], limit_train_batches=0.001)

    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':

    main()



