import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import losses
import torch
import torch.nn as nn
import torch.optim as optim
import util
from warmup_scheduler import GradualWarmupScheduler
from prep import printProgressBar
from WFTUNet import WFTUNet
import logging
from measure import compute_measure
from tqdm import tqdm



######### Loss ###########
criterion = losses.CharbonnierLoss()
criterion_char = losses.CharbonnierLoss()
criterion_edge = losses.EdgeLoss()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def split_arr(arr, patch_size, stride=32):
    pad = (16, 16, 16, 16)  # pad by (16, 16) on both sides
    arr = nn.functional.pad(arr, pad, "constant", 0)
    _, _, h, w = arr.shape
    num_h = (h - patch_size) // stride + 1
    num_w = (w - patch_size) // stride + 1
    arrs = torch.zeros(num_h * num_w, 1, patch_size, patch_size)

    for i in range(num_h):
        for j in range(num_w):
            arrs[i * num_w + j, 0] = arr[0, 0, i * stride:i * stride + patch_size, j * stride:j * stride + patch_size]
    return arrs

def agg_arr(arrs, size, stride=32):
    arr = torch.zeros(size, size)
    n, _, h, w = arrs.shape
    num = size // stride
    for i in range(num):
        for j in range(num):
            arr[i * stride:(i + 1) * stride, j * stride:(j + 1) * stride] = arrs[i * num + j, :, 16:48, 16:48]
    return arr.unsqueeze(0).unsqueeze(1)


class Solver(object):
    def __init__(self, args, data_loader, val_loader):
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.data_loader = data_loader
        self.val_loader = val_loader

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.result_fig = args.result_fig
        self.patch_size = args.patch_size
        self.criterion = nn.MSELoss()

        self.CTFormer = WFTUNet(in_c=1, out_c=1)

        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.CTFormer = nn.DataParallel(self.CTFormer)   ## data parallel  ,device_ids=[2,3]
        self.CTFormer.to(self.device)

        self.lr = args.lr
        self.optimizer = optim.Adam(self.CTFormer.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8,
                                    weight_decay=1e-8)

        # Scheduler setup
        warmup_epochs = 3
        self.scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.num_epochs - warmup_epochs + 40,
                                                  eta_min=1e-6)
        self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=warmup_epochs,
                                                after_scheduler=self.scheduler_cosine)

        log_file = os.path.join(self.save_path, 'training_log.txt')
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                            handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
        if args.resume:
            self.resume_training()
        self.best_psnr = 0  # 初始化 best_psnr
        self.best_epoch = -1  # 初始化 best_epoch
        self.start_epoch = 1

    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image


    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat

    def resume_training(self):
        path_chk_rest = util.get_last_path(self.save_path, '_latest.pth')
        util.load_checkpoint(self.CTFormer, path_chk_rest)
        self.start_epoch = util.load_start_epoch(path_chk_rest) + 1
        util.load_optim(self.optimizer, path_chk_rest)

        for i in range(1, self.start_epoch):
            self.scheduler.step()
        new_lr = self.scheduler.get_lr()[0]
        print('------------------------------------------------------------------------------')
        print(f"==> Resuming Training from epoch {self.start_epoch} with learning rate: {new_lr}")
        print('------------------------------------------------------------------------------')
        logging.info('------------------------------------------------------------------------------')
        logging.info(f"==> Resuming Training from epoch {self.start_epoch} with learning rate: {new_lr}")
        logging.info('------------------------------------------------------------------------------')

    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(self.save_path, 'fig', 'result_{}.png'.format(fig_name)))
        plt.close()

    def train(self):
        NumOfParam = count_parameters(self.CTFormer)
        print('trainable parameter:', NumOfParam)

        train_losses = []
        start_time = time.time()
        loss_all = []
        total_samples = len(self.data_loader.dataset)

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            self.CTFormer.train(True)
            epoch_loss = 0
            epoch_start_time = time.time()

            for iter_, (x, y) in enumerate(
                    tqdm(self.data_loader)):
                # add 1 channel
                x = x.unsqueeze(0).float().to(self.device)  # expand one dimension given the dimension 0 4->[1,4]
                y = y.unsqueeze(0).float().to(self.device)  # copy data to device

                if self.patch_size:  # patch training
                    x = x.view(-1, 1, self.patch_size, self.patch_size)  # similar to reshape
                    y = y.view(-1, 1, self.patch_size, self.patch_size)

                pred = self.CTFormer(x)
                # loss_char = torch.sum(torch.stack([criterion_char(pred[j], y) for j in range(len(pred))]))
                # loss_edge = torch.sum(torch.stack([criterion_edge(pred[j], y) for j in range(len(pred))]))
                # loss = loss_char + (0.05 * loss_edge)
                loss = self.criterion(pred, y)*100 + 1e-4
                self.CTFormer.zero_grad()
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
                epoch_loss += loss.item()
                loss_all.append(loss.item())

            epoch_loss /= len(self.data_loader)
            if epoch % 5 == 0 or epoch in [1, 3, ]:
                self.val(epoch, detailed=True)
            if epoch % 2 == 0:
                self.val(epoch, detailed=False)

            self.scheduler.step()
            print("------------------------------------------------------------------")
            print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch,
                                                                                      time.time() - epoch_start_time,
                                                                                      epoch_loss,
                                                                                      self.scheduler.get_lr()[0]))
            print("------------------------------------------------------------------")
            logging.info("------------------------------------------------------------------")
            logging.info(
                f"Epoch: {epoch}\tTime: {time.time() - epoch_start_time:.4f}\tLoss: {epoch_loss:.4f}\tLearningRate {self.scheduler.get_lr()[0]:.6f}")
            logging.info("------------------------------------------------------------------")
            torch.save({'epoch': epoch,
                        'state_dict': self.CTFormer.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                        }, os.path.join(self.save_path, "model_latest.pth"))
            if epoch % 2 == 0:
                torch.save({'epoch': epoch,
                            'state_dict': self.CTFormer.state_dict(),
                            'optimizer': self.optimizer.state_dict()
                            }, os.path.join(self.save_path, f"model_epoch_{epoch}.pth"))

    def val(self, epoch, detailed=False):
        self.CTFormer.eval()
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
        with torch.no_grad():
            for i, (x, y) in enumerate(self.val_loader, 0):
                shape_ = x.shape[-1]
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)
                pred = self.CTFormer(x)
                pred = pred[0]
                x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))

                data_range = self.trunc_max - self.trunc_min

                original_result, pred_result = compute_measure(x, y, pred, data_range)
                ori_psnr_avg += original_result[0]
                pred_psnr_avg += pred_result[0]
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]
                if self.result_fig and detailed:
                    self.save_fig(x, y, pred, i, original_result, pred_result)

        ori_psnr_avg /= len(self.val_loader)
        pred_psnr_avg /= len(self.val_loader)
        pred_ssim_avg /= len(self.val_loader)
        pred_rmse_avg /= len(self.val_loader)
        print('\n')
        print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg,
                                                                                                pred_ssim_avg,
                                                                                                pred_rmse_avg))
        if pred_psnr_avg > self.best_psnr:
            self.best_psnr = pred_psnr_avg
            self.best_epoch = epoch
            torch.save({'epoch': epoch,
                        'state_dict': self.CTFormer.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                        }, os.path.join(self.save_path, "model_best.pth"))

        print("[epoch %d PSNR: %.4f --- best_epoch %d  Best_PSNR %.4f]" % (
            epoch, pred_psnr_avg, self.best_epoch, self.best_psnr))
        logging.info(
            f"[epoch {epoch} PSNR: {pred_psnr_avg:.4f} --- best_epoch {self.best_epoch}  Best_PSNR {self.best_psnr:.4f}]")

        self.CTFormer.train()