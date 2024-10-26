import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from warmup_scheduler import GradualWarmupScheduler
from prep import printProgressBar
from WFTUNet import WFTUNet
import logging
from measure import compute_measure
from tqdm import tqdm
from edgenet import Edge_Net
from loader import get_loader
import util

# 模拟参数类
class Args:
    def __init__(self):
        self.mode = 'test'
        self.load_mode = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.norm_range_min = -1024
        self.norm_range_max = 3072
        self.trunc_min = -160
        self.trunc_max = 240
        self.multi_gpu = False
        self.result_fig = True


args = Args()

val_loader = get_loader(mode='test',
                        load_mode=0,
                        saved_path=r'xxxxxxxx',
                        test_patient='L506',
                        transform=False,
                        batch_size=1,
                        shuffle=False, )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TestSolver:
    def __init__(self, args, data_loader, val_loader):
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.val_loader = val_loader
        self.device = torch.device(args.device)
        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.save_path = r'xxxxx'
        self.multi_gpu = args.multi_gpu
        self.CTFormer = None

        self.result_fig = args.result_fig
        self.model_path = r'xxxxxxxxxx'
        log_file = os.path.join(self.save_path, 'psnr_ssim.txt')
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                            handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image

    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat

    def load_model(self, model, model_path):
        util.load_checkpoint(model, model_path, )

    def save_fig(self, x, y, pred, fig_name, ):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()

        # Ensure directories exist
        quarter_dose_dir = os.path.join(self.save_path, 'Quarter-dose')
        result_dir = os.path.join(self.save_path, 'Result')
        full_dose_dir = os.path.join(self.save_path, 'Full-dose')
        os.makedirs(quarter_dose_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(full_dose_dir, exist_ok=True)

        # Save Quarter-dose image
        plt.imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        quarter_dose_path = os.path.join(quarter_dose_dir, 'quarter_dose_{}.png'.format(fig_name))
        plt.axis('off')
        plt.savefig(quarter_dose_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save Result image
        plt.imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        result_path = os.path.join(result_dir, 'result_{}.png'.format(fig_name))
        plt.axis('off')
        plt.savefig(result_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save Full-dose image
        plt.imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        full_dose_path = os.path.join(full_dose_dir, 'full_dose_{}.png'.format(fig_name))
        plt.axis('off')
        plt.savefig(full_dose_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def test(self):
        del self.CTFormer
        self.CTFormer = WFTUNet(in_c=1, out_c=1)

        if self.multi_gpu and torch.cuda.device_count() > 1:
            self.CTFormer = nn.DataParallel(self.CTFormer)  # data parallel
        self.CTFormer.to(self.device)
        self.load_model(self.CTFormer, self.model_path)

        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(self.val_loader)):
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
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]
                pred_psnr_avg += pred_result[0]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]

                logging.info(
                    f"Image {i} ----> PSNR: {pred_result[0]:.4f}, SSIM: {pred_result[1]:.4f}, RMSE: {pred_result[2]:.4f}")

                if self.result_fig:
                    self.save_fig(x, y, pred, i)

            print('\n')
            print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
                ori_psnr_avg / len(self.val_loader),
                ori_ssim_avg / len(self.val_loader),
                ori_rmse_avg / len(self.val_loader)))
            print('\n')
            print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
                pred_psnr_avg / len(self.val_loader),
                pred_ssim_avg / len(self.val_loader),
                pred_rmse_avg / len(self.val_loader)))
            logging.info(
                f"\nOriginal === \nPSNR avg: {ori_psnr_avg:.4f} \nSSIM avg: {ori_ssim_avg:.4f} \nRMSE avg: {ori_rmse_avg:.4f}")
            logging.info(
                f"\nPredictions === \nPSNR avg: {pred_psnr_avg:.4f} \nSSIM avg: {pred_ssim_avg:.4f} \nRMSE avg: {pred_rmse_avg:.4f}")


# 初始化并运行测试
test_solver = TestSolver(args, None, val_loader)
test_solver.test()
