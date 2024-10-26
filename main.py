import os
import argparse
from torch.backends import cudnn
from loader import get_loader
from solver import Solver
# import T2T_vit_model
from testdemo import TestSolver
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'  ## 1/2 ,multi GPU

def main(args):
    cudnn.benchmark = True

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    if args.result_fig:
        fig_path = os.path.join(args.save_path, 'fig')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))

    data_loader = get_loader(mode=args.mode,
                             load_mode=args.load_mode,
                             saved_path=args.saved_path,
                             test_patient=args.test_patient,
                             patch_n=(args.patch_n if args.mode=='train' else None),
                             patch_size=(args.patch_size if args.mode=='train' else None),
                             transform=args.transform,
                             batch_size=(args.batch_size if args.mode=='train' else 1),
                             shuffle=(True if args.mode=='train' else False),
                             )
    val_loader = get_loader(mode='test',
                             load_mode=args.load_mode,
                             saved_path=args.saved_path,
                             test_patient=args.test_patient,
                             transform=args.transform,
                             batch_size=1,
                             shuffle=False,)

    solver = Solver(args, data_loader, val_loader)
    testdemo = TestSolver(args, data_loader, val_loader)
    testdemo.test()
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--load_mode', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='xxx')
    parser.add_argument('--saved_path', type=str, default=r'Exxxx')   ##aapm_all_npy_3mm
    parser.add_argument('--save_path', type=str, default='save/')
    parser.add_argument('--test_patient', type=str, default='L506')
    parser.add_argument('--result_fig', type=bool, default=True)

    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)
    parser.add_argument('--trunc_min', type=float, default=-160.0)
    parser.add_argument('--trunc_max', type=float, default=240.0)

    parser.add_argument('--transform', type=bool, default=False)
    # if patch training, batch size is (--patch_n * --batch_size)
    parser.add_argument('--patch_n', type=int, default=2)   ## 10
    parser.add_argument('--patch_size', type=int, default=128)    ## 64
    parser.add_argument('--batch_size', type=int, default=1)   ## batch size has to be very small if size=512,16

    parser.add_argument('--num_epochs', type=int, default=50)  ## 200 or 2000
    ## the iterats~epochs*10 useless for now
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--device', type=str)  ##, default=[2,3]

    parser.add_argument('--multi_gpu', type=bool, default=True) ## 2/2 ,multi GPU

    args = parser.parse_args()
    main(args)
