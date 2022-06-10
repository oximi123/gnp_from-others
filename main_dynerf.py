import torch
import argparse

from dynerf.provider import DyNeRFDataset
# from dynerf.gui import NeRFGUI
from dynerf.utils import *
from dynerf.cnetwork import DyNeRFNetwork
from dynerf.combine_network import CombineDyNeRFNetwork
model_dict = {
    "DyNeRFNetwork" : DyNeRFNetwork,
    "CombineDyNeRFNetwork" : CombineDyNeRFNetwork,
}

#torch.autograd.set_detect_anomaly(True)

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--cuda_device', type=int, default=0, help="index of cuda device")
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--error_type', default = None, choices = ["error", "isg", "ist", None], 
        help="[experimental] use error map to sample rays")
    parser.add_argument('--downscale', type=int, default=2, help="down sample")
    parser.add_argument('--epoch', type=int, default=300, help="epoch")
    parser.add_argument('--eval_interval', type=int, default=100, help="epoch")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    ### dataset options
    parser.add_argument('--mode', type=str, default='colmap', help="dataset mode, supports (colmap, blender)")
    parser.add_argument('--model_static', action='store_true', help="")
    parser.add_argument('--model', default="DyNerfNetwork", help="", choices= list(model_dict.keys()))
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    parser.add_argument('--rand_pose_interval', type=int, default=0, help="[experimental] sample one random poses every $ steps, for sparse view regularization. 0 disables this feature.")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.1, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--video_frame_num', type = int, default = 150, help="number of frames to train")
    parser.add_argument('--cuda_ray_density_time_slice_interval', type = int, default = 1, help="")
    parser.add_argument('--video_frame_start', type = int, default = 0, help="number of frames to train")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### Model options
    parser.add_argument('--hidden_dim', type=int, default=256, help="GUI width")
    parser.add_argument('--geo_feat_dim', type=int, default=128, help="GUI width")

    opt = parser.parse_args()

    if opt.O:
        opt.cuda_ray = True
        opt.preload = True

    print(opt)
    return opt

if __name__ == '__main__':

    opt = get_argparse()  

    seed_everything(opt.seed)

    model = model_dict[opt.model](
        bound=opt.bound,
        hidden_dim = opt.hidden_dim,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        static = opt.model_static,
        time_slice = int(opt.video_frame_num/opt.cuda_ray_density_time_slice_interval)
    )
    
    print(model)

    criterion = torch.nn.MSELoss(reduction='none')

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    
    if opt.test:

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=[PSNRMeter()], use_checkpoint='latest')

        if opt.gui:
            # gui = NeRFGUI(opt, trainer)
            # gui.render()
            pass
        
        else:
            test_loader = DyNeRFDataset(opt, device=device, type='test', downscale=opt.downscale).dataloader()

            if opt.mode == 'blender':
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            else:
                trainer.test(test_loader) # colmap doesn't have gt, so just test.
            
            trainer.save_mesh(resolution=256, threshold=10)
    
    else:
        
        # you can tune the optimizer ! The default one is suitable for CombineDyNeRFNetwork
        # optimizer = lambda model: torch.optim.Adam([
        #     {'name': 'encoding', 'params': list(model.encoder.parameters()), 'lr': 1e-2},
        #     {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': 0, 'lr': 5e-3},
        # ], betas=(0.9, 0.99), eps=1e-15)
        optimizer = lambda model: torch.optim.Adam(model.parameters(), lr = 5e-3, betas=(0.9, 0.99), eps=1e-15)
        scheduler = lambda optimizer: optim.lr_scheduler.ExponentialLR(optimizer, 0.985,last_epoch=-1)

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, 
            criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, metrics=[PSNRMeter()], 
            use_checkpoint='latest', eval_interval=opt.eval_interval)

        if opt.gui:
            train_loader = DyNeRFDataset(opt, device=device, type='all', downscale=opt.downscale).dataloader()
            trainer.train_loader = train_loader # attach dataloader to trainer
        
        else:
            train_loader = DyNeRFDataset(opt, device=device, type='train', downscale=opt.downscale).dataloader()
            valid_loader = DyNeRFDataset(opt, device=device, type='val', downscale=opt.downscale).dataloader()

            trainer.train(train_loader, valid_loader, opt.epoch)

            # also test
            test_loader = DyNeRFDataset(opt, device=device, type='test', downscale=opt.downscale).dataloader()
            
            if opt.mode == 'blender':
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            else:
                trainer.test(test_loader) # colmap doesn't have gt, so just test.
            
            trainer.save_mesh(resolution=256, threshold=10)
