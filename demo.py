import datetime
import time
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch.nn.functional as F
import saverloader
import torch
from fire import Fire
from nets.pips2 import Pips
from tensorboardX import SummaryWriter

import utils.improc
from utils.basic import print_, print_stats

def read_mp4(fn) -> list[np.ndarray]:
    vidcap = cv2.VideoCapture(fn)
    frames = []
    while(vidcap.isOpened()):
        ret, frame = vidcap.read()
        if ret == False:
            break
        frames.append(frame)
    vidcap.release()
    return frames

def run_model(model: Pips, rgbs: list[np.ndarray], S_max=128, N=64, iters=16, sw=None):
    rgbs = rgbs.cuda().float() # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    assert(B==1)

    # pick N points to track; we'll use a uniform grid
    N_ = np.sqrt(N).round().astype(np.int32)
    grid_y, grid_x = utils.basic.meshgrid2d(B, N_, N_, stack=False, norm=False, device='cuda')
    grid_y = 8 + grid_y.reshape(B, -1)/float(N_-1) * (H-16)
    grid_x = 8 + grid_x.reshape(B, -1)/float(N_-1) * (W-16)
    xy0 = torch.stack([grid_x, grid_y], dim=-1) # B, N_* N_, 2
    _, S, C, H, W = rgbs.shape

    # zero-vel init
    trajs_e = xy0.unsqueeze(1).repeat(1,S,1,1)

    iter_start_time = time.time()
    
    preds, preds_anim, _, _ = model(trajs_e, rgbs, iters=iters, feat_init=None, beautify=True)
    trajs_e = preds[-1]

    iter_time = time.time() - iter_start_time
    print(f'inference time: {iter_time:.2f} seconds ({S/iter_time:.1f} fps)')

    if sw is not None and sw.save_this:
        rgbs_prep = utils.improc.preprocess_color(rgbs)
        sw.summ_traj2ds_on_rgbs(
            name='outputs/trajs_on_rgbs', 
            trajs=trajs_e[0:1], 
            rgbs=utils.improc.preprocess_color(rgbs[0:1]), 
            cmap='hot', linewidth=1, show_dots=False,
        )
    
    return trajs_e


def main(
        filename='./stock_videos/camel.mp4',
        S=48, # seqlen
        N=1024, # number of points per clip
        stride=8, # spatial stride of the model
        timestride=1, # temporal stride of the model
        iters=16, # inference steps of the model
        image_size=(512,896), # input resolution
        max_iters=4, # number of clips to run
        shuffle=False, # dataset shuffling
        log_freq=1, # how often to make image summaries
        log_dir='./logs_demo',
        init_dir='./reference_model',
        device_ids=[0],
):

    # the idea in this file is to run the model on a demo video,
    # and return some visualizations
    
    exp_name = 'de00' # copy from dev repo

    print(f'{filename=}')
    name = Path(filename).stem
    # print('name', name)
    
    rgbs = read_mp4(filename)
    rgbs = np.stack(rgbs, axis=0)  # nframes,H,W,3
    rgbs = rgbs[:,:,:,::-1].copy()  # BGR->RGB
    rgbs = rgbs[::timestride]
    nframes, H, W, C = rgbs.shape
    print(f'{rgbs.shape=}')

    # autogen a name
    model_name = "%s_%d_%d_%s" % (name, S, N, exp_name)
    
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print(f'{model_name=}')
    
    log_dir = 'logs_demo'
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    model = Pips(stride=8).cuda()
    parameters = list(model.parameters())
    if init_dir and os.path.exists(init_dir):
        _ = saverloader.load(init_dir, model)
    model.eval()

    idx = list(range(0, max(nframes - S, 1), S))
    if max_iters:
        idx = idx[:max_iters]
    print(f'{idx=}')
    
    global_step = 0
    for si in idx:
        global_step += 1
        
        iter_start_time = time.time()

        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=16,
            scalar_freq=int(log_freq/2),
            just_gif=True)

        rgb_seq = rgbs[si:si+S]
        rgb_seq = torch.from_numpy(rgb_seq).permute(0,3,1,2).to(torch.float32) # S,3,H,W
        rgb_seq = F.interpolate(rgb_seq, image_size, mode='bilinear').unsqueeze(0) # 1,S,3,H,W
        
        with torch.no_grad():
            _trajs_e = run_model(model, rgb_seq, S_max=S, N=N, iters=iters, sw=sw_t)

        iter_time = time.time()-iter_start_time
        
        print(f'{model_name=}; step {global_step=:06}/{max_iters}; {iter_time=:.2f}s')
        
    writer_t.close()
            

if __name__ == '__main__':
    Fire(main)
