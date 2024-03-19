"""
Part two of PostFocus pipeline: restore images
version 1. By KLW, 20240319
"""

import argparse
import os
import random

import numpy as np
from skimage import io, exposure
import torch as th
from torchvision import transforms

from datasets_augment import DeblurDs
from guided_diffusion_restoration_2 import dist_util, logger
from guided_diffusion_restoration_2.script_util_2ch import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def main():
    '''
    Run restoration using the text file generated in step one
    '''
    args = create_argparser().parse_args()
    outputdir = args.outputdir
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )


    img_ds = DeblurDs(blur_image_files = args.dsdir_GT,\
                      sharp_image_files = args.dsdir_GT,\
                      root_dir = '',crop_size = args.img_size,\
                      crop_size2 = args.img_size,\
                      transform = transforms.Compose([transforms.ToTensor() ]))

    gtfile = open(args.dsdir_GT,'r').readlines()
    gt_imgnames = [f.split('\n')[0].split('/')[-1] for f in gtfile]
    gt_blocknames = [f.split('/images/crops_8bit_s/')[0].split('/B')[1] for f in gtfile]

    dist = int(len(img_ds)//75)+1
    start = (args.subsetID-1)*dist
    end = min((args.subsetID)*dist, len(img_ds))

    sub_ds = th.utils.data.Subset(img_ds,range(start,end))

    datal = th.utils.data.DataLoader(
        sub_ds,
        batch_size=1,
        shuffle=False)
    data = iter(datal)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    index=start

    while True:
        data_b, _ = next(data)
        data_c = th.randn_like(data_b[:, :1, ...])
        img = th.cat((data_b, data_c), dim=1)
        new_name = 'B'+gt_blocknames[index]+'_'+gt_imgnames[index]
        io.imsave(f'{outputdir}{new_name}_1.png',img[0,0,...])
        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)

        i=0
        while i < args.num_ensemble:
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, _, _ = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            output = sample.cpu()[0,0,...].numpy()
            output = exposure.rescale_intensity(output, out_range=np.uint8)
            if np.amax(output)!=0:
                io.imsave(f'{outputdir}{new_name}_output{str(i)}.png', output)
                i+=1
        index+=1

def create_argparser():
    '''
    define arguments for the parser
    '''
    defaults = dict(
        data_dir="/project/varadarajan/kwu14/repo/Diffusion-based-Segmentation-main/data/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        num_ensemble=5
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputdir', type=str, default='Restored_imgs/')
    parser.add_argument('--model_path', type=str, default="DDPM_trained.pt")
    parser.add_argument('--dsdir_GT', type=str)
    parser.add_argument('--subsetID', type=int)
    parser.add_argument('--img_size', type=int, default=224)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
