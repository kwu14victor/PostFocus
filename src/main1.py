'''
Part one of PostFocus pipeline: find OOF image through CoatNet classifier
version 1. By KLW, 20240315
'''
import argparse
import os

import numpy as np
import torch
import pandas as pd
from PIL import Image
from skimage import io, exposure
from torchvision import transforms

from MyModels import CoatNet
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_TIMING_block_list(homedir):
    '''
    Read TIMING data following the schema from Lu et al
    Load blocks into a list
    '''
    blocks = [os.path.join(homedir,f) for f in os.listdir(homedir) if 'B' in f]
    blocks = [b for b in blocks if os.path.isdir(b)]
    return blocks

def get_TIMING_stack_list(blockdir):
    '''
    Read TIMING data following the schema from Lu et al
    Load nanowells into a list
    '''
    stacks = os.listdir(os.path.join(blockdir, 'images', 'crops_8bit_s'))
    stacks = [f for f in stacks if 'CH0' in f]
    stacks = [os.path.join(blockdir,'images', 'crops_8bit_s',f) for f in stacks]
    return stacks

def get_TIMING_imgstack(stackdir, size=73):
    '''
    Read TIMING data following the schema from Lu et al
    Load all images from a video into a list
    '''
    prefix = os.listdir(stackdir)[0].split('_')[0]
    imgs = [prefix+f'_t{k+1}.tif' for k in range(size)]
    imgs = [os.path.join(stackdir, f) for f in imgs]
    imgs = [f for f in imgs if os.path.isfile(f)]
    return imgs

def stack_2_imgs(imgstack):
    '''
    Read all images from the list only when necessary
    '''
    imgarray = [io.imread(f) for f in imgstack]
    imgarray = [exposure.rescale_intensity(f, out_range=np.uint8) for f in imgarray]
    return imgarray

def stack_2_imgs_restored(imgstack, restored_list, restored_dir, num_ensemble=1):
    '''
    Read the restored TIMING images
    '''
    imgarray = []
    for meta in imgstack:
        b_number = meta.split('/B')[1].split('/')[0]
        img_name = meta.split('.tif')[0].split('/')[-1]
        f_name = f'B{b_number}_{img_name}'
        if f_name in restored_list:
            try:
                imgs = [io.imread(restored_dir+\
                    f_name+f'.tif_output{k}.png') for k in range(num_ensemble)]
                imgs = np.mean(np.stack(imgs), axis=0).astype(np.uint8)
                imgarray.append(imgs)
            except:
                imgarray.append(io.imread(meta))
        else:
            imgarray.append(io.imread(meta))
    imgarray = [exposure.rescale_intensity(meta, out_range=np.uint8) for f in imgarray]
    return imgarray

def infer_imgstack(imgstack, model, transform_func):
    '''
    Classify images
    '''
    imgs = stack_2_imgs(imgstack)
    tf_func = transform_func
    answer = []
    for img in imgs:
        frame = Image.fromarray(img).convert('RGB')
        ans = np.argmax(model((tf_func(frame).unsqueeze_(0).cuda()).float()).detach().cpu().numpy())
        answer.append(ans)
    return answer

def infer_imgstack_restored(imgstack, model, transform_func,\
    restored_list, restored_dir, num_ensemble):
    '''
    Classify restored images
    '''
    imgs = stack_2_imgs_restored(imgstack, restored_list, restored_dir, num_ensemble)
    tf_func = transform_func
    answer = []
    for img in imgs:
        frame = Image.fromarray(img).convert('RGB')
        ans = model((tf_func(frame).unsqueeze_(0).cuda()).float())
        ans = np.argmax(ans.detach().cpu().numpy())
        answer.append(ans)
    return answer

def TIMING_stack_infer(args, model, restored=False):
    '''
    Main function that generates list of OOF images
    '''
    model.eval()
    model.cuda()
    model.load_state_dict(torch.load(args.weights))
    tf_func = transforms.Compose([transforms.Resize((args.size,args.size)),\
        transforms.ToTensor(),\
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    blocks = get_TIMING_block_list(args.TIMING_dir)
    b_data, s_data, val_data = [],[],[]
    if restored:
        restored_list = os.listdir(args.restore_dir)
        restored_list = [f.split('.tif')[0] for f in restored_list]
        restored_list = list(set(restored_list))
    with open(f'{args.prefix}_for_deblur.txt',"a") as note:
        for block in blocks:
            stacks = get_TIMING_stack_list(block)
            if stacks!=[]:
                for stack in stacks:
                    run = False
                    if args.TIMING_subset==1:
                        if os.path.exists(stack.replace('images/crops_8bit_s/',\
                            'labels/TRACK/EZ/FRCNN-Fast/').replace('CH0','')):
                            run = True
                    elif args.TIMING_subset==2:
                        cond1 = os.path.exists(stack.replace('images/crops_8bit_s/',\
                            'labels/DET/FRCNN-Fast/clean/').replace('CH0',''))
                        cond2 = os.path.exists(stack.replace('images/crops_8bit_s/',\
                            'labels/TRACK/EZ/FRCNN-Fast/').replace('CH0',''))
                        if cond1 and not cond2:
                            run = True
                    if run:
                        if args.first_hr==0:
                            imgs = get_TIMING_imgstack(stack)
                        elif args.first_hr==1:
                            imgs = get_TIMING_imgstack(stack, size=13)
                        if not restored:
                            answer = infer_imgstack(imgs, model, tf_func)
                        else:
                            answer = infer_imgstack_restored(imgs, model, tf_func,\
                                restored_list, args.restore_dir, args.num_ensemble)
                        b_data.append(block.split('/')[-1])
                        s_data.append(stack.split('/')[-1])
                        val_data.append(sum(answer)/len(answer))
                        for ind in range(len(answer)):
                            if answer[ind]==1:
                                note.writelines(imgs[ind]+'\n')
    pd.DataFrame([b_data, s_data,val_data]).T.to_csv(f'{args.prefix}_data.csv')
    note.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pytorch CNN practice.')
    parser.add_argument('--mode', default= 'TIMING_stack', type=str)
    parser.add_argument('--model', default= 'COAT', type=str)
    parser.add_argument('--size', default=281, type=int)
    parser.add_argument('--weights', default='COAT_testweights.pth', type=str)
    parser.add_argument('--TIMING_dir',\
        default='/project/varadarajan/kwu14/DT-HPC/NALM6_CARTCD28-02/', type=str)
    parser.add_argument('--restore_dir', type=str)
    parser.add_argument('--prefix', default='NALM6_CARTCD28-02', type=str)
    parser.add_argument('--TIMING_subset', default=1, type=int)
    parser.add_argument('--num_ensemble', default=1, type=int)
    parser.add_argument('--first_hr', default=0, type=int)
    args = parser.parse_args()

    if args.model=='COAT':
        model = CoatNet(args)
    else:
        print('model not defined')
    if args.mode=='TIMING_stack':
        TIMING_stack_infer(args, model)
    else:
        print('function not defined')
