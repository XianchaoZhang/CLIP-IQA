# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os, glob


import torch

from mmedit.apis import init_model, restoration_inference, init_coop_model
from mmedit.core import tensor2img, srocc, plcc

import pandas as pd
from tqdm import tqdm

# import megengine as mge
# mge.dtr.eviction_threshold = "10GB"
# mge.dtr.enable()

def parse_args():
    parser = argparse.ArgumentParser(description='ClipIQA demo')
    parser.add_argument('--config', default='configs/clipiqa/clipiqa_attribute_test.py', help='test config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--file_path', default='..\\dataset\\macro\\HDRAuto&Off', help='path to input image file')
    parser.add_argument('--csv_path', default='../dataset/macro/HDRAuto&Off/tmp.csv', help='path to output csv file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    attribute_list = ['Quality', 'Brightness', 'Sharpness', 'Noisiness', 'Colorfulness', 'Color', 'Contrast', 'Aesthetic', 'Happy', 'Natural', 'Scary', 'Complex']
    #out_cols = ['image_name', *attribute_list]

    img_test = glob.glob(args.file_path + "\\*.jpg")

    #print(img_test)
    #exit(0)
    #gc.collect()
    #torch.cuda.empty_cache()
    #print(torch.cuda.memory_summary())

    out_idx =[]
    attr_rows = []
    for i in tqdm(range(len(img_test))):
        f = img_test[i]
        fp, fn = os.path.split(f)
        #print('\n', fn)

        output, attributes = restoration_inference(model, f, return_attributes=True)
        attributes = attributes.float().detach().cpu().numpy()[0]
        out_idx.append(fn)
        attr_rows.append(attributes)
        #torch.cuda.empty_cache()

    # end for loop    
    #exit(0)
    out_df = pd.DataFrame(attr_rows, columns=attribute_list, index=out_idx)
    out_df.to_csv(args.csv_path)



if __name__ == '__main__':
    main()