# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import argparse

import torch
from torch import nn
import copy

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--load_path', type=str, required=True,
    )
    parser.add_argument(
        '--save_path', type=str, required=True,
    )
    parser.add_argument(
        '--dataset', type=str, default='hico',
    )

    args = parser.parse_args()

    return args


def main(args):
    ps = torch.load(args.load_path)

    obj_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
               82, 84, 85, 86, 87, 88, 89, 90]

    # For no pair
    obj_ids.append(91)
    path_name=['','_HItoO','_HOtoI','_OItoH']
    for path in path_name:
        ps['model']['query_embed'+path+'.weight'] = ps['model']['query_embed.weight'].clone()
    
        ps['model']['query_embed'+path+'2'+'.weight'] = ps['model']['query_embed.weight'].clone()
        
        ps['model']['sub_bbox_embed'+path+'.layers.0.weight'] = ps['model']['bbox_embed.layers.0.weight'].clone()
        ps['model']['sub_bbox_embed'+path+'.layers.0.bias'] = ps['model']['bbox_embed.layers.0.bias'].clone()
        ps['model']['sub_bbox_embed'+path+'.layers.1.weight'] = ps['model']['bbox_embed.layers.1.weight'].clone()
        ps['model']['sub_bbox_embed'+path+'.layers.1.bias'] = ps['model']['bbox_embed.layers.1.bias'].clone()
        ps['model']['sub_bbox_embed'+path+'.layers.2.weight'] = ps['model']['bbox_embed.layers.2.weight'].clone()
        ps['model']['sub_bbox_embed'+path+'.layers.2.bias'] = ps['model']['bbox_embed.layers.2.bias'].clone()

        ps['model']['obj_bbox_embed'+path+'.layers.0.weight'] = ps['model']['bbox_embed.layers.0.weight'].clone()
        ps['model']['obj_bbox_embed'+path+'.layers.0.bias'] = ps['model']['bbox_embed.layers.0.bias'].clone()
        ps['model']['obj_bbox_embed'+path+'.layers.1.weight'] = ps['model']['bbox_embed.layers.1.weight'].clone()
        ps['model']['obj_bbox_embed'+path+'.layers.1.bias'] = ps['model']['bbox_embed.layers.1.bias'].clone()
        ps['model']['obj_bbox_embed'+path+'.layers.2.weight'] = ps['model']['bbox_embed.layers.2.weight'].clone()
        ps['model']['obj_bbox_embed'+path+'.layers.2.bias'] = ps['model']['bbox_embed.layers.2.bias'].clone()

        ps['model']['obj_class_embed'+path+'.weight'] = ps['model']['class_embed.weight'].clone()[obj_ids]
        ps['model']['obj_class_embed'+path+'.bias'] = ps['model']['class_embed.bias'].clone()[obj_ids]

        if args.dataset == 'vcoco':
            l = nn.Linear(ps['model']['obj_class_embed'+path+'.weight'].shape[1], 1)
            l.to(ps['model']['obj_class_embed'+path+'.weight'].device)
            ps['model']['obj_class_embed'+path+'.weight'] = torch.cat((
                ps['model']['obj_class_embed'+path+'.weight'][:-1], l.weight, ps['model']['obj_class_embed'+path+'.weight'][[-1]]))
            ps['model']['obj_class_embed'+path+'.bias'] = torch.cat(
                (ps['model']['obj_class_embed'+path+'.bias'][:-1], l.bias, ps['model']['obj_class_embed'+path+'.bias'][[-1]]))
    path_name=['HOtoI_','HItoO_','OItoH_']
    
    for i in range(2):
        for key in ps['model'].copy().keys():
            
            for path in path_name:
            
                if 'transformer.decoder' in key:
                    ps['model'][path+str(i+1)+key.split('.',1)[1]] = ps['model'][key].clone()
                    
    torch.save(ps, args.save_path)


if __name__ == '__main__':
    args = get_args()
    main(args)