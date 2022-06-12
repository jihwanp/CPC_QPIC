# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from scipy.optimize import linear_sum_assignment
import pdb
import torch
from torch import nn
import torch.nn.functional as F
import copy
import itertools

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)


class DETRHOI(nn.Module):

    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, aux_loss=False,use_type=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.use_type=use_type
        
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.transformer.decoder.parameters():
            p.requires_grad_(True)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        #HO->I
        self.HOtoI_1decoder=copy.deepcopy(self.transformer.decoder)
        self.HOtoI_2decoder=copy.deepcopy(self.transformer.decoder)
        self.query_embed_HOtoI = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_HOtoI2 = nn.Embedding(num_queries, hidden_dim)
        self.obj_class_embed_HOtoI = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed_HOtoI = nn.Linear(hidden_dim, num_verb_classes)
        self.interactiveness_HOtoI =nn.Linear(hidden_dim,1)
        self.sub_bbox_embed_HOtoI = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed_HOtoI = MLP(hidden_dim, hidden_dim, 4, 3)
        #HI->O
        self.HItoO_1decoder=copy.deepcopy(self.transformer.decoder)
        self.HItoO_2decoder=copy.deepcopy(self.transformer.decoder)
        self.query_embed_HItoO = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_HItoO2 = nn.Embedding(num_queries, hidden_dim)
        self.obj_class_embed_HItoO = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed_HItoO = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed_HItoO = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed_HItoO = MLP(hidden_dim, hidden_dim, 4, 3)
        #OI->H
        self.OItoH_1decoder=copy.deepcopy(self.transformer.decoder)
        self.OItoH_2decoder=copy.deepcopy(self.transformer.decoder)
        self.query_embed_OItoH = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_OItoH2 = nn.Embedding(num_queries, hidden_dim)
        self.obj_class_embed_OItoH = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed_OItoH = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed_OItoH = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed_OItoH = MLP(hidden_dim, hidden_dim, 4, 3)

        

        if use_type:
            self.q_type_embed=nn.EmbeddingBag(3,hidden_dim) #H,O,I,pos
            self.out_type_embed=nn.EmbeddingBag(3,hidden_dim) #H,O,I

        self.aux_loss = aux_loss

        
    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        bs = features[-1].tensors.shape[0]
        src, mask = features[-1].decompose()
        assert mask is not None
        # pdb.set_trace()
        
        # q_pos=torch.tensor([[3]],device=self.q_type_embed.weight.device)
        import pdb;pdb.set_trace()
        hs,memory= self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])
        outputs_obj_class_base = self.obj_class_embed(hs)
        outputs_verb_class_base = self.verb_class_embed(hs)
        outputs_sub_coord_base = self.sub_bbox_embed(hs).sigmoid()
        outputs_obj_coord_base = self.obj_bbox_embed(hs).sigmoid()
                
    
        tgt = torch.zeros_like(self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1))
        hs_HItoO=self.HItoO_1decoder(tgt, memory, memory_key_padding_mask=mask.flatten(1), pos=pos[-1].flatten(2).permute(2, 0, 1), query_pos=self.query_embed_HItoO.weight.unsqueeze(1).repeat(1, bs, 1)).transpose(1,2)
        hs2_HItoO=self.HItoO_2decoder(hs_HItoO.transpose(1,2)[-1], memory, memory_key_padding_mask=mask.flatten(1), pos=pos[-1].flatten(2).permute(2, 0, 1), query_pos=self.query_embed_HItoO2.weight.unsqueeze(1).repeat(1, bs, 1)).transpose(1,2) 
        outputs_obj_class_HItoO = self.obj_class_embed_HItoO(hs2_HItoO)
        outputs_verb_class_HItoO = self.verb_class_embed_HItoO(hs_HItoO)
        outputs_sub_coord_HItoO = self.sub_bbox_embed_HItoO(hs_HItoO).sigmoid()
        outputs_obj_coord_HItoO = self.obj_bbox_embed_HItoO(hs2_HItoO).sigmoid()
        #OItoH
        hs_OItoH=self.OItoH_1decoder(tgt, memory, memory_key_padding_mask=mask.flatten(1), pos=pos[-1].flatten(2).permute(2, 0, 1), query_pos=self.query_embed_OItoH.weight.unsqueeze(1).repeat(1, bs, 1)).transpose(1,2)
        hs2_OItoH=self.OItoH_2decoder(hs_OItoH.transpose(1,2)[-1], memory, memory_key_padding_mask=mask.flatten(1), pos=pos[-1].flatten(2).permute(2, 0, 1), query_pos=self.query_embed_OItoH2.weight.unsqueeze(1).repeat(1, bs, 1)).transpose(1,2) 
        outputs_obj_class_OItoH = self.obj_class_embed_OItoH(hs_OItoH)
        outputs_verb_class_OItoH = self.verb_class_embed_OItoH(hs_OItoH)
        outputs_sub_coord_OItoH = self.sub_bbox_embed_OItoH(hs2_OItoH).sigmoid()
        outputs_obj_coord_OItoH = self.obj_bbox_embed_OItoH(hs_OItoH).sigmoid()
        #HOtoI
        hs_HOtoI=self.HOtoI_1decoder(tgt, memory, memory_key_padding_mask=mask.flatten(1), pos=pos[-1].flatten(2).permute(2, 0, 1), query_pos=self.query_embed_HOtoI.weight.unsqueeze(1).repeat(1, bs, 1)).transpose(1,2)
        hs2_HOtoI=self.HOtoI_2decoder(hs_HOtoI.transpose(1,2)[-1], memory, memory_key_padding_mask=mask.flatten(1), pos=pos[-1].flatten(2).permute(2, 0, 1), query_pos=self.query_embed_HOtoI2.weight.unsqueeze(1).repeat(1, bs, 1)).transpose(1,2) 
        outputs_obj_class_HOtoI = self.obj_class_embed_HOtoI(hs_HOtoI)
        outputs_verb_class_HOtoI = self.verb_class_embed_HOtoI(hs2_HOtoI)
        outputs_interactiveness_HOtoI=self.interactiveness_HOtoI(hs_HOtoI)
        # outputs_verb_class_HOtoI=torch.cat([outputs_verb_class_HOtoI_,outputs_interactiveness_HOtoI],dim=-1)
        outputs_sub_coord_HOtoI = self.sub_bbox_embed_HOtoI(hs_HOtoI).sigmoid()
        outputs_obj_coord_HOtoI = self.obj_bbox_embed_HOtoI(hs_HOtoI).sigmoid()
        
        outputs_obj_class=torch.stack([outputs_obj_class_base,outputs_obj_class_HItoO,outputs_obj_class_OItoH,outputs_obj_class_HOtoI],dim=2) #(dec_l,bs,4,q,num_obj_class)
        outputs_verb_class=torch.stack([outputs_verb_class_base,outputs_verb_class_HItoO,outputs_verb_class_OItoH,outputs_verb_class_HOtoI],dim=2) #(dec_l,bs,4,q,num_verb_class)
        outputs_sub_coord=torch.stack([outputs_sub_coord_base,outputs_sub_coord_HItoO,outputs_sub_coord_OItoH,outputs_sub_coord_HOtoI],dim=2) #(dec_l,bs,4,q,4)
        outputs_obj_coord=torch.stack([outputs_obj_coord_base,outputs_obj_coord_HItoO,outputs_obj_coord_OItoH,outputs_obj_coord_HOtoI],dim=2) #(dec_l,bs,4,q,4)

        # outputs_obj_class=torch.stack([outputs_obj_class_base,outputs_obj_class_HOtoI],dim=2) #(dec_l,bs,4,q,num_obj_class)
        # outputs_verb_class=torch.stack([outputs_verb_class_base,outputs_verb_class_HOtoI],dim=2) #(dec_l,bs,4,q,num_verb_class)
        # outputs_sub_coord=torch.stack([outputs_sub_coord_base,outputs_sub_coord_HOtoI],dim=2) #(dec_l,bs,4,q,4)
        # outputs_obj_coord=torch.stack([outputs_obj_coord_base,outputs_obj_coord_HOtoI],dim=2) #(dec_l,bs,4,q,4)

            #cdn
            # hs_HItoO=self.transformer(self.input_proj(src), mask, self.query_embed_HItoO.weight, pos[-1])[0]
            # hs2_HItoO=self.transformer(self.input_proj(src),mask,hs_HItoO.transpose(1,2)[-1], pos[-1])[0]
            # outputs_obj_class_HItoO = self.obj_class_embed_HItoO(hs2_HItoO)
            # outputs_verb_class_HItoO = self.verb_class_embed_HItoO(hs_HItoO)
            # outputs_sub_coord_HItoO = self.sub_bbox_embed_HItoO(hs_HItoO).sigmoid()
            # outputs_obj_coord_HItoO = self.obj_bbox_embed_HItoO(hs2_HItoO).sigmoid()
            # #OItoH
            # hs_OItoH=self.transformer(self.input_proj(src), mask, self.query_embed_OItoH.weight, pos[-1])[0]
            # hs2_OItoH=self.transformer(self.input_proj(src),mask,hs_OItoH.transpose(1,2)[-1], pos[-1])[0]
            # outputs_obj_class_OItoH = self.obj_class_embed_OItoH(hs_OItoH)
            # outputs_verb_class_OItoH = self.verb_class_embed_OItoH(hs_OItoH)
            # outputs_sub_coord_OItoH = self.sub_bbox_embed_OItoH(hs2_OItoH).sigmoid()
            # outputs_obj_coord_OItoH = self.obj_bbox_embed_OItoH(hs_OItoH).sigmoid()
            # #HOtoI
            # hs_HOtoI=self.transformer(self.input_proj(src), mask, self.query_embed_HOtoI.weight, pos[-1])[0]
            # hs2_HOtoI=self.transformer(self.input_proj(src),mask,hs_HOtoI.transpose(1,2)[-1], pos[-1])[0]
            # outputs_obj_class_HOtoI = self.obj_class_embed_HOtoI(hs_HOtoI)
            # outputs_verb_class_HOtoI = self.verb_class_embed_HOtoI(hs2_HOtoI)
            # outputs_interactiveness_HOtoI=self.interactiveness_HOtoI(hs_HOtoI)
            # outputs_sub_coord_HOtoI = self.sub_bbox_embed_HOtoI(hs_HOtoI).sigmoid()
            # outputs_obj_coord_HOtoI = self.obj_bbox_embed_HOtoI(hs_HOtoI).sigmoid()
        
        # outputs_obj_class=torch.stack([outputs_obj_class_base,outputs_obj_class_HItoO,outputs_obj_class_OItoH,outputs_obj_class_HOtoI],dim=2) #(dec_l,bs,4,q,num_obj_class)
        # outputs_verb_class=torch.stack([outputs_verb_class_base,outputs_verb_class_HItoO,outputs_verb_class_OItoH,outputs_verb_class_HOtoI],dim=2) #(dec_l,bs,4,q,num_verb_class)
        # outputs_sub_coord=torch.stack([outputs_sub_coord_base,outputs_sub_coord_HItoO,outputs_sub_coord_OItoH,outputs_sub_coord_HOtoI],dim=2) #(dec_l,bs,4,q,4)
        # outputs_obj_coord=torch.stack([outputs_obj_coord_base,outputs_obj_coord_HItoO,outputs_obj_coord_OItoH,outputs_obj_coord_HOtoI],dim=2) #(dec_l,bs,4,q,4)
        
        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1],
               'pred_HOtoI_interactiveness':outputs_interactiveness_HOtoI[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                    outputs_sub_coord, outputs_obj_coord,outputs_interactiveness_HOtoI)
        #pdb.set_trace()
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord,outputs_interactiveness_HOtoI):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d,'pred_HOtoI_interactiveness':e}
                for a, b, c, d,e in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1],outputs_interactiveness_HOtoI[:-1])]


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, verb_loss_type):
        super().__init__()

        assert verb_loss_type == 'bce' or verb_loss_type == 'focal'

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.verb_loss_type = verb_loss_type

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, use_consis,log=True):
        assert 'pred_obj_logits' in outputs
        
        src_logits = outputs['pred_obj_logits'].flatten(0,1)
        nu,q,hd=src_logits.shape
        hoi_ind=list(itertools.chain.from_iterable(indices))
        idx = self._get_src_permutation_idx(hoi_ind)

        
        
        target_classes_o = torch.cat([t['obj_labels'][J] for t, indice in zip(targets, indices) for (_,J) in indice])
        # target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
        #                             dtype=torch.int64, device=src_logits.device)
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

        #consistency
        if use_consis:
            
            consistency_idxs=[self._get_consistency_src_permutation_idx(indice) for indice in indices ]
            src_logits_inputs=[F.softmax(outputs['pred_obj_logits'][i][consistency_idx[0]],-1) for i,consistency_idx in enumerate(consistency_idxs)]
            src_logits_targets=[F.softmax(outputs['pred_obj_logits'][i][consistency_idx[1]],-1) for i,consistency_idx in enumerate(consistency_idxs)]
            # import pdb;pdb.set_trace()
            # src_logits_means=[(0.5*(src_logits_input+src_logits_target)) for src_logits_input,src_logits_target in zip(src_logits_inputs,src_logits_targets)]
            loss_obj_consistency=[0.5*(F.kl_div(src_logits_input.log(),src_logits_target.clone().detach(),reduction='batchmean')+F.kl_div(src_logits_target.log(),src_logits_input.clone().detach(),reduction='batchmean')) if src_logits_input.nelement()!=0 else src_logits_input.sum() for src_logits_input,src_logits_target in zip(src_logits_inputs,src_logits_targets)]
            loss_obj_consistency=torch.mean(torch.stack(loss_obj_consistency))
            # if loss_obj_consistency:
            #   loss_obj_consistency=torch.mean(torch.stack(loss_obj_consistency))
            # else: 
            #   loss_obj_consistency=torch.tensor(0.,device=src_logits.device)
            # pdb.set_trace()
            losses = {'loss_obj_ce': loss_obj_ce,'loss_obj_label_consistency':loss_obj_consistency}
        else: 
            losses = {'loss_obj_ce': loss_obj_ce}
        # pdb.set_trace()
        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions,use_consis):
        pred_logits = outputs['pred_obj_logits'].transpose(0,1).flatten(0,1)
        _,num_tf=outputs['pred_obj_logits'].shape[:2]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device).repeat(num_tf)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions,use_consis):
        assert 'pred_verb_logits' in outputs
        _,num_tf=outputs['pred_verb_logits'].shape[:2]
        verb_prob=outputs['pred_verb_logits'].sigmoid()
        interactiveness_prob=outputs['pred_HOtoI_interactiveness'].sigmoid()
        verb_prob=verb_prob*torch.cat([torch.stack([torch.ones_like(interactiveness_prob)]*(num_tf-1),dim=1),interactiveness_prob.unsqueeze(1)],dim=1)

        # import pdb;pdb.set_trace()
        # src_logits = outputs['pred_verb_logits'].flatten(0,1)
        src_logits = verb_prob.flatten(0,1)
        
        hoi_ind=list(itertools.chain.from_iterable(indices))
        idx = self._get_src_permutation_idx(hoi_ind)
        

        target_classes_o = torch.cat([t['verb_labels'][J] for t, indice in zip(targets, indices) for (_,J) in indice])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        if self.verb_loss_type == 'bce':
            loss_verb_ce = F.binary_cross_entropy(src_logits, target_classes)
            
        elif self.verb_loss_type == 'focal':
            # src_logits = src_logits.sigmoid()
            loss_verb_ce = self._neg_loss(src_logits, target_classes)
        if use_consis:
            consistency_idxs=[self._get_consistency_src_permutation_idx(indice) for indice in indices]
            src_action_inputs=[(verb_prob[i][consistency_idx[0]]).log() for i,consistency_idx in enumerate(consistency_idxs)]
            src_action_targets=[(verb_prob[i][consistency_idx[1]]).log() for i,consistency_idx in enumerate(consistency_idxs)]
            # loss_action_consistency=[F.mse_loss(src_action_input,src_action_target) for src_action_input,src_action_target in zip(src_action_inputs,src_action_targets) if src_action_input.nelement()!=0]
            # if loss_action_consistency:
            #     loss_action_consistency=torch.mean(torch.stack(loss_action_consistency))
            # else:
            #     loss_action_consistency=torch.tensor(0.,device=src_logits.device)
            loss_action_consistency=[F.mse_loss(src_action_input,src_action_target) if src_action_input.nelement()!=0 else src_action_input.sum() for src_action_input,src_action_target in zip(src_action_inputs,src_action_targets)]
            
            loss_action_consistency=torch.mean(torch.stack(loss_action_consistency))
        
            
            losses = {'loss_verb_ce': loss_verb_ce,'loss_verb_label_consistency':loss_action_consistency}
        # # import pdb;pdb.set_trace()
        else:
            losses = {'loss_verb_ce': loss_verb_ce}
        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions,use_consis):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        
        hoi_ind=list(itertools.chain.from_iterable(indices))
        idx = self._get_src_permutation_idx(hoi_ind)
        _,num_tf=outputs['pred_sub_boxes'].shape[:2]
        # pdb.set_trace()
        src_sub_boxes = outputs['pred_sub_boxes'].flatten(0,1)[idx]
        src_obj_boxes = outputs['pred_obj_boxes'].flatten(0,1)[idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, indice in zip(targets, indices) for (_,i) in indice], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, indice in zip(targets, indices) for (_,i) in indice], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
            if use_consis:
                # losses['loss_sub_bbox_consistency']=src_sub_boxes.sum()
                # losses['loss_sub_giou_consistency']=src_sub_boxes.sum()
                # losses['loss_obj_bbox_consistency']=src_obj_boxes.sum()
                # losses['loss_obj_giou_consistency']=src_obj_boxes.sum()
                losses['loss_sub_reg_label_consistency']=src_sub_boxes.sum()
                losses['loss_obj_reg_label_consistency']=src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / (num_interactions*num_tf)
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / (num_interactions*num_tf)
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
            if use_consis:
                # consistency_sub_idxs=[self._get_consistency_src_permutation_idx(indice) for indice in indices ]
                # consistency_obj_idxs=[self._get_consistency_src_permutation_idx(indice,target_obj=target['obj_boxes']) for indice,target in zip(indices,targets)]
                # src_sub_inputs=[outputs['pred_sub_boxes'][i][consistency_idx[0]] for i,consistency_idx in enumerate(consistency_sub_idxs)]
                # src_sub_targets=[outputs['pred_sub_boxes'][i][consistency_idx[1]] for i,consistency_idx in enumerate(consistency_sub_idxs)]
                # src_obj_inputs=[outputs['pred_obj_boxes'][i][consistency_idx[0]] for i,consistency_idx in enumerate(consistency_obj_idxs)]
                # src_obj_targets=[outputs['pred_obj_boxes'][i][consistency_idx[1]] for i,consistency_idx in enumerate(consistency_obj_idxs)]
                # loss_sub_bbox_consistency=[F.l1_loss(src_sub_input,src_sub_target, reduction='none').sum()/float(len(src_sub_input)) if src_sub_input.nelement()!=0 else src_sub_input.sum() for src_sub_input,src_sub_target in zip(src_sub_inputs,src_sub_targets) ]
                # loss_sub_giou_consistency=[(1-torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_input),box_cxcywh_to_xyxy(src_sub_target)))).sum()/float(len(src_sub_input)) if src_sub_input.nelement()!=0 else src_sub_input.sum() for src_sub_input,src_sub_target in zip(src_sub_inputs,src_sub_targets) ]
                # loss_obj_bbox_consistency=[F.l1_loss(src_obj_input,src_obj_target, reduction='none').sum()/float(len(src_obj_input)) if src_obj_input.nelement()!=0 else src_obj_input.sum() for src_obj_input,src_obj_target in zip(src_obj_inputs,src_obj_targets) ]
                # loss_obj_giou_consistency=[(1-torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_input),box_cxcywh_to_xyxy(src_obj_target)))).sum()/float(len(src_obj_input)) if src_obj_input.nelement()!=0 else src_obj_input.sum() for src_obj_input,src_obj_target in zip(src_obj_inputs,src_obj_targets) ]
                # losses['loss_sub_bbox_consistency']=torch.mean(torch.stack(loss_sub_bbox_consistency))
                # losses['loss_sub_giou_consistency']=torch.mean(torch.stack(loss_sub_giou_consistency))
                # losses['loss_obj_bbox_consistency']=torch.mean(torch.stack(loss_obj_bbox_consistency))
                # losses['loss_obj_giou_consistency']=torch.mean(torch.stack(loss_obj_giou_consistency))
                # 
                consistency_sub_idxs=[self._get_consistency_src_permutation_idx(indice) for indice in indices ]
                consistency_obj_idxs=[self._get_consistency_src_permutation_idx(indice,target_obj=target['obj_boxes']) for indice,target in zip(indices,targets)]
                src_sub_inputs=[outputs['pred_sub_boxes'][i][consistency_idx[0]] for i,consistency_idx in enumerate(consistency_sub_idxs)]
                src_sub_targets=[outputs['pred_sub_boxes'][i][consistency_idx[1]] for i,consistency_idx in enumerate(consistency_sub_idxs)]
                src_obj_inputs=[outputs['pred_obj_boxes'][i][consistency_idx[0]] for i,consistency_idx in enumerate(consistency_obj_idxs)]
                src_obj_targets=[outputs['pred_obj_boxes'][i][consistency_idx[1]] for i,consistency_idx in enumerate(consistency_obj_idxs)]
                loss_sub_consistency=[F.mse_loss(src_sub_input,src_sub_target) if src_sub_input.nelement()!=0 else src_sub_input.sum() for src_sub_input,src_sub_target in zip(src_sub_inputs,src_sub_targets) ]
                loss_obj_consistency=[F.mse_loss(src_obj_input,src_obj_target) if src_obj_input.nelement()!=0 else src_obj_input.sum() for src_obj_input,src_obj_target in zip(src_obj_inputs,src_obj_targets) ]
                losses['loss_sub_reg_label_consistency']=torch.mean(torch.stack(loss_sub_consistency))
                losses['loss_obj_reg_label_consistency']=torch.mean(torch.stack(loss_obj_consistency))
            
            
            # pdb.set_trace()
        


        return losses

    def _neg_loss(self, pred, gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_consistency_src_permutation_idx(self, indices,target_obj=None):
        all_tgt=torch.cat([j for(_,j) in indices]).unique()
        
        if torch.is_tensor(target_obj): 
            exist_obj_gt_index=all_tgt[(target_obj[all_tgt]!=0).any(dim=1)]
        else:
            exist_obj_gt_index=all_tgt
        path_idxs=[torch.cat([torch.tensor([i]) for i,(_,t)in enumerate(indices) if (t==tgt).any()]) for tgt in exist_obj_gt_index]
        q_idxs=[torch.cat([s[t==tgt] for (s,t)in indices]) for tgt in exist_obj_gt_index]
        path_idxs_=[torch.combinations(path_idx) for path_idx in path_idxs if len(path_idx)>1]
        q_idxs_=[torch.combinations(q_idx) for q_idx in q_idxs if len(q_idx)>1]
        assert len(path_idxs_)==len(q_idxs_)
        if len(path_idxs_)>0:
            path_idxs=torch.cat(path_idxs_)
            q_idxs=torch.cat(q_idxs_)
        # import pdb;pdb.set_trace()
            return (path_idxs[:,0],q_idxs[:,0]),(path_idxs[:,1],q_idxs[:,1])
        else:
            return ([],[]),([],[])

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num,use_consis, **kwargs):
        loss_map = {
            'obj_labels': self.loss_obj_labels,
            'obj_cardinality': self.loss_obj_cardinality,
            'verb_labels': self.loss_verb_labels,
            'sub_obj_boxes': self.loss_sub_obj_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num,use_consis, **kwargs)

    def forward(self, outputs, targets,log,use_consis):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets,log)

        num_interactions = sum(len(t['obj_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions,use_consis))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'obj_labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions,use_consis, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcessHOI(nn.Module):

    def __init__(self, subject_category_id):
        super().__init__()
        self.subject_category_id = subject_category_id

    @torch.no_grad()
    def forward(self, outputs, target_sizes,path_id):
        
        out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = outputs['pred_obj_logits'][:,path_id,...], \
                                                                        outputs['pred_verb_logits'][:,path_id,...], \
                                                                        outputs['pred_sub_boxes'][:,path_id,...], \
                                                                        outputs['pred_obj_boxes'][:,path_id,...]

        
        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        
        obj_prob = F.softmax(out_obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        verb_scores = out_verb_logits.sigmoid()

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for os, ol, vs, sb, ob in zip(obj_scores, obj_labels, verb_scores, sub_boxes, obj_boxes):
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            vs = vs * os.unsqueeze(1)

            ids = torch.arange(b.shape[0])

            results[-1].update({'verb_scores': vs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                'obj_ids': ids[ids.shape[0] // 2:]})

        return results
