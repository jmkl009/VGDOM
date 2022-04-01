import numpy as np
import torch
import torch.nn as nn
import torchvision

import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from dgl.nn.pytorch import GraphConv

from dgl import DGLGraph
import dgl.function as fn
from functools import partial

from utils import count_parameters
from TCN import TemporalConvNet


class VAMWOD(nn.Module):
    def __init__(self, roi_output_size, img_H, n_classes, backbone='resnet', use_context=True, use_attention=True, hidden_dim=384, 
                 use_bbox_feat=True, bbox_hidden_dim=32, n_additional_features=0, trainable_convnet=True, drop_prob=0.2, class_names=None):
        """
        Implementation of our Visual Attention-based Model for Webpage Object Detection (VAMWOD)

        roi_output_size: Tuple (int, int) which will be output of the roi_pool layer for each channel of convnet_feature
        img_H: height of image given as input to the convnet. Image assumed to be of same W and H
        n_classes: num of classes for BBoxes
        backbone: string stating which convnet feature extractor to use. Allowed values: [alexnet, resnet (default)]
        use_context: if True, use context for context_representation along with own_features (default: True) 
        use_attention: if True, learn scores for all n_context contexts and take weighted avg for context_representation
            NOTE: this parameter is not used if use_context=False
        hidden_dim: size of hidden contextual representation, used when use_attention=True (default: 384)
        use_bbox_feat: if True, then use [x,y,w,h,asp_ratio] with convnet visual features (default: True)
        bbox_hidden_dim: size of hidden representation of 5 bbox features, used when use_bbox_feat=True (default: 32)
        n_additional_feat: num of additional features for each bbox to be used along with visual and bbox features
        trainable_convnet: if False then convnet weights will be freezed while training (default: True)
        drop_prob: dropout probability (default: 0.2)
        class_names: list of n_classes string elements containing names of the classes (default: [0, 1, ..., n_classes-1])
        """
        super(VAMWOD, self).__init__()

        self.n_classes = n_classes
        self.use_context = use_context
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        self.use_bbox_feat = use_bbox_feat
        self.bbox_hidden_dim = bbox_hidden_dim
        self.n_additional_features = n_additional_features
        self.class_names = np.arange(self.n_classes).astype(str) if class_names is None else class_names

        ##### REPRESENTATION NETWORK (RN) #####
        if backbone == 'resnet':
            self.convnet = torchvision.models.resnet18(pretrained=True)
            modules = list(self.convnet.children())[:-5] # remove last few layers!
        elif backbone == 'alexnet':
            self.convnet = torchvision.models.alexnet(pretrained=True)
            modules = list(self.convnet.features.children())[:7] # remove last few layers!

        self.convnet = nn.Sequential(*modules)
        if trainable_convnet == False:
            for p in self.convnet.parameters(): # freeze weights
                p.requires_grad = False

        _imgs = torch.autograd.Variable(torch.Tensor(1, 3, img_H, img_H))
        _conv_feat = self.convnet(_imgs)
        _convnet_output_size = _conv_feat.shape # [1, C, H, W]
        spatial_scale = _convnet_output_size[2]/img_H

        self.roi_pool = torchvision.ops.RoIPool(roi_output_size, spatial_scale)

        self.n_visual_feat = _convnet_output_size[1] * roi_output_size[0] * roi_output_size[1]
        self.n_bbox_feat = self.bbox_hidden_dim if self.use_bbox_feat else 0 # [x,y,w,h,asp_rat] of BBox encoded to n_bbox_feat
        self.n_own_feat = self.n_visual_feat + self.n_bbox_feat + self.n_additional_features

        if self.use_bbox_feat:
            self.bbox_feat_encoder = nn.Sequential(
                nn.Linear(3, self.n_bbox_feat),
                nn.BatchNorm1d(self.n_bbox_feat),
                nn.ReLU(),
            )
        
        if self.n_additional_features > 0:
            self.bn_additional_feat = nn.BatchNorm1d(self.n_additional_features)
        else:
            self.bn_additional_feat = lambda x: x

        ##### CONTEXT ATTENTIVE NETWORK (CAN) #####
        self.n_context_feat = self.n_own_feat if self.use_context else 0
        if self.use_context and self.use_attention:
            self.q_encoder = nn.Linear(self.n_own_feat, self.hidden_dim)
            self.k_encoder = nn.Linear(self.n_context_feat, self.hidden_dim)
            
            self.attention_layer = nn.Linear(2*self.hidden_dim, 1)
            with torch.no_grad():
                    self.attention_layer.weight.fill_(0)
            
            self.context_encoder = nn.Sequential(
                nn.Linear(self.n_context_feat, self.n_context_feat),
                nn.BatchNorm1d(self.n_context_feat),
                nn.ReLU(),
            )
        
        ##### FC LAYERS #####
        self.n_total_feat = self.n_own_feat + self.n_context_feat
        self.decoder = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(self.n_total_feat, self.n_total_feat),
            nn.BatchNorm1d(self.n_total_feat),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.n_total_feat, self.n_classes),
        )

        print('Model Parameters:', count_parameters(self))

    def to_parallel(self):
        self.convnet = nn.DataParallel(self.convnet)

    def num_bboxes_per_img(self, bboxes_img_indices):
        assert bboxes_img_indices[0] == 0
        last_bbox_img_idx = 0
        num_bboxes = []
        counter = 0
        for bbox_img_idx in bboxes_img_indices:
            if bbox_img_idx == last_bbox_img_idx:
                counter += 1
            else:
                last_bbox_img_idx = bbox_img_idx
                num_bboxes.append(counter)
                counter = 1
        num_bboxes.append(counter)
        # print(num_bboxes)
        # print(bboxes_img_indices)
        assert len(num_bboxes) == bboxes_img_indices[-1] + 1
        return num_bboxes
    
    def forward(self, images, bboxes, additional_features, context_indices):
        """
        images: torch.Tensor of size [batch_size, 3, img_H, img_H]
        bboxes: torch.Tensor [total_n_bboxes_in_batch, 5]
            each of [batch_img_index, top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        additional_features: torch.Tensor [total_n_bboxes_in_batch, n_additional_features]
        context_indices: Torch.LongTensor [total_n_bboxes_in_batch, n_context]
            i.e. bbox indices (0-indexed) of contexts for all n bboxes. If not enough found, rest are -1
        
        Returns:
            prediction_scores: torch.Tensor of size [total_n_bboxes_in_batch, n_classes]
        """
        batch_size = bboxes.shape[0]

        ##### BBOX FEATURES #####
        if self.use_bbox_feat:
            bbox_features = bboxes[:, 1:].clone() # discard batch_img_index column
            bbox_features[:, 2:] -= bbox_features[:, :2] # convert to [top_left_x, top_left_y, width, height]
            
            bbox_asp_ratio = (bbox_features[:, 2]/bbox_features[:, 3]).view(batch_size, 1)
            bbox_features = torch.cat((bbox_features[:, 2:], bbox_asp_ratio), dim=1)
            
            bbox_features = self.bbox_feat_encoder(bbox_features)
        else:
            bbox_features = bboxes[:, :0] # size [n_bboxes, 0]
        
        ##### OWN VISUAL + BBOX FEATURES + ADDITIONAL FEATURES #####
        own_features = self.roi_pool(self.convnet(images), bboxes).view(batch_size, self.n_visual_feat)
        additional_features = self.bn_additional_feat(additional_features)
        own_features = torch.cat((own_features, bbox_features, additional_features), dim=1)

        ##### CONTEXT FEATURES USING SELF-ATTENTION #####
        if self.use_context:
            n_context = context_indices.shape[1]
            zero_feat = torch.zeros((1, self.n_own_feat)).to(images.device) # for -1 contexts i.e. extra padded
            own_feat_padded = torch.cat((own_features, zero_feat), dim=0)
            value = own_feat_padded[context_indices.view(-1)].view(batch_size, n_context, self.n_context_feat) # context_features

            if self.use_attention:
                query = self.q_encoder(own_features) # [batch_size, hidden_dim]

                attention_wts = []
                for c in range(n_context):
                    key = self.k_encoder(value[:, c, :]) # [batch_size, hidden_dim]
                    curr_attention_wt = self.attention_layer(torch.cat((query, key), dim=1)) # [batch_size, 1]
                    attention_wts.append(curr_attention_wt)
                attention_wts = torch.softmax(torch.cat(attention_wts, dim=1), dim=1) # [batch_size, n_context]
                
                context_representation = (attention_wts.unsqueeze(-1) * value).sum(1) # weighted avg of contexts [batch_size, n_context_feat]
                context_representation = self.context_encoder(context_representation) # [batch_size, n_context_feat]
            else: # average of context features for context representation [batch_size, n_context_feat]
                context_representation = value.sum(dim=1) / (context_indices != -1).sum(dim=1).view(batch_size, 1)
        else:
            context_representation = own_features[:, :0] # size [n_bboxes, 0]

        ##### FINAL FEATURE VECTOR #####
        combined_feat = torch.cat((own_features, context_representation), dim=1)
        output = self.decoder(combined_feat)

        return output

class RCNNGCN(nn.Module):
    def __init__(self, roi_output_size, img_H, n_classes, num_candidates, num_rels, device, backbone='resnet', use_context=True, use_attention=True, use_gru=True, hidden_dim=384, 
                 use_bbox_feat=True, bbox_hidden_dim=32, n_additional_features=0, trainable_convnet=True, drop_prob=0.2, class_names=None):
        """
        Implementation of our Visual Attention-based Model for Webpage Object Detection (VAMWOD)

        roi_output_size: Tuple (int, int) which will be output of the roi_pool layer for each channel of convnet_feature
        img_H: height of image given as input to the convnet. Image assumed to be of same W and H
        n_classes: num of classes for BBoxes
        backbone: string stating which convnet feature extractor to use. Allowed values: [alexnet, resnet (default)]
        use_context: if True, use context for context_representation along with own_features (default: True) 
        use_attention: if True, learn scores for all n_context contexts and take weighted avg for context_representation
            NOTE: this parameter is not used if use_context=False
        hidden_dim: size of hidden contextual representation, used when use_attention=True (default: 384)
        use_bbox_feat: if True, then use [x,y,w,h,asp_ratio] with convnet visual features (default: True)
        bbox_hidden_dim: size of hidden representation of 5 bbox features, used when use_bbox_feat=True (default: 32)
        n_additional_feat: num of additional features for each bbox to be used along with visual and bbox features
        trainable_convnet: if False then convnet weights will be freezed while training (default: True)
        drop_prob: dropout probability (default: 0.2)
        class_names: list of n_classes string elements containing names of the classes (default: [0, 1, ..., n_classes-1])
        """
        super(RCNNGCN, self).__init__()

        self.n_classes = n_classes
        self.use_context = use_context
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        self.use_bbox_feat = use_bbox_feat
        self.bbox_hidden_dim = bbox_hidden_dim
        self.n_additional_features = n_additional_features
        self.class_names = np.arange(self.n_classes).astype(str) if class_names is None else class_names
        self.use_gru = use_gru

        ##### REPRESENTATION NETWORK (RN) #####
        if backbone == 'resnet':
            self.convnet = torchvision.models.resnet18(pretrained=True)
            modules = list(self.convnet.children())[:-5] # remove last few layers!
        elif backbone == 'alexnet':
            self.convnet = torchvision.models.alexnet(pretrained=True)
            modules = list(self.convnet.features.children())[:7] # remove last few layers!

        self.convnet = nn.Sequential(*modules)
        if trainable_convnet == False:
            for p in self.convnet.parameters(): # freeze weights
                p.requires_grad = False

        _imgs = torch.autograd.Variable(torch.Tensor(1, 3, img_H, img_H))
        _conv_feat = self.convnet(_imgs)
        _convnet_output_size = _conv_feat.shape # [1, C, H, W]
        spatial_scale = _convnet_output_size[2]/img_H

        self.roi_pool = torchvision.ops.RoIPool(roi_output_size, spatial_scale)

        self.n_visual_feat = _convnet_output_size[1] * roi_output_size[0] * roi_output_size[1]
        self.n_bbox_feat = self.bbox_hidden_dim if self.use_bbox_feat else 0 # [x,y,w,h,asp_rat] of BBox encoded to n_bbox_feat
        self.n_own_feat = self.n_visual_feat + self.n_bbox_feat + self.n_additional_features

        if self.use_bbox_feat:
            self.bbox_feat_encoder = nn.Sequential(
                nn.Linear(3, self.n_bbox_feat),
                nn.BatchNorm1d(self.n_bbox_feat),
                nn.ReLU(),
            )
        
        if self.n_additional_features > 0:
            self.bn_additional_feat = nn.BatchNorm1d(self.n_additional_features)
        else:
            self.bn_additional_feat = lambda x: x
        
        ##### FC LAYERS #####
        self.n_total_feat0 = self.n_own_feat
        self.n_total_feat= self.hidden_dim
        self.decoder = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(self.n_total_feat0, self.n_total_feat),
            nn.BatchNorm1d(self.n_total_feat),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.n_total_feat, self.n_classes),
        )

        ##### GCN #####
        self.num_candidates = num_candidates
        self.gcn_output_dim = self.hidden_dim
        self.gcn_hidden_dim = self.gcn_output_dim
        self.gcn = RGCN(self.num_candidates * 3,
              self.gcn_hidden_dim,
              self.gcn_output_dim,
              num_rels, device=device, 
              num_hidden_layers=0, 
              embedding_provided=True, 
              embedding_dim=self.gcn_output_dim + 4 + self.n_bbox_feat, 
              use_edge_nn=False).to(device)

        self.intermediate_output_encoder = nn.Sequential (
            nn.Dropout(drop_prob),
            nn.Linear(self.n_total_feat0, self.gcn_output_dim)
        )

        self.gcn_plus_intermediate_output_decoder = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(self.gcn_output_dim * 2, 2 * self.gcn_output_dim),
            nn.BatchNorm1d(2 * self.gcn_output_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(2 * self.gcn_output_dim, self.n_classes),
        )

        self.gcn_output_decoder = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(self.gcn_output_dim, self.gcn_output_dim),
            nn.BatchNorm1d(self.gcn_output_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.gcn_output_dim, self.n_classes),
        )
        

        print('Model Parameters:', count_parameters(self))

    def to_parallel(self):
        self.convnet = nn.DataParallel(self.convnet)

    def num_bboxes_per_img(self, bboxes_img_indices):
        assert bboxes_img_indices[0] == 0
        last_bbox_img_idx = 0
        num_bboxes = []
        counter = 0
        for bbox_img_idx in bboxes_img_indices:
            if bbox_img_idx == last_bbox_img_idx:
                counter += 1
            else:
                last_bbox_img_idx = bbox_img_idx
                num_bboxes.append(counter)
                counter = 1
        num_bboxes.append(counter)
        # print(num_bboxes)
        # print(bboxes_img_indices)
        assert len(num_bboxes) == bboxes_img_indices[-1] + 1
        return num_bboxes

    def encode(self, images, bboxes, additional_features, context_indices):
        """
        images: torch.Tensor of size [batch_size, 3, img_H, img_H]
        bboxes: torch.Tensor [total_n_bboxes_in_batch, 5]
            each of [batch_img_index, top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        additional_features: torch.Tensor [total_n_bboxes_in_batch, n_additional_features]
        context_indices: Torch.LongTensor [total_n_bboxes_in_batch, n_context]
            i.e. bbox indices (0-indexed) of contexts for all n bboxes. If not enough found, rest are -1
        
        Returns:
            prediction_scores: torch.Tensor of size [total_n_bboxes_in_batch, n_classes]
        """
        batch_size = bboxes.shape[0]

        ##### BBOX FEATURES #####
        if self.use_bbox_feat:
            bbox_features = bboxes[:, 1:].clone() # discard batch_img_index column
            bbox_features[:, 2:] -= bbox_features[:, :2] # convert to [top_left_x, top_left_y, width, height]
            # bbox_features[:, 0] += np.random.uniform(1280) - 640
            # bbox_features[:, 1] += np.random.uniform(1280) - 640
            bbox_asp_ratio = (bbox_features[:, 2]/bbox_features[:, 3]).view(batch_size, 1)
            bbox_features = torch.cat((bbox_features[:, 2:], bbox_asp_ratio), dim=1)
            
            bbox_features = self.bbox_feat_encoder(bbox_features)
        else:
            bbox_features = bboxes[:, :0] # size [n_bboxes, 0]
        
        ##### OWN VISUAL + BBOX FEATURES + ADDITIONAL FEATURES #####
        own_features = self.roi_pool(self.convnet(images), bboxes).view(batch_size, self.n_visual_feat)
        additional_features = self.bn_additional_feat(additional_features)
        own_features = torch.cat((own_features, bbox_features, additional_features), dim=1)

        
        return own_features

    def semi_decode(self, encoded_features):
        # print(self.decoder[:4])
        return self.decoder[:4](encoded_features)

    def final_decode(self, semi_decoded_features):
        return self.decoder[4:](semi_decoded_features)

    def decode(self, encoded_features):
        return self.decoder(encoded_features)
        
    def forward(self, images, bboxes, additional_features, context_indices):
        """
        images: torch.Tensor of size [batch_size, 3, img_H, img_H]
        bboxes: torch.Tensor [total_n_bboxes_in_batch, 5]
            each of [batch_img_index, top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        additional_features: torch.Tensor [total_n_bboxes_in_batch, n_additional_features]
        context_indices: Torch.LongTensor [total_n_bboxes_in_batch, n_context]
            i.e. bbox indices (0-indexed) of contexts for all n bboxes. If not enough found, rest are -1
        
        Returns:
            prediction_scores: torch.Tensor of size [total_n_bboxes_in_batch, n_classes]
        """
        combined_feat = self.encode(images, bboxes, additional_features, context_indices)
        output = self.decoder(combined_feat)

        return output

    def gcn_forward(self, graph, encoded_features):
        gcn_output = self.gcn(graph)[:3*self.num_candidates]
        gcn_plus_intermediate = torch.cat((gcn_output, self.intermediate_output_encoder(encoded_features)), dim=1)
        return (self.gcn_output_decoder(gcn_output),
                self.gcn_plus_intermediate_output_decoder(gcn_plus_intermediate))

class VAMWODGCN(nn.Module):
    def __init__(self, roi_output_size, img_H, n_classes, num_candidates, num_rels, device, backbone='resnet', use_context=True, use_attention=True, use_gru=True, hidden_dim=384, 
                 use_bbox_feat=True, bbox_hidden_dim=32, n_additional_features=0, trainable_convnet=True, drop_prob=0.2, class_names=None):
        """
        Implementation of our Visual Attention-based Model for Webpage Object Detection (VAMWOD)

        roi_output_size: Tuple (int, int) which will be output of the roi_pool layer for each channel of convnet_feature
        img_H: height of image given as input to the convnet. Image assumed to be of same W and H
        n_classes: num of classes for BBoxes
        backbone: string stating which convnet feature extractor to use. Allowed values: [alexnet, resnet (default)]
        use_context: if True, use context for context_representation along with own_features (default: True) 
        use_attention: if True, learn scores for all n_context contexts and take weighted avg for context_representation
            NOTE: this parameter is not used if use_context=False
        hidden_dim: size of hidden contextual representation, used when use_attention=True (default: 384)
        use_bbox_feat: if True, then use [x,y,w,h,asp_ratio] with convnet visual features (default: True)
        bbox_hidden_dim: size of hidden representation of 5 bbox features, used when use_bbox_feat=True (default: 32)
        n_additional_feat: num of additional features for each bbox to be used along with visual and bbox features
        trainable_convnet: if False then convnet weights will be freezed while training (default: True)
        drop_prob: dropout probability (default: 0.2)
        class_names: list of n_classes string elements containing names of the classes (default: [0, 1, ..., n_classes-1])
        """
        super(VAMWODGCN, self).__init__()

        self.n_classes = n_classes
        self.use_context = use_context
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        self.use_bbox_feat = use_bbox_feat
        self.bbox_hidden_dim = bbox_hidden_dim
        self.n_additional_features = n_additional_features
        self.class_names = np.arange(self.n_classes).astype(str) if class_names is None else class_names
        self.use_gru = use_gru

        ##### REPRESENTATION NETWORK (RN) #####
        if backbone == 'resnet':
            self.convnet = torchvision.models.resnet18(pretrained=True)
            modules = list(self.convnet.children())[:-5] # remove last few layers!
        elif backbone == 'alexnet':
            self.convnet = torchvision.models.alexnet(pretrained=True)
            modules = list(self.convnet.features.children())[:7] # remove last few layers!

        self.convnet = nn.Sequential(*modules)
        if trainable_convnet == False:
            for p in self.convnet.parameters(): # freeze weights
                p.requires_grad = False

        _imgs = torch.autograd.Variable(torch.Tensor(1, 3, img_H, img_H))
        _conv_feat = self.convnet(_imgs)
        _convnet_output_size = _conv_feat.shape # [1, C, H, W]
        spatial_scale = _convnet_output_size[2]/img_H

        self.roi_pool = torchvision.ops.RoIPool(roi_output_size, spatial_scale)

        self.n_visual_feat = _convnet_output_size[1] * roi_output_size[0] * roi_output_size[1]
        self.n_bbox_feat = self.bbox_hidden_dim if self.use_bbox_feat else 0 # [x,y,w,h,asp_rat] of BBox encoded to n_bbox_feat
        self.n_own_feat = self.n_visual_feat + self.n_bbox_feat + self.n_additional_features

        if self.use_bbox_feat:
            self.bbox_feat_encoder = nn.Sequential(
                nn.Linear(3, self.n_bbox_feat),
                nn.BatchNorm1d(self.n_bbox_feat),
                nn.ReLU(),
            )
        
        if self.n_additional_features > 0:
            self.bn_additional_feat = nn.BatchNorm1d(self.n_additional_features)
        else:
            self.bn_additional_feat = lambda x: x

        ##### CONTEXT ATTENTIVE NETWORK (CAN) #####
        if not self.use_gru:
            self.n_context_feat = self.n_own_feat if self.use_context else 0
            if self.use_context and self.use_attention:
                self.q_encoder = nn.Linear(self.n_own_feat, self.hidden_dim)
                self.k_encoder = nn.Linear(self.n_context_feat, self.hidden_dim)
                
                self.attention_layer = nn.Linear(2*self.hidden_dim, 1)
                with torch.no_grad():
                        self.attention_layer.weight.fill_(0)
                
                self.context_encoder = nn.Sequential(
                    nn.Linear(self.n_context_feat, self.n_context_feat),
                    nn.BatchNorm1d(self.n_context_feat),
                    nn.ReLU(),
                )
        else:
            self.gru = nn.GRU(self.n_own_feat,
                                self.hidden_dim,
                                num_layers=1,
                                bidirectional=True,
                                batch_first=True, dropout = drop_prob)
            h0_init_nums = 2
            h0 = torch.zeros(h0_init_nums, 1, self.hidden_dim)
            nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
            self.h0 = nn.Parameter(h0, requires_grad=True)
        
        ##### FC LAYERS #####
        if not self.use_gru:
            self.n_total_feat0 = self.n_own_feat + self.n_context_feat
            self.n_total_feat= self.n_total_feat0
        else:
            self.n_total_feat0 = 2 * self.hidden_dim
            self.n_total_feat= self.hidden_dim
        self.decoder = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(self.n_total_feat0, self.n_total_feat),
            nn.BatchNorm1d(self.n_total_feat),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.n_total_feat, self.n_classes),
        )

        ##### GCN #####
        self.num_candidates = num_candidates
        self.gcn_output_dim = 200
        self.gcn_hidden_dim = self.gcn_output_dim
        self.gcn = RGCN(self.num_candidates * 3,
              self.gcn_hidden_dim,
              self.gcn_output_dim,
              num_rels, device=device, 
              num_hidden_layers=0, 
              embedding_provided=True, 
              embedding_dim=self.gcn_output_dim + 4 + self.n_bbox_feat, 
              use_edge_nn=False).to(device)

        self.intermediate_output_encoder = nn.Sequential (
            nn.Dropout(drop_prob),
            nn.Linear(self.n_total_feat0, self.gcn_output_dim)
        )

        self.gcn_plus_intermediate_output_decoder = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(self.gcn_output_dim * 2, 2 * self.gcn_output_dim),
            nn.BatchNorm1d(2 * self.gcn_output_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(2 * self.gcn_output_dim, self.n_classes),
        )

        self.gcn_output_decoder = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(self.gcn_output_dim, self.gcn_output_dim),
            nn.BatchNorm1d(self.gcn_output_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.gcn_output_dim, self.n_classes),
        )
        

        print('Model Parameters:', count_parameters(self))

    def to_parallel(self):
        self.convnet = nn.DataParallel(self.convnet)

    def num_bboxes_per_img(self, bboxes_img_indices):
        assert bboxes_img_indices[0] == 0
        last_bbox_img_idx = 0
        num_bboxes = []
        counter = 0
        for bbox_img_idx in bboxes_img_indices:
            if bbox_img_idx == last_bbox_img_idx:
                counter += 1
            else:
                last_bbox_img_idx = bbox_img_idx
                num_bboxes.append(counter)
                counter = 1
        num_bboxes.append(counter)
        # print(num_bboxes)
        # print(bboxes_img_indices)
        assert len(num_bboxes) == bboxes_img_indices[-1] + 1
        return num_bboxes

    def encode(self, images, bboxes, additional_features, context_indices):
        """
        images: torch.Tensor of size [batch_size, 3, img_H, img_H]
        bboxes: torch.Tensor [total_n_bboxes_in_batch, 5]
            each of [batch_img_index, top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        additional_features: torch.Tensor [total_n_bboxes_in_batch, n_additional_features]
        context_indices: Torch.LongTensor [total_n_bboxes_in_batch, n_context]
            i.e. bbox indices (0-indexed) of contexts for all n bboxes. If not enough found, rest are -1
        
        Returns:
            prediction_scores: torch.Tensor of size [total_n_bboxes_in_batch, n_classes]
        """
        batch_size = bboxes.shape[0]

        ##### BBOX FEATURES #####
        if self.use_bbox_feat:
            bbox_features = bboxes[:, 1:].clone() # discard batch_img_index column
            bbox_features[:, 2:] -= bbox_features[:, :2] # convert to [top_left_x, top_left_y, width, height]
            # bbox_features[:, 0] += np.random.uniform(1280) - 640
            # bbox_features[:, 1] += np.random.uniform(1280) - 640
            bbox_asp_ratio = (bbox_features[:, 2]/bbox_features[:, 3]).view(batch_size, 1)
            bbox_features = torch.cat((bbox_features[:, 2:], bbox_asp_ratio), dim=1)
            
            bbox_features = self.bbox_feat_encoder(bbox_features)
        else:
            bbox_features = bboxes[:, :0] # size [n_bboxes, 0]
        
        ##### OWN VISUAL + BBOX FEATURES + ADDITIONAL FEATURES #####
        own_features = self.roi_pool(self.convnet(images), bboxes).view(batch_size, self.n_visual_feat)
        additional_features = self.bn_additional_feat(additional_features)
        own_features = torch.cat((own_features, bbox_features, additional_features), dim=1)

        if not self.use_gru:
            ##### CONTEXT FEATURES USING SELF-ATTENTION #####
            if self.use_context:
                n_context = context_indices.shape[1]
                zero_feat = torch.zeros((1, self.n_own_feat)).to(images.device) # for -1 contexts i.e. extra padded
                own_feat_padded = torch.cat((own_features, zero_feat), dim=0)
                value = own_feat_padded[context_indices.view(-1)].view(batch_size, n_context, self.n_context_feat) # context_features

                if self.use_attention:
                    query = self.q_encoder(own_features) # [batch_size, hidden_dim]

                    attention_wts = []
                    for c in range(n_context):
                        key = self.k_encoder(value[:, c, :]) # [batch_size, hidden_dim]
                        curr_attention_wt = self.attention_layer(torch.cat((query, key), dim=1)) # [batch_size, 1]
                        attention_wts.append(curr_attention_wt)
                    attention_wts = torch.softmax(torch.cat(attention_wts, dim=1), dim=1) # [batch_size, n_context]
                    
                    context_representation = (attention_wts.unsqueeze(-1) * value).sum(1) # weighted avg of contexts [batch_size, n_context_feat]
                    context_representation = self.context_encoder(context_representation) # [batch_size, n_context_feat]
                else: # average of context features for context representation [batch_size, n_context_feat]
                    context_representation = value.sum(dim=1) / (context_indices != -1).sum(dim=1).view(batch_size, 1)
            else:
                context_representation = own_features[:, :0] # size [n_bboxes, 0]
                ##### FINAL FEATURE VECTOR #####
            combined_feat = torch.cat((own_features, context_representation), dim=1)
            return combined_feat

        batch_size = bboxes.shape[0]
        num_bboxes = self.num_bboxes_per_img(bboxes[:, 0])
        curr_idx = 0
        own_features_partitioned = []
        max_num_bbox = np.max(num_bboxes)
        for num_bbox in num_bboxes:
            padded_own_features = torch.cat((
                own_features[curr_idx:curr_idx + num_bbox], 
                torch.zeros(max_num_bbox - num_bbox, own_features.shape[1]).to(own_features.device)), dim=0)
            own_features_partitioned.append(padded_own_features)
            curr_idx += num_bbox
        own_features_rebatched = torch.stack(own_features_partitioned)

        packed_own_features = pack_padded_sequence(own_features_rebatched, num_bboxes, batch_first=True, enforce_sorted=False)
        output, _ = self.gru(packed_own_features, self.h0.repeat(1, batch_size, 1))

        
        output_unpacked, output_lengths = pad_packed_sequence(output, batch_first=True) 
        output_feature_partitioned = []
        for output_unpacked_for_an_img, num_bbox in zip(output_unpacked, num_bboxes):
            output_feature_partitioned.append(output_unpacked_for_an_img[:num_bbox])
        out = torch.cat(output_feature_partitioned, dim=0)

        return out

    def semi_decode(self, encoded_features):
        # print(self.decoder[:4])
        return self.decoder[:4](encoded_features)

    def final_decode(self, semi_decoded_features):
        return self.decoder[4:](semi_decoded_features)

    def decode(self, encoded_features):
        return self.decoder(encoded_features)
        
    def forward(self, images, bboxes, additional_features, context_indices):
        """
        images: torch.Tensor of size [batch_size, 3, img_H, img_H]
        bboxes: torch.Tensor [total_n_bboxes_in_batch, 5]
            each of [batch_img_index, top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        additional_features: torch.Tensor [total_n_bboxes_in_batch, n_additional_features]
        context_indices: Torch.LongTensor [total_n_bboxes_in_batch, n_context]
            i.e. bbox indices (0-indexed) of contexts for all n bboxes. If not enough found, rest are -1
        
        Returns:
            prediction_scores: torch.Tensor of size [total_n_bboxes_in_batch, n_classes]
        """
        combined_feat = self.encode(images, bboxes, additional_features, context_indices)
        output = self.decoder(combined_feat)

        return output

    def gcn_forward(self, graph, encoded_features):
        gcn_output = self.gcn(graph)[:3*self.num_candidates]
        gcn_plus_intermediate = torch.cat((gcn_output, self.intermediate_output_encoder(encoded_features)), dim=1)
        return (self.gcn_output_decoder(gcn_output),
                self.gcn_plus_intermediate_output_decoder(gcn_plus_intermediate))


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False, use_provided_embedding=False, use_edge_nn=False, drop_prob=0.2):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer
        self.use_provided_embedding = use_provided_embedding
        self.use_edge_nn = use_edge_nn
        self.dropout = nn.Dropout(drop_prob)

        # if self.is_input_layer:
        #     self.batchnorm = nn.BatchNorm1d(self.in_feat)

        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # weight bases in equation (3)
        if not use_edge_nn:
            self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))

            if self.num_bases < self.num_rels:
            # linear combination coefficients in equation (3)
                self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

            # add bias
            if bias:
                self.bias = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat))
        else:
            self.weight1 = nn.Parameter(torch.Tensor(self.num_rels, self.in_feat,
                                                self.out_feat))
            self.weight2 = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat,
                                                self.out_feat))
            self.bias1 = nn.Parameter(torch.Tensor(self.num_rels, out_feat))
            self.bias2 = nn.Parameter(torch.Tensor(self.num_rels, out_feat))

        # init trainable parameters
        if not use_edge_nn:
            nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
            # nn.init.xavier_uniform_(self.attn_fc,
            #                         gain=nn.init.calculate_gain('leaky_relu'))
            if self.num_bases < self.num_rels:
                nn.init.xavier_uniform_(self.w_comp,
                                        gain=nn.init.calculate_gain('relu'))
            if bias:
                nn.init.xavier_uniform_(self.bias,
                                        gain=nn.init.calculate_gain('relu'))
        else:
            nn.init.xavier_uniform_(self.weight1,
                                gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.weight2,
                                gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.bias1,
                                gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.bias2,
                                gain=nn.init.calculate_gain('relu'))
        # self.batchnorm = nn.BatchNorm1d(in_feat)
        # if self.is_input_layer:
        #     self.batchnorm = nn.BatchNorm1d(self.in_feat)

    def message_func(self, edges):
        if self.use_edge_nn:
            h = edges.src['h']
            # h = self.batchnorm(h)
            
            w1 = self.weight1[edges.data['rel_type']]
            out1 = torch.bmm(h.unsqueeze(1), w1).squeeze()
            b1 = self.bias1[edges.data['rel_type']]
            out1 += b1
            if self.activation:
                h1 = self.activation(out1)
            else:
                h1 = out1
            h1 = self.dropout(h1)

            w2 = self.weight2[edges.data['rel_type']]
            out2 = torch.bmm(h1.unsqueeze(1), w2).squeeze()
            b2 = self.bias2[edges.data['rel_type']]
            out2 += b2

            msg = out2 * edges.data['norm']
            return {'msg': msg}
        else:
            w = self.weight[edges.data['rel_type']]
            h = edges.src['h'].unsqueeze(1)
            msg = torch.bmm(h, w).squeeze()
            msg = msg * edges.data['norm']
            return {'msg': msg}

    def forward(self, g):
        if not self.use_edge_nn:
            if self.num_bases < self.num_rels:
                # generate all weights from bases (equation (3))
                weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
                weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                            self.in_feat, self.out_feat)
            else:
                weight = self.weight

        if self.is_input_layer:
            def message_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                if self.use_provided_embedding:
                    return self.message_func(edges)

                embed = weight.view(-1, self.out_feat)
                index = edges.data['rel_type'] * self.in_feat + edges.src['id']
                return {'msg': embed[index] * edges.data['norm']}
        else:
            message_func = self.message_func

        def apply_func(nodes):
            h = nodes.data['h']
            # print(h)
            # if self.bias:
            #     h = h + self.bias
            if self.activation and not self.use_edge_nn:
                # Output Layer has no activation and thus is not dropouted
                h = self.activation(h)
                h = self.dropout(h)
            return {'h': h}

        # print('Before g.update_all')
        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)

class RGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, embedding_provided=False, embedding_dim = 0, device='cpu',
                 num_bases=-1, num_hidden_layers=1, use_edge_nn=False, drop_prob=0.2):
        super(RGCN, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_provided = embedding_provided
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.device = device
        self.use_edge_nn = use_edge_nn
        self.drop_prob = drop_prob

        # create rgcn layers
        self.build_model()
        if self.num_hidden_layers == -1:
            self.fc = nn.Linear(self.h_dim, self.out_dim)

        # create initial features
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        if self.num_hidden_layers == -1:
            self.layers.append(
                RGCNLayer(self.embedding_dim, self.h_dim, self.num_rels, self.num_bases, bias=True,
                         activation=None, is_input_layer=True, use_provided_embedding=True, use_edge_nn=self.use_edge_nn, drop_prob=self.drop_prob))
            return  
        # input to hidden
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer()
        self.layers.append(h2o)

        # self.fn = nn.Linear(h_dim, out_dim)

    # initialize feature for each node
    def create_features(self):
        features = torch.arange(self.num_nodes).to(self.device)
        return features

    def build_input_layer(self):
        if self.embedding_provided:
            return  RGCNLayer(self.embedding_dim, self.h_dim, self.num_rels, self.num_bases, bias=True,
                         activation=F.relu, is_input_layer=True, use_provided_embedding=True, use_edge_nn=self.use_edge_nn, drop_prob=self.drop_prob)
        return RGCNLayer(self.num_nodes, self.h_dim, self.num_rels, self.num_bases, bias=True,
                         activation=F.relu, is_input_layer=True, use_edge_nn=self.use_edge_nn, drop_prob=self.drop_prob)

    def build_hidden_layer(self):
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases, bias=True,
                         activation=F.relu, use_edge_nn=self.use_edge_nn, drop_prob=self.drop_prob)

    def build_output_layer(self):
        return RGCNLayer(self.h_dim, self.out_dim, self.num_rels, self.num_bases, bias=True,
                         activation=None, use_edge_nn=self.use_edge_nn, drop_prob=self.drop_prob)

    def forward(self, g):
        # print('Enter forward')
        # if self.features is not None:
        #     g.ndata['id'] = self.features
        # print('Finished features')
        for layer in self.layers:
            # print('Layer Forward')
            layer(g)
        if self.num_hidden_layers == -1:
            return self.fc(F.relu(g.ndata['h']))
        return g.ndata['h']

class TreeNet(nn.Module):
    def __init__(self, roi_output_size, img_H, n_classes, device, backbone='resnet', hidden_dim=384, 
                 use_bbox_feat=True, bbox_hidden_dim=32, n_additional_features=0, trainable_convnet=True, drop_prob=0.2, class_names=None):
        super(TreeNet, self).__init__()

        self.n_classes = n_classes
        self.device = device
        self.hidden_dim = hidden_dim
        self.use_bbox_feat = use_bbox_feat
        self.bbox_hidden_dim = bbox_hidden_dim
        self.n_additional_features = n_additional_features
        self.class_names = np.arange(self.n_classes).astype(str) if class_names is None else class_names

        ##### REPRESENTATION NETWORK (RN) #####
        if backbone == 'resnet':
            self.convnet = torchvision.models.resnet18(pretrained=True)
            modules = list(self.convnet.children())[:-5] # remove last few layers!
        elif backbone == 'alexnet':
            self.convnet = torchvision.models.alexnet(pretrained=True)
            modules = list(self.convnet.features.children())[:7] # remove last few layers!

        self.convnet = nn.Sequential(*modules)
        if trainable_convnet == False:
            for p in self.convnet.parameters(): # freeze weights
                p.requires_grad = False

        _imgs = torch.autograd.Variable(torch.Tensor(1, 3, img_H, img_H))
        _conv_feat = self.convnet(_imgs)
        _convnet_output_size = _conv_feat.shape # [1, C, H, W]
        spatial_scale = _convnet_output_size[2]/img_H

        self.roi_pool = torchvision.ops.RoIPool(roi_output_size, spatial_scale)

        self.n_visual_feat = _convnet_output_size[1] * roi_output_size[0] * roi_output_size[1]
        self.n_bbox_feat = self.bbox_hidden_dim if self.use_bbox_feat else 0 # [x,y,w,h,asp_rat] of BBox encoded to n_bbox_feat
        self.n_own_feat = self.n_visual_feat + self.n_bbox_feat + self.n_additional_features

        if self.use_bbox_feat:
            self.bbox_feat_encoder = nn.Sequential(
                nn.Linear(3, self.n_bbox_feat),
                nn.BatchNorm1d(self.n_bbox_feat),
                nn.ReLU(),
            )
        
        if self.n_additional_features > 0:
            self.bn_additional_feat = nn.BatchNorm1d(self.n_additional_features)
        else:
            self.bn_additional_feat = lambda x: x

        ##### Tree Information Aggregation Network #####
        # self.lstm = ChildSumTreeLSTM(self.n_own_feat, self.n_own_feat)
        self.lstm = PassDownTreeLSTM(self.n_own_feat, self.n_own_feat)

        self.gru = nn.GRU(self.n_own_feat,
                            self.hidden_dim,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True, dropout = drop_prob)
        h0_init_nums = 2
        h0 = torch.zeros(h0_init_nums, 1, self.hidden_dim)
        nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
        self.h0 = nn.Parameter(h0, requires_grad=True)

        self.output_dim = 2 * self.hidden_dim 
        
        ##### FC LAYERS #####
        # self.n_total_feat = self.n_own_feat
        self.decoder = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(self.output_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.output_dim, self.n_classes),
        )

        print('Model Parameters:', count_parameters(self))

    def to_parallel(self):
        self.convnet = nn.DataParallel(self.convnet)

    def num_bboxes_per_img(self, bboxes_img_indices):
        assert bboxes_img_indices[0] == 0
        last_bbox_img_idx = 0
        num_bboxes = []
        counter = 0
        for bbox_img_idx in bboxes_img_indices:
            if bbox_img_idx == last_bbox_img_idx:
                counter += 1
            else:
                last_bbox_img_idx = bbox_img_idx
                num_bboxes.append(counter)
                counter = 1
        num_bboxes.append(counter)
        # print(num_bboxes)
        # print(bboxes_img_indices)
        assert len(num_bboxes) == bboxes_img_indices[-1] + 1
        return num_bboxes

    def extract_tree_states(self, tree, shape):
        # outputs = torch.zeros(shape).to(self.device)
        # stack = [tree]
        # while len(stack) > 0:
        #     curr = stack.pop()
        #     stack += curr.children
        #     outputs[curr.idx] = curr.state[0]
        # return outputs
        outputs = torch.zeros(shape).to(self.device)
        stack = [tree]
        max_idx = -1
        while len(stack) > 0:
            curr = stack.pop()
            stack += curr.children
            if curr.num_children == 0:
                outputs[curr.idx] = curr.state[0]
                if curr.idx > max_idx:
                    max_idx = curr.idx
        # print(max_idx + 1)
        return outputs[:max_idx + 1], max_idx+1

    def forward(self, images, bboxes, additional_features, trees):
        """
        images: torch.Tensor of size [batch_size, 3, img_H, img_H]
        bboxes: torch.Tensor [total_n_bboxes_in_batch, 5]
            each of [batch_img_index, top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        additional_features: torch.Tensor [total_n_bboxes_in_batch, n_additional_features]
        
        Returns:
            prediction_scores: torch.Tensor of size [total_n_bboxes_in_batch, n_classes]
        """
        batch_size = bboxes.shape[0]

        ##### BBOX FEATURES #####
        if self.use_bbox_feat:
            bbox_features = bboxes[:, 1:].clone() # discard batch_img_index column
            bbox_features[:, 2:] -= bbox_features[:, :2] # convert to [top_left_x, top_left_y, width, height]
            
            bbox_asp_ratio = (bbox_features[:, 2]/bbox_features[:, 3]).view(batch_size, 1)
            bbox_features = torch.cat((bbox_features[:, 2:], bbox_asp_ratio), dim=1)
            
            bbox_features = self.bbox_feat_encoder(bbox_features)
        else:
            bbox_features = bboxes[:, :0] # size [n_bboxes, 0]
        
        ##### OWN VISUAL + BBOX FEATURES + ADDITIONAL FEATURES #####
        own_features = self.roi_pool(self.convnet(images), bboxes).view(batch_size, self.n_visual_feat)
        additional_features = self.bn_additional_feat(additional_features)
        own_features = torch.cat((own_features, bbox_features, additional_features), dim=1)

        ##### TREE LSTM #####
        num_bboxes = self.num_bboxes_per_img(bboxes[:, 0])
        curr_idx = 0
        tree_features = []
        leaf_nums = []
        assert len(num_bboxes) == len(trees)
        for i, num_bbox in enumerate(num_bboxes):
            inputs = own_features[curr_idx:curr_idx + num_bbox]
            
            self.lstm(trees[i], inputs)
            # print(tree_features)
            tree_feature, leaf_num = self.extract_tree_states(trees[i], (num_bbox, self.n_own_feat))
            tree_features.append(tree_feature)
            leaf_nums.append(leaf_num)
            # tree_features.append(torch.zeros((num_bbox, self.n_own_feat)).to(self.device))
            curr_idx += num_bbox

        max_num_leafs = np.max(leaf_nums)
        tree_features_partitioned = []
        for tree_feature in tree_features:
            padded_tree_features = torch.cat((
                tree_feature, 
                torch.zeros(max_num_leafs - tree_feature.shape[0], tree_feature.shape[1]).to(own_features.device)), dim=0)
            tree_features_partitioned.append(padded_tree_features)
        tree_features_rebatched = torch.stack(tree_features_partitioned)
        packed_tree_features = pack_padded_sequence(tree_features_rebatched, leaf_nums, batch_first=True, enforce_sorted=False)
        output, _ = self.gru(packed_tree_features, self.h0.repeat(1, batch_size, 1))
        output_unpacked, output_lengths = pad_packed_sequence(output, batch_first=True)

        output_feature_partitioned = []
        for output_unpacked_for_an_img, leaf_num in zip(output_unpacked, leaf_nums):
            output_feature_partitioned.append(output_unpacked_for_an_img[:leaf_num])
        out = torch.cat(output_feature_partitioned, dim=0)
        # print(tree_features.shape)

        ##### FINAL FEATURE VECTOR #####
        output = self.decoder(out)

        return output

class PassDownTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(PassDownTreeLSTM, self).__init__()
        self.n_bbox_feat = 32
        self.rel_bbox_encoder = nn.Sequential(
            nn.BatchNorm1d(9),
            nn.Linear(9, self.n_bbox_feat),
            nn.BatchNorm1d(self.n_bbox_feat),
            nn.ReLU(),
        )

        self.gru = nn.GRUCell(in_dim + self.n_bbox_feat, mem_dim)
        self.in_dim = in_dim + self.n_bbox_feat
        self.mem_dim = mem_dim

        # self.horizontal_gru = nn.GRU(self.mem_dim, # self.n_own_feat,
        #                     self.mem_dim,
        #                     num_layers=1,
        #                     bidirectional=True,
        #                     batch_first=True)
        # self.compressor = nn.Linear(2*self.mem_dim, self.mem_dim)
        # hh0 = torch.zeros(2, 1, self.mem_dim)
        # nn.init.xavier_normal_(hh0, gain=nn.init.calculate_gain('relu'))
        # self.hh0 = nn.Parameter(hh0, requires_grad=True)

        h0 = torch.zeros(1, 1, self.mem_dim)
        nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
        self.h0 = nn.Parameter(h0, requires_grad=True)
        # self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        # self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        # self.fx = nn.Linear(self.in_dim, self.mem_dim)
        # self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        # self.i2h = nn.Linear(in_dim + mem_dim, mem_dim)
        # self.i2o = nn.Linear(in_dim + mem_dim, in_dim)

    # def node_forward(self, inputs, parent_c, parent_h):

        # iou = self.ioux(inputs) + self.iouh(parent_h)
        # i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        # i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        # f = F.sigmoid(
        #     self.fh(parent_h) +
        #     self.fx(inputs).repeat(1, 1)
        # )
        # fc = torch.mul(f, parent_c)

        # c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        # h = torch.mul(o, F.tanh(c))
        # return c, h
        

    def node_forward(self, inputs, parent_h):
        # print(inputs.shape)
        # print(parent_h.shape)
        # h = self.gru(inputs.view(1, -1), parent_h.view(1, -1))
        # print(c.shape)
        # print(h.shape)
        # return h.view(-1)
        return self.gru(inputs, parent_h)

    def create_rel_bboxes_info(self, rel_bboxes):
        self.relative_bbox_info = self.rel_bbox_encoder(rel_bboxes)

    def forward(self, tree, inputs):

        curr_level = [tree]
        while len(curr_level) > 0:
            next_level = []
            parent_hs = torch.stack(
                [self.h0.view(-1) if curr.parent is None else curr.parent.state for curr in curr_level])
            # print([(len(inputs), curr.idx) for curr in curr_level])
            inputs_batch = torch.stack(
                [torch.cat((inputs[curr.idx], self.relative_bbox_info[curr.idx])) for curr in curr_level])
            curr_level_states = self.node_forward(inputs_batch, parent_hs)
            assert curr_level_states.shape[0] == len(curr_level)
            for curr, curr_state in zip(curr_level, curr_level_states):
                curr.state = curr_state

            for curr in curr_level:
                # curr.state = self.node_forward(
                #     torch.cat((inputs[curr.idx], self.relative_bbox_info[curr.idx])), parent_h)

                for idx in range(curr.num_children):
                    assert curr.children[idx].parent == curr # Sanity check
                    next_level.append(curr.children[idx])
            # print([curr.state for curr in curr_level])
            # output, _ = self.horizontal_gru(torch.cat([curr.state for curr in curr_level]).view(1, len(curr_level), self.mem_dim), self.hh0)
            # output = self.compressor(output[0])
            # assert output.shape[0] == len(curr_level)
            # for state, curr in zip(output, curr_level):
            #     curr.state = state
            curr_level = next_level


# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        # self.i2h = nn.Linear(in_dim + mem_dim, mem_dim)
        # self.i2o = nn.Linear(in_dim + mem_dim, in_dim)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

        # print(inputs.shape, child_h_sum.shape)
        # combined = torch.cat((inputs.view(1, -1), child_h_sum), dim=1)
        # hidden = self.i2h(combined)
        # output = self.i2o(combined)
        # return output, hidden




    def forward(self, tree, inputs):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        if tree.num_children == 0:
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        return tree.state

class TreeGCN(nn.Module):
    def __init__(self, roi_output_size, img_H, n_classes, num_candidates, 
                        num_rels, device, backbone='resnet', hidden_dim=300, 
                 use_bbox_feat=True, bbox_hidden_dim=32, n_additional_features=0, trainable_convnet=True, drop_prob=0.2, class_names=None):
        super(TreeGCN, self).__init__()

        self.n_classes = n_classes
        self.device = device
        self.hidden_dim = hidden_dim
        self.use_bbox_feat = use_bbox_feat
        self.bbox_hidden_dim = bbox_hidden_dim
        self.n_additional_features = n_additional_features
        self.class_names = np.arange(self.n_classes).astype(str) if class_names is None else class_names
        self.num_candidates = num_candidates
        self.num_rels = num_rels

        ##### REPRESENTATION NETWORK (RN) #####
        if backbone == 'resnet':
            self.convnet = torchvision.models.resnet18(pretrained=True)
            modules = list(self.convnet.children())[:-5] # remove last few layers!
        elif backbone == 'alexnet':
            self.convnet = torchvision.models.alexnet(pretrained=True)
            modules = list(self.convnet.features.children())[:7] # remove last few layers!

        self.convnet = nn.Sequential(*modules)
        if trainable_convnet == False:
            for p in self.convnet.parameters(): # freeze weights
                p.requires_grad = False

        _imgs = torch.autograd.Variable(torch.Tensor(1, 3, img_H, img_H))
        _conv_feat = self.convnet(_imgs)
        _convnet_output_size = _conv_feat.shape # [1, C, H, W]
        spatial_scale = _convnet_output_size[2]/img_H

        self.roi_pool = torchvision.ops.RoIPool(roi_output_size, spatial_scale)

        self.n_visual_feat = _convnet_output_size[1] * roi_output_size[0] * roi_output_size[1]
        self.n_bbox_feat = self.bbox_hidden_dim if self.use_bbox_feat else 0 # [x,y,w,h,asp_rat] of BBox encoded to n_bbox_feat
        self.n_own_feat = self.n_visual_feat + self.n_bbox_feat + self.n_additional_features

        if self.use_bbox_feat:
            self.bbox_feat_encoder = nn.Sequential(
                nn.BatchNorm1d(3),
                nn.Linear(3, self.n_bbox_feat),
                nn.BatchNorm1d(self.n_bbox_feat),
                nn.ReLU(),
            )
        
        if self.n_additional_features > 0:
            self.bn_additional_feat = nn.BatchNorm1d(self.n_additional_features)
        else:
            self.bn_additional_feat = lambda x: x

        ##### Tree Information Aggregation Network #####
        # self.lstm = ChildSumTreeLSTM(self.n_own_feat, self.n_own_feat)
        self.tree_rnn = PassDownTreeLSTM(self.n_own_feat, self.hidden_dim)

        #self.tree_visual_feature_merger = nn.Linear(self.n_own_feat*2, self.n_own_feat)
        self.leaf_rnn = nn.GRU(self.hidden_dim, # self.n_own_feat,
                            self.hidden_dim,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True, dropout = drop_prob)
        h0_init_nums = 2
        h0 = torch.zeros(h0_init_nums, 1, self.hidden_dim)
        nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
        self.h0 = nn.Parameter(h0, requires_grad=True)

        self.decoder_hidden_dim = 300
        
        ##### FC LAYERS #####
        # self.n_total_feat = self.n_own_feat
        self.decoder = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(2 * self.hidden_dim, self.decoder_hidden_dim),
            nn.BatchNorm1d(self.decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.decoder_hidden_dim, self.n_classes),
        )

        # self.tree_decoder = nn.Sequential(
        #     nn.Dropout(drop_prob),
        #     nn.Linear(3 * self.hidden_dim, self.decoder_hidden_dim),
        #     nn.BatchNorm1d(self.decoder_hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(drop_prob),
        #     nn.Linear(self.decoder_hidden_dim, self.n_classes),
        # )

        ##### GCN #####
        self.num_candidates = num_candidates
        self.gcn_output_dim = 300
        self.gcn_hidden_dim = self.gcn_output_dim
        self.gcn = RGCN(self.num_candidates * 3,
              self.gcn_hidden_dim,
              self.gcn_output_dim,
              num_rels, device=device, 
              num_hidden_layers=0, 
              embedding_provided=True, 
              embedding_dim=self.gcn_output_dim + 4 + self.n_bbox_feat, 
              use_edge_nn=False).to(device)

        self.intermediate_output_encoder = nn.Sequential (
            nn.Dropout(drop_prob),
            # nn.Linear(3*self.hidden_dim, self.gcn_output_dim)
            nn.Linear(2*self.hidden_dim, self.gcn_output_dim)
        )

        self.gcn_plus_intermediate_output_decoder = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(self.gcn_output_dim * 2, self.gcn_output_dim * 2),
            nn.BatchNorm1d(self.gcn_output_dim * 2),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.gcn_output_dim * 2, self.n_classes),
        )

        self.gcn_output_decoder = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(self.gcn_output_dim, self.gcn_output_dim),
            nn.BatchNorm1d(self.gcn_output_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.gcn_output_dim, self.n_classes),
        )

        print('Model Parameters:', count_parameters(self))

    def to_parallel(self):
        self.convnet = nn.DataParallel(self.convnet)

    def num_bboxes_per_img(self, bboxes_img_indices):
        assert bboxes_img_indices[0] == 0
        last_bbox_img_idx = 0
        num_bboxes = []
        counter = 0
        for bbox_img_idx in bboxes_img_indices:
            if bbox_img_idx == last_bbox_img_idx:
                counter += 1
            else:
                last_bbox_img_idx = bbox_img_idx
                num_bboxes.append(counter)
                counter = 1
        num_bboxes.append(counter)
        # print(num_bboxes)
        # print(bboxes_img_indices)
        assert len(num_bboxes) == bboxes_img_indices[-1] + 1
        return num_bboxes

    def extract_tree_states(self, tree, shape):
        # outputs = torch.zeros(shape).to(self.device)
        # stack = [tree]
        # while len(stack) > 0:
        #     curr = stack.pop()
        #     stack += curr.children
        #     outputs[curr.idx] = curr.state[0]
        # return outputs
        outputs = [[]] * shape[0]
        stack = [tree]
        max_idx = -1
        while len(stack) > 0:
            curr = stack.pop()
            stack += curr.children
            if curr.num_children == 0:
                outputs[curr.idx] = curr.state
                if curr.idx > max_idx:
                    max_idx = curr.idx
        # print(max_idx + 1)
        # print(outputs[0].shape)
        return torch.stack(outputs[:max_idx + 1]), max_idx+1

    def encode(self, images, bboxes, additional_features, trees, rel_bboxes):
        """
        images: torch.Tensor of size [batch_size, 3, img_H, img_H]
        bboxes: torch.Tensor [total_n_bboxes_in_batch, 5]
            each of [batch_img_index, top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        additional_features: torch.Tensor [total_n_bboxes_in_batch, n_additional_features]
        
        Returns:
            prediction_scores: torch.Tensor of size [total_n_bboxes_in_batch, n_classes]
        """
        batch_size = bboxes.shape[0]

        ##### BBOX FEATURES #####
        if self.use_bbox_feat:
            bbox_features = bboxes[:, 1:].clone() # discard batch_img_index column
            bbox_features[:, 2:] -= bbox_features[:, :2] # convert to [top_left_x, top_left_y, width, height]
            
            bbox_asp_ratio = (bbox_features[:, 2]/bbox_features[:, 3]).view(batch_size, 1)
            bbox_features = torch.cat((bbox_features[:, 2:], bbox_asp_ratio), dim=1)
            
            bbox_features = self.bbox_feat_encoder(bbox_features)
        else:
            bbox_features = bboxes[:, :0] # size [n_bboxes, 0]
        
        ##### OWN VISUAL + BBOX FEATURES + ADDITIONAL FEATURES #####
        own_features = self.roi_pool(self.convnet(images), bboxes).view(batch_size, self.n_visual_feat)
        additional_features = self.bn_additional_feat(additional_features)
        own_features = torch.cat((own_features, bbox_features, additional_features), dim=1)

        ##### TREE GRU #####
        num_bboxes = self.num_bboxes_per_img(bboxes[:, 0])
        curr_idx = 0
        tree_features = []
        leaf_nums = []
        assert len(num_bboxes) == len(trees)
        
        for i, num_bbox in enumerate(num_bboxes):
            inputs = own_features[curr_idx:curr_idx + num_bbox]
            rel_bbox_segment = rel_bboxes[curr_idx:curr_idx + num_bbox]
            
            self.tree_rnn.create_rel_bboxes_info(rel_bbox_segment)
            self.tree_rnn(trees[i], inputs)
            # print(tree_features)
            tree_feature, leaf_num = self.extract_tree_states(trees[i], (num_bbox, self.n_own_feat))
            tree_features.append(tree_feature)
            leaf_nums.append(leaf_num)
            # tree_features.append(torch.zeros((num_bbox, self.n_own_feat)).to(self.device))

            curr_idx += num_bbox

        max_num_leafs = np.max(leaf_nums)
        # own_features_partitioned = []
        # curr_idx = 0
        # for i, (num_bbox, leaf_num) in enumerate(zip(num_bboxes, leaf_nums)):
        #     inputs = own_features[curr_idx:curr_idx + num_bbox]
        #     padded_own_features = torch.cat((
        #         inputs[:leaf_num], 
        #         torch.zeros(max_num_leafs - leaf_num, self.n_own_feat).to(own_features.device)), dim=0)
        #     own_features_partitioned.append(padded_own_features)
        #     curr_idx += num_bbox
        # own_features_rebatched = torch.stack(own_features_partitioned)
        # packed_own_features = pack_padded_sequence(own_features_rebatched, leaf_nums, batch_first=True, enforce_sorted=False)
        # output, _ = self.leaf_rnn(packed_own_features, self.h0.repeat(1, batch_size, 1))
        # output_unpacked, output_lengths = pad_packed_sequence(output, batch_first=True)

        tree_features_partitioned = []
        for tree_feature in tree_features:
            padded_tree_features = torch.cat((
                tree_feature, 
                torch.zeros(max_num_leafs - tree_feature.shape[0], tree_feature.shape[1]).to(own_features.device)), dim=0)
            tree_features_partitioned.append(padded_tree_features)
        tree_features_rebatched = torch.stack(tree_features_partitioned)
        packed_tree_features = pack_padded_sequence(tree_features_rebatched, leaf_nums, batch_first=True, enforce_sorted=False)
        output, _ = self.leaf_rnn(packed_tree_features, self.h0.repeat(1, batch_size, 1))
        output_unpacked, output_lengths = pad_packed_sequence(output, batch_first=True)

        output_feature_partitioned = []
        for output_unpacked_for_an_img, leaf_num in zip(output_unpacked, leaf_nums):
            output_feature_partitioned.append(output_unpacked_for_an_img[:leaf_num])
        out1 = torch.cat(output_feature_partitioned, dim=0)
        out2 = out1
        # tree_features = torch.cat(tree_features, dim=0)
        # assert tree_features.shape[0] == out1.shape[0]

        # out2 = torch.cat((out1.detach(), tree_features), dim=1)
        #print(out1.shape, out2.shape)

        return out1, out2

    def semi_decode(self, encoded_features):
        # print(self.decoder[:4])
        return self.decoder[:4](encoded_features)

    def tree_semi_decode(self, encoded_features):
        # print(self.decoder[:4])
        return self.tree_decoder[:4](encoded_features)

    def final_decode(self, semi_decoded_features):
        return self.decoder[4:](semi_decoded_features)

    def tree_final_decode(self, semi_decoded_features):
        return self.tree_decoder[4:](semi_decoded_features)

    def decode(self, encoded_features):
        return self.decoder(encoded_features)

    def tree_decode(self, encoded_features):
        return self.tree_decoder(encoded_features)
        
    def forward(self, images, bboxes, additional_features, trees, rel_bboxes):
        """
        images: torch.Tensor of size [batch_size, 3, img_H, img_H]
        bboxes: torch.Tensor [total_n_bboxes_in_batch, 5]
            each of [batch_img_index, top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        additional_features: torch.Tensor [total_n_bboxes_in_batch, n_additional_features]
        context_indices: Torch.LongTensor [total_n_bboxes_in_batch, n_context]
            i.e. bbox indices (0-indexed) of contexts for all n bboxes. If not enough found, rest are -1
        
        Returns:
            prediction_scores: torch.Tensor of size [total_n_bboxes_in_batch, n_classes]
        """
        combined_feat1, combined_feat2 = self.encode(images, bboxes, additional_features, trees, rel_bboxes)
        output1 = self.decoder(combined_feat1)
        output2 = self.tree_decoder(combined_feat2)

        return output1, output2

    def gcn_forward(self, graph, encoded_features):
        gcn_output = self.gcn(graph)[:3*self.num_candidates]
        gcn_plus_intermediate = torch.cat((gcn_output.detach(), self.intermediate_output_encoder(encoded_features)), dim=1)
        return (self.gcn_output_decoder(gcn_output),
                self.gcn_plus_intermediate_output_decoder(gcn_plus_intermediate))

class BboxOnlyModel(nn.Module):
    def __init__(self):
        super(BboxOnlyModel, self).__init__()

        self.n_bbox_feat = 32

        self.bbox_feat_encoder = nn.Sequential(
                nn.BatchNorm1d(4),
                nn.Linear(4, self.n_bbox_feat),
                nn.BatchNorm1d(self.n_bbox_feat),
                nn.ReLU(),
            )

        self.decoder_input_dim = self.n_bbox_feat
        self.decoder_hidden_dim = self.decoder_input_dim
        drop_prob = 0.2
        self.n_classes = 4

        self.decoder = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(self.decoder_input_dim, self.decoder_hidden_dim),
            nn.BatchNorm1d(self.decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.decoder_hidden_dim, self.n_classes),
        )

    def encode(self, bboxes):
        batch_size = bboxes.shape[0]
        bbox_features = bboxes[:, 1:].clone() # discard batch_img_index column
        bbox_features[:, 2:] -= bbox_features[:, :2] # convert to [top_left_x, top_left_y, width, height]
        
        # bbox_asp_ratio = (bbox_features[:, 2]/bbox_features[:, 3]).view(batch_size, 1)
        # bbox_features = torch.cat((bbox_features[:, 2:], bbox_asp_ratio), dim=1)
        
        bbox_features = self.bbox_feat_encoder(bbox_features)
        return bbox_features

    def num_bboxes_per_img(self, bboxes_img_indices):
        assert bboxes_img_indices[0] == 0
        last_bbox_img_idx = 0
        num_bboxes = []
        counter = 0
        for bbox_img_idx in bboxes_img_indices:
            if bbox_img_idx == last_bbox_img_idx:
                counter += 1
            else:
                last_bbox_img_idx = bbox_img_idx
                num_bboxes.append(counter)
                counter = 1
        num_bboxes.append(counter)
        # print(num_bboxes)
        # print(bboxes_img_indices)
        assert len(num_bboxes) == bboxes_img_indices[-1] + 1
        return num_bboxes

    def decode(self, encoded_features):
        return self.decoder(encoded_features)

    def forward(self, bboxes, texts, chars, tags, trees, rel_bboxes):
       return self.decode(self.encode(bboxes, texts, chars, tags, trees, rel_bboxes))

class TextGCN(nn.Module):
    def __init__(self, char_vocab_size, tag_vocab_size, n_classes, num_candidates, 
                num_rels, device, pretrained_wordvecs, hidden_dim=300, 
                 use_bbox_feat=True, char_embed_dim=100, tag_embed_dim=100, bbox_hidden_dim=32, trainable_convnet=True, drop_prob=0.2, class_names=None):
        super(TextGCN, self).__init__()

        self.n_classes = n_classes
        self.device = device
        self.hidden_dim = 300
        self.use_bbox_feat = use_bbox_feat
        self.bbox_hidden_dim = bbox_hidden_dim
        self.class_names = np.arange(self.n_classes).astype(str) if class_names is None else class_names
        self.num_candidates = num_candidates
        self.num_rels = num_rels
        self.char_vocab_size = char_vocab_size
        self.char_embed_dim = char_embed_dim
        self.tag_vocab_size = tag_vocab_size
        self.tag_embed_dim = tag_embed_dim

        ##### Character Embedding #####
        self.char_channel_dim = 20
        self.char_emb = nn.Embedding(self.char_vocab_size, char_embed_dim)
        self.char_conv = nn.Sequential(
            nn.Conv2d(1, self.char_channel_dim, (self.char_embed_dim, 5)),
            nn.ReLU()
        )
        # self.char_lstm = nn.LSTM(input_size=self.char_embed_dim,
        #                         hidden_size=self.char_embed_dim,
        #                         bidirectional=True,
        #                         batch_first=True,
        #                         dropout=drop_prob)


        ##### Word Embedding #####
        self.word_emb = nn.Embedding.from_pretrained(pretrained_wordvecs, freeze=False)
        self.word_embed_dim = pretrained_wordvecs.shape[1]
        # self.char_embed_dim = 0
        ##### Combine Character and Word #####
        # self.char_embed_dim = 0
        self.text_embed_dim = self.word_embed_dim + self.char_channel_dim
        self.text_lstm = nn.LSTM(input_size=self.text_embed_dim,
                                hidden_size=self.text_embed_dim,
                                bidirectional=True,
                                batch_first=True,
                                dropout=drop_prob)

        ##### Tag Embedding #####
        self.tag_emb = nn.Embedding(self.tag_vocab_size, tag_embed_dim)
        self.tag_lstm = nn.LSTM(input_size=self.tag_embed_dim,
                                hidden_size=self.tag_embed_dim,
                                bidirectional=False,
                                batch_first=True,
                                dropout=drop_prob)
        # self.tag_embed_dim = 83
        # self.tag_encoder = nn.Linear(4 * self.tag_embed_dim, self.tag_embed_dim)

        ##### Bounding Box Embedding #####
        self.n_bbox_feat = self.bbox_hidden_dim if self.use_bbox_feat else 0 # [x,y,w,h,asp_rat] of BBox encoded to n_bbox_feat
        if self.use_bbox_feat:
            self.bbox_feat_encoder = nn.Sequential(
                nn.BatchNorm1d(3),
                nn.Linear(3, self.n_bbox_feat),
                nn.BatchNorm1d(self.n_bbox_feat),
                nn.ReLU(),
            )

        self.hidden_dim = self.text_embed_dim * 2 + self.tag_embed_dim + self.n_bbox_feat

        self.decoder_hidden_dim = self.hidden_dim
        #self.tree_visual_feature_merger = nn.Linear(self.n_own_feat*2, self.n_own_feat)
        self.leaf_rnn = nn.LSTM(input_size=self.hidden_dim, # self.n_own_feat,
                            hidden_size=self.decoder_hidden_dim,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True, dropout = drop_prob)
        h0_init_nums = 2
        h0 = torch.zeros(h0_init_nums, 1, self.hidden_dim)
        nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
        self.h0 = nn.Parameter(h0, requires_grad=True)

        self.pre_leaf_rnn_decoder = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(self.hidden_dim, self.decoder_hidden_dim),
            nn.BatchNorm1d(self.decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.decoder_hidden_dim, self.n_classes),
        )

        # self.decoder_hidden_dim = 300
        self.decoder_input_dim = self.decoder_hidden_dim*2
        ##### FC LAYERS #####
        # self.n_total_feat = self.n_own_feat
        self.decoder = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(self.decoder_input_dim, self.decoder_hidden_dim),
            nn.BatchNorm1d(self.decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.decoder_hidden_dim, self.n_classes),
        )

        ##### GCN #####
        self.num_candidates = num_candidates
        self.gcn_output_dim = self.hidden_dim
        self.gcn_hidden_dim = self.gcn_output_dim
        self.gcn = RGCN(self.num_candidates * 3,
              self.gcn_hidden_dim,
              self.gcn_output_dim,
              num_rels, device=device, 
              num_hidden_layers=0, 
              embedding_provided=True, 
              embedding_dim=self.gcn_hidden_dim + 4 + self.n_bbox_feat, 
              use_edge_nn=False).to(device)

        self.gcn_output_decoder = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(self.gcn_output_dim, self.gcn_output_dim),
            nn.BatchNorm1d(self.gcn_output_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.gcn_output_dim, self.n_classes),
        )

        self.intermediate_output_encoder = nn.Sequential (
            nn.Dropout(drop_prob),
            nn.Linear(self.decoder_input_dim, self.gcn_hidden_dim)
        )

        self.dropout = nn.Dropout(0.2)

        print('Model Parameters:', count_parameters(self))

    # def to_parallel(self):
    #     self.convnet = nn.DataParallel(self.convnet)
    def to_parallel(self):
        self.char_conv = nn.DataParallel(self.char_conv)


    def num_bboxes_per_img(self, bboxes_img_indices):
        assert bboxes_img_indices[0] == 0
        last_bbox_img_idx = 0
        num_bboxes = []
        counter = 0
        for bbox_img_idx in bboxes_img_indices:
            if bbox_img_idx == last_bbox_img_idx:
                counter += 1
            else:
                last_bbox_img_idx = bbox_img_idx
                num_bboxes.append(counter)
                counter = 1
        num_bboxes.append(counter)
        # print(num_bboxes)
        # print(bboxes_img_indices)
        assert len(num_bboxes) == bboxes_img_indices[-1] + 1
        return num_bboxes

    def encode(self, bboxes, texts, chars, tags, trees, rel_bboxes):
        """
        Returns:
            prediction_scores: torch.Tensor of size [total_n_bboxes_in_batch, n_classes]
        """
        batch_size = bboxes.shape[0]

        ##### Character Features #####

        def char_emb_layer(chars, device):
            max_word_len = np.max([len(word) for sentence in chars for word in sentence])
            max_sentence_len = np.max([len(sentence) for sentence in chars])
            for i, sentence in enumerate(chars):
                if len(sentence) > 0:
                    for word in sentence:
                        word.extend([self.char_vocab_size - 1] * (max_word_len - len(word))) # padding
                else:
                    chars[i] = [[self.char_vocab_size - 1] * max_word_len]
            for sentence in chars:
                sentence.extend([[self.char_vocab_size - 1] * max_word_len for _ in range(max_sentence_len - len(sentence))])

            x = self.dropout(self.char_emb(torch.tensor(chars).to(device)))
            x = x.transpose(2, 3)
            x = x.view(-1, self.char_embed_dim, x.size(3)).unsqueeze(1)
            x = self.char_conv(x).squeeze()
            x = F.max_pool1d(x, x.size(2)).squeeze()
            x = x.view(batch_size, -1, self.char_channel_dim)
            return x

        def word_emb_layer(texts, device):
            max_sentence_len = np.max([len(sentence) for sentence in texts])
            text_lens = []
            for sentence in texts:
                text_lens.append(len(sentence))
                sentence.extend([0] * (max_sentence_len - len(sentence))) # padding

            x = self.word_emb(torch.tensor(texts).to(device))
            return x, text_lens

        def tag_emb_layer(tags, device):
            max_tag_path_len = np.max([len(tag_path) for tag_path in tags])
            tag_path_lens = []
            for tag_path in tags:
                tag_path_lens.append(len(tag_path))
                tag_path.extend([self.tag_vocab_size-1] * (max_tag_path_len - len(tag_path))) # padding

            x = self.tag_emb(torch.tensor(tags).to(device))
            return x, tag_path_lens

        char_embs = char_emb_layer(chars, self.device)
        word_embs, text_lens = word_emb_layer(texts, self.device)
        wc_embs = torch.cat((char_embs, word_embs), dim=2)
        # wc_embs = word_embs
        for i, text_len in enumerate(text_lens):
            if text_len == 0:
                text_lens[i] = 1
        packed_text_features = pack_padded_sequence(wc_embs, text_lens, batch_first=True, enforce_sorted=False)
        text_out, _ = self.text_lstm(packed_text_features)
        text_output_unpacked, output_lengths = pad_packed_sequence(text_out, batch_first=True) 
        text_feature_partitioned = []
        for output_unpacked_for_a_node, text_len in zip(text_output_unpacked, text_lens):   
            text_feature_partitioned.append(output_unpacked_for_a_node[:text_len].mean(dim=0))
        text_feature = torch.stack(text_feature_partitioned)

        #### Tag Features ####
        tag_embs, tag_path_lens = tag_emb_layer(tags, self.device)
        packed_tag_features = pack_padded_sequence(tag_embs, tag_path_lens, batch_first=True, enforce_sorted=False)
        tag_out, _ = self.tag_lstm(packed_tag_features)
        tag_output_unpacked, output_lengths = pad_packed_sequence(tag_out, batch_first=True) 
        tag_path_feature_partitioned = []
        for output_unpacked_for_a_node, tag_path_len in zip(tag_output_unpacked, tag_path_lens):
            tag_path_feature_partitioned.append(output_unpacked_for_a_node[tag_path_len - 1])
        tag_path_feature = torch.stack(tag_path_feature_partitioned)

        ##### BBOX FEATURES #####
        if self.use_bbox_feat:
            bbox_features = bboxes[:, 1:].clone() # discard batch_img_index column
            bbox_features[:, 2:] -= bbox_features[:, :2] # convert to [top_left_x, top_left_y, width, height]
            
            bbox_asp_ratio = (bbox_features[:, 2]/bbox_features[:, 3]).view(batch_size, 1)
            bbox_features = torch.cat((bbox_features[:, 2:], bbox_asp_ratio), dim=1)
            
            bbox_features = self.bbox_feat_encoder(bbox_features)
        else:
            bbox_features = bboxes[:, :0] # size [n_bboxes, 0]
        
        ##### text+path+bbox #####
        own_features = torch.cat((text_feature, tag_path_feature, bbox_features), dim=1)
        # print('return not leaf rnn')
        pre_leaf_rnn_result = self.pre_leaf_rnn_decoder(own_features.detach())

        # return own_features, own_features

        ##### LEAF RNN #####

        batch_size = bboxes.shape[0]
        num_bboxes = self.num_bboxes_per_img(bboxes[:, 0])
        curr_idx = 0
        own_features_partitioned = []
        max_num_bbox = np.max(num_bboxes)
        for num_bbox in num_bboxes:
            padded_own_features = torch.cat((
                own_features[curr_idx:curr_idx + num_bbox], 
                torch.zeros(max_num_bbox - num_bbox, own_features.shape[1]).to(own_features.device)), dim=0)
            own_features_partitioned.append(padded_own_features)
            curr_idx += num_bbox
        own_features_rebatched = torch.stack(own_features_partitioned)

        packed_own_features = pack_padded_sequence(own_features_rebatched, num_bboxes, batch_first=True, enforce_sorted=False)
        output, _ = self.leaf_rnn(packed_own_features)

        
        output_unpacked, output_lengths = pad_packed_sequence(output, batch_first=True) 
        output_feature_partitioned = []
        for output_unpacked_for_an_img, num_bbox in zip(output_unpacked, num_bboxes):
            output_feature_partitioned.append(output_unpacked_for_an_img[:num_bbox])
        out1 = torch.cat(output_feature_partitioned, dim=0)

        return out1, pre_leaf_rnn_result

    def semi_decode(self, encoded_features):
        # print(self.decoder[:4])
        return self.decoder[:4](encoded_features)

    def tree_semi_decode(self, encoded_features):
        # print(self.decoder[:4])
        return self.tree_decoder[:4](encoded_features)

    def final_decode(self, semi_decoded_features):
        return self.decoder[4:](semi_decoded_features)

    def tree_final_decode(self, semi_decoded_features):
        return self.tree_decoder[4:](semi_decoded_features)

    def decode(self, encoded_features):
        return self.decoder(encoded_features)

    def tree_decode(self, encoded_features):
        return self.tree_decoder(encoded_features)
        
    def forward(self, bboxes, texts, chars, tags, trees, rel_bboxes):
        """
        images: torch.Tensor of size [batch_size, 3, img_H, img_H]
        bboxes: torch.Tensor [total_n_bboxes_in_batch, 5]
            each of [batch_img_index, top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        additional_features: torch.Tensor [total_n_bboxes_in_batch, n_additional_features]
        context_indices: Torch.LongTensor [total_n_bboxes_in_batch, n_context]
            i.e. bbox indices (0-indexed) of contexts for all n bboxes. If not enough found, rest are -1
        
        Returns:
            prediction_scores: torch.Tensor of size [total_n_bboxes_in_batch, n_classes]
        """
        combined_feat1, combined_feat2 = self.encode(bboxes, texts, chars, tags, trees, rel_bboxes)
        output1 = self.decoder(combined_feat1)
        output2 = self.tree_decoder(combined_feat2)

        return output1, output2

    def gcn_forward(self, graph, encoded_features):
        gcn_output = self.gcn(graph)[:3*self.num_candidates]
        return self.gcn_output_decoder(gcn_output)

class VisTextGCN(nn.Module):
    def __init__(self, char_vocab_size, tag_vocab_size, n_classes, num_candidates, 
                num_rels, device, pretrained_wordvecs, hidden_dim=300, 
                 use_bbox_feat=True, char_embed_dim=100, tag_embed_dim=100, bbox_hidden_dim=32, trainable_convnet=True, drop_prob=0.2, class_names=None, splitted=False):
        super(VisTextGCN, self).__init__()

        self.n_classes = n_classes
        self.device = device
        self.hidden_dim = hidden_dim
        self.use_bbox_feat = use_bbox_feat
        self.bbox_hidden_dim = bbox_hidden_dim
        self.class_names = np.arange(self.n_classes).astype(str) if class_names is None else class_names
        self.num_candidates = num_candidates
        self.num_rels = num_rels
        self.char_vocab_size = char_vocab_size
        self.char_embed_dim = char_embed_dim
        self.tag_vocab_size = tag_vocab_size
        self.tag_embed_dim = tag_embed_dim
        self.splitted = splitted

        second_device = 'cuda:1' if self.splitted else device

        self.convnet = torchvision.models.resnet18(pretrained=True)
        modules = list(self.convnet.children())[:-5]
        self.convnet = nn.Sequential(*modules)
        img_H = 1280
        _imgs = torch.autograd.Variable(torch.Tensor(1, 3, img_H, img_H))
        with torch.no_grad():
            _conv_feat = self.convnet(_imgs)
        _convnet_output_size = _conv_feat.shape # [1, C, H, W]
        spatial_scale = _convnet_output_size[2]/img_H
        roi_output_size = (3, 3)

#         self.convnet = self.convnet.to(second_device)
        self.roi_pool = torchvision.ops.RoIPool(roi_output_size, spatial_scale)
        self.n_visual_feat = _convnet_output_size[1] * roi_output_size[0] * roi_output_size[1]

        # self.n_visual_feat = 1
        # print("No character embedding!!!!!!!")

        ##### Character Embedding #####
        self.char_channel_dim = 30
        self.char_emb = nn.Embedding(self.char_vocab_size, char_embed_dim)
        self.char_conv = nn.Sequential(
            nn.Conv2d(1, self.char_channel_dim, (self.char_embed_dim, 5)),
            nn.ReLU()
        )

        ##### Word Embedding #####
        self.word_emb = nn.Embedding.from_pretrained(pretrained_wordvecs, freeze=False)
        self.word_embed_dim = pretrained_wordvecs.shape[1]

        ##### Combine Character and Word #####
        self.text_embed_dim = self.word_embed_dim + self.char_channel_dim
        self.text_lstm = nn.LSTM(input_size=self.text_embed_dim,
                                hidden_size=self.text_embed_dim,
                                bidirectional=True,
                                batch_first=True,
                                dropout=drop_prob)

        ##### Tag Embedding #####
        self.tag_emb = nn.Embedding(self.tag_vocab_size, tag_embed_dim).to(device)
        self.tag_hidden_size = 100
        self.tag_lstm = nn.LSTM(input_size=self.tag_embed_dim,
                                hidden_size=self.tag_hidden_size,
                                bidirectional=True,
                                batch_first=True,
                                dropout=drop_prob)
        
        self.leaf_tag_dim = 20
        self.leaf_tag_embed = nn.Embedding(self.tag_vocab_size, self.leaf_tag_dim)

        ##### Bounding Box Embedding #####
        self.n_bbox_feat = self.bbox_hidden_dim if self.use_bbox_feat else 0 # [x,y,w,h,asp_rat] of BBox encoded to n_bbox_feat
        if self.use_bbox_feat:
            self.bbox_feat_encoder = nn.Sequential(
                nn.BatchNorm1d(5),
                nn.Linear(5, self.n_bbox_feat),
                nn.BatchNorm1d(self.n_bbox_feat),
                nn.ReLU(),
            )

        self.n_node_feat = self.n_visual_feat + self.n_bbox_feat
        self.leaf_input_dim = self.text_embed_dim * 2 + self.tag_hidden_size * 2 + self.hidden_dim + self.leaf_tag_dim

        ##### Tree Information Aggregation Network #####
        self.tree_rnn = PassDownTreeLSTM(self.n_node_feat, self.hidden_dim)

        ##### Leaf Level Preorder Information Aggregation Network #####
        self.leaf_rnn = nn.GRU(self.leaf_input_dim,
                            self.hidden_dim,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True, dropout = drop_prob)
        h0_init_nums = 2
        h0 = torch.zeros(h0_init_nums, 1, self.hidden_dim)
        nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
        self.h0 = nn.Parameter(h0, requires_grad=True)

        self.decoder_hidden_dim = self.hidden_dim
        self.decoder_input_dim = self.hidden_dim * 2

        ##### FC LAYERS #####
        self.decoder = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(self.decoder_input_dim, self.decoder_hidden_dim),
            nn.BatchNorm1d(self.decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.decoder_hidden_dim, self.n_classes),
        )

        ##### GCN #####
        self.num_candidates = num_candidates
        self.gcn_output_dim = self.decoder_hidden_dim
        self.gcn_hidden_dim = self.gcn_output_dim
        self.gcn = RGCN(self.num_candidates * 3,
              self.gcn_hidden_dim,
              self.gcn_output_dim,
              num_rels, device=device, 
              num_hidden_layers=0, 
              embedding_provided=True, 
              embedding_dim=self.gcn_hidden_dim + 4 + 4 + self.n_bbox_feat, 
              use_edge_nn=False, 
              drop_prob=drop_prob)

        self.gcn_output_decoder = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(self.gcn_output_dim, self.gcn_output_dim),
            nn.BatchNorm1d(self.gcn_output_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.gcn_output_dim, self.n_classes),
        )

        self.intermediate_output_encoder = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(self.decoder_input_dim, self.gcn_hidden_dim)
        )
        
        self.char_conv_dropout = nn.Dropout(0.1).to(device)
        self.word_emb_dropout = nn.Dropout(0.2).to(device)
        self.tag_emb_dropout = nn.Dropout(0.2).to(device)

        print('Model Parameters:', count_parameters(self))
        print('Conv net Parameters:', count_parameters(self.char_conv))

    def split(self):
        # self.bbox_feat_encoder = self.bbox_feat_encoder.to('cuda:1')
        self.tree_rnn = self.tree_rnn.to('cuda:1')
        self.leaf_rnn = self.leaf_rnn.to('cuda:1')
        self.decoder = self.decoder.to('cuda:1')
        self.gcn = self.gcn.to('cuda:1')
        self.gcn_output_decoder = self.gcn_output_decoder.to('cuda:1')
        self.intermediate_output_encoder = self.intermediate_output_encoder.to('cuda:1')
        self.splitted=True
        gc.collect()

    def num_bboxes_per_img(self, bboxes_img_indices):
        assert bboxes_img_indices[0] == 0
        last_bbox_img_idx = 0
        num_bboxes = []
        counter = 0
        for bbox_img_idx in bboxes_img_indices:
            if bbox_img_idx == last_bbox_img_idx:
                counter += 1
            else:
                last_bbox_img_idx = bbox_img_idx
                num_bboxes.append(counter)
                counter = 1
        num_bboxes.append(counter)
        assert len(num_bboxes) == bboxes_img_indices[-1] + 1
        return num_bboxes

    def extract_tree_states(self, tree, shape):
        outputs = [[]] * shape[0]
        stack = [tree]
        max_idx = -1
        while len(stack) > 0:
            curr = stack.pop()
            stack += curr.children
            if curr.num_children == 0:
                outputs[curr.idx] = curr.state
                if curr.idx > max_idx:
                    max_idx = curr.idx
        return torch.stack(outputs[:max_idx + 1]), max_idx+1

    def encode(self, images, leaf_bboxes, extended_bboxes, texts, chars, leaf_tags, tags, trees, rel_bboxes):
        """
        Returns:
            prediction_scores: torch.Tensor of size [total_n_bboxes_in_batch, n_classes]
        """
        batch_size = leaf_bboxes.shape[0]

        ##### Character Features #####

        def char_emb_layer(chars, device):
            max_word_len = np.max([len(word) for sentence in chars for word in sentence])
            max_sentence_len = np.max([len(sentence) for sentence in chars])
            for i, sentence in enumerate(chars):
                if len(sentence) > 0:
                    for word in sentence:
                        word.extend([self.char_vocab_size - 1] * (max_word_len - len(word))) # padding
                else:
                    chars[i] = [[self.char_vocab_size - 1] * max_word_len]
            for sentence in chars:
                sentence.extend([[self.char_vocab_size - 1] * max_word_len for _ in range(max_sentence_len - len(sentence))])

            x = self.char_conv_dropout(self.char_emb(torch.tensor(chars).to(device)))
            x = x.transpose(2, 3)
            x = x.view(-1, self.char_embed_dim, x.size(3)).unsqueeze(1)
            x = self.char_conv(x).squeeze()
            x = F.max_pool1d(x, x.size(2)).squeeze()
            x = x.view(batch_size, -1, self.char_channel_dim)
            return x

        def word_emb_layer(texts, device):
            max_sentence_len = np.max([len(sentence) for sentence in texts])
            text_lens = []
            for sentence in texts:
                text_lens.append(len(sentence))
                sentence.extend([0] * (max_sentence_len - len(sentence))) # padding

            x = self.word_emb_dropout(self.word_emb(torch.tensor(texts).to(device)))
            return x, text_lens

        def tag_emb_layer(tags, device):
            max_tag_path_len = np.max([len(tag_path) for tag_path in tags])
            tag_path_lens = []
            for tag_path in tags:
                tag_path_lens.append(len(tag_path))
                tag_path.extend([self.tag_vocab_size-1] * (max_tag_path_len - len(tag_path))) # padding

            x = self.tag_emb_dropout(self.tag_emb(torch.tensor(tags).to(device)))
            if self.splitted:
                x = x.to('cuda:1')
            return x, tag_path_lens

        char_embs = char_emb_layer(chars, self.device)
        word_embs, text_lens = word_emb_layer(texts, self.device)
        wc_embs = torch.cat((char_embs, word_embs), dim=2)
        # wc_embs = word_embs
        for i, text_len in enumerate(text_lens):
            if text_len == 0:
                text_lens[i] = 1
        packed_text_features = pack_padded_sequence(wc_embs, text_lens, batch_first=True, enforce_sorted=False)
        text_out, _ = self.text_lstm(packed_text_features)
        text_output_unpacked, output_lengths = pad_packed_sequence(text_out, batch_first=True) 
        text_feature_partitioned = []
        for output_unpacked_for_a_node, text_len in zip(text_output_unpacked, text_lens):   
            text_feature_partitioned.append(output_unpacked_for_a_node[:text_len].mean(dim=0))
        text_feature = torch.stack(text_feature_partitioned)

        #### Tag Features ####
        tag_embs, tag_path_lens = tag_emb_layer(tags, self.device)
        packed_tag_features = pack_padded_sequence(tag_embs, tag_path_lens, batch_first=True, enforce_sorted=False)
        tag_out, _ = self.tag_lstm(packed_tag_features)
        tag_output_unpacked, output_lengths = pad_packed_sequence(tag_out, batch_first=True) 
        tag_path_feature_partitioned = []
        for output_unpacked_for_a_node, tag_path_len in zip(tag_output_unpacked, tag_path_lens):
            tag_path_feature_partitioned.append(output_unpacked_for_a_node[tag_path_len - 1])
        tag_path_feature = torch.stack(tag_path_feature_partitioned)

        ##### text+tag_path #####
        tag_leaf_features = self.leaf_tag_embed(leaf_tags)
        text_features = torch.cat((text_feature, tag_path_feature, tag_leaf_features), dim=1)

        ##### BBOX FEATURES #####
        if self.use_bbox_feat:
            bbox_features = extended_bboxes[:, 1:].clone() # discard batch_img_index column
            bbox_features[:, 2:] -= bbox_features[:, :2] # convert to [top_left_x, top_left_y, width, height]
            
            bbox_asp_ratio = (bbox_features[:, 2]/bbox_features[:, 3]).view(extended_bboxes.shape[0], 1)
            bbox_features = torch.cat((bbox_features, bbox_asp_ratio), dim=1)
            
            bbox_features = self.bbox_feat_encoder(bbox_features)
        else:
            bbox_features = extended_bboxes[:, :0] # size [n_bboxes, 0]

        ##### NODE VISUAL + BBOX FEATURES #####
        if self.splitted:
            images = images.to('cuda:1')
            extended_bboxes = extended_bboxes.to('cuda:1')
            bbox_features = bbox_features.to('cuda:1')
            rel_bboxes = rel_bboxes.to('cuda:1')
        visual_features = self.roi_pool(self.convnet(images), extended_bboxes).view(extended_bboxes.shape[0], self.n_visual_feat)
        # visual_features = torch.zeros((extended_bboxes.shape[0], self.n_visual_feat)).to(self.device)
        extended_node_features = torch.cat((visual_features, bbox_features), dim=1)

        ##### TREE GRU #####
        num_extended_bboxes = self.num_bboxes_per_img(extended_bboxes[:, 0])
        num_leaf_bboxes = self.num_bboxes_per_img(leaf_bboxes[:, 0])
        curr_idx = 0
        curr_leaf_bbox_idx = 0
        leaf_features = []
        leaf_nums = []
        assert len(num_extended_bboxes) == len(trees)
        
        for i, (num_extended_bbox, num_leaf_bbox) in enumerate(zip(num_extended_bboxes, num_leaf_bboxes)):
            inputs = extended_node_features[curr_idx:curr_idx + num_extended_bbox]
            rel_bbox_segment = rel_bboxes[curr_idx:curr_idx + num_extended_bbox]
            
            self.tree_rnn.create_rel_bboxes_info(rel_bbox_segment)
            self.tree_rnn(trees[i], inputs)
            tree_feature, leaf_num = self.extract_tree_states(trees[i], (num_extended_bbox, self.n_node_feat))
            assert leaf_num == num_leaf_bbox

            text_feature_segment = text_features[curr_leaf_bbox_idx:curr_leaf_bbox_idx + num_leaf_bbox]
            leaf_features.append(torch.cat((tree_feature, text_feature_segment), dim=1))
            leaf_nums.append(leaf_num)

            curr_idx += num_extended_bbox
            curr_leaf_bbox_idx += num_leaf_bbox

        max_num_leafs = np.max(leaf_nums)

        leaf_features_partitioned = []
        for leaf_feature in leaf_features:
            if self.splitted:
                padded_leaf_features = torch.cat((
                    leaf_feature, 
                    torch.zeros(max_num_leafs - leaf_feature.shape[0], leaf_feature.shape[1]).to('cuda:1')), dim=0)
            else:
                padded_leaf_features = torch.cat((
                    leaf_feature, 
                    torch.zeros(max_num_leafs - leaf_feature.shape[0], leaf_feature.shape[1]).to(self.device)), dim=0)
            leaf_features_partitioned.append(padded_leaf_features)
        leaf_features_rebatched = torch.stack(leaf_features_partitioned)
        packed_leaf_features = pack_padded_sequence(leaf_features_rebatched, leaf_nums, batch_first=True, enforce_sorted=False)
        output, _ = self.leaf_rnn(packed_leaf_features, self.h0.repeat(1, batch_size, 1))
        output_unpacked, output_lengths = pad_packed_sequence(output, batch_first=True)

        output_feature_partitioned = []
        for output_unpacked_for_an_img, leaf_num in zip(output_unpacked, leaf_nums):
            output_feature_partitioned.append(output_unpacked_for_an_img[:leaf_num])
        out1 = torch.cat(output_feature_partitioned, dim=0)

        return out1

    def semi_decode(self, encoded_features):
        # print(self.decoder[:4])
        return self.decoder[:4](encoded_features)

    def tree_semi_decode(self, encoded_features):
        # print(self.decoder[:4])
        return self.tree_decoder[:4](encoded_features)

    def final_decode(self, semi_decoded_features):
        return self.decoder[4:](semi_decoded_features)

    def tree_final_decode(self, semi_decoded_features):
        return self.tree_decoder[4:](semi_decoded_features)

    def decode(self, encoded_features):
        return self.decoder(encoded_features)

    def tree_decode(self, encoded_features):
        return self.tree_decoder(encoded_features)
        
    def forward(self, bboxes, texts, chars, leaf_tags, tags, trees, rel_bboxes):
        """
        images: torch.Tensor of size [batch_size, 3, img_H, img_H]
        bboxes: torch.Tensor [total_n_bboxes_in_batch, 5]
            each of [batch_img_index, top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        additional_features: torch.Tensor [total_n_bboxes_in_batch, n_additional_features]
        context_indices: Torch.LongTensor [total_n_bboxes_in_batch, n_context]
            i.e. bbox indices (0-indexed) of contexts for all n bboxes. If not enough found, rest are -1
        
        Returns:
            prediction_scores: torch.Tensor of size [total_n_bboxes_in_batch, n_classes]
        """
        combined_feat1, combined_feat2 = self.encode(bboxes, texts, chars, leaf_tags, tags, trees, rel_bboxes)
        output1 = self.decoder(combined_feat1)
        output2 = self.tree_decoder(combined_feat2)

        return output1, output2

    def gcn_forward(self, graph, encoded_features):
        gcn_output = self.gcn(graph)[:3*self.num_candidates]
        return self.gcn_output_decoder(gcn_output)