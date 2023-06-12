from cv2 import CAP_PROP_XI_SENSOR_OUTPUT_CHANNEL_COUNT
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from sklearn import metrics
np.set_printoptions(precision=3)
import time
import os
import pandas as pd
import copy
import time
import datetime
import wandb
import pickle
from torch import autograd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
from argparse import ArgumentParser
from lib.object_detector import detector
from lib.config import Config
from lib.evaluation_recall_all_rel_cats import BasicSceneGraphEvaluator
from lib.AdamW import AdamW
# from torch.optim import AdamW
# from lib.sttran import STTran
# from lib.binary_cross_entropy import BinaryCrossEntropy
from meta import *
# from torchviz import make_dot
from lib.model import *
from lib.cosine_lr import CosineLRScheduler


def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class AG(Dataset):

    def __init__(self, mode, data_path=None):

        self.features_path = os.path.join(data_path, 'features/')
        if mode == 'train':
            with open('train_videos_v2.txt', 'r') as f:
                self.video_list = f.read().splitlines()
        elif mode == 'test':
            with open('test_videos.txt', 'r') as f:
                self.video_list = f.read().splitlines()
        elif mode == 'val':
            with open('val_videos_v1.txt', 'r') as f:
                self.video_list = f.read().splitlines()
        self.object_classes = ['__background__']

        with open(os.path.join(data_path, 'annotations/object_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.object_classes.append(line)
        f.close()
        self.object_classes[9] = 'closet/cabinet'
        self.object_classes[11] = 'cup/glass/bottle'
        self.object_classes[23] = 'paper/notebook'
        self.object_classes[24] = 'phone/camera'
        self.object_classes[31] = 'sofa/couch'

        # collect relationship classes
        self.relationship_classes = []
        with open(os.path.join(data_path, 'annotations/relationship_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.relationship_classes.append(line)
        self.relationship_classes[0] = 'looking_at'
        self.relationship_classes[1] = 'not_looking_at'
        self.relationship_classes[5] = 'in_front_of'
        self.relationship_classes[7] = 'on_the_side_of'
        self.relationship_classes[10] = 'covered_by'
        self.relationship_classes[11] = 'drinking_from'
        self.relationship_classes[13] = 'have_it_on_the_back'
        self.relationship_classes[15] = 'leaning_on'
        self.relationship_classes[16] = 'lying_on'
        self.relationship_classes[17] = 'not_contacting'
        self.relationship_classes[18] = 'other_relationship'
        self.relationship_classes[19] = 'sitting_on'
        self.relationship_classes[20] = 'standing_on'
        self.relationship_classes[25] = 'writing_on'

        self.attention_relationships = self.relationship_classes[0:3]
        self.spatial_relationships = self.relationship_classes[3:9]
        self.contacting_relationships = self.relationship_classes[9:]

    def __getitem__(self, index):
        video_name = self.video_list[index]
        feat_file = os.path.join(self.features_path, video_name + '.pkl')
        with open(feat_file, 'rb') as f:
            entry = pickle.load(f)
        return entry

    def __len__(self):
        return len(self.video_list)
    
def cuda_collate_fn(batch):
    """
    don't need to zip the tensor

    """
    return batch[0]


class STTran(nn.Module):

    def __init__(self):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        """
        super(STTran, self).__init__()

        self.a_rel_compress = nn.Linear(1936, 3)  # 3: looking at, not looking at, unsure
        self.s_rel_compress = nn.Linear(1936, 6)  # 6: in front of, behind, on the side of, above, beneath, in
        self.c_rel_compress = nn.Linear(1936, 17)  # 17: carrying, drinking from, have it on the back, ...

    def forward(self, x):

        entry = {}
        global_output = x
        assert not torch.any(torch.isnan(global_output))

        entry["attention_distribution"] = self.a_rel_compress(global_output)  # [pair_num, 3]
        entry["spatial_distribution"] = self.s_rel_compress(global_output)  # [pair_num, 6]
        entry["contacting_distribution"] = self.c_rel_compress(global_output)  # [pair_num, 17]

        entry["spatial_distribution"] = torch.sigmoid(entry["spatial_distribution"])
        entry["contacting_distribution"] = torch.sigmoid(entry["contacting_distribution"])

        return entry


def main():
    wandb.init(project="STTran")
    # torch.autograd.set_detect_anomaly(True)  # only for debug, could lower speed
    # fix random seeds for reproducibility
    SEED = 15
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    """------------------------------------some settings----------------------------------------"""
    parser = ArgumentParser(description='training code')
    parser.add_argument('-mode', dest='mode', help='predcls/sgcls/sgdet', default='predcls', type=str)
    parser.add_argument('-save_path', default='data/', type=str)
    parser.add_argument('-model_path', default=None, type=str)
    parser.add_argument('-data_path', default='/data/scene_understanding/action_genome/', type=str)
    # parser.add_argument('-datasize', dest='datasize', help='mini dataset or whole', default='large', type=str)
    parser.add_argument('-ckpt', dest='ckpt', help='checkpoint', default=None, type=str)
    parser.add_argument('-optimizer', help='adamw/adam/sgd', default='adamw', type=str)
    parser.add_argument('-lr', dest='lr', help='learning rate', default=1e-5, type=float)
    parser.add_argument('-nepoch', help='epoch number', default=10, type=int)
    # parser.add_argument('-enc_layer', dest='enc_layer', help='spatial encoder layer', default=1, type=int)
    # parser.add_argument('-dec_layer', dest='dec_layer', help='temporal decoder layer', default=3, type=int)
    parser.add_argument('-bce_loss', action='store_true')
    parser.add_argument('-log_freq', type=int, default=100, help='print log frequency')
    conf = parser.parse_args()
    log_freq = conf.log_freq
    print('The CKPT saved here:', conf.save_path)
    if not os.path.exists(conf.save_path):
        os.mkdir(conf.save_path)

    for i in vars(conf):
        print(f'{i} : {getattr(conf, i)}')
    # obj_loss_coef = 1.0
    # print(f'object loss coefficient: {obj_loss_coef}')
    """-----------------------------------------------------------------------------------------"""

    # imbalanced training dataset
    AG_dataset_train = AG(mode="train", data_path=conf.data_path)
    dataloader_train = torch.utils.data.DataLoader(AG_dataset_train, shuffle=True, num_workers=0, collate_fn=cuda_collate_fn)
    AG_dataset_test = AG(mode="test", data_path=conf.data_path)
    dataloader_test = torch.utils.data.DataLoader(AG_dataset_test, shuffle=False, num_workers=0, collate_fn=cuda_collate_fn)
    # a balanced meta dataset for weighting samples (which is a subset of training dataset)
    # AG_dataset_val = AG(mode='val', data_path=conf.data_path)
    # dataloader_val = DataLoader(AG_dataset_val, shuffle=True, num_workers=0, collate_fn=cuda_collate_fn)
    print('Reweight samples separately on attention, spatial, and contact relationship!')

    gpu_device = torch.device("cuda:0")
    # freeze the detection backbone
    # object_detector = detector(train=True, object_classes=AG_dataset_train.object_classes, use_SUPPLY=True, mode=conf.mode).to(device=gpu_device)
    # object_detector.eval()

    model = STTran().to(device=gpu_device)

    wandb.watch(model, log='all')
    meta_net = MLP(hidden_size=100, num_layers=1).to(device=gpu_device)  # weight function
    wandb.watch(meta_net, log='all')
    print('meta net hidden size: 100')
    evaluator =BasicSceneGraphEvaluator(mode=conf.mode,
                                        AG_object_classes=AG_dataset_train.object_classes,
                                        AG_all_predicates=AG_dataset_train.relationship_classes,
                                        AG_attention_predicates=AG_dataset_train.attention_relationships,
                                        AG_spatial_predicates=AG_dataset_train.spatial_relationships,
                                        AG_contacting_predicates=AG_dataset_train.contacting_relationships,
                                        iou_threshold=0.5,
                                        constraint='with')
    evaluator_semi =BasicSceneGraphEvaluator(mode=conf.mode,
                                        AG_object_classes=AG_dataset_train.object_classes,
                                        AG_all_predicates=AG_dataset_train.relationship_classes,
                                        AG_attention_predicates=AG_dataset_train.attention_relationships,
                                        AG_spatial_predicates=AG_dataset_train.spatial_relationships,
                                        AG_contacting_predicates=AG_dataset_train.contacting_relationships,
                                        iou_threshold=0.5,
                                        constraint='semi', semithreshold=0.9)
    evaluator_no =BasicSceneGraphEvaluator(mode=conf.mode,
                                        AG_object_classes=AG_dataset_train.object_classes,
                                        AG_all_predicates=AG_dataset_train.relationship_classes,
                                        AG_attention_predicates=AG_dataset_train.attention_relationships,
                                        AG_spatial_predicates=AG_dataset_train.spatial_relationships,
                                        AG_contacting_predicates=AG_dataset_train.contacting_relationships,
                                        iou_threshold=0.5,
                                        constraint='no')
    # for name, param in model.named_parameters():
    #     if 'compress' in name:
    #         pass  # freeze all layers except last linear classifier
    #     else:
    #         param.requires_grad = False

    # # inverse class frequency loss weight
    # rel_inv_freq = torch.cuda.FloatTensor(AG_dataset_train.rel_inv_freq)
    # attention_weight = rel_inv_freq[:3] / rel_inv_freq[:3].sum()
    # spatial_weight = rel_inv_freq[3:9] / rel_inv_freq[3:9].sum()
    # contact_weight = rel_inv_freq[9:] / rel_inv_freq[9:].sum()
    # print('Use inverse class frequence weight for CE loss and BCE loss!')

    # loss function, default Multi-label margin loss
    if conf.bce_loss:
        ce_loss = nn.CrossEntropyLoss()
        # bce_loss = nn.BCELoss()
        # bce_loss = BinaryCrossEntropy()  # avoid double-backwards error in PyTorch 1.1.0, https://github.com/pytorch/pytorch/issues/18945
        bce_loss = nn.BCEWithLogitsLoss()
    else:
        ce_loss = nn.CrossEntropyLoss()
        mlm_loss = nn.MultiLabelMarginLoss()

    # optimizer
    if conf.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=conf.lr)
        # optimizer = AdamW(model.parameters(), lr=conf.lr, eps=1e-10, weight_decay=1e-3)
        # print(f'AdamW lr: {conf.lr}, eps: 1e-10, weight_decay: 1e-3')
    elif conf.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=conf.lr)
    elif conf.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=conf.lr, momentum=0.9, weight_decay=0.01)

    # meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=1e-2, weight_decay=1e-4)
    meta_lr = 0.01
    meta_optimizer = optim.SGD(meta_net.parameters(), lr=meta_lr, momentum=0.9, weight_decay=0.01)
    print(f'meta optimizer SGD lr: {meta_lr}, momentum: 0.9, weight_decay: 0.01')
    # print(f'meta optimizer lr: 1e-2, weight_decay: 1e-4')
    # meta_optimizer = AdamW(meta_net.parameters(), lr=1e-3, weight_decay=1e-4)
    # print(f'meta net optimizer AdamW: lr: 1e-3, weight_decay: 1e-4')
    scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4, threshold_mode="abs", min_lr=1e-7)
    # scheduler = CosineLRScheduler(
    #             optimizer,
    #             t_initial=conf.nepoch,
    #             lr_min=1e-6,
    #             warmup_lr_init=0.0001,
    #             warmup_t=2,
    #             k_decay=1.0,
    #         )
    # print('Cosine LR schedule with warmup, cycle/restarts, noise, k-decay.')
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    # print(f'StepLR scheduler')

    # some parameters
    tr = []

    # if use_amp:
    #     from torch.cuda.amp import autocast, GradScaler
    #     scaler = GradScaler()

    print(f"Start training for {conf.nepoch} epochs")
    start_time = time.time()
    rel_num_per_cls = np.asarray([144631, 193370,  38777,   4653,  58226, 254631,  46413,  69895,
                            12939,   4012,   4087,   4378,   3214,    314, 157010,  11530,
                            3410, 105135,   8749,  40576,   7609,  52201,     88,   6766,
                            772,   1102])
    # rel_inv_freq = [3.638e-04, 2.721e-04, 1.357e-03, 1.131e-02, 9.037e-04, 2.066e-04,
    #    1.134e-03, 7.528e-04, 4.067e-03, 1.312e-02, 1.287e-02, 1.202e-02,
    #    1.637e-02, 1.676e-01, 3.351e-04, 4.564e-03, 1.543e-02, 5.005e-04,
    #    6.014e-03, 1.297e-03, 6.915e-03, 1.008e-03, 5.979e-01, 7.777e-03,
    #    6.816e-02, 4.775e-02]  # normalized relationship weight: inverse of class frequency
    rel_inv_freq = 1.0 / rel_num_per_cls
    # rel_inv_freq = torch.FloatTensor(rel_inv_freq).to(device=gpu_device)
    val_attention_weight = rel_inv_freq[:3]
    val_attention_weight = val_attention_weight / val_attention_weight.sum()
    val_attention_weight = torch.FloatTensor(val_attention_weight).to(device=gpu_device)
    val_spatial_weight = rel_inv_freq[3:9]
    val_spatial_weight = val_spatial_weight / val_spatial_weight.sum()
    val_spatial_weight = torch.FloatTensor(val_spatial_weight).to(device=gpu_device)
    val_contact_weight = rel_inv_freq[9:]
    val_contact_weight = val_contact_weight / val_contact_weight.sum()
    val_contact_weight = torch.FloatTensor(val_contact_weight).to(device=gpu_device)
    val_data = pickle.load(open('data/val_videos_v2_features.pkl', 'rb'))
    for epoch in range(conf.nepoch):
        # if epoch > 0: continue
        # model.train()
        model.train()
        start = time.time()
        train_iter = iter(dataloader_train)
        test_iter = iter(dataloader_test)
        # val_iter = cycle(dataloader_val)  # len(val) << len(train), https://github.com/pytorch/pytorch/issues/23900#issuecomment-518858050
        # with torch.profiler.profile(
        #     schedule=torch.profiler.schedule(
        #         wait=2,
        #         warmup=2,
        #         active=6,
        #         repeat=1,
        #     ),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
        #     with_stack=True,
        # ) as profiler:
        for b in range(len(dataloader_train)):
            # if b >= 2: continue
            data = next(train_iter)
            feat = data['global_output']
            feat = torch.from_numpy(feat).float().to(gpu_device)
            attention_label = torch.from_numpy(data['attention_gt']).long().to(device=gpu_device)  # [pair_num]
            spatial_label = torch.from_numpy(data['spatial_gt']).float().to(device=gpu_device)  # [pair_num, 6]
            try:
                contact_label = torch.from_numpy(data['contacting_gt']).float().to(device=gpu_device)  # [pair, 17]
            except KeyError:
                contact_label = torch.from_numpy(data['contact_gt']).float().to(device=gpu_device)

            if (b + 1) % 1 == 0:  # meta interval
            # with higher.innerloop_ctx(model, optimizer) as (meta_model, meta_optimizer):
            # 1. Update meta model on training data
                pseudo_model = STTran().to(device=gpu_device)
                pseudo_model.load_state_dict(model.state_dict())
                pseudo_model.train()
                pseudo_train_pred = pseudo_model(feat)
                # make_dot(meta_model(entry), params=dict(list(meta_model.named_parameters())), show_attrs=True, show_saved=True)
                meta_attention_distribution = pseudo_train_pred["attention_distribution"]  # [pair_num, 3]
                meta_spatial_distribution = pseudo_train_pred["spatial_distribution"]  # [pair_num, 6]
                meta_contact_distribution = pseudo_train_pred["contacting_distribution"]  # [pair_num, 17]

                meta_train_losses = {}
                ce_loss.weight = None
                ce_loss.reduction = 'none'
                bce_loss.reduction = 'none'
                
                # pseudo_attention_loss_vector = ce_loss(meta_attention_distribution, attention_label)  # * is torch.mul(), https://blog.csdn.net/weixin_39228381/article/details/108982594
                attention_label = torch.nn.functional.one_hot(attention_label, num_classes=3).float()
                pseudo_attention_loss_vector = bce_loss(meta_attention_distribution, attention_label)
                # pseudo_attention_loss_vector = pseudo_attention_loss_vector.mean(dim=1)
                pseudo_attention_loss_vector_reshape = pseudo_attention_loss_vector.unsqueeze(dim=-1)
                # pseudo_attention_loss_vector_reshape = torch.reshape(pseudo_attention_loss_vector, (-1, 1))
                pseudo_attention_weight = meta_net(pseudo_attention_loss_vector_reshape.data)  # [pair_num, 3, 1]
                # norm_a = torch.sum(pseudo_attention_weight)
                # if norm_a != 0:
                    # pseudo_attention_weight = pseudo_attention_weight / norm_a
                # pseudo_attention_loss = torch.sum(pseudo_attention_weight * pseudo_attention_loss_vector_reshape)
                # pseudo_attention_loss = torch.mean(pseudo_attention_weight * pseudo_attention_loss_vector_reshape)
                pseudo_attention_loss_tensor = pseudo_attention_weight * pseudo_attention_loss_vector_reshape  # [pair_num, 3, 1]
                pseudo_spatial_loss_vector = bce_loss(meta_spatial_distribution, spatial_label)  # [pair_num, 6]
                # pseudo_spatial_loss_vector = pseudo_spatial_loss_vector.mean(dim=1)  # [pair_num]
                # pseudo_spatial_loss_vector_reshape = torch.reshape(pseudo_spatial_loss_vector, (-1, 1))
                pseudo_spatial_loss_vector_reshape = pseudo_spatial_loss_vector.unsqueeze(dim=-1)
                pseudo_spatial_weight = meta_net(pseudo_spatial_loss_vector_reshape.data)  # [pair_num, 6, 1]
                # norm_s = torch.sum(pseudo_attention_weight)
                # if norm_s != 0:
                    # pseudo_spatial_weight = pseudo_spatial_weight / norm_s
                # pseudo_spatial_loss = torch.sum(pseudo_spatial_weight * pseudo_spatial_loss_vector_reshape)
                # pseudo_spatial_loss = torch.mean(pseudo_spatial_weight * pseudo_spatial_loss_vector_reshape)
                pseudo_spatial_loss_tensor = pseudo_spatial_weight * pseudo_spatial_loss_vector_reshape  # [pair_num, 6, 1]
                pseudo_contact_loss_vector = bce_loss(meta_contact_distribution, contact_label)  # [pair_num, 17]
                # pseudo_contact_loss_vector = pseudo_contact_loss_vector.mean(dim=1)
                # pseudo_contact_loss_vector_reshape = torch.reshape(pseudo_contact_loss_vector, (-1, 1))
                pseudo_contact_loss_vector_reshape = pseudo_contact_loss_vector.unsqueeze(dim=-1)
                pseudo_contact_weight = meta_net(pseudo_contact_loss_vector_reshape.data)
                # norm_c = torch.sum(pseudo_contact_weight)
                # if norm_c != 0:
                    # pseudo_contact_weight = pseudo_contact_weight / norm_c
                # pseudo_contact_loss = torch.sum(pseudo_contact_weight * pseudo_contact_loss_vector_reshape)
                # pseudo_contact_loss = torch.mean(pseudo_contact_weight * pseudo_contact_loss_vector_reshape)
                pseudo_contact_loss_tensor = pseudo_contact_weight * pseudo_contact_loss_vector_reshape  # [pair_num, 17, 1]
                # pseudo_loss = pseudo_attention_loss + pseudo_spatial_loss + pseudo_contact_loss
                pseudo_loss = (pseudo_attention_loss_tensor.sum() + pseudo_spatial_loss_tensor.sum() + pseudo_contact_loss_tensor.sum()) / (pseudo_attention_weight.sum() + pseudo_spatial_weight.sum() + pseudo_contact_weight.sum())
                pseudo_model.zero_grad()
                pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_model.parameters(), create_graph=True, allow_unused=True)
                if epoch > 0 and epoch % 5 == 0:
                    meta_lr = meta_lr * 0.5
                pseudo_optimizer = MetaSGD(pseudo_model, pseudo_model.parameters(), lr=meta_lr)
                pseudo_optimizer.load_state_dict(optimizer.state_dict())
                pseudo_optimizer.meta_step(pseudo_grads)
                # meta_model.update_params(0.1, source_params=grads)
                # for tgt, src in zip(meta_model.
                del pseudo_grads
                # 2. Comute grads of eps on meta validation data
                # val_data = next(val_iter)

                val_feat = val_data['global_output']
                val_feat = torch.from_numpy(val_feat).float().to(gpu_device)

                val_pred = pseudo_model(val_feat)
                val_attention_distribution = val_pred["attention_distribution"]  # [pair_num, 3]
                val_spatial_distribution = val_pred["spatial_distribution"]  # [pair_num, 6]
                val_contact_distribution = val_pred["contacting_distribution"]  # [pair_num, 17]

                val_dist = torch.cat((val_attention_distribution, val_spatial_distribution, val_contact_distribution), dim=1)
                val_gt = torch.from_numpy(val_data['gt']).long().to(device=gpu_device)
                # val_attention_label = torch.from_numpy(val_data["attention_gt"]).long().to(device=val_attention_distribution.device).squeeze()  # [pair_num]
                # val_attention_label = torch.nn.functional.one_hot(val_attention_label, num_classes=3).float()  # [pair_num, 3]
                # val_spatial_label = torch.from_numpy(val_data["spatial_gt"]).float().to(device=val_attention_distribution.device)  # [pair_num, 6]
                # try:
                #     val_contact_label = torch.from_numpy(val_data["contacting_gt"]).float().to(device=val_attention_distribution.device)  # [pair, 17]
                # except KeyError:
                #     val_contact_label = torch.from_numpy(val_data["contact_gt"]).float().to(device=val_attention_distribution.device)
                
                # get loss weights for attention, spatial and contact in meta validation set
                # attention_weights = torch.zeros(3, dtype=torch.float).to(device=gpu_device)
                # att_output, att_counts = val_attention_label.unique(return_counts=True)
                # att_output = att_output.type(torch.long)  # tensors used as indices must be long, byte or bool tensors
                # for idx, count in zip(att_output, att_counts):
                #     attention_weights[idx] = count
                # for idx in range(attention_weights.shape[0]):
                #     if attention_weights[idx] != 0:
                #         attention_weights[idx] = 1.0 / attention_weights[idx]
                # attention_weights /= attention_weights.sum()  # normalize
                
                # spatial_weights = torch.zeros(6, dtype=torch.float).to(device=gpu_device)
                # for this_sample in val_spatial_label:
                #     for this_index, this_label in enumerate(this_sample):
                #         if this_label == 1:
                #             spatial_weights[this_index] += 1
                # for idx in range(spatial_weights.shape[0]):
                #     if spatial_weights[idx] != 0:
                #         spatial_weights[idx] = 1 / spatial_weights[idx]
                # spatial_weights /= spatial_weights.sum()

                # contact_weights = torch.zeros(17, dtype=torch.float).to(device=gpu_device)
                # for this_sample in val_contact_label:
                #     for this_index, this_label in enumerate(this_sample):
                #         if this_label == 1:
                #             contact_weights[this_index] += 1
                # for idx in range(contact_weights.shape[0]):
                #     if contact_weights[idx] != 0:
                #         contact_weights[idx] = 1.0 / contact_weights[idx]
                # contact_weights /= contact_weights.sum()

                ce_loss.weight = None  # weight is applied on the GT class in each sample's softmax, so cannot use torch.sum() here
                ce_loss.reduction = 'mean'  # default, mean on each sample
                # bce_loss.reduction = 'none'
                # meta_val_losses = {}
                # meta_val_losses["attention_relation_loss"] = ce_loss(val_attention_distribution, val_attention_label)
                # val_attention_loss = bce_loss(val_attention_distribution, val_attention_label) * val_attention_weight  # [pair_num, class_num]

                # meta_val_losses["spatial_relation_loss"] = (bce_loss(val_spatial_distribution, val_spatial_label) * spatial_weights).sum(dim=1).mean()  # first weighted sum on class dim then mean on each sample, follow CE loss weight (reduction='mean'), mean on sample dim
                # meta_val_losses["contact_relation_loss"] = (bce_loss(val_contact_distribution, val_contact_label) * contact_weights).sum(dim=1).mean()  # [pair_num, class_num] * [class_num] = [pair_num, class_num], https://blog.csdn.net/weixin_39228381/article/details/108982594
                # val_spatial_loss = bce_loss(val_spatial_distribution, val_spatial_label) * val_spatial_weight  # [pair_num, 6]
                # val_contact_loss = bce_loss(val_contact_distribution, val_contact_label) * val_contact_weight  # [pair_num, 17]
                # meta_val_loss = sum(meta_val_losses.values())
                # meta_val_loss = (val_attention_loss.sum(dim=1) + val_spatial_loss.sum(dim=1) + val_contact_loss.sum(dim=1)).mean(dim=0)
                # wandb.log({'val attention loss': val_attention_loss.sum(1).mean()})
                # wandb.log({'val spatial loss': val_spatial_loss.sum(1).mean()})
                # wandb.log({'val contact loss': val_contact_loss.sum(1).mean()})
                meta_val_loss = ce_loss(val_dist, val_gt)
                wandb.log({'val loss': meta_val_loss})
                y_true = val_gt.data.cpu().numpy()
                _, y_pred = val_dist.data.cpu().topk(1, dim=1)
                y_pred = y_pred.numpy().squeeze()
                # val_attention_preds = F.one_hot(val_attention_distribution.argmax(dim=1), num_classes=3)  # [pair_num, 3]
                # val_spatial_preds = torch.where(val_spatial_distribution>0.5, torch.tensor(1.0).to(gpu_device), torch.tensor(0.0).to(gpu_device)).long()  # binarize sigmoid output
                # val_contact_preds = torch.where(val_contact_distribution>0.5, torch.tensor(1.0).to(gpu_device), torch.tensor(0.0).to(gpu_device)).long()  # [pair_num, 17]
                # val_preds = torch.cat((val_attention_preds, val_spatial_preds, val_contact_preds), dim=1).data.cpu().numpy()  # (pair_num, 26)
                # val_gts = torch.cat((val_attention_label, val_spatial_label, val_contact_label), dim=1).data.cpu().numpy()  # (pair_num, 26)
                # val_recall = metrics.recall_score(val_gts, val_preds, average='micro')
                val_mean_recall = metrics.recall_score(y_true, y_pred, average='macro')  # mean recall
                wandb.log({'val mean recall': val_mean_recall})
                # val_all_recalls = metrics.recall_score(val_gts, val_preds, average=None)
                # ce_loss.weight = None
                # ce_loss.reduction = 'mean'
                # bce_loss.reduction = 'mean'
                # val_attention_loss = ce_loss(val_attention_distribution, val_attention_label)
                # val_spatial_loss = bce_loss(val_spatial_distribution, val_spatial_label)
                # val_contact_loss = bce_loss(val_contact_distribution, val_contact_label)
                # meta_val_loss = val_attention_loss + val_spatial_loss + val_contact_loss
                # https://discuss.pytorch.org/t/runtimeerror-trying-to-backward-through-the-graph-a-second-time-but-the-buffers-have-already-been-freed-specify-retain-graph-true-when-calling-backward-the-first-time/6795/2
                meta_optimizer.zero_grad()
                meta_val_loss.backward()
                meta_optimizer.step()

            pred = model(feat)
            attention_distribution = pred["attention_distribution"]  # [pair_num, 3]
            spatial_distribution = pred["spatial_distribution"]  # [pair_num, 6]
            contact_distribution = pred["contacting_distribution"]  # [pair_num, 17]

            losses = {}
            # ce_loss.weight = None
            # ce_loss.reduction = 'none'
            bce_loss.reduction = 'none'
            # attention_loss_vector = ce_loss(attention_distribution, attention_label)  # [pair_num]
            attention_loss_vector = bce_loss(attention_distribution, attention_label)
            # attention_loss_vector = attention_loss_vector.mean(dim=1)
            wandb.log({'v0_attention_loss': attention_loss_vector.mean()})  # log original loss
            # attention_loss_vector_reshape = torch.reshape(attention_loss_vector, (-1, 1))  # [pair_num, 1]
            attention_loss_vector_reshape = attention_loss_vector.unsqueeze(dim=-1)

            spatial_loss_vector = bce_loss(spatial_distribution, spatial_label)  # [pair_num, class_num]
            # spatial_loss_vector = spatial_loss_vector.mean(dim=1)  # [pair_num, 1]
            wandb.log({'v0_spatial_loss': spatial_loss_vector.mean()})
            # spatial_loss_vector_reshape = torch.reshape(spatial_loss_vector, (-1, 1))
            spatial_loss_vector_reshape = spatial_loss_vector.unsqueeze(dim=-1)

            contact_loss_vector = bce_loss(contact_distribution, contact_label)  # [pair_num, class_num]
            # contact_loss_vector = contact_loss_vector.mean(dim=1)
            wandb.log({'v0_contact_loss': contact_loss_vector.mean()})
            # contact_loss_vector_reshape = torch.reshape(contact_loss_vector, (-1, 1))  # [pair_num, 1]
            contact_loss_vector_reshape = contact_loss_vector.unsqueeze(dim=-1)
            with torch.no_grad():
                attention_weight = meta_net(attention_loss_vector_reshape)
                spatial_weight = meta_net(spatial_loss_vector_reshape)
                contact_weight = meta_net(contact_loss_vector_reshape)
            
            # attention_loss = (attention_weight * attention_loss_vector_reshape).mean()
            # spatial_loss = (spatial_weight * spatial_loss_vector_reshape).mean()
            # contact_loss = (contact_weight * contact_loss_vector_reshape).mean()
            # if attention_weight.sum() != 0:
            #     attention_weight = attention_weight / attention_weight.sum()
            # if spatial_weight.sum() != 0:
            #     spatial_weight = spatial_weight / spatial_weight.sum()
            # if contact_weight.sum() != 0:
            #     contact_weight = contact_weight / contact_weight.sum()
            # attention_loss = torch.sum(attention_weight * attention_loss_vector_reshape)
            # spatial_loss = torch.sum(spatial_weight * spatial_loss_vector_reshape)
            # contact_loss = torch.sum(contact_weight * contact_loss_vector_reshape)
            # loss = attention_loss + spatial_loss + contact_loss

            attention_loss = (attention_weight * attention_loss_vector_reshape)
            spatial_loss = (spatial_weight * spatial_loss_vector_reshape)
            contact_loss = (contact_weight * contact_loss_vector_reshape)
            loss = (attention_loss.sum() + spatial_loss.sum() + contact_loss.sum()) / (attention_weight.sum() + spatial_weight.sum() + contact_weight.sum())

            wandb.log({'loss': loss})
            # wandb.log({'attention loss': attention_loss})
            # wandb.log({'spatial loss': spatial_loss})
            # wandb.log({'contact loss': contact_loss})
            optimizer.zero_grad()
            loss.backward()
            # scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            # scaler.step(optimizer)
            # scaler.update()
            # profiler.step()

            # tr.append(pd.Series({x: y.item() for x, y in losses.items()}))

            if b % log_freq == 0 and b >= log_freq:
                time_per_batch = (time.time() - start) / log_freq
                print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b, len(dataloader_train),
                                                                                    time_per_batch, len(dataloader_train) * time_per_batch / 60))
                print(f'lr: {optimizer.param_groups[0]["lr"]}')
                start = time.time()

        # torch.save({"state_dict": model.state_dict()}, os.path.join(conf.save_path, "model_{}.tar".format(epoch)))
        ckpt_save_path = os.path.join('results', wandb.run.id)
        if not os.path.exists(ckpt_save_path):
            os.makedirs(ckpt_save_path)
        torch.save({"state_dict": model.state_dict()}, os.path.join(ckpt_save_path, "model_{}.tar".format(epoch)))
        print("*" * 40)
        print("save the checkpoint after {} epochs".format(epoch))

        model.eval()
        with torch.no_grad():
            for b in range(len(dataloader_test)):
                # if b >= 2: continue
                data = next(test_iter)
                feat = data['global_output']
                feat = torch.from_numpy(feat).float().to(gpu_device)
                gt_annotation = data['gt_annotation']
                pred = model(feat)
                pred.update(data)
                evaluator.evaluate_scene_graph(gt_annotation, pred)
                if epoch == conf.nepoch-1:  # only evaluate with semi and no on last epoch
                    evaluator_semi.evaluate_scene_graph(gt_annotation, pred)
                    evaluator_no.evaluate_scene_graph(gt_annotation, pred)
            print('-----------', flush=True)
        # score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
        # score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20]['all_rel_cates'])
        # score = np.mean(list(evaluator.result_dict[conf.mode + "_recall"][20].values()))
        print('-------------------------with constraint-------------------------------')
        with_outputs = evaluator.print_stats()
        if epoch == conf.nepoch-1:
            print('-------------------------semi constraint-------------------------------')
            semi_outputs = evaluator_semi.print_stats()
            print('-------------------------no constraint-------------------------------')
            no_outputs = evaluator_no.print_stats()
        score = with_outputs[10][0]
        wandb.log({'mR@10': score})
        # wandb.log({'mR@20': score})
        # wandb.log({'mR@50': np.mean(list(evaluator.result_dict[conf.mode + "_recall"][50].values()))})
        wandb.log({'epoch': epoch})
        # wandb log recall table
        # recall_table = wandb.Table(
        #         columns=['constraint', 'topK', 'mean', 'all', 'attention mean', 'spatial mean', 'contact mean', 
        #         'not_looking_at', 'looking_at',	'unsure', 'in_front_of', 'on_the_side_of', 'beneath', 'behind', 
        #         'above', 'have_it_on_the_back', 'holding', 'not_contacting', 'touching', 'sitting_on', 'in', 
        #         'leaning_on', 'other_relationship', 'standing_on', 'wearing', 'drinking_from', 'covered_by', 
        #         'carrying', 'eating', 'lying_on', 'writing_on', 'wiping', 'twisting']
        #         )
        # for k, v in with_outputs.items():
        #     recall_table.add_data(*(['with'] + [k] + v))  # https://docs.wandb.ai/guides/data-vis/log-tables#add-data
        # for k, v in semi_outputs.items():
        #     recall_table.add_data(*(['semi'] + [k] + v))
        # for k, v in no_outputs.items():
        #     recall_table.add_data(*(['no'] + [k] + v))
        # wandb.log({"table_key": recall_table})
        # print('scheduler.step(mR@K)')
       
        evaluator.reset_result()
        evaluator_semi.reset_result()
        evaluator_no.reset_result()
        scheduler.step(score)
        # scheduler.step(epoch)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    main()
