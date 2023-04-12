from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
from torch.optim import lr_scheduler
import yaml
import time
import cv2
import h5py
import random
import logging
import argparse
import numpy as np
from PIL import Image
from attrdict import AttrDict
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from data_provider_ours import Provider, Validation
# from data_provider_ours_aug import Provider, Validation
from utils.show import show_affs, val_show
from model_2d.unet2d_residual import ResidualUNet2D_embedding as ResidualUNet2D_affs
# from unet2d_residual_attention import ResidualUNet2D_embedding_attention as ResidualUNet2D_affs
from utils.utils import setup_seed
from loss.loss2 import WeightedMSE, WeightedBCE
from loss.loss2 import MSELoss, BCELoss, BCE_loss_func
from loss.loss_embedding_mse import embedding_loss,embedding2affs
from loss.loss_discriminative import discriminative_loss
# from utils.evaluate import BestDice, AbsDiffFGLabels
from lib.evaluate.CVPPP_evaluate import BestDice, AbsDiffFGLabels, SymmetricBestDice
from utils.seg_mutex import seg_mutex
from utils.affinity_ours import multi_offset
from utils.emb2affs import embeddings_to_affinities
from postprocessing import merge_small_object, merge_func
from data.data_segmentation import relabel
from utils.cluster import cluster_ms
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
from utils.utils_rag_fc6 import construct_graph, calculate_self_node_similarity, calculate_mutual_node_similarity
import warnings

warnings.filterwarnings("ignore")


def init_project(cfg):
    def init_logging(path):
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            datefmt='%m-%d %H:%M',
            filename=path,
            filemode='w')

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    # seeds
    setup_seed(cfg.TRAIN.random_seed)
    if cfg.TRAIN.if_cuda:
        if torch.cuda.is_available() is False:
            raise AttributeError('No GPU available')

    prefix = cfg.time
    if cfg.TRAIN.resume:
        model_name = cfg.TRAIN.model_name
    else:
        model_name = prefix + '_' + cfg.NAME
    cfg.cache_path = os.path.join(cfg.TRAIN.cache_path, model_name)
    cfg.save_path = os.path.join(cfg.TRAIN.save_path, model_name)
    # cfg.record_path = os.path.join(cfg.TRAIN.record_path, 'log')
    cfg.record_path = os.path.join(cfg.save_path, model_name)
    cfg.valid_path = os.path.join(cfg.save_path, 'valid')
    if cfg.TRAIN.resume is False:
        if not os.path.exists(cfg.cache_path):
            os.makedirs(cfg.cache_path)
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path)
        if not os.path.exists(cfg.record_path):
            os.makedirs(cfg.record_path)
        if not os.path.exists(cfg.valid_path):
            os.makedirs(cfg.valid_path)
    init_logging(os.path.join(cfg.record_path, prefix + '.log'))
    logging.info(cfg)
    writer = SummaryWriter(cfg.record_path)
    writer.add_text('cfg', str(cfg))
    return writer


def load_dataset(cfg):
    print('Caching datasets ... ', end='', flush=True)
    t1 = time.time()
    train_provider = Provider('train', cfg)
    if cfg.TRAIN.if_valid:
        valid_provider = Validation(cfg, mode='validation')
    else:
        valid_provider = None
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider


def build_model(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')

    model = ResidualUNet2D_affs(in_channels=cfg.MODEL.input_nc,
                                out_channels=cfg.MODEL.output_nc,
                                nfeatures=cfg.MODEL.filters,
                                emd=cfg.MODEL.emd,
                                if_sigmoid=cfg.MODEL.if_sigmoid,
                                show_feature=True).to(device)

    model_T = ResidualUNet2D_affs(in_channels=cfg.MODEL_T.input_nc,
                                out_channels=cfg.MODEL_T.output_nc,
                                nfeatures=cfg.MODEL_T.filters,
                                emd=cfg.MODEL_T.emd,
                                if_sigmoid=cfg.MODEL_T.if_sigmoid,
                                show_feature=True).to(device)
                                
    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(model)
        else:
            raise AttributeError(
                'Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    if cfg.TRAIN.if_KD:
        print('Load Teacher pretrained Model')
        model_path = os.path.join(cfg.TRAIN.model_T_path, 'model-%06d.ckpt' % cfg.TRAIN.model_T_id)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            model_T.load_state_dict(checkpoint['model_weights'])
            # optimizer.load_state_dict(checkpoint['optimizer_weights'])
        else:
            raise AttributeError('No checkpoint found at %s' % model_path)
        print('Done (time: %.2fs)' % (time.time() - t1))
        print('valid %d' % checkpoint['current_iter'])
        for k, v in model_T.named_parameters():
            v.requires_grad = False
    return model, model_T


def resume_params(cfg, model, optimizer, resume):
    if resume:
        t1 = time.time()
        model_path = os.path.join(cfg.save_path, 'model-%06d.ckpt' % cfg.TRAIN.model_id)

        print('Resuming weights from %s ... ' % model_path, end='', flush=True)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_weights'])
            # optimizer.load_state_dict(checkpoint['optimizer_weights'])
        else:
            raise AttributeError('No checkpoint found at %s' % model_path)
        print('Done (time: %.2fs)' % (time.time() - t1))
        print('valid %d' % checkpoint['current_iter'])
        return model, optimizer, checkpoint['current_iter']
    else:
        return model, optimizer, 0


def calculate_lr(iters):
    if iters < cfg.TRAIN.warmup_iters:
        current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(float(iters) / cfg.TRAIN.warmup_iters,
                                                                  cfg.TRAIN.power) + cfg.TRAIN.end_lr
    else:
        if iters < cfg.TRAIN.decay_iters:
            current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(
                1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
        else:
            current_lr = cfg.TRAIN.end_lr
    return current_lr


def loop(cfg, train_provider, valid_provider, model, model_T, criterion, optimizer, iters, writer):
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    f_valid_txt = open(os.path.join(cfg.record_path, 'valid.txt'), 'a')
    rcd_time = []
    sum_time = 0
    sum_loss = 0
    sum_loss_aff = 0.0
    sum_loss_affinity = 0.0
    sum_loss_graph = 0.0
    sum_loss_node = 0.0
    sum_loss_edge = 0.0
    sum_loss_CI_affinity = 0.0
    sum_loss_CI_graph = 0.0
    sum_loss_CI_node = 0.0
    sum_loss_CI_edge = 0.0
    sum_loss_mask = 0.0
    device = torch.device('cuda:0')
    offsets = multi_offset(list(cfg.DATA.shifts), neighbor=cfg.DATA.neighbor)

    if cfg.TRAIN.loss_func == 'MSELoss':
        criterion = MSELoss()
    elif cfg.TRAIN.loss_func == 'BCELoss':
        criterion = BCELoss()
    elif cfg.TRAIN.loss_func == 'WeightedBCELoss':
        criterion = WeightedBCE()
    elif cfg.TRAIN.loss_func == 'WeightedMSELoss':
        criterion = WeightedMSE()
    else:
        raise AttributeError("NO this criterion")
    criterion_dis = discriminative_loss
    criterion_mask = BCE_loss_func
    valid_mse = MSELoss()
    valid_bce = BCELoss()

    lr_strategies = ['steplr', 'multi_steplr', 'explr', 'lambdalr']
    if cfg.TRAIN.lr_mode == 'steplr':
        print('Step LR')
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.TRAIN.step_size, gamma=cfg.TRAIN.gamma)
    elif cfg.TRAIN.lr_mode == 'multi_steplr':
        print('Multi step LR')
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000, 150000],
                                                            gamma=cfg.TRAIN.gamma)
    elif cfg.TRAIN.lr_mode == 'explr':
        print('Exponential LR')
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    elif cfg.TRAIN.lr_mode == 'lambdalr':
        print('Lambda LR')
        lambda_func = lambda epoch: (1.0 - epoch / cfg.TRAIN.total_iters) ** 0.9
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)
    else:
        print('Other LR scheduler')

    while iters <= cfg.TRAIN.total_iters:
        # train
        model.train()
        model_T.eval()

        iters += 1
        t1 = time.time()
        batch_data = train_provider.next()
        inputs = batch_data['image'].cuda()
        target = batch_data['affs'].cuda()
        weightmap = batch_data['wmap'].cuda()
        target_ins = batch_data['seg'].cuda()
        affs_mask = batch_data['mask'].cuda()

        if cfg.TRAIN.lr_mode in lr_strategies:
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = cfg.TRAIN.base_lr

        optimizer.zero_grad()
        if cfg.MODEL.model_type==2:
            x5, x_emb1, x_emb2, x_emb3, embedding, pred_mask = model(inputs)
        else:
            x5, x_emb1, x_emb2, x_emb3, embedding = model(inputs)

        if cfg.MODEL_T.model_type==2:
            x5_T, x_emb1_T, x_emb2_T, x_emb3_T, embedding_T, pred_mask_T = model_T(inputs)
        else:
            x5_T, x_emb1_T, x_emb2_T, x_emb3_T, embedding_T = model_T(inputs)

        ##############################
        # LOSS
        # loss = criterion(pred, target, weightmap)

        loss_aff, pred,_ = embedding_loss(embedding, target, weightmap, affs_mask, criterion, offsets,
                                              affs0_weight=cfg.TRAIN.dis_weight)
        _, pred_T,_ = embedding_loss(embedding_T, target, weightmap, affs_mask, criterion, offsets,
                                              affs0_weight=cfg.TRAIN.dis_weight)

        s1 = torch.prod(torch.tensor(pred.size()[-2:]).float())
        s2 = pred.size()[0]
        norm_term = (s1 * s2).cuda()
        loss_aff_KD = torch.sum(((pred - pred_T)) ** 2)/norm_term * cfg.TRAIN.affinity_weight

        loss_graph = torch.zeros(1).to(device)
        loss_node = torch.zeros(1).to(device)
        loss_edge = torch.zeros(1).to(device)

        
        for emb,emb_T in zip([embedding,x_emb3,x_emb2],[embedding_T,x_emb3_T,x_emb2_T]):
            if emb_T.shape != emb.shape:
                transform = nn.Conv2d(emb.shape[1], emb_T.shape[1], 1, bias=False).to(device)
                transform.weight.data.uniform_(-0.005, 0.005)
                emb = transform(emb)

            if emb_T.shape != emb.shape:
                pred_tmp = embedding2affs(emb, offsets)
                pred_T_tmp = embedding2affs(emb_T, offsets)

                s1 = torch.prod(torch.tensor(pred_tmp.size()[-2:]).float())
                s2 = pred_tmp.size()[0]
                norm_term = (s1 * s2).cuda()
                loss_aff_KD_tmp = torch.sum(((pred_tmp - pred_T_tmp)) ** 2)/norm_term * cfg.TRAIN.affinity_weight
            else:
                loss_aff_KD_tmp = torch.zeros(1).to(device)

            h_list, edge_list = construct_graph(target_ins, [emb], if_adjacent=cfg.TRAIN.if_neighbor)

            h_list_T, edge_list_T = construct_graph(target_ins, [emb_T])



            loss_graph_tmp, loss_node_tmp, loss_edge_tmp = calculate_mutual_node_similarity(h_list_T, h_list, edge_list,
                                                        if_node=cfg.TRAIN.if_node,
                                                        if_edge_discrepancy=cfg.TRAIN.if_edge_discrepancy,
                                                        if_edge_relation=cfg.TRAIN.if_edge_relation,
                                                        if_neighbor=cfg.TRAIN.if_neighbor,
                                                        node_weight = cfg.TRAIN.node_weight,
                                                        edge_weight = cfg.TRAIN.edge_weight)
            loss_graph = loss_graph + loss_graph_tmp
            loss_node = loss_node + loss_node_tmp
            loss_edge = loss_edge + loss_edge_tmp
            loss_aff_KD = loss_aff_KD + loss_aff_KD_tmp


        loss = loss_aff + loss_aff_KD + loss_graph 


        loss.backward()
        # pred = F.relu(pred)
        ##############################
        optimizer.step()
        if cfg.TRAIN.weight_decay is not None:
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-cfg.TRAIN.weight_decay * group['lr'], param.data)

        if cfg.TRAIN.lr_mode in lr_strategies:
            lr_scheduler.step()

        sum_loss += loss.item()
        sum_loss_aff += loss_aff.item()
        sum_loss_affinity += loss_aff_KD.item()
        sum_loss_graph += loss_graph.item()
        sum_loss_node += loss_node.item()
        sum_loss_edge += loss_edge.item()

        sum_loss_mask = 0.0
        # sum_loss_mask += loss_mask.item()
        sum_time += time.time() - t1

        # log train
        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            rcd_time.append(sum_time)
            if iters == 1:
                logging.info(
                    'step %d, loss=%.6f, loss_aff=%.6f,loss_affinity=%.6f,loss_graph=%.6f, loss_node=%.6f, loss_edge=%.6f, loss_CI_affinity=%6f, loss_CI_graph=%6f, loss_CI_node=%6f, loss_CI_edge=%6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                    % (iters, sum_loss, sum_loss_aff, sum_loss_affinity,sum_loss_graph,sum_loss_node, sum_loss_edge, sum_loss_CI_affinity, sum_loss_CI_graph, sum_loss_CI_node, sum_loss_CI_edge, current_lr, sum_time,
                       (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss, iters)

            else: 
                logging.info(
                    'step %d, loss=%.6f, loss_aff=%.6f,loss_affinity=%.6f,loss_graph=%.6f, loss_node=%.6f, loss_edge=%.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)' \
                    % (iters, sum_loss / cfg.TRAIN.display_freq, \
                       sum_loss_aff / cfg.TRAIN.display_freq, \
                       sum_loss_affinity / cfg.TRAIN.display_freq, \
                       sum_loss_graph / cfg.TRAIN.display_freq, \
                       sum_loss_node / cfg.TRAIN.display_freq, \
                       sum_loss_edge / cfg.TRAIN.display_freq, \
                        current_lr, sum_time, \
                       (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                # logging.info('step %d, loss_dis=%.6f, loss_emd=%.6f' % (iters, loss_embedding_dis, loss_embedding))
                writer.add_scalar('sum_loss', sum_loss / cfg.TRAIN.display_freq, iters)
                writer.add_scalar('sum_loss_aff', sum_loss_aff / cfg.TRAIN.display_freq, iters)
                writer.add_scalar('sum_loss_affinity', sum_loss_affinity / cfg.TRAIN.display_freq, iters)
                writer.add_scalar('sum_loss_graph', sum_loss_graph / cfg.TRAIN.display_freq, iters)
                writer.add_scalar('sum_loss_node', sum_loss_node / cfg.TRAIN.display_freq, iters)
                writer.add_scalar('sum_loss_edge', sum_loss_edge / cfg.TRAIN.display_freq, iters)
            # f_loss_txt.write('step = ' + str(iters) + ', loss = ' + str(sum_loss / cfg.TRAIN.display_freq))
            f_loss_txt.write('step = %d, loss = %.6f, loss_aff=%.6f,l oss_affinity=%.6f,loss_graph=%.6f, loss_node=%.6f, loss_edge=%.6f' % \
                             (iters, sum_loss / cfg.TRAIN.display_freq, sum_loss_aff / cfg.TRAIN.display_freq,
                              sum_loss_affinity / cfg.TRAIN.display_freq,sum_loss_graph / cfg.TRAIN.display_freq,sum_loss_node / cfg.TRAIN.display_freq,sum_loss_edge / cfg.TRAIN.display_freq))
            f_loss_txt.write('\n')
            f_loss_txt.flush()
            sys.stdout.flush()
            sum_time = 0.0
            sum_loss = 0.0
            sum_loss_aff = 0.0
            sum_loss_affinity = 0.0
            sum_loss_graph = 0.0
            sum_loss_node = 0.0
            sum_loss_edge = 0.0
            sum_loss_mask = 0.0
        # display
        if iters % cfg.TRAIN.valid_freq == 0 or iters == 1:
            show_affs(iters, batch_data['image'], pred[:, -1], batch_data['affs'][:, -1], cfg.cache_path)

        # valid
        if cfg.TRAIN.if_valid:
            if iters % cfg.TRAIN.save_freq == 0 or iters == 1:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                model.eval()
                dataloader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                                         shuffle=False, drop_last=False, pin_memory=True)
                losses_valid = []
                dice = []
                diff = []
                all_voi = []
                all_arand = []
                all_mse = []
                all_bce = []
                for k, batch in enumerate(dataloader, 0):
                    batch_data = batch
                    inputs = batch_data['image'].cuda()
                    target = batch_data['affs'].cuda()
                    weightmap = batch_data['wmap'].cuda()
                    target_ins = batch_data['seg'].cuda().float()
                    affs_mask = batch_data['mask'].cuda().float()
                    with torch.no_grad():
                        x5, x_emb1, x_emb2, x_emb3, embedding, _ = model(inputs)

                    loss_embedding, pred, _ = embedding_loss(embedding, target, weightmap, affs_mask, criterion,
                                                             offsets, affs0_weight=cfg.TRAIN.dis_weight)
                    tmp_loss = loss_embedding
                    losses_valid.append(tmp_loss.item())
                    # pred = F.relu(pred)
                    temp_mse = valid_mse(pred * affs_mask, target * affs_mask)
                    temp_bce = valid_bce(torch.clamp(pred, 0.0, 1.0) * affs_mask, target * affs_mask)
                    all_mse.append(temp_mse.item())
                    all_bce.append(temp_bce.item())
                    out_affs = np.squeeze(pred.data.cpu().numpy())

                    # post-processing
                    gt_ins = np.squeeze(batch_data['seg'].numpy()).astype(np.uint8)
                    gt_mask = gt_ins.copy()
                    gt_mask[gt_mask != 0] = 1
                    if cfg.TEST.if_mutex == True:
                        pred_seg = seg_mutex(out_affs, offsets=offsets, strides=list(cfg.DATA.strides),
                                             mask=gt_mask).astype(np.uint16)
                        print('PostPorcess mode: Embedding2Affinity-Mutex')
                    else:
                        embedding_np = np.squeeze(embedding.data.cpu().numpy())
                        pred_seg = cluster_ms(embedding_np, bandwidth=0.5, semantic_mask=gt_mask).astype(np.uint16)

                    pred_seg = merge_func(pred_seg)
                    pred_seg = relabel(pred_seg)
                    pred_seg = pred_seg.astype(np.uint16)
                    gt_ins = gt_ins.astype(np.uint16)

                    # evaluate
                    temp_dice = SymmetricBestDice(pred_seg, gt_ins)
                    temp_diff = AbsDiffFGLabels(pred_seg, gt_ins)
                    arand = adapted_rand_ref(gt_ins, pred_seg, ignore_labels=(0))[0]
                    voi_split, voi_merge = voi_ref(gt_ins, pred_seg, ignore_labels=(0))
                    voi_sum = voi_split + voi_merge
                    all_voi.append(voi_sum)
                    all_arand.append(arand)
                    dice.append(temp_dice)
                    diff.append(temp_diff)
                    if k == 0:
                        affs_gt = batch_data['affs'].numpy()[0, -1]
                        val_show(iters, out_affs[-1], affs_gt, pred_seg, gt_ins, cfg.valid_path)
                epoch_loss = sum(losses_valid) / len(losses_valid)
                sbd = sum(dice) / len(dice)
                # sbd = 0.0
                dic = sum(diff) / len(diff)
                mean_voi = sum(all_voi) / len(all_voi)
                mean_arand = sum(all_arand) / len(all_arand)
                mean_mse = sum(all_mse) / len(all_mse)
                mean_bce = sum(all_bce) / len(all_bce)

                # out_affs[out_affs <= 0.5] = 0
                # out_affs[out_affs > 0.5] = 1
                # whole_f1 = f1_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - out_affs.astype(np.uint8).flatten())
                print('model-%d, valid-loss=%.6f, SBD=%.6f, DiC=%.6f, VOI=%.6f, ARAND=%.6f, MSE=%.6f, BCE=%.6f' % \
                      (iters, epoch_loss, sbd, dic, mean_voi, mean_arand, mean_mse, mean_bce), flush=True)
                writer.add_scalar('valid/epoch_loss', epoch_loss, iters)
                writer.add_scalar('valid/SBD', sbd, iters)
                writer.add_scalar('valid/DiC', dic, iters)
                writer.add_scalar('valid/VOI', mean_voi, iters)
                writer.add_scalar('valid/ARAND', mean_arand, iters)
                writer.add_scalar('valid/MSE', mean_mse, iters)
                writer.add_scalar('valid/BCE', mean_bce, iters)
                f_valid_txt.write(
                    'model-%d, valid-loss=%.6f, SBD=%.6f, DiC=%.6f, VOI=%.6f, ARAND=%.6f, MSE=%.6f, BCE=%.6f' % \
                    (iters, epoch_loss, sbd, dic, mean_voi, mean_arand, mean_mse, mean_bce))
                f_valid_txt.write('\n')
                f_valid_txt.flush()
                torch.cuda.empty_cache()

        # save
        if iters % cfg.TRAIN.save_freq == 0:
            states = {'current_iter': iters, 'valid_result': None,
                      'model_weights': model.state_dict()}
            torch.save(states, os.path.join(cfg.save_path, 'model-%06d.ckpt' % iters))
            print('***************save modol, iters = %d.***************' % (iters), flush=True)
    f_loss_txt.close()
    f_valid_txt.close()


if __name__ == "__main__":
    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='seg_inpainting', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='path to config file')
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    print('mode: ' + args.mode)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))

    from data_provider_ours import Provider, Validation

    print('*' * 20 + 'import data_provider_ours' + '*' * 20)

    timeArray = time.localtime()
    time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S', timeArray)
    print('time stamp:', time_stamp)

    cfg.path = cfg_file
    cfg.time = time_stamp

    if args.mode == 'train':
        writer = init_project(cfg)
        train_provider, valid_provider = load_dataset(cfg)
        model, model_T = build_model(cfg, writer)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999),
                                     eps=0.01, weight_decay=1e-6, amsgrad=True)
        # optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
        # optimizer = optim.Adamax(model.parameters(), lr=cfg.TRAIN.base_l, eps=1e-8)
        model, optimizer, init_iters = resume_params(cfg, model, optimizer, cfg.TRAIN.resume)
        loop(cfg, train_provider, valid_provider, model, model_T, nn.L1Loss(), optimizer, init_iters, writer)
        writer.close()
    else:
        pass
    print('***Done***')