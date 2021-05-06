import argparse
import json
import os
from pathlib import Path
from threading import Thread
import traceback

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
# from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized
import time
from torch.cuda.amp import autocast as autocast
from utils.loss import ComputeLoss

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_model(opt):
    device = select_device(opt.device, batch_size=opt.batch_size)
    # Load model
    print(device)
    model = attempt_load(opt.weights, map_location='cpu')  # load FP32 model
    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # if half:
    #     model.half()
    #     model.cuda()
    model.eval()
    return model

def get_dataloader(opt,model):
    device = select_device(opt.device, batch_size=opt.batch_size)
    # Load model
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.img_size, s=gs)  # check img_size

    # Configure
    with open(opt.data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
    check_dataset(data)  # check
    nc =  int(data['nc'])  # number of classes
    
    # no augment
    dataloader = create_dataloader(data[opt.task] , imgsz, opt.batch_size, gs, opt, pad=0.5, rect=True,
                                   prefix=colorstr(opt.task + ':'))[0]
    return dataloader,nc

def run_model(opt,model,dataloader,nc,batch_idx_range,TF=None,C_param=None):
    device = select_device(opt.device, batch_size=opt.batch_size)
    
    half = device.type != 'cpu'
    if half:
        iouv = torch.linspace(0.5, 0.95, 10).cuda()
    else:
        iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    seen = 0
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    stats, ap, ap_class = [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        if batch_i<batch_idx_range[0]:continue
        elif batch_i>=batch_idx_range[1]:break
        # perform transformation
        if TF is not None:
            tf_imgs = None
            for th_img in img:
                np_img = th_img.permute(1,2,0).numpy()
                tf_img = TF.transform(image=np_img[:,:,(2,1,0)], C_param=C_param)
                tf_img = torch.from_numpy(tf_img[:,:,(2,1,0)]).float().permute(2,0,1).unsqueeze(0)
                if tf_imgs is None:
                    tf_imgs = tf_img
                else:
                    tf_imgs = torch.cat((tf_imgs,tf_img),0)
            img = tf_imgs
        # end transformation
        if half: img = img.cuda()
        # img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if half:targets = targets.cuda()
        # targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            try:
                out, train_out = model(img, augment=opt.augment)  # inference and training outputs
            except Exception as e:
                print(traceback.format_exc())
                print(batch_i,img.shape,C_param)
            
            t0 += time_synchronized() - t

            # Run NMS
            if half:
                targets[:, 2:] *= torch.Tensor([width, height, width, height]).cuda()
            else:
                targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if opt.save_hybrid else []  # for autolabelling
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, labels=lb, multi_label=True)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Assign all predictions as incorrect
            if half:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool).cuda()
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    return map50

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def evaluate_threshold(thresh):
    from compression.deepcod import DeepCOD, orthorgonal_regularizer, init_weights
    sim_train = Simulator(train=True,use_model=True)
    sim_test = Simulator(train=False,use_model=False)

    opt = sim_train.opt
    device = select_device(opt.device, batch_size=opt.batch_size)
    half = opt.device != 'cpu'
    use_subsampling=True
    # data
    test_loader = sim_test.dataloader
    train_loader = sim_train.dataloader

    # vision app
    app_model = sim_train.model
    if half:
        app_model = app_model.cuda()
        for layer in app_model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()
    app_model.eval()

    # encoder+decoder
    gen_model = DeepCOD(use_subsampling=use_subsampling)
    gen_model.apply(init_weights)
    if half:
        gen_model = gen_model.cuda()
        for layer in gen_model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    criterion_mse = nn.MSELoss()
    scaler_g = torch.cuda.amp.GradScaler(enabled=half)
    optimizer_g = torch.optim.Adam(gen_model.parameters(), lr=0.0001)
    max_map = 0
    max_cr = 0
    thresh = torch.FloatTensor(thresh)
    if half: thresh = thresh.cuda()
    for epoch in range(1,5):
        # train
        gen_model.train()
        if half:
            iouv = torch.linspace(0.5, 0.95, 10).cuda()
        else:
            iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        nc = sim_train.nc
        seen = 0
        names = {k: v for k, v in enumerate(app_model.names if hasattr(app_model, 'names') else app_model.module.names)}
        coco91class = coco80_to_coco91_class()
        # s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        p, r, f1, mp, mr, map50, map, = 0., 0., 0., 0., 0., 0., 0.
        stats, ap, ap_class = [], [], []
        train_iter = tqdm(train_loader)
        rlcr = AverageMeter()
        for batch_i, (img, targets, paths, shapes) in enumerate(train_iter):
            img = img.type(torch.FloatTensor).cuda() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if half:targets = targets.cuda()
            nb, _, height, width = img.shape  # batch size, channels, height, width

            # generator update
            optimizer_g.zero_grad()
            with autocast():
                if use_subsampling:
                    recon,res = gen_model((img,thresh))
                else:
                    recon,r = gen_model(img)
                pred,recon_features = app_model(recon, augment=opt.augment, extract_features=True)
                _,origin_features = app_model(img, augment=opt.augment, extract_features=True)
                loss = criterion_mse(img,recon)
                loss += orthorgonal_regularizer(gen_model.encoder.sample.weight,0.0001,half)
                for origin_feat,recon_feat in zip(origin_features,recon_features):
                    if origin_feat is None:continue
                    loss += criterion_mse(origin_feat,recon_feat)
                if use_subsampling:
                    esti_cr,real_cr,std = res
                    # loss += esti_cr - 0.0001*std

            scaler_g.scale(loss).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            rlcr.update(real_cr if use_subsampling else r)

            # Run NMS
            if half:
                targets[:, 2:] *= torch.Tensor([width, height, width, height]).cuda()
            else:
                targets[:, 2:] *= torch.Tensor([width, height, width, height])

            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if opt.save_hybrid else []  # for autolabelling

            out = non_max_suppression(pred[0], conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, labels=lb, multi_label=True)


            # Statistics per image
            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                predn = pred.clone()
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

                # Assign all predictions as incorrect
                if half:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool).cuda()
                else:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels

                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                            # Append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in image
                                        break

                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].detach().cpu(), pred[:, 5].detach().cpu(), tcls))

            if batch_i%100==0 or batch_i==(len(train_loader)-1):
                metric = stat_to_map(stats,names,nc)
                if use_subsampling:
                    train_iter.set_description(
                        f"Train: {epoch:3}. Thresh: {thresh.cpu().numpy()[0]:.3f}. "
                        f"map50: {metric[3]:.2f}. map: {metric[4]:.2f}. "
                        f"MP: {metric[1]:.2f}. MR: {metric[2]:.2f}. "
                        f"loss: {loss.cpu().item():.3f}. "
                        f"cr: {rlcr.avg:.5f}. "
                        )
                else:
                    train_iter.set_description(
                        f"Train: {epoch:3}. "
                        f"map50: {metric[3]:.2f}. map: {metric[4]:.2f}. "
                        f"MP: {metric[1]:.2f}. MR: {metric[2]:.2f}. "
                        f"loss: {loss.cpu().item():.3f}. "
                        f"cr: {rlcr.avg:.5f}. "
                        )
        train_iter.close()

        # eval
        gen_model.eval()
        if half:
            iouv = torch.linspace(0.5, 0.95, 10).cuda()
        else:
            iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        nc = sim_test.nc
        seen = 0
        names = {k: v for k, v in enumerate(app_model.names if hasattr(app_model, 'names') else app_model.module.names)}
        coco91class = coco80_to_coco91_class()
        # s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        p, r, f1, mp, mr, map50, map = 0., 0., 0., 0., 0., 0., 0.
        stats, ap, ap_class = [], [], []
        test_iter = tqdm(test_loader)
        rlcr = AverageMeter()
        for batch_i, (img, targets, paths, shapes) in enumerate(test_iter):
            img = img.type(torch.FloatTensor).cuda() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if half:targets = targets.cuda()
            nb, _, height, width = img.shape  # batch size, channels, height, width

            # Run model
            with torch.no_grad():
                if use_subsampling:
                    recon,res = gen_model((img,thresh))
                else:
                    recon,r = gen_model(img)
                pred,recon_features = app_model(recon, augment=opt.augment, extract_features=True)
                _,origin_features = app_model(img, augment=opt.augment, extract_features=True)
                loss = criterion_mse(img,recon)
                loss += orthorgonal_regularizer(gen_model.encoder.sample.weight,0.0001,half)
                for origin_feat,recon_feat in zip(origin_features,recon_features):
                    if origin_feat is None:continue
                    loss += criterion_mse(origin_feat,recon_feat)
                if use_subsampling:
                    esti_cr,real_cr,std = res
                    loss += esti_cr - 0.01*std

            rlcr.update(real_cr if use_subsampling else r)
                
            # Run NMS
            if half:
                targets[:, 2:] *= torch.Tensor([width, height, width, height]).cuda()
            else:
                targets[:, 2:] *= torch.Tensor([width, height, width, height])

            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if opt.save_hybrid else []  # for autolabelling

            out = non_max_suppression(pred[0], conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, labels=lb, multi_label=True)


            # Statistics per image
            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                predn = pred.clone()
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

                # Assign all predictions as incorrect
                if half:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool).cuda()
                else:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels

                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                            # Append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in image
                                        break

                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].detach().cpu(), pred[:, 5].detach().cpu(), tcls))

            if batch_i%100==0 or batch_i==(len(test_loader)-1):
                metric = stat_to_map(stats,names,nc)
                if use_subsampling:
                    test_iter.set_description(
                        f"Test: {epoch:3}. Thresh: {thresh.cpu().numpy()[0]:.3f}. "
                        f"map50: {metric[3]:.2f}. map: {metric[4]:.2f}. "
                        f"MP: {metric[1]:.2f}. MR: {metric[2]:.2f}. "
                        f"loss: {loss.cpu().item():.3f}. "
                        f"cr: {rlcr.avg:.5f}. "
                        )
                else:
                    test_iter.set_description(
                        f"Test: {epoch:3}. "
                        f"map50: {metric[3]:.2f}. map: {metric[4]:.2f}. "
                        f"MP: {metric[1]:.2f}. MR: {metric[2]:.2f}. "
                        f"loss: {loss.cpu().item():.3f}. "
                        f"cr: {rlcr.avg:.5f}. "
                        )
        test_iter.close()
        if metric[3] > max_map:
            max_map = metric[3]
            max_cr = rlcr.avg
    return float(max_map),max_cr

# 1. get an average estimate
# 2.1 finetune CCO-S:use one selected cfg per model
# 2.2 finetune CCO-A:use a set of selected cfgs
def deepcod_main():
    from compression.deepcod import DeepCOD, orthorgonal_regularizer, init_weights
    sim_train = Simulator(train=True,use_model=True)
    sim_test = Simulator(train=False,use_model=False)

    opt = sim_train.opt
    device = select_device(opt.device, batch_size=opt.batch_size)
    half = opt.device != 'cpu'
    use_subsampling=True
    # data
    test_loader = sim_test.dataloader
    train_loader = sim_train.dataloader

    # vision app
    app_model = sim_train.model
    if half:
        app_model = app_model.cuda()
        for layer in app_model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()
    app_model.eval()

    # encoder+decoder
    PATH = 'backup/CCO-A.pth' if use_subsampling else 'backup/deepcod_soft_c8.pth'
    gen_model = DeepCOD(use_subsampling=use_subsampling)
    gen_model.apply(init_weights)
    if half:
        gen_model = gen_model.cuda()
        for layer in gen_model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    criterion_mse = nn.MSELoss()
    scaler_g = torch.cuda.amp.GradScaler(enabled=half)
    optimizer_g = torch.optim.Adam(gen_model.parameters(), lr=0.0001, betas=(0,0.9))
    max_map = 0

    thresh = torch.FloatTensor([0.1])
    if half: thresh = thresh.cuda()
    for epoch in range(1,7):
        rlcr = AverageMeter()
        # train
        gen_model.train()
        if half:
            iouv = torch.linspace(0.5, 0.95, 10).cuda()
        else:
            iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        nc = sim_train.nc
        seen = 0
        names = {k: v for k, v in enumerate(app_model.names if hasattr(app_model, 'names') else app_model.module.names)}
        coco91class = coco80_to_coco91_class()
        # s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        p, r, f1, mp, mr, map50, map, = 0., 0., 0., 0., 0., 0., 0.
        stats, ap, ap_class = [], [], []
        train_iter = tqdm(train_loader)
        # assign threshold
        cnt = 0
        for batch_i, (img, targets, paths, shapes) in enumerate(train_iter):
            img = img.type(torch.FloatTensor).cuda() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if half:targets = targets.cuda()
            nb, _, height, width = img.shape  # batch size, channels, height, width

            # generator update
            optimizer_g.zero_grad()
            with autocast():
                if use_subsampling:
                    recon,res = gen_model((img,thresh))
                else:
                    recon,r = gen_model(img)
                pred,recon_features = app_model(recon, augment=opt.augment, extract_features=True)
                _,origin_features = app_model(img, augment=opt.augment, extract_features=True)
                loss = criterion_mse(img,recon)
                loss += orthorgonal_regularizer(gen_model.encoder.sample.weight,0.0001,half)
                for origin_feat,recon_feat in zip(origin_features,recon_features):
                    if origin_feat is None:continue
                    loss += criterion_mse(origin_feat,recon_feat)
                if use_subsampling:
                    filter_loss,real_cr,entropy = res
                    loss += 0.01*filter_loss# + 0.0001* entropy

            scaler_g.scale(loss).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            rlcr.update(real_cr if use_subsampling else r)


            # Run NMS
            if half:
                targets[:, 2:] *= torch.Tensor([width, height, width, height]).cuda()
            else:
                targets[:, 2:] *= torch.Tensor([width, height, width, height])

            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if opt.save_hybrid else []  # for autolabelling

            out = non_max_suppression(pred[0], conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, labels=lb, multi_label=True)


            # Statistics per image
            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                predn = pred.clone()
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

                # Assign all predictions as incorrect
                if half:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool).cuda()
                else:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels

                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                            # Append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in image
                                        break

                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].detach().cpu(), pred[:, 5].detach().cpu(), tcls))

            if batch_i%100==0 or batch_i==(len(train_loader)-1):
                metric = stat_to_map(stats,names,nc)
                if use_subsampling:
                    train_iter.set_description(
                        f"Train: {epoch:3}. Thresh: {thresh.cpu().numpy()[0]:.3f}. "
                        f"map50: {metric[3]:.2f}. map: {metric[4]:.2f}. "
                        f"MP: {metric[1]:.2f}. MR: {metric[2]:.2f}. "
                        f"loss: {loss.cpu().item():.3f}. "
                        f"cr: {rlcr.avg:.5f}. "
                        )
                else:
                    train_iter.set_description(
                        f"Train: {epoch:3}. "
                        f"map50: {metric[3]:.2f}. map: {metric[4]:.2f}. "
                        f"MP: {metric[1]:.2f}. MR: {metric[2]:.2f}. "
                        f"loss: {loss.cpu().item():.3f}. "
                        f"r: {rlcr.avg:.5f}. "
                        )
        train_iter.close()

        # eval
        rlcr = AverageMeter()
        gen_model.eval()
        if half:
            iouv = torch.linspace(0.5, 0.95, 10).cuda()
        else:
            iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        nc = sim_test.nc
        seen = 0
        names = {k: v for k, v in enumerate(app_model.names if hasattr(app_model, 'names') else app_model.module.names)}
        coco91class = coco80_to_coco91_class()
        # s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        p, r, f1, mp, mr, map50, map = 0., 0., 0., 0., 0., 0., 0.
        stats, ap, ap_class = [], [], []
        test_iter = tqdm(test_loader)
        for batch_i, (img, targets, paths, shapes) in enumerate(test_iter):
            img = img.type(torch.FloatTensor).cuda() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if half:targets = targets.cuda()
            nb, _, height, width = img.shape  # batch size, channels, height, width

            # Run model
            with torch.no_grad():
                if use_subsampling:
                    recon,res = gen_model((img,thresh))
                else:
                    recon,r = gen_model(img)
                pred,recon_features = app_model(recon, augment=opt.augment, extract_features=True)
                _,origin_features = app_model(img, augment=opt.augment, extract_features=True)
                loss = criterion_mse(img,recon)
                loss += orthorgonal_regularizer(gen_model.encoder.sample.weight,0.0001,half)
                for origin_feat,recon_feat in zip(origin_features,recon_features):
                    if origin_feat is None:continue
                    loss += criterion_mse(origin_feat,recon_feat)
                if use_subsampling:
                    esti_cr,real_cr,std = res
                    # loss += esti_cr - 0.01*std

            rlcr.update(real_cr if use_subsampling else r)
                
            # Run NMS
            if half:
                targets[:, 2:] *= torch.Tensor([width, height, width, height]).cuda()
            else:
                targets[:, 2:] *= torch.Tensor([width, height, width, height])

            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if opt.save_hybrid else []  # for autolabelling

            out = non_max_suppression(pred[0], conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, labels=lb, multi_label=True)


            # Statistics per image
            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                predn = pred.clone()
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

                # Assign all predictions as incorrect
                if half:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool).cuda()
                else:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels

                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                            # Append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in image
                                        break

                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].detach().cpu(), pred[:, 5].detach().cpu(), tcls))

            if batch_i%100==0 or batch_i==(len(test_loader)-1):
                metric = stat_to_map(stats,names,nc)
                if use_subsampling:
                    test_iter.set_description(
                        f"Test: {epoch:3}. Thresh: {thresh.cpu().numpy()[0]:.3f}. "
                        f"map50: {metric[3]:.2f}. map: {metric[4]:.2f}. "
                        f"MP: {metric[1]:.2f}. MR: {metric[2]:.2f}. "
                        f"loss: {loss.cpu().item():.3f}. "
                        f"cr: {rlcr.avg:.5f}. "
                        )
                else:
                    test_iter.set_description(
                        f"Test: {epoch:3}. "
                        f"map50: {metric[3]:.2f}. map: {metric[4]:.2f}. "
                        f"MP: {metric[1]:.2f}. MR: {metric[2]:.2f}. "
                        f"loss: {loss.cpu().item():.3f}. "
                        f"r: {rlcr.avg:.5f}. "
                        )
        test_iter.close()
        if metric[3] > max_map:
            torch.save(gen_model.state_dict(), PATH)
            max_map = metric[3]

# validate original, CCO-R, CCO-A
def deepcod_validate():
    from compression.deepcod import DeepCOD, orthorgonal_regularizer, init_weights
    sim = Simulator(train=False,use_model=True)

    opt = sim.opt
    device = select_device(opt.device, batch_size=opt.batch_size)
    half = opt.device != 'cpu'
    use_subsampling=True
    # data
    test_loader = sim.dataloader

    # vision app
    app_model = sim.model
    if half:
        app_model = app_model.cuda()
        for layer in app_model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()
    app_model.eval()

    # encoder+decoder
    PATH = 'backup/CCO-A.pth' if use_subsampling else 'backup/deepcod_soft_c8.pth'
    gen_model = DeepCOD(use_subsampling=use_subsampling)
    gen_model.load_state_dict(torch.load(PATH,map_location='cpu'))
    if args.device != 'cpu':
        gen_model = gen_model.cuda()

    # thresh to evaluate
    thresh_list = []
    if use_subsampling:
        for th1 in range(11):
            thresh = torch.FloatTensor([th1/10.0])
            thresh_list.append(thresh)
    else:
        thresh_list.append(None)

    # eval
    gen_model.eval()

    for thresh in thresh_list:
        if half: thresh = thresh.cuda()
        if half:
            iouv = torch.linspace(0.5, 0.95, 10).cuda()
        else:
            iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        nc = sim.nc
        seen = 0
        names = {k: v for k, v in enumerate(app_model.names if hasattr(app_model, 'names') else app_model.module.names)}
        coco91class = coco80_to_coco91_class()
        p, r, f1, mp, mr, map50, map = 0., 0., 0., 0., 0., 0., 0.
        stats, ap, ap_class = [], [], []
        test_iter = tqdm(test_loader)
        rlcr = AverageMeter()
        for batch_i, (img, targets, paths, shapes) in enumerate(test_iter):
            img = img.type(torch.FloatTensor).cuda() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if half:targets = targets.cuda()
            nb, _, height, width = img.shape  # batch size, channels, height, width

            # Run model
            with torch.no_grad():
                if use_subsampling:
                    recon,res = gen_model((img,thresh))
                else:
                    recon,r = gen_model(img)
                pred,recon_features = app_model(recon, augment=opt.augment, extract_features=True)
                _,origin_features = app_model(img, augment=opt.augment, extract_features=True)
               
                if use_subsampling:
                    esti_cr,real_cr,std = res
            rlcr.update(real_cr if use_subsampling else r)
                
            # Run NMS
            if half:
                targets[:, 2:] *= torch.Tensor([width, height, width, height]).cuda()
            else:
                targets[:, 2:] *= torch.Tensor([width, height, width, height])

            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if opt.save_hybrid else []  # for autolabelling

            out = non_max_suppression(pred[0], conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, labels=lb, multi_label=True)


            # Statistics per image
            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                predn = pred.clone()
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

                # Assign all predictions as incorrect
                if half:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool).cuda()
                else:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels

                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                            # Append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in image
                                        break

                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].detach().cpu(), pred[:, 5].detach().cpu(), tcls))

            if batch_i%100==0 or batch_i==(len(test_loader)-1):
                metric = stat_to_map(stats,names,nc)
                if use_subsampling:
                    test_iter.set_description(
                        f"Test: {epoch:3}. Thresh: {thresh.cpu().numpy()[0]:.3f}. "
                        f"map50: {metric[3]:.2f}. map: {metric[4]:.2f}. "
                        f"MP: {metric[1]:.2f}. MR: {metric[2]:.2f}. "
                        f"loss: {loss.cpu().item():.3f}. "
                        f"cr: {rlcr.avg:.5f}. "
                        )
                else:
                    test_iter.set_description(
                        f"Test: {epoch:3}. "
                        f"map50: {metric[3]:.2f}. map: {metric[4]:.2f}. "
                        f"MP: {metric[1]:.2f}. MR: {metric[2]:.2f}. "
                        f"loss: {loss.cpu().item():.3f}. "
                        f"r: {rlcr.avg:.5f}. "
                        )
        with open("raw_eval.log" if use_subsampling else "original_eval.log", "a") as f:
            f.write(f"{metric[3]:.3f} {rlcr.avg:.5f}\n")
        test_iter.close()

        #0.49,0.00489


def run_model_multi_range(opt,model,dataloader,nc,ranges,TF=None,C_param=None):
    device = select_device(opt.device, batch_size=opt.batch_size)
    
    half = device.type != 'cpu'
    if half:
        iouv = torch.linspace(0.5, 0.95, 10).cuda()
    else:
        iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    seen = 0
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    stats, ap, ap_class = [], [], []
    crs = []
    metrics = []
    test_iter = tqdm(dataloader, disable=False)
    for batch_i, (img, targets, paths, shapes) in enumerate(test_iter):
        if batch_i>=ranges[-1]:break
        # perform transformation
        if TF is not None:
            tf_imgs = None
            for th_img in img:
                np_img = th_img.permute(1,2,0).numpy()
                tf_img = TF.transform(image=np_img[:,:,(2,1,0)], C_param=C_param)
                tf_img = torch.from_numpy(tf_img[:,:,(2,1,0)]).float().permute(2,0,1).unsqueeze(0)
                if tf_imgs is None:
                    tf_imgs = tf_img
                else:
                    tf_imgs = torch.cat((tf_imgs,tf_img),0)
            img = tf_imgs
        # end transformation
        if half: img = img.cuda()
        # img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if half:targets = targets.cuda()
        # targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            out, train_out = model(img, augment=opt.augment)  # inference and training outputs
            t0 += time_synchronized() - t

            # Run NMS
            if half:
                targets[:, 2:] *= torch.Tensor([width, height, width, height]).cuda()
            else:
                targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if opt.save_hybrid else []  # for autolabelling
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, labels=lb, multi_label=True)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Assign all predictions as incorrect
            if half:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool).cuda()
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
        if (batch_i+1) in ranges:
            crs += [TF.get_compression_ratio() if TF is not None else 0]
            metrics += [stat_to_map(stats,names,nc)]
        cr = crs[-1] if crs else 0
        metric = metrics[-1] if metrics else [np.array([0]),0,0,0,0]
        test_iter.set_description(
                f"Test Iter: {batch_i+1:3}/{len(dataloader):3}. "
                f"NT: {metric[0].sum():3}. CR: {cr:.2f}. "
                f"map50: {metric[3]:.2f}. map: {metric[4]:.2f}. "
                f"MP: {metric[1]:.2f}. MR: {metric[2]:.2f}. ")

    test_iter.close()

    map50s = [m[3] for m in metrics]

    return map50s,crs

def stat_to_map(stats,names,nc):
    mp, mr, map50, map = 0.,0.,0.,0.
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)
    return (nt,mp,mr,map50,map)

def setup_opt():
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='train', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    # check_requirements()
    return opt

class Simulator:
    def __init__(self,train=True,use_model=True):
        self.opt = setup_opt()
        self.opt.task = 'train' if train else 'val'
        self.model = get_model(self.opt)
        self.dataloader,self.nc = get_dataloader(self.opt,self.model)
        self.num_batches = len(self.dataloader)
        if not use_model:
            self.model = None

    def get_one_point(self, datarange, TF=None, C_param=None):
        # start counting the compressed size
        if TF is not None: TF.reset()
        map50 = run_model(self.opt,self.model,self.dataloader,self.nc,datarange,TF,C_param)
        # get the compression ratio
        cr = TF.get_compression_ratio() if TF is not None else 0
        return map50,cr

    def get_multi_point(self, ranges, TF=None, C_param=None):
        if TF is not None: TF.reset()
        map50s,crs = run_model_multi_range(self.opt,self.model,self.dataloader,self.nc,ranges,TF,C_param)
        return map50s,crs

    def test(self):
        feature_trainer(self.opt,self.dataloader)


if __name__ == '__main__':
    sim = Simulator()
    r = sim.get_one_point((0,10))
    print(r)