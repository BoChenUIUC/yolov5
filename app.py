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

def get_model(opt):
    device = select_device(opt.device, batch_size=opt.batch_size)
    # Load model
    print(device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()
    model.eval()
    if device.type != 'cpu':
        model.cuda()
        model(torch.zeros(1, 3, opt.img_size, opt.img_size).to(device).type_as(next(model.parameters())))  # run once
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
class OneLayer(nn.Module):
    def __init__(self):
        super(OneLayer, self).__init__()
        num_features = 32
        self.conv1 = nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.final = nn.Conv2d(num_features, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.final(x)
        x = x.view(x.size(0), -1)
        x = F.tanh(x)
        x = x * 0.5 + 0.5
        return x

class TwoLayer(nn.Module):
    def __init__(self):
        super(TwoLayer, self).__init__()
        num_features = 16
        self.conv1 = nn.Conv2d(3, num_features, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features)
        self.final = nn.Conv2d(num_features, 1, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.final(x)
        x = x.view(x.size(0), -1)
        x = F.tanh(x)
        x = x * 0.5 + 0.5
        return x

def feature_main():
    sim_train = Simulator(train=True,use_model=False)
    sim_test = Simulator(train=False,use_model=False)
    opt = sim_train.opt
    half = opt.device != 'cpu'

    net = TwoLayer()
    if half: net = net.cuda()
    # for i in range(10):
    #     s = time.perf_counter()
    #     print(net(torch.randn(1, 3, 320, 672)).shape)
    #     print(time.perf_counter()-s)
    # return 
    
    PATH = 'backup/sf.pth'
    # net.load_state_dict(torch.load(PATH))

    for epoch in range(100):
        feature_trainer(sim_train.dataloader,net,half,epoch)
        feature_tester(sim_test.dataloader,net,half,epoch)
        torch.save(net.state_dict(), PATH)

def feature_trainer(dataloader,net,half,epoch):
    running_loss = 0.0
    tp,gt,dt = 0.0,0.0,0.0
    toMacroBlock = nn.MaxPool2d(kernel_size=8, stride=8, padding=0, ceil_mode=True)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train_iter = tqdm(dataloader)
    for batch_i, (img, targets, paths, shapes) in enumerate(train_iter):
        if half: img = img.cuda()
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = img.shape  # batch size, channels, height, width
        if half:
            targets = targets.cuda()
        if half:
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).cuda()
        else:
            targets[:, 2:] *= torch.Tensor([width, height, width, height])
        gt_ft_map = torch.zeros(nb, height, width)
        for si in range(nb):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            if nl:
                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                # xyxy
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                for x1,y1,x2,y2 in tbox:
                    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                    gt_ft_map[si,y1:y2,x1:x2] = 1
        gt_ft_map = toMacroBlock(gt_ft_map)
        gt_ft_map = gt_ft_map.view(gt_ft_map.size(0),-1)

        inputs = torch.FloatTensor(img)
        labels = torch.FloatTensor(gt_ft_map)
        if half:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # zero gradient
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        assert(labels.size(1) == outputs.size(1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.cpu().item()
        tp += torch.sum(labels[outputs>0.5]==1)
        gt += torch.sum(labels==1)
        dt += torch.sum(outputs>0.5)
        prec = tp/dt
        rec = tp/gt
        f1_score = 2*prec*rec/(prec+rec)
        train_iter.set_description(
            f"Epoch: {epoch+1:3}. Batch: {batch_i:3}. "
            f"Loss: {running_loss/(1+batch_i):.6f}. Prec: {prec:.6f}. Rec: {rec:.6f}. F1: {f1_score:.6f}. ")

def feature_tester(dataloader,net,half,epoch):
    running_loss = 0.0
    tp,gt,dt = 0.0,0.0,0.0
    toMacroBlock = nn.MaxPool2d(kernel_size=8, stride=8, padding=0, ceil_mode=True)
    criterion = nn.BCELoss()
    test_iter = tqdm(dataloader)
    for batch_i, (img, targets, paths, shapes) in enumerate(test_iter):
        if half: img = img.cuda()
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = img.shape  # batch size, channels, height, width
        if half:
            targets = targets.cuda()
        if half:
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).cuda()
        else:
            targets[:, 2:] *= torch.Tensor([width, height, width, height])
        gt_ft_map = torch.zeros(nb, height, width)
        for si in range(nb):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            if nl:
                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                # xyxy
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                for x1,y1,x2,y2 in tbox:
                    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                    gt_ft_map[si,y1:y2,x1:x2] = 1
        gt_ft_map = toMacroBlock(gt_ft_map)
        gt_ft_map = gt_ft_map.view(gt_ft_map.size(0),-1)

        inputs = torch.FloatTensor(img)
        labels = torch.FloatTensor(gt_ft_map)
        if half:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # forward + backward + optimize
        with torch.no_grad():
            outputs = net(inputs) 
            loss = criterion(outputs, labels)
        
        # print statistics
        running_loss += loss.cpu().item()
        tp += torch.sum(labels[outputs>0.5]==1)
        gt += torch.sum(labels==1)
        dt += torch.sum(outputs>0.5)
        prec = tp/dt
        rec = tp/gt
        f1_score = 2*prec*rec/(prec+rec)
        test_iter.set_description(
            f"Epoch: {epoch+1:3}. Batch: {batch_i:3}. "
            f"Loss: {running_loss/(1+batch_i):.6f}. Prec: {prec:.6f}. Rec: {rec:.6f}. F1: {f1_score:.6f}. ")


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
    mp, mr, map50 = 0.,0.,0.
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
    # opt = setup_opt()
    # print(opt)

    # if opt.task in ['val', 'test']:  # run normally
    #     test(opt.data,
    #          opt.weights,
    #          opt.batch_size,
    #          opt.img_size,
    #          opt.conf_thres,
    #          opt.iou_thres,
    #          opt.save_json,
    #          opt.single_cls,
    #          opt.augment,
    #          opt.verbose,
    #          save_txt=opt.save_txt | opt.save_hybrid,
    #          save_hybrid=opt.save_hybrid,
    #          save_conf=opt.save_conf,
    #          )

    # elif opt.task == 'speed':  # speed benchmarks
    #     for w in opt.weights:
    #         test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False)

    # elif opt.task == 'study':  # run over a range of settings and save/plot
    #     # python test.py --task study --data coco.yaml --iou 0.7 --weights yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
    #     x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
    #     for w in opt.weights:
    #         f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
    #         y = []  # y axis
    #         for i in x:  # img-size
    #             print(f'\nRunning {f} point {i}...')
    #             r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
    #                            plots=False)
    #             y.append(r + t)  # results and times
    #         np.savetxt(f, y, fmt='%10.4g')  # save
    #     os.system('zip -r study.zip study_*.txt')
    #     plot_study_txt(x=x)  # plot
