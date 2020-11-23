import cv2
import numpy as np
from easydict import EasyDict as edict
import yaml
import random
import torch.nn as nn
import torch

class AverageMeter(object):
    """ Computes ans stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def save_model(state, is_best, is_last, filename):
    if is_last:
        torch.save(state, filename + '_pretrained.pth.tar')
    else:
        if is_best:
            torch.save(state, filename + '_best.pth.tar')
        else:
            torch.save(state, filename + '_latest.pth.tar')

def Config(filename):
    with open(filename, 'r') as f:
        parser = edict(yaml.load(f))
    for x in parser:
        print('{}: {}'.format(x, parser[x]))
    return parser

def normalize_tensor(tensor, mean, std):
    for t in tensor:
        t.sub_(mean).div_(std)
    return tensor

def compute_iou(pred, target, box_l = None, box_r = None, is_idx = False):
    n_classes = len(np.unique(target.cpu().data.numpy()))
    if n_classes == 1:
         pred_unique_np = np.unique(pred.cpu().data.numpy())
         if len(pred_unique_np) == 1 and pred_unique_np[0] == 0:
             return np.array([1.0])
         else:
             return np.array([0.0])
    ious = []
    if not pred.shape[2] == target.shape[1]:
        pred = nn.functional.interpolate(pred, size = (target.shape[1], target.shape[2]), mode = 'bilinear', align_corners = True)
        
    if not is_idx:
        pred = torch.argmax(pred, dim=1)
    
    # If region of interest is provided, clear values outside of it
    if box_l is not None and box_r is not None:
        assert pred.shape[0] == 1 and target.shape[0] == 1, "IoU for RoI computation only supports batch size of 1"
        for pred_i, target_i, box_l_i, box_r_i in zip(pred, target, box_l, box_r):
            for hand_label, box_j in enumerate([box_l_i, box_r_i]):
                hand_label = hand_label + 1
                region_mask = torch.zeros_like(pred_i, dtype=torch.bool)
                region_mask[box_j[0]:box_j[1], box_j[2]:box_j[3]] = True
                pred_i[(pred_i == hand_label) & (region_mask == False)] = 0
                target_i[(target_i == hand_label) & (region_mask == False)] = 0
        
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes):#xrange(1, n_classes):
        pred_inds = pred == cls
        #print(np.unique(pred_inds.cpu().data.numpy()))
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(1.0)  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(union))
    return np.array(ious)
    
def compute_ap(energy_l_pred, energy_r_pred, box_l_gt, box_r_gt, th = 0.5, scale_factor = 1.0, close_kernel_size = 15):
    assert energy_l_pred.shape[0] == 1 and energy_r_pred.shape[0] == 1, "AP computation only supports batch size of 1"
    ap_list = []
    box_pred_list = []
    for energy_l_pred_i, energy_r_pred_i, box_l_gt_i, box_r_gt_i in zip(energy_l_pred, energy_r_pred, box_l_gt, box_r_gt):
        for energy_j, box_gt_j in zip([energy_l_pred_i, energy_r_pred_i], [box_l_gt_i, box_r_gt_i]):
            energy_j = np.squeeze(energy_j)
            val_th_l = 0.5#np.amax(energy_j)*0.5
            positives_l = (energy_j > val_th_l).astype(np.uint8)
            # Close operation to remove small noise
            positives_l = cv2.erode(positives_l, np.ones((close_kernel_size, close_kernel_size)))
            positives_l = cv2.dilate(positives_l, np.ones((close_kernel_size, close_kernel_size)))
            #positives_l = cv2.morphologyEx(positives_l, cv2.MORPH_CLOSE, np.ones((15, 15)))
            coords = np.where(positives_l.astype(bool))
            if np.amax(box_gt_j) == 0:
                # If the hand is missing in ground truth
                if coords[0].size == 0:
                    ap_list.append(1.0)
                else:
                    ap_list.append(0.0)
            else:
                if coords[0].size == 0:
                    ap_list.append(0.0)
                else:
                    row_min, row_max, col_min, col_max = np.min(coords[0]), np.max(coords[0]), np.min(coords[1]), np.max(coords[1])
                    #print("pred {}, {}, {}, {}".format(row_min, row_max, col_min, col_max))
                    #print("box  {}".format(box_gt_j))
                    box_pred_list.append((row_min, row_max, col_min, col_max))
                    iou = bb_intersection_over_union([row_min, row_max, col_min, col_max], box_gt_j)
                    if iou >= th:
                        ap_list.append(1.0)
                    else:
                        ap_list.append(0.0)
    return ap_list

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[2], boxB[2])
    xB = min(boxA[1], boxB[1])
    yB = min(boxA[3], boxB[3])
    

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # rectangles
    boxAArea = abs((boxA[1] - boxA[0]) * (boxA[3] - boxA[2]))
    boxBArea = abs((boxB[1] - boxB[0]) * (boxB[3] - boxB[2]))
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
   
    return iou

def seg_augmentation_wo_kpts(img, seg, energy):
    img_h, img_w = img.shape[:2]
    fg_mask = seg.copy()
    
    coords1 = np.where(fg_mask)
    img_top, img_bot = np.min(coords1[0]), np.max(coords1[0])
    
    shift_range_ratio = 0.2
    # down shift
    down_shift = True if not fg_mask[0, :].any() else False
    if down_shift:
        down_space = int((img_h - img_top)*shift_range_ratio)
        old_bot = img_h
        down_offset = random.randint(0, down_space)
        old_bot -= down_offset

        old_top = 0
        cut_height = old_bot - old_top

        new_bot = img_h
        new_top = new_bot - cut_height
    else:
        old_bot, old_top = img_h, 0
        new_bot, new_top = old_bot, old_top
    
    coords2 = np.where(fg_mask[old_top:old_bot,:])
    img_left, img_right = np.min(coords2[1]), np.max(coords2[1])
    
    # Left shift or right shift    
    left_shift = True if not fg_mask[old_top:old_bot, -1].any() else False
    right_shift = True if not fg_mask[old_top:old_bot, 0].any() else False
    if left_shift and right_shift:
        if random.random() > 0.5:
            right_shift = False
        else:
            left_shift = False
            
    if left_shift:
        left_space = int(img_right*shift_range_ratio)
        old_left = 0
        left_offset = random.randint(0, left_space)
        old_left += left_offset
        
        old_right = img_w
        cut_width = old_right - old_left
        
        new_left = 0
        new_right = new_left + cut_width
        
    if right_shift:
        right_space = int((img_w - img_left)*shift_range_ratio)
        old_right = img_w
        right_offset = random.randint(0, right_space)
        old_right -= right_offset
        
        old_left = 0
        cut_width = old_right - old_left
        
        new_right = img_w
        new_left = new_right - cut_width
    
    if not (left_shift or right_shift):
        old_left, old_right = 0, img_w
        new_left, new_right = old_left, old_right

    img_new = np.zeros_like(img)
    seg_new = np.zeros_like(seg)
    energy_new = np.zeros_like(energy)

    img_new[new_top:new_bot, new_left:new_right] = img[old_top:old_bot, old_left:old_right]
    seg_new[new_top:new_bot, new_left:new_right] = seg[old_top:old_bot, old_left:old_right]
    energy_new[new_top:new_bot, new_left:new_right] = energy[old_top:old_bot, old_left:old_right]
    return img_new, seg_new, energy_new

def get_bounding_box_from_energy(energy, close_kernel_size = 15, close_op = True):
    energy_positives = (energy > 0.5).astype(np.uint8)
    if close_op:
        energy_positives = cv2.erode(energy_positives, np.ones((close_kernel_size, close_kernel_size)))
        energy_positives = cv2.dilate(energy_positives, np.ones((close_kernel_size, close_kernel_size)))
    coords = np.where(energy_positives.astype(bool))
    if coords[0].size != 0:
        row_min, row_max, col_min, col_max = np.min(coords[0]), np.max(coords[0]), np.min(coords[1]), np.max(coords[1])
    else:
        row_min, row_max, col_min, col_max = 0, 0, 0, 0
    return np.array([row_min, row_max, col_min, col_max])

def index_to_color(img_idx, unsure_idx, palette):
  img_color = np.zeros(img_idx.shape + (3,), dtype=np.uint8)
  idx_list = np.unique(img_idx).astype(np.uint8)
  for idx in idx_list:
    if idx != unsure_idx:
      img_color[img_idx == idx] = palette[idx][::-1]
    else:
      img_color[img_idx == idx] = palette[-1][::-1]
  return img_color
    
def visualize_seg_detection(img, seg, energy_l, energy_r, box_gt_l = None, box_gt_r = None, close_kernel_size = 7):
    palette = [(0, 0, 0), (128, 0, 0), (0, 128, 0)]
    color_box_l = (128, 128, 255)
    color_box_r = (160, 255, 160)
    color_box_l_gt = (0, 0, 255)
    color_box_r_gt = (0, 255, 0)
    alpha = 0.5
    thickness = 2
    img_vis = img.copy()
    if seg is not None:
        seg_vis = index_to_color(seg, 255, palette)
        # Overlay seg on img
        seg_positives = seg > 0
        img_vis[seg_positives] = (img[seg_positives]*alpha + seg_vis[seg_positives]*(1-alpha)).astype('uint8')
    # Draw detection boxes
    if energy_l is not None:
        box_l = get_bounding_box_from_energy(energy_l, close_kernel_size = close_kernel_size)
        if box_l.any():
            img_vis = cv2.rectangle(img_vis, (box_l[2], box_l[0]), (box_l[3], box_l[1]), color_box_l, thickness)
        if box_gt_l is not None and box_gt_l.any():
            img_vis = cv2.rectangle(img_vis, (box_gt_l[2], box_gt_l[0]), (box_gt_l[3], box_gt_l[1]), color_box_l_gt, thickness)
    if energy_r is not None:
        box_r = get_bounding_box_from_energy(energy_r, close_kernel_size = close_kernel_size)
        if box_r.any():
            img_vis = cv2.rectangle(img_vis, (box_r[2], box_r[0]), (box_r[3], box_r[1]), palette[2], thickness)
        if box_gt_r is not None and box_gt_r.any():
            img_vis = cv2.rectangle(img_vis, (box_gt_r[2], box_gt_r[0]), (box_gt_r[3], box_gt_r[1]), color_box_r_gt, thickness)
    return img_vis