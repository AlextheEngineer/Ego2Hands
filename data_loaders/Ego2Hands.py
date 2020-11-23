import os
import numpy as np
import torch
import torch.utils.data as data
import cv2
import random
import math
from utils import *

def random_bg_augment(img, img_path = "", bg_adapt = False, brightness_aug = True, flip_aug = True):
    if brightness_aug:
        if bg_adapt:
            brightness_mean = int(np.mean(img))
            brightness_val = random.randint(brightness_mean - 50, brightness_mean + 50)
            img = change_mean_brightness(img, None, brightness_val, 20, img_path)
        else:
            brightness_val = random.randint(35, 220)
            img = change_mean_brightness(img, None, brightness_val, 20, img_path)
    
    img = img.astype("uint8")
    
    if flip_aug:
        do_flip = bool(random.getrandbits(1))
        if do_flip:
            img = cv2.flip(img, 1)
    return img

def resize_bg(fg_shape, bg_img, bg_adapt):
    fg_h, fg_w = fg_shape[:2]
    if not bg_adapt:
        bg_h, bg_w = bg_img.shape[:2]
        
        if bg_h < fg_h or bg_w < fg_w:
            fb_h_ratio = float(fg_h)/bg_h
            fb_w_ratio = float(fg_w)/bg_w
            bg_resize_ratio = max(fb_h_ratio, fb_w_ratio)
            bg_img = cv2.resize(bg_img, (int(math.ceil(bg_img.shape[1]*bg_resize_ratio)), int(math.ceil(bg_img.shape[0]*bg_resize_ratio))))
        bg_h, bg_w = bg_img.shape[:2]

        # Get row/col offsets
        bg_h_offset_range = max(bg_h - fg_h, 0)
        bg_w_offset_range = max(bg_w - fg_w, 0)

        bg_h_offset = random.randint(0, bg_h_offset_range)
        bg_w_offset = random.randint(0, bg_w_offset_range)
        bg_img = bg_img[bg_h_offset:bg_h_offset+fg_h, bg_w_offset:bg_w_offset+fg_w, :3]
    else:
        bg_img = cv2.resize(bg_img, (fg_w, fg_h))
    return bg_img

def add_alpha_image_to_bg(alpha_img, bg_img):
    alpha_s = np.repeat((alpha_img[:,:,3]/255.0)[:,:,np.newaxis], 3, axis=2)
    alpha_l = 1.0 - alpha_s
    combined_img = np.multiply(alpha_s ,alpha_img[:,:,:3]) + np.multiply(alpha_l, bg_img)
    return combined_img
    
def add_alpha_border(hand_img):
    fg_mask = (hand_img[:,:,-1] == 0).astype(np.uint8)
    fg_mask = cv2.dilate(fg_mask, np.ones((3, 3)))
    alpha_mask = fg_mask * 255
    alpha_mask = 255 - cv2.GaussianBlur(alpha_mask, (7, 7), 0)
    #alpha_mask[np.logical_not(fg_mask)] = 255
    hand_img[:,:,-1] = alpha_mask
    hand_seg = alpha_mask > 200
    hand_all_seg = alpha_mask > 0
    return hand_img, hand_seg, hand_all_seg

def merge_hands(top_hand_img, bot_hand_img, bg_img, bg_adapt, bg_resize = True):
    if top_hand_img is not None and bot_hand_img is not None:
        bot_hand_img, _, _ = add_alpha_border(bot_hand_img)
        top_hand_img, _, _ = add_alpha_border(top_hand_img)
        bg_img_resized = resize_bg(bot_hand_img.shape, bg_img, bg_adapt) if bg_resize else bg_img
        combined_hand_img = add_alpha_image_to_bg(bot_hand_img, bg_img_resized)
        combined_hand_img = add_alpha_image_to_bg(top_hand_img, combined_hand_img)
    else:
        top_hand_img, _, _ = add_alpha_border(top_hand_img)
        bg_img_resized = resize_bg(top_hand_img.shape, bg_img, bg_adapt) if bg_resize else bg_img
        combined_hand_img = add_alpha_image_to_bg(top_hand_img, bg_img_resized)
    return combined_hand_img, bg_img_resized
    
def change_mean_brightness(img, seg, brightness_val, jitter_range = 20, img_path = ""):
    if seg is not None:
        old_mean_val = np.mean(img[seg])
    else:
        old_mean_val = np.mean(img)
    assert old_mean_val != 0, "ERROR: {} has mean of 0".format(img_path)
    new_mean_val = brightness_val + random.uniform(-jitter_range/2, jitter_range/2)
    img *= (new_mean_val/old_mean_val)
    img = np.clip(img, 0, 255)
    return img
    
def random_smoothness(img, smooth_rate = 0.3):
    smooth_rate_tick = smooth_rate/5
    rand_val = random.random()
    if rand_val < smooth_rate:
        if rand_val < smooth_rate_tick:
            kernel_size = 3
        elif rand_val < smooth_rate_tick*2:
            kernel_size = 5
        elif rand_val < smooth_rate_tick*3:
            kernel_size = 7
        elif rand_val < smooth_rate_tick*4:
            kernel_size = 9
        else:
            kernel_size = 11
        img[:,:,:3] = cv2.blur(img[:,:,:3], (kernel_size, kernel_size))
    return img

def read_ego2hands_files(ego2hands_root_dir):
    #root_dir = 'C:/School/Alex/Ego2Hands'
    img_path_list = []
    energy_path_list = []
    
    for root, dirs, files in os.walk(ego2hands_root_dir):
        for file_name in files:
            if file_name.endswith(".png") and "energy" not in file_name and "vis" not in file_name:
                img_path = os.path.join(root, file_name)
                img_path_list.append(img_path)
                energy_path_list.append(img_path.replace(".png", "_energy.png"))

    return img_path_list, energy_path_list
    
def read_bg_data(args, config, bg_adapt, seq_i):
    if not args.custom:
        if not bg_adapt:
            root_bg_dir = config.bg_all_dir
        else:
            root_bg_dir = os.path.join(config.dataset_eval_dir, "eval_seq{}_bg".format(seq_i))
    else:
        assert config.custom_bg_dir != "", "Error: custom bg dir not set. Please set \"custom_bg_dir\" in the config file."
        root_bg_dir = config.custom_bg_dir
        
    # backgrounds
    bg_path_list = []
    for root, dirs, files in os.walk(root_bg_dir):
        for file_name in files:
            if file_name.endswith(".jpg") or file_name.endswith(".png"):
                bg_path_list.append(os.path.join(root, file_name))
    return bg_path_list

def read_test_sequences(args, config, seq_i):
    if not args.custom:
        test_seq_dir = os.path.join(config.dataset_eval_dir, "eval_seq{}_imgs".format(seq_i))
    else:
        assert config.custom_eval_dir != "", "Error: custom eval dir not set. Please set \"custom_eval_dir\" in the config file."
        test_seq_dir = config.custom_eval_dir
    
    img_path_list = []
    seg_gt_path_list = []
    energy_l_path_list = []
    energy_r_path_list = []
    
    for root, dirs, files in os.walk(test_seq_dir):
        for file_name in files:
            if file_name.endswith(".png") and not "e" in file_name:
                if not args.custom and os.path.exists(os.path.join(root, file_name.replace(".png", "_seg.png"))):
                    img_path_list.append(os.path.join(root, file_name))
                    seg_gt_path_list.append(os.path.join(root, file_name.replace(".png", "_seg.png")))
                    energy_l_path_list.append(os.path.join(root, file_name.replace(".png", "_e_l.png")))
                    energy_r_path_list.append(os.path.join(root, file_name.replace(".png", "_e_r.png")))
                else:
                    img_path_list.append(os.path.join(root, file_name))
                    seg_gt_path_list.append(None)
                    energy_l_path_list.append(None)
                    energy_r_path_list.append(None)
                
    return img_path_list, seg_gt_path_list, energy_l_path_list, energy_r_path_list

def get_random_brightness_for_scene(args, config, bg_adapt, seq_i):
    dark_lighting_set = [5]
    normal_lighting_set = [1, 3, 4, 6, 7]
    bright_lighting_set = [2, 8]
    brightness_map = {"dark": (0, 55), "normal": (55, 200), "bright": (55, 255)}
    if not bg_adapt:
        return random.randint(15, 240)
    else:
        if not args.custom:
            if seq_i in dark_lighting_set:
                return random.randint(*brightness_map["dark"])
            elif seq_i in normal_lighting_set:
                return random.randint(*brightness_map["normal"])
            elif seq_i in bright_lighting_set:
                return random.randint(*brightness_map["bright"])
        else:
            assert config.custom_scene_brightness != "", "Error: custom scene brightness not set. Please set \"custom_scene_brightness\" in the config file."
            assert config.custom_scene_brightness in brightness_map, "Error: unrecognized brightness {} (valid options [\"dark\", \"normal\", \"bright\"]".format(config.custom_scene_brightness)
            return random.randint(*brightness_map[config.custom_scene_brightness])

LEFT_IDX = 1
RIGHT_IDX = 2

class Ego2HandsData(data.Dataset):
    
    def __init__(self, args, config, mode, seq_i = -1):
        self.args = args
        self.config = config
        self.mode = mode
        self.bg_adapt = args.adapt
        self.seq_i = seq_i
        self.input_edge = args.input_edge
        self.bg_list = read_bg_data(self.args, self.config, self.bg_adapt, seq_i)
        if self.mode == "train_seg":
            self.img_path_list, self.energy_path_list = read_ego2hands_files(self.config.dataset_train_dir)
        elif self.mode == "test_seg":
            self.img_path_list, self.seg_gt_path_list, self.energy_l_gt_path_list, self.energy_r_gt_path_list = read_test_sequences(self.args, self.config, seq_i)
            self.bg_list = []
        else:
            print("Unknown mode: {}".format(mode))
            sys.exit()
            
        self.img_h, self.img_w = 288, 512
        self.valid_hand_seg_th = 5000
        self.EMPTY_IMG_ARRAY = np.zeros((1, 1))
        self.EMPTY_BOX_ARRAY = np.zeros([0, 0, 0, 0])
        print("Loading finished")
        print("#hand imgs: {}".format(len(self.img_path_list)))
        print("#bg imgs: {}".format(len(self.bg_list)))

    def __getitem__(self, index):
        if self.mode == "train_seg":
            # Left hand
            left_i = random.randint(0, self.__len__() - 1)
            left_img = cv2.imread(self.img_path_list[left_i], cv2.IMREAD_UNCHANGED)
            assert left_img is not None, "Error, image not found: {}".format(self.img_path_list[left_i])
            left_img = left_img.astype(np.float32)
            left_img = cv2.resize(left_img, (self.img_w, self.img_h))
            left_img = cv2.flip(left_img, 1)
            left_seg = left_img[:,:,-1] > 128
            left_energy = cv2.imread(self.energy_path_list[left_i], 0)
            left_energy = cv2.resize(left_energy, (self.img_w, self.img_h)).astype(np.float32)/255.0
            left_energy = cv2.flip(left_energy, 1)
            left_img_orig = left_img.copy()
            # Augmentation with random translation
            left_img, left_seg, left_energy = seg_augmentation_wo_kpts(left_img, left_seg, left_energy)
            # Augmentation
            brightness_val = get_random_brightness_for_scene(self.args, self.config, self.bg_adapt, self.seq_i)
            left_img = change_mean_brightness(left_img, left_seg, brightness_val, 20, self.img_path_list[left_i])
            left_img = random_smoothness(left_img)

            # Right hand
            right_i = random.randint(0, self.__len__() - 1)
            right_img = cv2.imread(self.img_path_list[right_i], cv2.IMREAD_UNCHANGED)
            assert right_img is not None, "Error, image not found: {}".format(self.img_path_list[right_i])
            right_img = right_img.astype(np.float32)
            right_img = cv2.resize(right_img, (self.img_w, self.img_h))
            right_seg = right_img[:,:,-1] > 128
            right_energy = cv2.imread(self.energy_path_list[right_i], 0)
            right_energy = cv2.resize(right_energy, (self.img_w, self.img_h)).astype(np.float32)/255.0
            right_img_orig = right_img.copy()
            # Augmentation with random translation
            right_img, right_seg, right_energy = seg_augmentation_wo_kpts(right_img, right_seg, right_energy)
            # Augmentation
            right_img = change_mean_brightness(right_img, right_seg, brightness_val, 20, self.img_path_list[right_i])
            right_img = random_smoothness(right_img)

            # Find background
            bg_img = None
            while(bg_img is None):
                bg_i = random.randint(0, len(self.bg_list) - 1)
                bg_img = cv2.imread(self.bg_list[bg_i]).astype(np.float32)
                if not self.bg_adapt:
                    bg_img = random_bg_augment(bg_img, self.bg_list[bg_i], bg_adapt = self.bg_adapt)
                else:
                    bg_img = random_bg_augment(bg_img, self.bg_list[bg_i], bg_adapt = self.bg_adapt, flip_aug = False)
                bg_img = random_smoothness(bg_img)
            merge_mode = random.randint(0, 9)
            if merge_mode < 8:
                if np.sum(left_energy) > np.sum(right_energy):
                    # left hand first
                    merge_mode = 0
                else:
                    # right hand first
                    merge_mode = 4
            
            # Merge hands
            if merge_mode < 4:
                # left hand top, right hand bottom
                img_real, bg_img_resized = merge_hands(left_img, right_img, bg_img, self.bg_adapt)
                img_real_orig, _ = merge_hands(left_img_orig, right_img_orig, bg_img_resized, self.bg_adapt, False)
                seg_real = np.zeros((img_real.shape[:2]), dtype=np.uint8)
                seg_real[right_seg] = RIGHT_IDX
                seg_real[left_seg] = LEFT_IDX
                # Check for hand with insufficient size
                right_mask = seg_real == RIGHT_IDX
                if right_mask.sum() < self.valid_hand_seg_th:
                    seg_real[right_mask] = 0
                    right_energy.fill(0.0)
            elif merge_mode >= 4 and merge_mode < 8:
                # left hand bottom, right hand top
                img_real, bg_img_resized = merge_hands(right_img, left_img, bg_img, self.bg_adapt)
                img_real_orig, _ = merge_hands(right_img_orig, left_img_orig, bg_img_resized, self.bg_adapt, False)
                seg_real = np.zeros((img_real.shape[:2]), dtype=np.uint8)
                seg_real[left_seg] = LEFT_IDX
                seg_real[right_seg] = RIGHT_IDX
                # Check for hand with insufficient size
                left_mask = seg_real == LEFT_IDX
                if left_mask.sum() < self.valid_hand_seg_th:
                    seg_real[left_mask] = 0
                    left_energy.fill(0.0)
            elif merge_mode == 8:
                # drop left hand, right hand only
                img_real, bg_img_resized = merge_hands(right_img, None, bg_img, self.bg_adapt)
                img_real_orig, _ = merge_hands(right_img_orig, None, bg_img_resized, self.bg_adapt, False)
                seg_real = np.zeros((img_real.shape[:2]), dtype=np.uint8)
                seg_real[right_seg] = RIGHT_IDX
                left_energy.fill(0.0)
            elif merge_mode == 9:
                # drop right ahnd, left hand only
                img_real, bg_img_resized = merge_hands(left_img, None, bg_img, self.bg_adapt)
                img_real_orig, _ = merge_hands(left_img_orig, None, bg_img_resized, self.bg_adapt, False)
                seg_real = np.zeros((img_real.shape[:2]), dtype=np.uint8)
                seg_real[left_seg] = LEFT_IDX
                right_energy.fill(0.0)
            
            # For multi-resoluiton
            seg_real2 = cv2.resize(seg_real, (self.img_w//2, self.img_h//2), interpolation=cv2.INTER_NEAREST)
            seg_real4 = cv2.resize(seg_real, (self.img_w//4, self.img_h//4), interpolation=cv2.INTER_NEAREST)
            
            left_energy2 = cv2.resize(left_energy, (self.img_w//2, self.img_h//2))
            left_energy4 = cv2.resize(left_energy2, (self.img_w//4, self.img_h//4))
            
            right_energy2 = cv2.resize(right_energy, (self.img_w//2, self.img_h//2))
            right_energy4 = cv2.resize(right_energy2, (self.img_w//4, self.img_h//4))
            
            bg_energy = 1.0 - np.maximum(left_energy, right_energy)
            bg_energy2 = 1.0 - np.maximum(left_energy2, right_energy2)
            bg_energy4 = 1.0 - np.maximum(left_energy4, right_energy4)
            
            energy_gt = np.stack([bg_energy, left_energy, right_energy], 0)
            energy_gt2 = np.stack([bg_energy2, left_energy2, right_energy2], 0)
            energy_gt4 = np.stack([bg_energy4, left_energy4, right_energy4], 0)

            img_real_orig_tensor = torch.from_numpy(img_real_orig)
            img_real = cv2.cvtColor(img_real, cv2.COLOR_RGB2GRAY)
            
            # For input edge map
            if self.input_edge:
                img_edge = cv2.Canny(img_real.astype(np.uint8), 25, 100).astype(np.float32)
                img_real = np.stack((img_real, img_edge), -1)
            else:
                img_real = np.expand_dims(img_real, -1)
                
            # Prepare tensors
            img_id = torch.from_numpy(np.array([0]))
            img_real_tensor = normalize_tensor(torch.from_numpy(img_real.transpose(2, 0, 1)), 128.0, 256.0)
            seg_real_tensor = torch.from_numpy(seg_real).long()
            seg_real2_tensor = torch.from_numpy(seg_real2).long()
            seg_real4_tensor = torch.from_numpy(seg_real4).long()
            energy_gt_tensor = torch.from_numpy(energy_gt)
            energy_gt2_tensor = torch.from_numpy(energy_gt2)
            energy_gt4_tensor = torch.from_numpy(energy_gt4)

            return img_id, img_real_orig_tensor, img_real_tensor, seg_real_tensor, seg_real2_tensor, seg_real4_tensor, energy_gt_tensor, energy_gt2_tensor, energy_gt4_tensor
        elif self.mode == "test_seg":
            # Prepare image
            img_real_test = cv2.imread(self.img_path_list[index]).astype(np.float32)
            img_real_test = cv2.resize(img_real_test, (self.img_w, self.img_h))
            img_real_orig = img_real_test.copy()
            img_real_test = cv2.cvtColor(img_real_test, cv2.COLOR_RGB2GRAY)
            
            # For input edge map
            if self.input_edge:
                img_edge = cv2.Canny(img_real_test.astype(np.uint8), 25, 100).astype(np.float32)
                img_real_test = np.stack((img_real_test, img_edge), -1)
            else:
                img_real_test = np.expand_dims(img_real_test, -1)
            
            if not self.args.custom:
                # Prepare segmentation gt
                seg_gt_test = (cv2.imread(self.seg_gt_path_list[index], 0)/50).astype(np.uint8)
                seg_gt_test = cv2.resize(seg_gt_test, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)

                # Prepare bounding box gt
                energy_l_gt = cv2.resize(cv2.imread(self.energy_l_gt_path_list[index]), (self.img_w, self.img_h)).astype(np.float32)/255.0
                box_l_np = get_bounding_box_from_energy(energy_l_gt, close_op = False)
                energy_r_gt = cv2.resize(cv2.imread(self.energy_r_gt_path_list[index]), (self.img_w, self.img_h)).astype(np.float32)/255.0
                box_r_np = get_bounding_box_from_energy(energy_r_gt, close_op = False)
            else:
                seg_gt_test = self.EMPTY_IMG_ARRAY
                energy_l_gt = self.EMPTY_IMG_ARRAY
                box_l_np = self.EMPTY_BOX_ARRAY
                energy_r_gt = self.EMPTY_IMG_ARRAY
                box_r_np = self.EMPTY_BOX_ARRAY

            img_real_orig_tensor = torch.from_numpy(img_real_orig)
            img_real_test_tensor = normalize_tensor(torch.from_numpy(img_real_test.transpose(2, 0, 1)), 128.0, 256.0)
            seg_gt_tensor = torch.from_numpy(seg_gt_test).long()
            box_l_tensor = torch.from_numpy(box_l_np)
            box_r_tensor = torch.from_numpy(box_r_np)
            
            return img_real_orig_tensor, img_real_test_tensor, seg_gt_tensor, box_l_tensor, box_r_tensor

    def __len__(self):
        return len(self.img_path_list)

