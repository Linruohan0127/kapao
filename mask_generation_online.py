import cv2
import numpy as np
import numpy.random as rd
from pdb import set_trace as st
import random
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.patches as patches
import torch


class RandomArguments:
    def __init__(self, joint=(), ori_img=np.array([[]]), box_size=0):
        # print("current_joint:", joint)
        # print("ori_img.shape", ori_img.shape)
        self.sizeseed1 = rd.randint(0, box_size * 0.2+1)
        self.width = rd.randint(self.sizeseed1, self.sizeseed1*2+1)  # 到底多大好呢？
        self.sizeseed2 = rd.randint(0, box_size * 0.2+1)
        self.height = rd.randint(self.sizeseed2, self.sizeseed2*2+1)
        self.color_joint = rd.randint(1, ori_img.shape[0]+1)-1, rd.randint(1, ori_img.shape[1]+1)-1
        # self.color = ori_img[color_joint]
        # self.color = (int(self.color[0]), int(self.color[1]), int(self.color[2]))
        bias_x = rd.randint(-self.width/2, self.width/2+1)
        bias_y = rd.randint(-self.height/2, self.height/2+1)
        self.x1 = max(0, int(joint[0] - self.width / 2 + bias_x))
        self.y1 = max(0, int(joint[1] - self.height / 2 + bias_y))
        self.x2 = min(ori_img.shape[1], int(joint[0] + self.width / 2 + bias_x))
        self.y2 = min(ori_img.shape[0], int(joint[1] + self.height / 2 + bias_y))

    def show(self):
        print(self.x1, ", ", self.y1)
        print(self.x2, ", ", self.y2)
        print(self.width, ", ", self.height)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def get_person_box(im, person_object_list=np.array([])):
    h, w = im.shape[:2]
    # print(person_object_list[1:])
    b = person_object_list[1:]  # * [w, h, w, h]
    # b[2:] = b[2:] * 1.2 + 3  # pad??
    box_w, box_h = b[2:]
    # print(box_w, box_h)
    # print(b)
    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)
    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
    # cv2.rectangle(im, b, (255,0,0), -1)
    # cv2.circle(im, (int(b[0]), int(b[1])), 5, (255, 0, 0), 2)
    # cv2.circle(im, (int(b[2]), int(b[3])), 5, (255, 0, 0), 2)
    # cv2.imwrite("/nvme/duhanwen/kapao-master/data/datasets/coco/masked_images/000000008844_box.jpg", im)
    # cv2.imwrite("/nvme/duhanwen/kapao-master/000000008844.jpg", im)
    return b, box_w, box_h


def draw_rectangles(image, joints):
    mask_of_current_image = []
    for i, current_joint in enumerate(joints):
        person_object = joints[0]
        for k in range(i, -1, -1):
            if joints[k, 0] == 0:
                person_object = joints[k]
                break

        _, box_w, box_h = get_person_box(image, person_object)
        args = RandomArguments(joint=current_joint[1:], ori_img=image, box_size=max(box_w, box_h))
        cutout_ratio = 10/np.shape(joints)[0]
        color = image[args.color_joint]
        color = (int(color[0]), int(color[1]), int(color[2]))
        # args.show()
        prob = rd.random()
        if (prob > 1 - cutout_ratio) and (current_joint[0] > 4):
            cv2.rectangle(image, (args.x1, args.y1), (args.x2, args.y2), color, -1)
            current_mask = [args.x1, args.y1, args.x2, args.y2]
            mask_of_current_image.append(current_mask)
            # st()
        # cv2.circle(image, (int(current_joint[0]), int(current_joint[1])), 5, (0, 0, 255), 2)  # 画出关节点
    return image,  mask_of_current_image


def get_image_and_joints(cache, image_label=""):
    # st()
    ori_img = cv2.imread("/nvme/duhanwen/kapao-master/" + image_label)  # 读取图片
    cache_joints = cache[image_label][0][:, :5]  # 读取joints坐标们
    # st()
    img_size = cache[image_label][1]  # 读取图片尺寸
    for i in range(len(cache_joints)):
        cache_joints[i,1] = cache_joints[i, 1] * img_size[0]
        cache_joints[i,2] = cache_joints[i, 2] * img_size[1]
        cache_joints[i, 3] = cache_joints[i, 3] * img_size[0]
        cache_joints[i, 4] = cache_joints[i, 4] * img_size[1]
    # cache_joints[i] = cache_joints[i]*np.array([img_size[0],img_size[1],img_size[0],img_size[1]])
    return ori_img, cache_joints  # y, x


def whether_mask(target_xywh, mask_area, img):
    xywh = target_xywh * np.array([img.size[0], img.size[1], img.size[0], img.size[1]])
    for _, area in enumerate(mask_area):
        if mask_area[0] < xywh[0] < mask_area[2] and mask_area[1] < xywh[1] < mask_area[3]:
            return True
    return False


def mask_imgs(img, img_label):
    cache_path = "/nvme/duhanwen/kapao-master/data/datasets/coco/kp_labels/img_txt/train2017.cache"
    cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
    current_image, current_person_joints = get_image_and_joints(cache, img_label)

    assert img == current_image

    if len(current_person_joints):
        print(len(current_person_joints))
        drawn_image, mask_area = draw_rectangles(img, current_person_joints)
    return drawn_image, mask_area
