import cv2
import numpy as np

from enum import Enum, auto


g_win_size = (416, 768)
g_win_name = 'LabelPose v1.0 by Inzapp'


class Limb(Enum):
    HEAD = 0
    NECK = auto()
    RIGHT_SHOULDER = auto()
    RIGHT_ELBOW = auto()
    RIGHT_WRIST = auto()
    LEFT_SHOULDER = auto()
    LEFT_ELBOW = auto()
    LEFT_WRIST = auto()
    RIGHT_HIP = auto()
    RIGHT_KNEE = auto()
    RIGHT_ANKLE = auto()
    LEFT_HIP = auto()
    LEFT_KNEE = auto()
    LEFT_ANKLE = auto()
    CHEST = auto()
    BACK = auto()


class LabelPose:
    def __init__(self):
        self.image_paths = self.init_image_paths()
        if len(self.image_paths) == 0:
            print('No image files in path.')
            exit(0)
        self.raw = None
        self.guide_img = None
        self.show_skeleton = True
        self.cur_image_path = ''
        self.cur_label_path = ''
        self.max_limb_size = 16
        self.limb_index = 0
        self.cur_label = self.reset_label()
        self.guide_label = self.reset_label()

    def init_image_paths(self):
        import natsort
        import tkinter as tk
        from glob import glob
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        image_path = filedialog.askdirectory()
        print(image_path)
        image_paths = natsort.natsorted(glob(f'{image_path}/*.jpg'))
        for i in range(len(image_paths)):
            image_paths[i] = image_paths[i].replace('\\', '/')
        return image_paths

    def resize(self, img, size):
        img_height, img_width = img.shape[:2]
        if img_width > size[0] or img_height > size[1]:
            return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        else:
            return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

    def reset_label(self):
        return [[0, 0, 0] for _ in range(self.max_limb_size)]  # use flag for index 0

    def circle(self, img, x, y):
        img = cv2.circle(img, (x, y), 8, (128, 255, 128), thickness=2, lineType=cv2.LINE_AA)
        img = cv2.circle(img, (x, y), 3, (32, 32, 192), thickness=-1, lineType=cv2.LINE_AA)
        return img

    def line_if_valid(self, img, p1, p2):
        if p1[0] == 1 and p2[0] == 1:
            img = cv2.line(img, (p1[1], p1[2]), (p2[1], p2[2]), (64, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        return img

    def get_limb_guide_img(self):
        img = self.guide_img.copy()
        img = self.circle(img, self.guide_label[self.limb_index][1], self.guide_label[self.limb_index][2])
        return img

    def update(self):
        global g_win_name
        img = self.raw.copy()
        if self.show_skeleton:
            img = self.line_if_valid(img, self.cur_label[Limb.HEAD.value], self.cur_label[Limb.NECK.value])  # head to neck

            img = self.line_if_valid(img, self.cur_label[Limb.NECK.value], self.cur_label[Limb.RIGHT_SHOULDER.value])  # neck to right shoulder
            img = self.line_if_valid(img, self.cur_label[Limb.RIGHT_SHOULDER.value], self.cur_label[Limb.RIGHT_ELBOW.value])  # right shoulder to right elbow
            img = self.line_if_valid(img, self.cur_label[Limb.RIGHT_ELBOW.value], self.cur_label[Limb.RIGHT_WRIST.value])  # right elbow to right wrist

            img = self.line_if_valid(img, self.cur_label[Limb.NECK.value], self.cur_label[Limb.LEFT_SHOULDER.value])  # neck to left shoulder
            img = self.line_if_valid(img, self.cur_label[Limb.LEFT_SHOULDER.value], self.cur_label[Limb.LEFT_ELBOW.value])  # left shoulder to left elbow
            img = self.line_if_valid(img, self.cur_label[Limb.LEFT_ELBOW.value], self.cur_label[Limb.LEFT_WRIST.value])  # left elbow to left wrist

            img = self.line_if_valid(img, self.cur_label[Limb.RIGHT_HIP.value], self.cur_label[Limb.RIGHT_KNEE.value])  # right hip to right knee
            img = self.line_if_valid(img, self.cur_label[Limb.RIGHT_KNEE.value], self.cur_label[Limb.RIGHT_ANKLE.value])  # right knee to right anlke

            img = self.line_if_valid(img, self.cur_label[Limb.LEFT_HIP.value], self.cur_label[Limb.LEFT_KNEE.value])  # right hip to right knee
            img = self.line_if_valid(img, self.cur_label[Limb.LEFT_KNEE.value], self.cur_label[Limb.LEFT_ANKLE.value])  # right knee to right anlke

            img = self.line_if_valid(img, self.cur_label[Limb.NECK.value], self.cur_label[Limb.CHEST.value])  # neck to chest
            img = self.line_if_valid(img, self.cur_label[Limb.CHEST.value], self.cur_label[Limb.RIGHT_HIP.value])  # chest to right hip
            img = self.line_if_valid(img, self.cur_label[Limb.CHEST.value], self.cur_label[Limb.LEFT_HIP.value])  # chest to left hip

            img = self.line_if_valid(img, self.cur_label[Limb.NECK.value], self.cur_label[Limb.BACK.value])  # neck to back
            img = self.line_if_valid(img, self.cur_label[Limb.BACK.value], self.cur_label[Limb.RIGHT_HIP.value])  # back to right hip
            img = self.line_if_valid(img, self.cur_label[Limb.BACK.value], self.cur_label[Limb.LEFT_HIP.value])  # back to left hip
        for use, x, y in self.cur_label:
            if use == Limb.NECK.value:
                img = self.circle(img, x, y)
        img = np.append(img, self.get_limb_guide_img(), axis=1)
        cv2.imshow(g_win_name, img)

    def save_label(self):
        global g_win_size
        label_content = ''
        for use, x, y in self.cur_label:
            x = x / float(g_win_size[0] - 1)
            y = y / float(g_win_size[1] - 1)
            label_content += f'{use:.1f} {x:.6f} {y:.6f}\n'
        with open(self.cur_label_path, 'wt') as f:
            f.writelines(label_content)

    def load_label_if_exists(self, guide=False):
        import os
        global g_win_size
        label_path = './guide.txt' if guide else self.cur_label_path
        if os.path.exists(label_path) and os.path.isfile(label_path):
            with open(label_path, 'rt') as f:
                lines = f.readlines()
            for i in range(len(lines)):
                use, x, y = list(map(float, lines[i].split()))
                if guide:
                    self.guide_label[i] = [int(use), int(x * float(g_win_size[0])), int(y * float(g_win_size[1]))]
                else:
                    self.cur_label[i] = [int(use), int(x * float(g_win_size[0])), int(y * float(g_win_size[1]))]
        if not guide:
            self.update()

    def run(self):
        global g_win_name, g_win_size
        index = 0
        cv2.namedWindow(g_win_name)
        cv2.setMouseCallback(g_win_name, self.mouse_callback)
        self.guide_img = self.resize(cv2.imdecode(np.fromfile('./guide.jpg', dtype=np.uint8), cv2.IMREAD_COLOR), g_win_size)
        self.load_label_if_exists(guide=True)
        while True:
            self.cur_image_path = self.image_paths[index]
            print(f'[{index}] : {self.cur_image_path}')
            self.cur_label_path = f'{self.cur_image_path[:-4]}.txt'
            self.raw = self.resize(cv2.imdecode(np.fromfile(self.cur_image_path, dtype=np.uint8), cv2.IMREAD_COLOR), g_win_size)
            self.load_label_if_exists()
            self.update()
            while True:
                res = cv2.waitKey(0)
                if res == ord('d'):  # go to next if input key was 'd'
                    self.save_label()
                    if index == len(self.image_paths) - 1:
                        print('Current image is last image')
                    else:
                        self.limb_index = 0
                        self.cur_label = self.reset_label()
                        index += 1
                        break
                elif res == ord('a'):  # go to previous image if input key was 'a'
                    self.save_label()
                    if index == 0:
                        print('Current image is first image')
                    else:
                        self.limb_index = 0
                        self.cur_label = self.reset_label()
                        index -= 1
                        break
                elif res == ord('w'):  # toggle show skeleton
                    self.show_skeleton = not self.show_skeleton
                    break
                elif res == ord('e'):  # go to next limb
                    self.limb_index += 1
                    if self.limb_index == self.max_limb_size:
                        self.limb_index = 0
                    print(f'limb index : {self.limb_index}')
                    break
                elif res == ord('q'):  # go to prev limb
                    self.limb_index -= 1
                    if self.limb_index == -1:
                        self.limb_index = self.max_limb_size - 1
                    print(f'limb index : {self.limb_index}')
                    break
                elif res == 27:  # exit if input key was ESC
                    self.save_label()
                    exit(0)

    def mouse_callback(self, event, x, y, flag, _):
        if event == 0 and flag == 0:  # no click mouse moving
            pass
        elif event == 4 and flag == 0:  # left click end
            x, y = x, y  # get img position
            self.cur_label[self.limb_index] = [1, x, y]
            self.update()
            self.save_label()
        elif event == 5 and flag == 0:  # right click end
            self.cur_label[self.limb_index] = [0, 0, 0]
            self.update()
            self.save_label()


if __name__ == '__main__':
    LabelPose().run()
