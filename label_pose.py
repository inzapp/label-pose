import cv2
import numpy as np


g_win_size = (768, 768)
g_win_name = 'LabelPose v1.0 by Inzapp'


class LabelPose:
    def __init__(self):
        self.image_paths = self.init_image_paths()
        if len(self.image_paths) == 0:
            print('No image files in path.')
            exit(0)
        self.raw = None
        self.img = None
        self.guide_img = None
        self.cur_image_path = ''
        self.cur_label_path = ''
        self.max_limb_size = 16
        self.limb_index = 0
        self.cur_label = self.reset_label()

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
        # 0  : Head
        # 1  : Neck
        # 2  : Right Shoulder
        # 3  : Right Elbow
        # 4  : Right Wrist
        # 5  : Left Shoulder
        # 6  : Left Elbow
        # 7  : Left Wrist
        # 8  : Right Hip
        # 9  : Right Knee
        # 10 : Right Ankle
        # 11 : Left Hip
        # 12 : Left Knee
        # 13 : Left Ankle
        # 14 : Chest
        # 15 : Back
        return [[0, 0, 0] for _ in range(self.max_limb_size)]  # use flag for index 0

    def circle(self, img, x, y):
       img = cv2.circle(img, (x, y), 8, (128, 255, 128), thickness=2, lineType=cv2.LINE_AA)
       img = cv2.circle(img, (x, y), 3, (32, 32, 192), thickness=-1, lineType=cv2.LINE_AA)
       return img

    def update(self):
        img = self.raw.copy()
        for use, x, y in self.cur_label:
            if use == 1:
                self.img = self.circle(img, x, y)
        return img

    def save_label(self):
        global g_win_size
        label_content = ''
        for use, x, y in self.cur_label:
            x = x / float(g_win_size[0])
            y = y / float(g_win_size[1])
            label_content += f'{use:.1f} {x:.6f} {y:.6f}\n'
        with open(self.cur_label_path, 'wt') as f:
            f.writelines(label_content)

    def run(self):
        global g_win_name, g_win_size
        index = 0
        cv2.namedWindow(g_win_name)
        cv2.setMouseCallback(g_win_name, self.mouse_callback)
        while True:
            self.cur_image_path = self.image_paths[index]
            print(f'[{index}] : {self.cur_image_path}')
            self.cur_label_path = f'{self.cur_image_path[:-4]}.txt'
            self.raw = cv2.imdecode(np.fromfile(self.cur_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.raw = self.resize(self.raw, g_win_size)
            self.img = self.raw.copy()
            cv2.imshow(g_win_name, self.img)
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
                elif res == 27:  # exit if input key was ESC
                    self.save_label()
                    exit(0)

    def mouse_callback(self, event, x, y, flag, _):
        if event == 0 and flag == 0:  # no click mouse moving
            cv2.imshow(g_win_name, self.img)
        elif event == 4 and flag == 0:  # left click end
            x, y = x, y  # get img position
            self.cur_label[self.limb_index] = [1, x, y]
            self.img = self.update()
            self.limb_index += 1
            if self.limb_index == self.max_limb_size:
                self.limb_index = 0
        elif event == 5 and flag == 0:  # right click end
            self.cur_label[self.limb_index] = [0, 0, 0]
            self.img = self.update()


if __name__ == '__main__':
    LabelPose().run()
