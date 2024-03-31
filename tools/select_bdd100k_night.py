import shutil
import glob
import os
night_img = []
night_label = []
data_dir = r'data/bdd10k/'
dst_img_dir = r'data/bdd10k/images/10k/test_night'
dst_label_dir = r'data/bdd10k/labels/sem_seg/masks/test_night'
with open('tools/images_trainval_night_correct_filenames.txt') as f:
    night_img = f.read().splitlines()
with open('tools/gt_trainval_night_correct_filenames.txt') as f:
    night_label = f.read().splitlines()
for f in night_img:
    full_path = data_dir + f
    os.makedirs(dst_img_dir, exist_ok=True)
    shutil.copy(full_path, dst_img_dir)
for f in night_label:
    full_path = data_dir + f
    os.makedirs(dst_label_dir, exist_ok=True)
    shutil.copy(full_path, dst_label_dir)
