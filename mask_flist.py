import os
import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./csy/gt/', type=str,
                    help='path to the dataset')
parser.add_argument('--output1', default='./datasets/mask/train_celeba_csy.flist', type=str, help='path to the file list')
parser.add_argument('--output2', default='./datasets/mask/train_mask2.flist', type=str, help='path to the file list')
parser.add_argument('--output3', default='./datasets/mask/train_mask3.flist', type=str, help='path to the file list')
parser.add_argument('--output4', default='./datasets/mask/train_mask4.flist', type=str, help='path to the file list')
parser.add_argument('--output5', default='./datasets/mask/train_mask5.flist', type=str, help='path to the file list')
parser.add_argument('--output6', default='./datasets/mask/train_mask6.flist', type=str, help='path to the file list')
args = parser.parse_args()

ext = {'.jpg', '.png', '.JPG'}

path_0_1 = '/Disk2/nl/datasets/celeba_hq_256_train/'#
path_1_2 = '/Disk2/nl/datasets/celeba_hq_256_test/'#
path_2_3 = '/Disk2/nl/datasets/irr_Mask/Mask_2_3/'#
path_3_4 = '/Disk2/nl/datasets/irr_Mask/Mask_3_4/'#
path_4_5 = '/Disk2/nl/datasets/irr_Mask/Mask_4_5/'#
path_5_6 = '/Disk2/nl/datasets/irr_Mask/Mask_5_6/'#

if not os.path.exists('./datasets/mask'):
    os.mkdir('./datasets/mask')

if not os.path.exists(path_0_1):
    os.makedirs(path_0_1)
if not os.path.exists(path_1_2):
    os.makedirs(path_1_2)
if not os.path.exists(path_2_3):
    os.makedirs(path_2_3)
if not os.path.exists(path_3_4):
    os.makedirs(path_3_4)
if not os.path.exists(path_4_5):
    os.makedirs(path_4_5)
if not os.path.exists(path_5_6):
    os.makedirs(path_5_6)

images = []
images1 = []
images2 = []
images3 = []
images4 = []
images5 = []
images6 = []

for root, dirs, files in os.walk(args.path):
    print('loading ' + root)
    files = sorted(files)
    for file in files:
        if os.path.splitext(file)[1] in ext:
            x = cv2.imread(os.path.join(root, file))
            images.append(os.path.join(root, file))

# for root, dirs, files in os.walk(args.path):
#     print('loading ' + root)
#     i = 0
#     files = sorted(files)
#     for file in files:
#         if os.path.splitext(file)[1] in ext:
#             x = cv2.imread(os.path.join(root, file))
#             if i < 28000:
#                 cv2.imwrite(os.path.join(path_0_1, file), x)
#             else:
#                 cv2.imwrite(os.path.join(path_1_2, file), x)
#         i = i + 1


# for root, dirs, files in os.walk(args.path):
#     print('loading ' + root)
#     for file in files:
#         if os.path.splitext(file)[1] in ext:
#             temp = cv2.imread(os.path.join(root, file))
#             x = temp
#             th, temp = cv2.threshold(src=temp, thresh=127, maxval=1, type=cv2.THRESH_BINARY)
#             bili = np.sum(temp) / np.sum(np.ones_like(temp))
#             if 0 <= bili <= 0.1:
#                 images1.append(os.path.join(root, file))
#                 cv2.imwrite(os.path.join(path_0_1, file), x)
#             elif 0.1 < bili <= 0.2:
#                 images2.append(os.path.join(root, file))
#                 cv2.imwrite(os.path.join(path_1_2, file), x)
#             elif 0.2 < bili <= 0.3:
#                 images3.append(os.path.join(root, file))
#                 cv2.imwrite(os.path.join(path_2_3, file), x)
#             elif 0.3 < bili <= 0.4:
#                 images4.append(os.path.join(root, file))
#                 cv2.imwrite(os.path.join(path_3_4, file), x)
#             elif 0.4 < bili <= 0.5:
#                 images5.append(os.path.join(root, file))
#                 cv2.imwrite(os.path.join(path_4_5, file), x)
#             elif 0.5 < bili <= 0.6:
#                 images6.append(os.path.join(root, file))
#                 cv2.imwrite(os.path.join(path_5_6, file), x)
#             else:
#                 print('file')
#
# images1 = sorted(images1)
# images2 = sorted(images2)
# images3 = sorted(images3)
# images4 = sorted(images4)
# images5 = sorted(images5)
# images6 = sorted(images6)
#
np.savetxt(args.output1, images, fmt='%s')
# np.savetxt(args.output2, images2, fmt='%s')
# np.savetxt(args.output3, images3, fmt='%s')
# np.savetxt(args.output4, images4, fmt='%s')
# np.savetxt(args.output5, images5, fmt='%s')
# np.savetxt(args.output6, images6, fmt='%s')
