import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='/Disk2/nl/datasets/celeba-256/', type=str,
                    help='path to the dataset')
parser.add_argument('--output', default='./datasets/CelebA_HQ_256/CelebA_HQ_256_train.flist', type=str, help='path to the file list')
parser.add_argument('--output1', default='./datasets/CelebA_HQ_256/CelebA_HQ_256_val.flist', type=str, help='path to the file list')
args = parser.parse_args()

if not os.path.exists('./datasets/CelebA_HQ_256/'):
    os.makedirs('./datasets/CelebA_HQ_256/')

ext = {'.jpg', '.png', '.JPG'}

images = []
images1 = []
# for root, dirs, files in os.walk(args.path):
#     print('loading ' + root)
#     for file in files:
#         if os.path.splitext(file)[1] in ext:
#             images.append(os.path.join(root, file))
#
# images = sorted(images)
# np.savetxt(args.output, images, fmt='%s')


for root, dirs, files in os.walk(args.path):
    print('loading ' + root)
    i = 0
    files = sorted(files)
    for file in files:
        if i < 28000:
            if os.path.splitext(file)[1] in ext:
                images.append(os.path.join(root, file))
        else:
            if os.path.splitext(file)[1] in ext:
                images1.append(os.path.join(root, file))
        i = i + 1

images = sorted(images)
images1 = sorted(images1)
np.savetxt(args.output, images, fmt='%s')
np.savetxt(args.output1, images1, fmt='%s')