import os

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

def get_data(file_paths, landmark_paths, landmark_wanted, cache_path=None, separator = " "):
    with open(os.path.normpath(file_paths), 'r') as f:
        file_paths = f.read().splitlines()
    if landmark_paths is not None:
        with open(os.path.normpath(landmark_paths), 'r') as f:
            landmark_paths = f.read().splitlines()
    images = []
    labels = []
    for i, f in enumerate(file_paths):
        print(f"{i+1}/{len(file_paths)} - {f}")
        img = nib.load(f).get_fdata()
        scale = [128/d for d in img.shape]
        img = zoom(img, scale)
        if landmark_paths is not None:
            with open(os.path.normpath(landmark_paths[i]), 'r') as l:
                landmark = l.read().splitlines()
            landmark = np.array([float(x) for x in landmark[landmark_wanted].split(separator)])
            landmark = [landmark[i] * scale[i] for i in range(len(landmark))]
            label = create_label_single_landmark(img, landmark)
            labels.append(label)
        img = np.expand_dims(img, 0)
        images.append(img)
    return images, labels

# https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
def onehot_initialization(a):
    ncols = 6
    out = np.zeros((ncols, a.size), dtype=np.uint8)
    out[a.ravel(), np.arange(a.size)] = 1
    out.shape = (ncols,) + a.shape
    return out

def create_label_single_landmark(img, landmark):
    xp = np.arange(img.shape[0], dtype=np.float32) - landmark[0]
    xp = np.stack((xp,) * img.shape[1], axis = 1)
    xp = np.stack((xp,) * img.shape[2], axis = 2)

    xm = np.negative(xp)


    yp = np.arange(img.shape[1], dtype=np.float32) - landmark[1]
    yp = np.stack((yp,) * img.shape[0], axis = 0)
    yp = np.stack((yp,) * img.shape[2], axis = 2)

    ym = np.negative(yp)

    zp = np.arange(img.shape[2], dtype=np.float32) - landmark[2]
    zp = np.stack((zp,) * img.shape[0], axis = 0)
    zp = np.stack((zp,) * img.shape[1], axis = 1)

    zm = np.negative(zp)
    
    mask = np.stack([xp, xm, yp, ym, zp, zm])
    highest = np.argmax(mask, axis=0)
    
    # unique, counts = np.unique(highest, return_counts=True)
    # print(dict(zip(unique, counts)))

    label = highest  # onehot_initialization_v2(highest)

    return label


def create_label_multiple_landmark(img, landmarks):

    def onehot_initialization_v2(a, sign):
        ncols = 3
        out = np.zeros((a.size, ncols), dtype=np.uint8)
        sign[np.arange(a.size)]
        out[np.arange(a.size),a.ravel()] = 1
        out.shape = a.shape + (ncols,)
        return out

    print("img shape", img.shape)
    # label = np.zeros((*img.shape, len(landmarks), 3), dtype=np.uint8)

    x = (np.dstack((np.arange(img.shape[0]),) * landmarks.shape[0]) - landmarks[:, 0]).squeeze()
    x = np.stack((x,) * img.shape[1], axis = 1)
    x = np.stack((x,) * img.shape[2], axis = 2)
    x_abs = np.absolute(x, dtype=np.float32)
    x = np.sign(x, dtype=np.float32)
    print("finished x", x.shape)

    y = (np.dstack((np.arange(img.shape[0]),) * landmarks.shape[0]) - landmarks[:, 1]).squeeze()
    y = np.stack((y,) * img.shape[1], axis = 1)
    y = np.stack((y,) * img.shape[2], axis = 2)
    y_abs = np.absolute(y, dtype=np.float32)
    y = np.sign(y, dtype=np.float32)    
    print("finished y", y.shape)

    # z = (np.dstack((np.arange(img.shape[0]),) * landmarks.shape[0]) - landmarks[:, 0]).squeeze()
    # z = np.stack((z,) * img.shape[1], axis = 1)
    # z = np.stack((z,) * img.shape[2], axis = 2)
    # z_abs = np.absolute(z, dtype=np.float32)
    # z = np.sign(z, dtype=np.float32)  
    # print("finished z", z.shape)
    
    
    mask = np.stack([x_abs, y_abs]) #, z_abs])
    mask_sign = np.stack([x, y]) # , z])
    print("mask.shape", mask.shape)
    highest = np.argmax(mask, axis=0)
    print("highest.shape", highest.shape)

    label = onehot_initialization_v2(highest, mask_sign)
    print("label.shape", label.shape)
    print("label", label)

    return label
