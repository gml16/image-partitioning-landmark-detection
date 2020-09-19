import argparse

import torch
import numpy as np

from data_reader import get_data
from data_reader import onehot_initialization


def predict(action_map):
    def aggregate(v, dir):
        v = (v==dir)
        cumsum = np.where(v!=0,np.cumsum(v, axis=1+int(dir/2)),v)
        tot = np.count_nonzero(v, axis=1+int(dir/2))
        tot = np.expand_dims(tot, axis=1+int(dir/2))
        ag = 2*cumsum - tot
        if dir%2 == 0: 
            ag = np.negative(ag)
        return ag

    # aggregate
    v = sum([aggregate(action_map, i) for i in range(6)])
    pos = np.unravel_index(np.argmax(v, axis=None), v.shape)
    return pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_paths',
                        default="data/files.txt")
    parser.add_argument('--model_path')
    args = parser.parse_args()

    file_paths = args.file_paths
    model_path = args.model_path

    # Fakin data
    _, y = get_data("data/files.txt", "data/landmarks.txt", 0)
    y = np.array(y)

    # Dummy data
    # from data_reader import create_label_single_landmark
    # fake = np.zeros((128, 128, 128))
    # y = create_label_single_landmark(fake, [127, 127, 127])
    # y = np.expand_dims(y, axis=0)

    tensor_y = torch.Tensor(y)
    landmark = predict(tensor_y)
    print("lanfmakr", landmark)

if __name__ == "__main__":
    main()
