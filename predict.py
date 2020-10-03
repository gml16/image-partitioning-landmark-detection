import argparse

import torch
import numpy as np

from model import UNet
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
    x, _ = get_data(file_paths, None, 0)
    
    unet = UNet(in_dim=1, out_dim=6, num_filters=4)
    unet.load_state_dict(torch.load(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet.to(device)

    # Dummy data
    # from data_reader import create_label_single_landmark
    # fake = np.zeros((128, 128, 128))
    # y = create_label_single_landmark(fake, [127, 127, 127])
    # y = np.expand_dims(y, axis=0)


    tensor_x = torch.Tensor(x).to(device)
    for i, tx in enumerate(tensor_x):
      tx = tx.unsqueeze(0)
      y = unet(tx).cpu()
      print("Y", y.shape)
      landmark = predict(y)
      print(f"landmark {i}", landmark)

if __name__ == "__main__":
    main()
