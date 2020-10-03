import argparse

import torch
import numpy as np

from model import UNet
from data_reader import get_data, onehot_initialization, create_label_single_landmark
from visualisation import view_image


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

def predict_true_label():
    fake = np.zeros((128, 128, 128))
    y = create_label_single_landmark(fake, [50, 60, 30])
    y = np.expand_dims(y, axis=0)
    view_image(y)


def predict_using_model(file_paths, landmark, separator, model_path):
    x, y = get_data(file_paths, None, landmark_wanted=landmark, separator=separator)
    view_image(x)

    unet = UNet(in_dim=1, out_dim=6, num_filters=4)
    unet.load_state_dict(torch.load(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using as device", device)
    unet.to(device)

    tensor_x = torch.Tensor(x).to(device)
    for i, tx in enumerate(tensor_x):
        tx = tx.unsqueeze(0)
        view_image(tx.cpu())
        y = unet(tx).cpu()

        landmark = predict(y)
        print(f"landmark {i}", landmark)

        action_map = y.detach().numpy().squeeze()
        action_map = np.argmax(action_map, axis=0)
        view_image(action_map)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_paths',
                        default="data/files.txt")
    parser.add_argument('--model_path')
    parser.add_argument('--separator', default=",")
    parser.add_argument('--landmark', type=int, default=0)
    args = parser.parse_args()

    predict_true_label()
    #predict_using_model(args.file_paths, args.landmark, args.separator, args.model_path)
    

if __name__ == "__main__":
    main()
